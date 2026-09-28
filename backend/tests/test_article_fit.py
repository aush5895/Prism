"""Article fit (pipeline/article_fit.py): does the supplied article cover what the
customer described? Not graded; carried in meta, debug and guided mode.

The contract asserted here: a "covered" verdict needs a quote that is really in the
article; "unknown" is never presented as a verdict; the graded response is unchanged.
"""
from __future__ import annotations

import json
import re

import pytest

from app.contracts import TroubleshootRequest
from app.ir import ComplaintIssue, Extraction
from app.llm.replay import ReplayProvider
from app.main import run_pipeline
from app.pipeline import article_fit
from app.pipeline.ground import normalize_siis

from conftest import FIXTURES

ARTICLE = ("Touchscreen issues. Remove the screen protector if it is peeling. "
           "Tap Settings, then Display, then Touch sensitivity to enable it. "
           "Restart the device if the problem continues.")


def _extraction(*issues):
    return Extraction(goal_topic="Touch", title="Touch issues",
                      complaint_issues=[ComplaintIssue(**i) for i in issues])


def test_a_covered_claim_with_a_verbatim_quote_is_believed():
    fit = article_fit.assess(_extraction(
        {"issue": "touch is laggy", "covered": True,
         "evidence": "Tap Settings, then Display, then Touch sensitivity to enable it."}),
        ARTICLE)
    assert fit["fit"] == "full" and fit["covered"] == 1
    issue = fit["issues"][0]
    assert issue["quote_found"] is True and issue["match"] == "verbatim"
    assert ARTICLE[issue["span"][0]:issue["span"][1]] == issue["evidence"]


def test_a_covered_claim_with_an_invented_quote_is_not_believed():
    """The model is a parser, not an authority. Saying 'covered' is not enough."""
    fit = article_fit.assess(_extraction(
        {"issue": "screen flashes when charging", "covered": True,
         "evidence": "Replace the charging port to stop the screen flashing."}), ARTICLE)
    assert fit["fit"] == "none" and fit["covered"] == 0
    assert fit["issues"][0]["quote_found"] is False
    assert fit["unverified_claims"] == 1
    assert fit["issues"][0]["evidence"] == "", "an unverified quote is never shown"


def test_a_reformatted_quote_is_still_found():
    fit = article_fit.assess(_extraction(
        {"issue": "touch is laggy", "covered": True,
         "evidence": "Tap  Settings, then Display, then Touch sensitivity to enable it"}),
        ARTICLE)
    assert fit["issues"][0]["covered"] is True
    assert fit["issues"][0]["match"] in ("verbatim", "located")


def test_no_reported_problems_is_unknown_not_covered():
    fit = article_fit.assess(_extraction(), ARTICLE)
    assert fit["fit"] == "unknown" and fit["total"] == 0
    assert article_fit.uncovered(fit) == []


def test_partial_coverage_names_what_is_missing():
    fit = article_fit.assess(_extraction(
        {"issue": "touch is laggy", "covered": True,
         "evidence": "Tap Settings, then Display, then Touch sensitivity to enable it."},
        {"issue": "screen cracked at the fold", "covered": False, "evidence": ""}), ARTICLE)
    assert fit["fit"] == "partial" and (fit["covered"], fit["total"]) == (1, 2)
    assert article_fit.uncovered(fit) == ["screen cracked at the fold"]


def test_problem_list_is_capped_and_cleaned():
    many = [{"issue": f"problem {i} see https://x.example/y", "covered": False, "evidence": ""}
            for i in range(9)]
    fit = article_fit.assess(_extraction(*many), ARTICLE)
    assert fit["total"] == 4
    assert all("http" not in i["issue"] for i in fit["issues"])


# ----------------------------------------------------------------- through the pipeline
class _ReplayWithIssues(ReplayProvider):
    """The recorded row_21 extraction plus complaint issues, built from the article at
    test time. Only complaint_issues differs from the recording."""

    def __init__(self, issues):
        super().__init__(FIXTURES / "extraction_row21.json")
        self._issues = issues

    def extract(self, query, evidence):
        base = super().extract(query, evidence)
        return base.model_copy(update={"complaint_issues": [ComplaintIssue(**i)
                                                             for i in self._issues]})


@pytest.fixture
def row21_request(row21):
    return TroubleshootRequest(query=row21["original_query"],
                               siis_response=row21["siis_response"])


@pytest.fixture
def row21_sentence(row21):
    """A real sentence from row_21's article that mentions touch sensitivity."""
    text = normalize_siis(row21["siis_response"]).text
    return next(s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                if "sensitivity" in s.lower() and len(s) > 30)


def test_the_graded_response_is_identical_with_or_without_the_fit(row21_request,
                                                                  row21_sentence):
    """The fit is a sibling of `response`, never inside it (guide §4.2.4)."""
    plain = run_pipeline(row21_request,
                         provider=ReplayProvider(FIXTURES / "extraction_row21.json"),
                         use_cache=False)
    judged = run_pipeline(row21_request, provider=_ReplayWithIssues([
        {"issue": "touch is laggy", "covered": True, "evidence": row21_sentence},
        {"issue": "phone is overheating", "covered": False, "evidence": ""}]),
        use_cache=False)
    assert judged.response == plain.response
    assert "article_fit" not in json.dumps(judged.response)
    assert judged.meta.article_fit["fit"] == "partial"
    assert plain.meta.article_fit["fit"] == "unknown", "an older recording judges nothing"


def test_a_cache_hit_carries_the_fit_from_the_cold_run(row21_request, row21_sentence):
    provider = _ReplayWithIssues([{"issue": "touch is laggy", "covered": True,
                                   "evidence": row21_sentence}])
    cold = run_pipeline(row21_request, provider=provider, use_cache=True)
    warm, debug = None, {}
    warm = run_pipeline(row21_request, provider=provider, debug_sink=debug, use_cache=True)
    assert warm.meta.cache_hit is True
    assert warm.meta.article_fit["fit"] == cold.meta.article_fit["fit"] == "full"
    assert warm.meta.article_fit["from_cache"] is True
    assert debug["article_fit"]["fit"] == "full"


def test_the_prompt_asks_for_coverage_with_quoted_evidence():
    from app.llm.base import EXTRACTION_JSON_SCHEMA, SYSTEM_PROMPT
    assert "complaint_issues" in EXTRACTION_JSON_SCHEMA["required"]
    item = EXTRACTION_JSON_SCHEMA["properties"]["complaint_issues"]["items"]
    assert set(item["required"]) == {"issue", "covered", "evidence"}
    assert "character for character" in SYSTEM_PROMPT


# ----------------------------------------------------------------- evaluation
def test_the_fit_evaluation_never_scores_unflagged_rows_as_known_correct(repo_root):
    """Phase 0 listed only the pairings it judged wrong. An unlisted row is reported as
    'not flagged', and a row the provider could not judge is 'unknown', not a miss."""
    import sys
    sys.path.insert(0, str(repo_root))
    from evaluation.run_eval import _render_article_fit, run_article_fit_eval

    def row(rid, fit, covered, total):
        return {"id": rid, "article_fit": {"fit": fit, "covered": covered, "total": total,
                                           "unverified_claims": 0, "issues": []}}

    per_row = [row("row_1", "none", 0, 1), row("row_8", "full", 1, 1),
               row("row_2", "partial", 1, 2), row("row_3", "unknown", 0, 0)]
    result = run_article_fit_eval(per_row)
    assert result["judged_rows"] == 3 and result["unknown_rows"] == 1
    assert (result["phase0_wrong_judged"], result["phase0_wrong_flagged"]) == (2, 1)
    assert result["unflagged_partial"] == 1
    lines = []
    _render_article_fit(lines.append, result, "test:fixture")
    text = "\n".join(lines)
    assert "not \"known correct\"" in text and "4 rows is a small sample" in text
