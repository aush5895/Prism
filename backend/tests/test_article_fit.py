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
    assert fit["covered"] == 0 and fit["issues"][0]["quote_found"] is False
    assert fit["issues"][0]["status"] == "unverified" and fit["unverified_claims"] == 1
    assert fit["issues"][0]["evidence"] == "", "an unverified quote is never shown"
    # REGRESSION, found on review: this used to be fit "none" and listed as "not covered".
    # A misquote is not evidence the article misses the problem.
    assert fit["fit"] == "unknown"
    assert article_fit.uncovered(fit) == []
    assert article_fit.unverified(fit) == ["screen flashes when charging"]


def test_a_reformatted_quote_is_still_found():
    fit = article_fit.assess(_extraction(
        {"issue": "touch is laggy", "covered": True,
         "evidence": "Tap  Settings, then Display, then Touch sensitivity to enable it"}),
        ARTICLE)
    assert fit["issues"][0]["covered"] is True
    assert fit["issues"][0]["match"] == "normalised"


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


def test_a_cache_hit_never_gives_one_customer_another_customers_fit(row21_request,
                                                                   row21_sentence):
    """REGRESSION, found on review: a hit replayed the cold run's fit as this customer's.
    Two complaints can share a plan and not their problems, and no model read the new
    one, so a reworded complaint gets no verdict. The same words (a refresh, or guided
    mode starting from the plan just shown) get the cold run's fit, marked as replayed.
    The debug view keeps the cold run's fit either way, for reference."""
    provider = _ReplayWithIssues([{"issue": "touch is laggy", "covered": True,
                                   "evidence": row21_sentence}])
    cold = run_pipeline(row21_request, provider=provider, use_cache=True)
    assert cold.meta.article_fit["fit"] == "full"

    same = run_pipeline(row21_request, provider=provider, use_cache=True)
    assert same.meta.cache_hit is True
    assert same.meta.article_fit["fit"] == "full" and same.meta.article_fit["from_cache"]

    reworded = TroubleshootRequest(query="Please help: " + row21_request.query,
                                   siis_response=row21_request.siis_response)
    debug = {}
    other = run_pipeline(reworded, provider=provider, debug_sink=debug, use_cache=True)
    assert other.meta.cache_hit is True
    assert other.meta.article_fit["fit"] == "unknown"
    assert other.meta.article_fit["reason"] == "cache_hit"
    assert debug["article_fit"]["fit"] == "full"
    assert debug["grounding_from"]["cold_run_query"] == row21_request.query


def test_the_prompt_asks_for_coverage_with_quoted_evidence():
    from app.llm.base import EXTRACTION_JSON_SCHEMA, SYSTEM_PROMPT
    assert "complaint_issues" in EXTRACTION_JSON_SCHEMA["required"]
    item = EXTRACTION_JSON_SCHEMA["properties"]["complaint_issues"]["items"]
    assert set(item["required"]) == {"issue", "covered", "evidence"}
    assert "character for character" in SYSTEM_PROMPT
    # REGRESSION, found on review: the block sat between two STRUCTURE bullets, splitting
    # query_variations off from the fields it belongs with.
    assert SYSTEM_PROMPT.index("- query_variations:") < SYSTEM_PROMPT.index("COMPLAINT COVERAGE")


# ----------------------------------------------------------------- what counts as a quote
@pytest.mark.parametrize("quote", [".", "the", "Settings", "Touch sensitivity",
                                   "then Display, then Touch sensitivity to enable it."])
def test_a_fragment_is_not_evidence(quote):
    """REGRESSION, found on review: '.', 'the' and 'Settings' passed as verbatim quotes,
    since each occurs in the article. Evidence must be a whole sentence."""
    fit = article_fit.assess(_extraction(
        {"issue": "touch is laggy", "covered": True, "evidence": quote}), ARTICLE)
    assert fit["issues"][0]["status"] == "unverified"


def test_an_opposite_sentence_with_the_same_words_is_not_evidence():
    """REGRESSION, found on review: the located path accepted word overlap, so a sentence
    saying the opposite of the article's passed as a quote from it."""
    fit = article_fit.assess(_extraction(
        {"issue": "touch is laggy", "covered": True,
         "evidence": "Tap Settings, then Display, then Touch sensitivity to disable it."}),
        ARTICLE)
    assert fit["issues"][0]["status"] == "unverified"


def test_a_quote_carrying_a_url_is_refused_and_never_reaches_the_response():
    """REGRESSION, found on review: evidence was sliced from the article as-is, so a
    sentence with a link went out in meta.article_fit, breaking the zero-URL rule."""
    article = ("Touch problems. Visit https://www.samsung.com/support for touch help. "
               "Tap Settings, then Display, then Touch sensitivity to enable it.")
    fit = article_fit.assess(_extraction(
        {"issue": "touch is laggy", "covered": True,
         "evidence": "Visit https://www.samsung.com/support for touch help."}), article)
    assert fit["issues"][0]["status"] == "unverified"
    assert "http" not in json.dumps(fit) and "www." not in json.dumps(fit)


def test_curly_quotes_case_and_a_missing_full_stop_are_tolerated():
    article = "Battery drain. Don\u2019t leave Bluetooth on when you aren\u2019t using it."
    fit = article_fit.assess(_extraction(
        {"issue": "battery drains", "covered": True,
         "evidence": "don't leave bluetooth on when you aren't using it"}), article)
    issue = fit["issues"][0]
    assert issue["status"] == "covered"
    assert issue["evidence"] == article[issue["span"][0]:issue["span"][1]]
    assert issue["evidence"].endswith("it."), "the article's own sentence is shown"


def _verdict(quote, article):
    fit = article_fit.assess(_extraction(
        {"issue": "the problem", "covered": True, "evidence": quote}), article)
    return fit["issues"][0]["status"]


def test_a_character_whose_lower_case_is_longer_cannot_crash_the_request():
    """REGRESSION, found on re-review: "\u0130".lower() is two characters, which
    desynchronised the normalised-to-original offset map; the IndexError reached the
    graded endpoint as a 500. A valid quote after it must still be found exactly."""
    article = "\u0130\u0130\u0130. Touch.\nRestart the phone if the touch problem continues"
    fit = article_fit.assess(_extraction(
        {"issue": "touch", "covered": True,
         "evidence": "restart the phone if the touch problem continues"}), article)
    issue = fit["issues"][0]
    assert issue["status"] == "covered"
    assert issue["evidence"] == "Restart the phone if the touch problem continues"


def test_a_failing_fit_never_costs_the_customer_the_graded_plan(row21_request,
                                                                 monkeypatch):
    """The fit is not graded. Whatever goes wrong inside it, the response still comes
    back and the fit says unknown."""
    plain = run_pipeline(row21_request,
                         provider=ReplayProvider(FIXTURES / "extraction_row21.json"),
                         use_cache=False)

    def boom(*_a, **_k):
        raise IndexError("simulated")
    monkeypatch.setattr(article_fit, "assess", boom)
    broken = run_pipeline(row21_request,
                          provider=ReplayProvider(FIXTURES / "extraction_row21.json"),
                          use_cache=False)
    assert broken.response == plain.response and broken.response["contexts"]
    assert broken.meta.article_fit["fit"] == "unknown"
    assert broken.meta.article_fit["reason"] == "fit_error"


def test_several_sentences_or_the_whole_article_are_not_one_quote():
    """REGRESSION, found on re-review: the whole article as 'evidence' made all 20 rows
    'full' and put 3 KB of text in meta.article_fit."""
    assert _verdict(ARTICLE, ARTICLE) == "unverified"
    assert _verdict("Remove the screen protector if it is peeling. Tap Settings, then "
                    "Display, then Touch sensitivity to enable it.", ARTICLE) == "unverified"


@pytest.mark.parametrize("article, fragment", [
    ("Touch help. Do not (under any circumstances) factory reset the phone to fix this.",
     "factory reset the phone to fix this."),
    ("Touch help. Do not - unless support asks - factory reset the phone to fix this.",
     "factory reset the phone to fix this."),
    ("Calls drop. Turn on Wi-Fi calling to fix dropped calls on the phone.",
     "Fi calling to fix dropped calls on the phone."),
    ("Never restart            the phone to fix this issue now.",
     "the phone to fix this issue now."),
])
def test_a_fragment_after_a_mid_line_marker_is_not_a_sentence(article, fragment):
    """REGRESSION, found on re-review: '-', ')' and a 12-character look-behind window
    counted as sentence starts anywhere, so a fragment could drop the article's own
    negation ("Do not ... factory reset") and still be accepted as its evidence."""
    assert _verdict(fragment, article) == "unverified"


@pytest.mark.parametrize("quote", [
    "Step 2: Force a Restart",
    "Restart your phone and try again",
    "Hold the Side button for 20 seconds.",
])
def test_headings_bullets_and_numbered_lines_are_sentences(quote):
    article = ("Touch problems\n## Step 2: Force a Restart\n- Restart your phone and try "
               "again\n1. Hold the Side button for 20 seconds.\n")
    assert _verdict(quote, article) == "covered"


def test_every_occurrence_is_tried_not_only_the_first():
    article = ("If the app freezes, restart your phone and try again. "
               "Still frozen? Restart your phone and try again.")
    assert _verdict("Restart your phone and try again.", article) == "covered"


def test_a_misquote_is_not_counted_as_a_reported_mismatch(repo_root):
    """REGRESSION, found on re-review: one covered plus one unverified claim is a
    'partial' fit, and §6.2 counted it as the engine flagging a Phase 0 mismatch."""
    import sys
    sys.path.insert(0, str(repo_root))
    from evaluation.run_eval import run_article_fit_eval
    fit = {"fit": "partial", "covered": 1, "total": 2, "unverified_claims": 1, "issues": [
        {"issue": "a", "status": "covered", "covered": True, "quote_found": True},
        {"issue": "b", "status": "unverified", "covered": False, "quote_found": False}]}
    result = run_article_fit_eval([{"id": "row_1", "article_fit": fit}])
    assert result["phase0_wrong_judged"] == 1 and result["phase0_wrong_flagged"] == 0


def test_the_gemini_parser_degrades_a_malformed_issue_list_instead_of_failing():
    """complaint_issues is not graded; a bad item must cost the fit, not the plan."""
    from app.llm.gemini import _tolerate_complaint_issues
    payload = {"complaint_issues": [
        {"issue": "touch is laggy", "covered": "yes", "evidence": None},
        {"covered": True, "evidence": "x"}, "not a dict",
        {"issue": "screen flashes", "covered": True, "evidence": "Some sentence here."}]}
    _tolerate_complaint_issues(payload)
    assert payload["complaint_issues"] == [
        {"issue": "touch is laggy", "covered": False, "evidence": ""},
        {"issue": "screen flashes", "covered": True, "evidence": "Some sentence here."}]
    for bad in (None, "text", {"a": 1}):
        payload = {"complaint_issues": bad}
        _tolerate_complaint_issues(payload)
        assert payload["complaint_issues"] == []
    Extraction(goal_topic="Touch", title="Touch issues", **payload)


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
