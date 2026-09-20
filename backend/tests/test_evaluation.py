"""Guards on the measurement harness itself.

A benchmark that quietly measures the wrong thing is worse than no benchmark: it produces
confident numbers for a submission. These tests pin the two ways this gold set can be
silently corrupted (a gold label that cannot be hit, a template that hands back its own
answer), the determinism the ablation depends on, and the rule that docs/metrics.md may
only contain figures that exist in evaluation/report.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import synthetic  # noqa: E402
from evaluation.run_eval import render_metrics, score_cases  # noqa: E402

REPORT_PATH = ROOT / "evaluation" / "report.json"
METRICS_PATH = ROOT / "docs" / "metrics.md"

RESOLVABLE_TYPES = ("onURL", "offURL", "onClickURL", "updateURL")


# ----------------------------------------------------------------- catalog invariants
def test_catalog_shape_is_what_the_gold_set_assumes(catalog):
    """The gold set is generated FROM the catalog, so a catalog change silently changes
    every number in metrics.md. Assert the shape the measurements were built against."""
    entries = catalog.entries
    assert len(entries) == 578
    assert len({e["deeplink"] for e in entries}) == 578

    counts = defaultdict(int)
    for e in entries:
        counts[e.get("originalType")] += 1
    assert dict(counts) == {
        "onClickURL": 254, "onURL": 138, "offURL": 138,
        "updateURL": 36, None: 11, "placeholder": 1,
    }


def test_validation_asymmetry_between_on_and_off(catalog):
    """onURL carries full validation, offURL carries key only. The emitter copies these
    verbatim, so the gold set's assumption about them has to hold."""
    for e in catalog.entries:
        validation = e.get("validation") or {}
        if e.get("originalType") == "onURL":
            assert set(validation) == {"deeplink", "key", "resultType", "condition", "value"}
        elif e.get("originalType") == "offURL":
            assert set(validation) == {"deeplink", "key"}


# ----------------------------------------------------------------- trap (a): gold class
def test_gold_is_an_equivalence_class_not_a_single_id():
    """A step phrased from a shared `message` cannot single out one of the entries that
    share it, so gold must be every id in the class.

    Measured on the supplied catalog: 'View Notification Settings' covers 21 different
    screens. Scoring such a case against one arbitrary id measures a coin flip.
    """
    cases = synthetic.build_cases()
    assert cases

    by_class = defaultdict(list)
    for e in json.loads((ROOT / "data" / "deeplinks.json").read_text(encoding="utf-8"))["deeplinks"]:
        if e.get("originalType") in RESOLVABLE_TYPES:
            by_class[(e["message"], e["originalType"])].append(e["id"])

    for case in cases:
        expected = frozenset(by_class[(case.message, case.original_type)])
        assert case.gold == expected, f"gold is not the full class for {case.case_id}"
        assert case.class_size == len(expected)

    # and the ambiguity is real, not theoretical
    assert any(c.class_size > 1 for c in cases)
    assert max(c.class_size for c in cases) >= 10


def test_duplicate_stats_agree_with_the_catalog():
    stats = synthetic.duplicate_message_stats()
    assert stats["distinct_classes"] < stats["entries_resolvable"]
    assert stats["classes_with_duplicates"] > 0
    assert 0.0 < stats["share_entries_in_duplicate_classes"] < 1.0
    assert sum(int(k) * v for k, v in stats["class_size_histogram"].items()) == \
        stats["entries_resolvable"]


def test_ambiguous_enable_disable_pairs_are_found():
    """The pairs gate [2] exists to separate."""
    pairs = synthetic.ambiguous_pairs()
    assert pairs
    for pair in pairs:
        assert pair["onURL"] and pair["offURL"]
        assert not set(pair["onURL"]) & set(pair["offURL"])


# ----------------------------------------------------------------- trap (b): answer leak
def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def test_no_generated_step_reproduces_its_own_message():
    """A template that reproduces the catalog message turns resolution into string
    matching and the measurement becomes meaningless."""
    for case in synthetic.build_cases():
        assert _normalise(case.step) != _normalise(case.message), \
            f"{case.case_id} reproduces its message verbatim: {case.step!r}"


def test_no_template_reuses_the_catalogs_leading_verb():
    """Every catalog message opens with one of a small closed set of verbs. A generated
    step must not open with the same one, or the leading token alone gives the answer."""
    for case in synthetic.build_cases():
        step_verb = _normalise(case.step).split()[0]
        message_verb = _normalise(case.message).split()[0]
        assert step_verb != message_verb, \
            f"{case.case_id} leads with the catalog's own verb {step_verb!r}"
        assert step_verb not in synthetic.CATALOG_LEADING_VERBS, \
            f"{case.case_id} leads with a catalog verb {step_verb!r}"


def test_every_case_still_names_its_subject():
    """The counterpart guard: having removed the verb, the step must still contain the
    subject, or the case is unanswerable rather than merely hard."""
    for case in synthetic.build_cases():
        assert case.subject.lower() in case.step.lower()


# ----------------------------------------------------------------- determinism
def test_resolution_is_deterministic_across_identical_runs():
    """The ablation compares runs against each other. If resolution were not
    reproducible, every difference it reports would be noise."""
    sample = synthetic.build_cases()[:60]
    first = score_cases(sample)
    second = score_cases(sample)
    for key in ("correct", "wrong", "abstained", "accuracy_at_1_pct", "precision_pct"):
        assert first[key] == second[key]
    assert first["wrong_examples"] == second["wrong_examples"]


def test_case_generation_is_deterministic():
    a = [c.case_id for c in synthetic.build_cases()]
    synthetic.build_cases.cache_clear()
    b = [c.case_id for c in synthetic.build_cases()]
    assert a == b


# ----------------------------------------------------------------- metrics.md provenance
@pytest.mark.skipif(not REPORT_PATH.exists() or not METRICS_PATH.exists(),
                    reason="run `python -m evaluation.run_eval` first")
def test_every_percentage_in_metrics_md_exists_in_report_json():
    """docs/metrics.md may not contain a number that the harness did not measure.

    This is the guard behind the project rule that no performance figure is ever typed by
    hand: if a percentage appears in the document, the same value must be present in
    report.json.
    """
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    markdown = METRICS_PATH.read_text(encoding="utf-8")

    values: set[float] = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, bool):
            pass
        elif isinstance(node, (int, float)):
            values.add(round(float(node), 1))

    walk(report)

    printed = {round(float(m), 1) for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", markdown)}
    missing = printed - values
    assert not missing, f"metrics.md prints percentages absent from report.json: {sorted(missing)}"


@pytest.mark.skipif(not REPORT_PATH.exists(), reason="run `python -m evaluation.run_eval` first")
def test_metrics_md_is_exactly_what_the_renderer_produces():
    """Nothing may be hand-edited into metrics.md after generation."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    expected = render_metrics(report)
    actual = METRICS_PATH.read_text(encoding="utf-8")
    assert actual == expected, "docs/metrics.md is stale or hand-edited; re-run the harness"


@pytest.mark.skipif(not REPORT_PATH.exists(), reason="run `python -m evaluation.run_eval` first")
def test_report_records_which_provider_produced_it():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["compliance"]["provider"]
    assert report["provider_requested"]
    assert report["gold_set"]["n_cases"] == len(synthetic.build_cases())


@pytest.mark.skipif(not REPORT_PATH.exists(), reason="run `python -m evaluation.run_eval` first")
def test_margin_sweep_fits_and_reports_on_disjoint_halves():
    """Fitting and reporting on the same cases would be fitting on the test set."""
    sweep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["margin_sweep"]
    assert sweep["n_fit"] + sweep["n_holdout"] == len(synthetic.build_cases())
    assert sweep["n_fit"] > 0 and sweep["n_holdout"] > 0
    assert len(sweep["fit_sweep"]) >= 5
    # the held-out half is reported for the selected and shipped deltas only
    reported = {row["margin_delta"] for row in sweep["holdout"].values()}
    assert reported <= {sweep["selected_margin_delta"], sweep["shipped_margin_delta"]}
