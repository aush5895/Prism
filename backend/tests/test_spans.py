"""Source-span location and verification (contract §3.4, the span_coverage term).

THE TEST THAT WAS MISSING is `test_an_in_bounds_span_over_unrelated_text_is_not_verified`.
The previous check asked only whether an offset was in bounds, which every fabricated
offset satisfies. Measured on the recorded row_21 extraction, all 29 model-claimed spans
were fabricated and span_coverage still read a perfect 1.00 - and span_coverage is 40% of
the confidence score, so the score was inflated by exactly the amount the model was wrong.
The suite stayed green throughout, because nothing asserted that a span quotes text having
anything to do with its step.
"""
from __future__ import annotations

import json

import pytest

from app.ir import ExtractedAction, ExtractedStep, ExtractedStepGroup, Extraction
from app.pipeline import spans
from app.pipeline.assemble import ResolvedAction, compute_score

# Worded the way the supplied articles are, so the overlap a real extraction would have is
# the overlap under test. A step is a close paraphrase of its sentence, not a synonym of
# it: "a different, undamaged charger" shares only one content word with "a damaged
# charger might not be supplying enough power", which is genuinely not enough to call a
# step grounded.
ARTICLE = (
    "Touchscreen issues on a Galaxy phone.\n"
    "If your screen protector is peeling or has debris under it, please remove it.\n"
    "A damaged charger might not be supplying enough power to your device.\n"
    "If you suspect this is the case, please try using a different, undamaged charger.\n"
    "To clean the device, gently wipe the front and back with a lint-free microfiber cloth.\n"
)


def _extraction(*step_texts, claimed=None):
    steps = [ExtractedStep(text=t, source_span=claimed) for t in step_texts]
    return Extraction(
        goal_topic="Touchscreen", title="Touchscreen issues",
        actions=[ExtractedAction(
            action_name="Check the screen",
            description="It will address the screen on your device",
            step_groups=[ExtractedStepGroup(steps=steps)])])


# ------------------------------------------------------------------ the missing test
def test_an_in_bounds_span_over_unrelated_text_is_not_verified():
    """REGRESSION. `[0, 12)` is in bounds for every article ever supplied, and the old
    predicate accepted it. It quotes "Touchscreen " against a step about a charger."""
    step = "Try a different, undamaged charger."
    assert 0 <= 0 < 12 <= len(ARTICLE)            # in bounds, as the old check required
    assert spans.verifies((0, 12), step, ARTICLE) is False


def test_a_span_over_the_supporting_sentence_is_verified():
    step = "Try a different, undamaged charger."
    start = ARTICLE.index("If you suspect")
    end = ARTICLE.index("\n", start)
    assert spans.verifies((start, end), step, ARTICLE) is True


@pytest.mark.parametrize("span", [None, (), (5,), (10, 10), (-1, 20), (5, 10_000), (None, None)])
def test_malformed_and_out_of_range_spans_are_never_verified(span):
    assert spans.verifies(span, "Try a different, undamaged charger.", ARTICLE) is False


# ------------------------------------------------------------------ location
def test_a_fabricated_span_is_relocated_to_the_real_supporting_text():
    start, end, provenance, confidence = spans.locate(
        "Remove the peeling screen protector.", ARTICLE, claimed=(0, 12))
    assert provenance == "located"
    assert confidence >= spans.MIN_SUPPORT
    assert "protector" in ARTICLE[start:end]


def test_a_correct_model_span_is_kept_and_labelled_as_the_models():
    step = "Try a different, undamaged charger."
    start = ARTICLE.index("If you suspect")
    end = ARTICLE.index("\n", start)
    got_start, got_end, provenance, _c = spans.locate(step, ARTICLE, claimed=(start, end))
    assert provenance == "model"
    assert (got_start, got_end) == (start, end)


def test_a_step_absent_from_the_article_gets_no_span():
    """An honest blank beats a confident pointer at unrelated text."""
    start, end, provenance, _c = spans.locate(
        "Replace the refrigerator water filter cartridge.", ARTICLE, None)
    assert start is None and end is None and provenance == "unlocated"


def test_a_step_with_no_subject_vocabulary_is_not_claimed_as_grounded():
    """"Open Settings." is all UI verbs and stopwords. There is nothing to match on, so it
    is not evidence that the plan is grounded and must not be counted as such."""
    start, _end, provenance, _c = spans.locate("Open Settings.", ARTICLE, None)
    assert start is None and provenance == "unlocated"
    assert spans.verifies((0, 40), "Open Settings.", ARTICLE) is False


def test_the_tightest_supporting_window_wins():
    """On a tie the shortest window is the more informative highlight."""
    start, end, _p, _c = spans.locate(
        "Gently wipe with a lint-free microfiber cloth.", ARTICLE, None)
    quoted = ARTICLE[start:end]
    assert "microfiber" in quoted
    assert "charger" not in quoted, "window drifted into the preceding sentence"


def test_relocation_rewrites_the_ir_and_reports_provenance():
    extraction = _extraction("Remove the peeling screen protector.",
                             "Try a different, undamaged charger.",
                             claimed=(0, 12))
    records = spans.relocate_extraction(extraction, ARTICLE)
    assert len(records) == 2

    # Records are keyed by step IDENTITY, not position: the extraction is in article
    # order while the emitted plan is in tier order, so positional keys would label the
    # wrong step in the UI.
    for step in extraction.actions[0].step_groups[0].steps:
        record = records[id(step)]
        assert record["provenance"] == "located"
        assert record["claimed_start"] == 0, "the model's original claim is preserved"
        assert step.source_span == (record["start"], record["end"])
        assert spans.verifies(step.source_span, step.text, ARTICLE)


# ------------------------------------------------------------------ the score
def _score_of(extraction):
    resolved = [ResolvedAction(action=a, category="manual", tier=0, source_order=i)
                for i, a in enumerate(extraction.actions)]
    return compute_score(resolved, ARTICLE, alignment=0.5)


def test_fabricated_spans_no_longer_score_as_perfect_coverage():
    """The headline consequence: before this, span_coverage was 1.00 here."""
    extraction = _extraction("Remove the peeling screen protector.",
                             "Try a different, undamaged charger.",
                             claimed=(0, 12))
    _score, dbg = _score_of(extraction)
    assert dbg.span_coverage == 0.0, "in-bounds fabrications must count for nothing"


def test_coverage_counts_only_steps_that_verify():
    extraction = _extraction("Remove the peeling screen protector.",   # locatable
                             "Replace the refrigerator water filter.")  # not in the article
    spans.relocate_extraction(extraction, ARTICLE)
    _score, dbg = _score_of(extraction)
    assert dbg.span_coverage == pytest.approx(0.5)


def test_an_unlocatable_step_stays_in_the_denominator():
    """Dropping it from both sides would let one traceable step out of many report
    perfect grounding."""
    extraction = _extraction("Remove the peeling screen protector.",
                             *["Replace the refrigerator water filter."] * 9)
    spans.relocate_extraction(extraction, ARTICLE)
    _score, dbg = _score_of(extraction)
    assert dbg.span_coverage == pytest.approx(0.1)


def test_relocated_spans_raise_coverage_over_trusting_the_model():
    """Relocation is not merely stricter - it recovers grounding the model mislabelled."""
    texts = ("Remove the peeling screen protector.", "Try a different, undamaged charger.")
    trusted = _extraction(*texts, claimed=(0, 12))
    relocated = _extraction(*texts, claimed=(0, 12))
    spans.relocate_extraction(relocated, ARTICLE)
    assert _score_of(trusted)[1].span_coverage == 0.0
    assert _score_of(relocated)[1].span_coverage == pytest.approx(1.0)
