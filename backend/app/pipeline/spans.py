"""Source-span location and verification (contract §3.4 — the span_coverage term).

WHY THIS MODULE EXISTS
----------------------
The extractor reports a `[start, end)` character offset for every step. Those offsets are
GUESSES. Measured on the recorded row_21 extraction: **0 of 29 model-claimed spans quoted
text with any meaningful overlap with their own step.** Counting characters is not
something a language model does reliably, and nothing upstream can make it so.

That mattered well beyond the demo UI. `span_coverage` is 40% of the confidence score
(contract §3.4), and the check behind it tested only that an offset was in BOUNDS. An
offset of `[0, 12)` is in bounds for every article, so fabricated spans scored a perfect
1.00 and the reported confidence was inflated by exactly the amount the model was wrong.

So a span is now LOCATED and VERIFIED rather than believed:

  locate()    finds where a step is actually supported, by content-token overlap over the
              article's own sentences, and scores the model's claim the same way. Whichever
              genuinely supports the step wins; ties go to the tightest window.
  verifies()  is the predicate `assemble.compute_score` uses. It requires that the text at
              the span actually shares the step's subject vocabulary. Bounds alone is not
              verification.

A step that cannot be located anywhere gets NO span. It stays in the span_coverage
denominator and contributes nothing to the numerator, so an unlocatable plan scores low
rather than being quietly excused. Dropping it from both sides would let a plan with one
locatable step out of thirty report perfect grounding.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..text import content, tokens

# A span must share at least this fraction of the step's subject vocabulary with the text
# it points at. Below it, we do not claim the step is traceable to the article.
MIN_SUPPORT = 0.5

# 1- and 2-sentence windows. Two is enough for a step split across a clause boundary; more
# would let a long window score well by accident, which is the failure being fixed.
MAX_WINDOW_SENTENCES = 2

_SEGMENT = re.compile(r"[^.!?\n]+[.!?]?")


@lru_cache(maxsize=8)
def _segment_index(evidence: str) -> Tuple[Tuple[int, int, frozenset], ...]:
    """Sentence spans with their token sets, computed once per article.

    Cached because relocation runs over every step of a plan and every candidate window
    of the same article; re-tokenising the article per step made this the slowest thing
    in the request.
    """
    out = []
    for match in _SEGMENT.finditer(evidence):
        if match.group().strip():
            out.append((match.start(), match.end(), frozenset(tokens(match.group()))))
    return tuple(out)


def support(step_text: str, evidence: str, start: int, end: int) -> float:
    """Fraction of the STEP's subject vocabulary present in the quoted region."""
    subject = set(content(step_text))
    if not subject:
        return 0.0
    quoted = set(tokens(evidence[start:end]))
    return sum(1 for t in subject if t in quoted) / len(subject)


def verifies(span: Optional[Sequence[int]], step_text: str, evidence: str) -> bool:
    """The predicate behind span_coverage.

    REGRESSION this exists for: the previous check was `0 <= start < end <= len(evidence)`,
    which a fabricated offset satisfies trivially. A span must now quote text that actually
    supports the step it is attached to.
    """
    if not span or len(span) != 2 or span[0] is None or span[1] is None:
        return False
    start, end = int(span[0]), int(span[1])
    if not 0 <= start < end <= len(evidence):
        return False
    subject = set(content(step_text))
    if not subject:
        # A step with no subject vocabulary ("Open Settings.") cannot be verified either
        # way. It is not evidence of grounding, so it does not count as verified.
        return False
    return support(step_text, evidence, start, end) >= MIN_SUPPORT


def locate(step_text: str, evidence: str,
           claimed: Optional[Sequence[int]] = None) -> Tuple[Optional[int], Optional[int], str, float]:
    """Where the step is really supported. Returns (start, end, provenance, confidence).

    provenance is one of:
      "model"      the extractor's own offsets survived verification
      "located"    relocated here by content overlap
      "unlocated"  not supported anywhere in the article; no span is claimed
    """
    subject = set(content(step_text))
    if not subject or not evidence:
        return None, None, "unlocated", 0.0

    segments = _segment_index(evidence)

    # Rank by (support, tightness). Without the tightness term a two-sentence window
    # starting one sentence early keeps the position on a tie, and the highlight drifts.
    best_score, best_span = 0.0, None
    for i, (start, _end, _toks) in enumerate(segments):
        window: Set[str] = set()
        for j in range(i, min(i + MAX_WINDOW_SENTENCES, len(segments))):
            window |= segments[j][2]
            score = sum(1 for t in subject if t in window) / len(subject)
            if best_span is None or (score, -(segments[j][1] - start)) > \
                    (best_score, -(best_span[1] - best_span[0])):
                best_score, best_span = score, (start, segments[j][1])

    claim_score = 0.0
    if claimed and len(claimed) == 2 and claimed[0] is not None:
        c0, c1 = int(claimed[0]), int(claimed[1])
        if 0 <= c0 < c1 <= len(evidence):
            claim_score = support(step_text, evidence, c0, c1)
            if claim_score >= MIN_SUPPORT and claim_score >= best_score:
                return c0, c1, "model", round(claim_score, 3)

    if best_span is not None and best_score >= MIN_SUPPORT:
        return best_span[0], best_span[1], "located", round(best_score, 3)
    return None, None, "unlocated", round(max(best_score, claim_score), 3)


def relocate_extraction(extraction, evidence: str) -> Dict[int, dict]:
    """Rewrite every step's source_span to a VERIFIED one, in place.

    Returns provenance records keyed by `id(step)`, NOT by position. The extraction is in
    article order and the emitted plan is in tier order, so positional records would
    label the wrong step in the UI; the caller re-keys them as it emits.

    The in-place rewrite is deliberate: the confidence score and the UI should read one
    set of spans and it should be the trustworthy set. The model's original claim is
    preserved in the record rather than in the IR.
    """
    records: Dict[int, dict] = {}
    for action in extraction.actions:
        for group in action.step_groups:
            for step in group.steps:
                claimed = step.source_span
                start, end, provenance, confidence = locate(step.text, evidence, claimed)
                step.source_span = (start, end) if start is not None else None
                records[id(step)] = {
                    "start": start, "end": end,
                    "claimed_start": claimed[0] if claimed else None,
                    "claimed_end": claimed[1] if claimed else None,
                    "provenance": provenance, "confidence": confidence,
                }
    return records
