"""Assembly: IR + catalog resolutions -> Samsung-schema response (contract §3, §7).

Deliberate detail: the response is built as plain dicts and then VALIDATED through
Pydantic, rather than being serialised from Pydantic. Validation is the gate; the dict is
the payload. That keeps the emitted key set byte-stable (no `classes: null` appearing
because an optional field exists) while still proving the object satisfies schema.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .. import config
from ..ir import EnrichedQuery, Evidence, ExtractedAction, Extraction
from . import ordering, spans, validate
from .deeplinks import DeeplinkCatalog, Resolution


@dataclass
class ResolvedAction:
    action: ExtractedAction
    category: str
    tier: int
    source_order: int
    resolutions: List[Resolution] = field(default_factory=list)


@dataclass
class PlanDebug:
    """Internal only — never serialised into the graded response (guide §4.2.4)."""
    resolutions: List[Dict[str, Any]] = field(default_factory=list)
    spans: List[Dict[str, Any]] = field(default_factory=list)
    span_coverage: float = 0.0
    deeplink_precision: float = 0.0
    evidence_alignment: float = 0.0
    checks: List[str] = field(default_factory=list)


def _span_is_verified(span: Optional[Tuple[int, int]], step_text: str, evidence: str) -> bool:
    """Bounds alone is NOT verification.

    This used to be `0 <= start < end <= evidence_len`, which any fabricated offset
    satisfies -- and measured on the recorded row_21 extraction, every one of the 29
    model-claimed offsets was fabricated while scoring a perfect 1.00 span_coverage.
    Since span_coverage is 40% of the confidence score (contract §3.4), the reported
    confidence was inflated by exactly the amount the model was wrong. A span must now
    quote text that actually supports its step.
    """
    return spans.verifies(span, step_text, evidence)


def resolve_actions(extraction: Extraction, catalog: DeeplinkCatalog) -> List[ResolvedAction]:
    out: List[ResolvedAction] = []
    for i, action in enumerate(extraction.actions):
        category = ordering.categorize(action)
        # Gate [0]: wording-based (hardware buttons) OR operation-based (the device
        # restarts / enters safe mode). Either one means "not a Settings screen".
        physical = ordering.is_physical(action) or ordering.is_device_operation(action)
        resolutions = [
            catalog.resolve_step_group([s.text for s in g.steps], category, physical)
            for g in action.step_groups
        ]
        first_type = next(
            (r.deeplink["originalType"] for r in resolutions if r.deeplink), None
        )
        out.append(
            ResolvedAction(
                action=action,
                category=category,
                tier=ordering.tier_of(category, action, first_type),
                source_order=i,
                resolutions=resolutions,
            )
        )
    return out


def compute_score(resolved: List[ResolvedAction], evidence: str, alignment: float) -> Tuple[float, PlanDebug]:
    """Contract §3.4. The model never chooses this number."""
    dbg = PlanDebug(evidence_alignment=alignment)

    total_steps = verified_steps = 0
    for ra in resolved:
        for group in ra.action.step_groups:
            for step in group.steps:
                # A step that could not be located stays in the DENOMINATOR and adds
                # nothing to the numerator. Dropping it from both sides would let a plan
                # with one traceable step out of thirty report perfect grounding.
                total_steps += 1
                verified_steps += int(_span_is_verified(step.source_span, step.text, evidence))
    dbg.span_coverage = (verified_steps / total_steps) if total_steps else 0.0

    auto_groups = [r for ra in resolved if ra.category == "auto" for r in ra.resolutions]
    exact = sum(1 for r in auto_groups if r.is_exact)
    if auto_groups:
        dbg.deeplink_precision = exact / len(auto_groups)
        score = (config.SCORE_W_SPAN * dbg.span_coverage
                 + config.SCORE_W_DEEPLINK * dbg.deeplink_precision
                 + config.SCORE_W_EVIDENCE * alignment)
    else:
        # No auto actions means the deeplink term is undefined, not zero. Renormalise
        # rather than silently penalising a legitimately all-manual plan.
        dbg.deeplink_precision = 0.0
        weight = config.SCORE_W_SPAN + config.SCORE_W_EVIDENCE
        score = (config.SCORE_W_SPAN * dbg.span_coverage
                 + config.SCORE_W_EVIDENCE * alignment) / weight
    return round(min(max(score, 0.0), 1.0), 2), dbg


def build_response(
    extraction: Extraction,
    enriched: EnrichedQuery,
    evidence: Evidence,
    catalog: DeeplinkCatalog,
    alignment: float,
) -> Tuple[Dict[str, Any], PlanDebug]:
    # Replace the extractor's claimed offsets with verified ones BEFORE anything scores
    # or renders them, so the confidence score and the UI read the same trustworthy set.
    span_by_step = spans.relocate_extraction(extraction, evidence.text)

    resolved = resolve_actions(extraction, catalog)
    if not resolved:
        return {"contexts": []}, PlanDebug(evidence_alignment=alignment)

    ordered: List[ResolvedAction] = ordering.order(
        [(ra.tier, ra.source_order, ra) for ra in resolved]
    )

    score, dbg = compute_score(resolved, evidence.text, alignment)

    actions: List[Dict[str, Any]] = []
    for a_idx, ra in enumerate(ordered):
        groups: List[Dict[str, Any]] = []
        for g_idx, (group, res) in enumerate(zip(ra.action.step_groups, ra.resolutions)):
            groups.append({
                "steps": [validate.scrub_urls(s.text) for s in group.steps],
                "actionableDeeplink": res.deeplink,
                "validationDeeplink": res.validation,
            })
            # Re-key the span records to the EMITTED position. The extraction is in
            # article order and this loop is in tier order, so positional records from
            # the extraction would label the wrong step.
            for s_idx, step in enumerate(group.steps):
                record = span_by_step.get(id(step))
                if record is not None:
                    dbg.spans.append({"action_index": a_idx, "group_index": g_idx,
                                      "step_index": s_idx, **record})
            dbg.resolutions.append({
                "action": ra.action.action_name,
                "decision": res.decision,
                "reason": res.reason,
                "catalog_id": res.catalog_id,
                "rejected": [t.__dict__ for t in res.trace if t.verdict != "eligible"][:5],
            })
        actions.append({
            "actionName": validate.repair_action_name(ra.action.action_name),
            "description": validate.repair_description(ra.action.description),
            "category": ra.category,
            "stepGroups": groups,
        })

    response = {"contexts": [{
        "goal": validate.build_goal(extraction.goal_topic),
        "title": validate.repair_title(extraction.title),
        "score": score,
        "actions": actions,
    }]}
    return response, dbg
