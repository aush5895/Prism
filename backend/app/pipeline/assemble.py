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
from . import ordering, validate
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
    span_coverage: float = 0.0
    deeplink_precision: float = 0.0
    evidence_alignment: float = 0.0
    checks: List[str] = field(default_factory=list)


def _span_is_verified(span: Optional[Tuple[int, int]], evidence_len: int) -> bool:
    if not span or len(span) != 2:
        return False
    start, end = span
    return 0 <= start < end <= evidence_len


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


def compute_score(resolved: List[ResolvedAction], evidence_len: int, alignment: float) -> Tuple[float, PlanDebug]:
    """Contract §3.4. The model never chooses this number."""
    dbg = PlanDebug(evidence_alignment=alignment)

    total_steps = verified_steps = 0
    for ra in resolved:
        for group in ra.action.step_groups:
            for step in group.steps:
                total_steps += 1
                verified_steps += int(_span_is_verified(step.source_span, evidence_len))
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
    resolved = resolve_actions(extraction, catalog)
    if not resolved:
        return {"contexts": []}, PlanDebug(evidence_alignment=alignment)

    ordered: List[ResolvedAction] = ordering.order(
        [(ra.tier, ra.source_order, ra) for ra in resolved]
    )

    score, dbg = compute_score(resolved, len(evidence.text), alignment)

    actions: List[Dict[str, Any]] = []
    for ra in ordered:
        groups: List[Dict[str, Any]] = []
        for group, res in zip(ra.action.step_groups, ra.resolutions):
            groups.append({
                "steps": [validate.scrub_urls(s.text) for s in group.steps],
                "actionableDeeplink": res.deeplink,
                "validationDeeplink": res.validation,
            })
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
