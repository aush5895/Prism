"""Categorisation and 5-tier sequencing (contract §6, directive 13).

Two rules that are easy to get backwards:
  * The LLM's category_hint is ADVISORY. Lexicon rules override it (contract §6.2), because
    a mis-categorised action either loses a deeplink it should have (recoverable) or gains
    one it must not (guide §4.1 violation). Rules fail in the recoverable direction.
  * Tiers are INTERNAL. Only auto|manual|critical is ever emitted.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .. import config
from ..ir import ExtractedAction
from ..text import contains_any
from .deeplinks import load_lexicons

TIER_MANUAL_NONINVASIVE = 0
TIER_AUTO_TOGGLE = 1
TIER_AUTO_NAVIGATIONAL = 2
TIER_SERVICE_ESCALATION = 3
TIER_CRITICAL = 4


def action_text(action: ExtractedAction) -> str:
    parts = [action.action_name, action.description]
    for group in action.step_groups:
        parts.extend(s.text for s in group.steps)
    return " ".join(parts)


def is_physical(action: ExtractedAction) -> bool:
    """Hardware-button or hands-on sequence: can never be a Settings screen, so it can
    never carry a deeplink even when the category is `critical` (gate [0])."""
    return contains_any(action_text(action), load_lexicons()["physical_interaction"]) is not None


def is_device_operation(action: ExtractedAction) -> bool:
    """A state the DEVICE enters (restart, safe mode, recovery mode) rather than a screen
    Settings can open.

    is_physical() is not sufficient on its own because it keys on step WORDING. An
    extraction that phrases a reboot as "Tap Restart." contains no hardware-button phrase,
    so gate [0] let it through and the resolver matched it against the catalog — observed
    output: a restart action carrying "Open Restart under Restart again". This predicate
    keys on the action as a whole instead, so the same operation is caught however it is
    written. Factory reset is excluded from the lexicon: it is a real Settings screen.
    """
    return contains_any(action_text(action), load_lexicons()["critical_device_operation"]) is not None


def categorize(action: ExtractedAction) -> str:
    """Rules first, hint second, schema default last.

    `critical` is tested before `manual` on purpose: a factory reset whose steps mention
    backing up data would otherwise be demoted to manual and lose its last-place ordering.
    """
    lex = load_lexicons()
    text = action_text(action)

    if contains_any(text, lex["critical"]):
        return "critical"
    if contains_any(text, lex["manual"]):
        return "manual"
    if action.category_hint in ("auto", "manual", "critical"):
        return action.category_hint
    return "manual"  # schema.py's own default, and the safe direction (guide §4.1)


def tier_of(category: str, action: ExtractedAction, original_type: Optional[str]) -> int:
    if category == "critical":
        return TIER_CRITICAL
    if category == "manual":
        if contains_any(action_text(action), load_lexicons()["service_escalation"]):
            return TIER_SERVICE_ESCALATION
        return TIER_MANUAL_NONINVASIVE
    return TIER_AUTO_TOGGLE if original_type in config.TOGGLE_TYPES else TIER_AUTO_NAVIGATIONAL


def order(items: Sequence[Tuple[int, int, object]]) -> List[object]:
    """Stable sort by (tier, source_order) — within a tier the article's own narrative
    order survives, which is why a 'last resort' step stays last (contract §6.1)."""
    return [obj for _t, _o, obj in sorted(items, key=lambda x: (x[0], x[1]))]
