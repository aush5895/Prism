"""Audit of Samsung's deeplink catalog (data/deeplinks.json) — read-only.

Two consumers, one definition:
  - gate [4] asks `label_supported(entry)` before trusting a candidate (deeplinks.py)
  - tools/catalog_health.py renders `audit(entries)` into docs/catalog_health.md

WHY THE RESOLVER NEEDS THIS
---------------------------
The resolver matches a step against each entry's `message` (its label). Measured on the
kit: for 58 of 577 real entries the label shares no subject word with the entry's own
`description` or `qna_description`, and several are plainly a different screen:

    DL-0163  "Enable Grayscale"        -> "Enables mono audio via device Settings"
    DL-0312  "Enable WiFi"             -> "Enables Mobile Hotspot via device Settings"
    DL-0117  "View Timeout Settings"   -> "Opens the guest mode settings page"

So "Tap the switch next to Grayscale to enable it." resolved, confidently, to the mono
audio toggle. The gold set could not see it: gold is built FROM labels, so following a
wrong label scores as correct. On the 850-case gold set, 26 of 771 exact resolutions
landed on such an entry.

WHAT "SUPPORTED" MEANS
----------------------
At least one subject word of the label (its `message` minus the leading verb, function
words and UI verbs removed) appears in the description or QnA text, after:
  - joiners closed up ("Wi-Fi" == "WiFi"), as text.tokens already does
  - a plural "s" dropped ("sounds" == "sound")
  - two adjacent words joined ("auto sync" == "Auto-Sync")
  - a shared 6-letter stem ("magnification" == "magnifier")
This is deliberately LENIENT: an entry is only distrusted when its label and its own
description share nothing at all. A generic-but-honest label ("View Sound Settings" for
"mute all sounds") passes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

from ..text import content, subject_of, tokens

DUMMY_ID = "DL-DUMMY"
STEM = 6


def _norm(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def _body_forms(text: str) -> Set[str]:
    words = [_norm(t) for t in tokens(text)]
    return set(words) | {a + b for a, b in zip(words, words[1:])}


def label_supported(entry: Dict[str, Any]) -> bool:
    """Does the entry's own description or QnA text support what its label names?"""
    subject = {_norm(t) for t in content(subject_of(entry.get("message") or ""))}
    if not subject:
        return True          # nothing to contradict; other gates handle empty labels
    body = _body_forms(f"{entry.get('description') or ''} {entry.get('qna_description') or ''}")
    stems = {b[:STEM] for b in body if len(b) >= STEM}
    return any(t in body or (len(t) >= STEM and t[:STEM] in stems) for t in subject)


def unsupported_ids(entries: Iterable[Dict[str, Any]]) -> Set[str]:
    return {e["id"] for e in entries if e.get("id") != DUMMY_ID and not label_supported(e)}


def _text_key(entry: Dict[str, Any]) -> tuple:
    return tuple((entry.get(f) or "").strip().lower()
                 for f in ("message", "description", "qna_description"))


def audit(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Everything docs/catalog_health.md reports, computed from the catalog alone."""
    real = [e for e in entries if e.get("id") != DUMMY_ID]

    by_label: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for e in real:
        by_label[(e.get("message"), e.get("originalType"))].append(e)
    shared = {k: v for k, v in by_label.items() if len(v) > 1}

    by_text: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for e in real:
        by_text[(e.get("originalType"),) + _text_key(e)].append(e)
    identical = [sorted(x["id"] for x in v) for v in by_text.values() if len(v) > 1]

    on = {subject_of(e["message"]).lower() for e in real if e.get("originalType") == "onURL"}
    off = {subject_of(e["message"]).lower() for e in real if e.get("originalType") == "offURL"}

    unsupported = [e for e in real if not label_supported(e)]

    return {
        "n_entries": len(real),
        "n_unique_uris": len({e.get("deeplink") for e in real}),
        "type_counts": dict(Counter(e.get("originalType") or "none" for e in real)),
        "shared_label_groups": len(shared),
        "shared_label_entries": sum(len(v) for v in shared.values()),
        "largest_shared_labels": [
            {"message": k[0], "type": k[1], "n": len(v)}
            for k, v in sorted(shared.items(), key=lambda kv: (-len(kv[1]), kv[0][0] or ""))[:8]
        ],
        "identical_text_groups": sorted(identical),
        "toggle_subjects_on": len(on),
        "toggle_subjects_off": len(off),
        "toggle_subjects_both": len(on & off),
        "on_without_off": sorted(on - off),
        "off_without_on": sorted(off - on),
        "missing_type": sorted(e["id"] for e in real if not e.get("originalType")),
        "missing_validation": sorted(e["id"] for e in real if not e.get("validation")),
        "label_unsupported": [
            {"id": e["id"], "message": e.get("message"), "type": e.get("originalType"),
             "description": e.get("description")}
            for e in sorted(unsupported, key=lambda x: x["id"])
        ],
    }


def entry_ids_in(group: Optional[Iterable[str]]) -> Set[str]:
    return set(group or [])
