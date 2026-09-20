"""Labelled gold set for resolver measurement, generated FROM the catalog.

There is no human-labelled deeplink set in the supplied kit, so the gold set is derived:
every catalog entry with a resolvable `originalType` is turned back into the step text a
troubleshooting article would have used to reach it, and the entry itself is the label.
Nothing here is hand-written per scenario (directive 17) — the subjects come from the
catalog's own `message` strings at build time.

TWO PROPERTIES THIS MODULE EXISTS TO GET RIGHT
----------------------------------------------
1. GOLD IS AN EQUIVALENCE CLASS, NOT ONE ID.
   Measured on the supplied catalog: 566 entries carry a resolvable type but form only
   427 distinct (message, originalType) classes. 193 of those entries live in a class
   with more than one member — "View Notification Settings" alone appears 21 times, each
   opening a genuinely different screen (reminder alerts, pop-ups, old-notification
   filter, minimised-notification filter), separable only by `description`, which gate
   [3] deliberately does not read.

   A step phrased from the shared `message` therefore CANNOT single one member out, and
   no resolver could. Scoring such a case against one arbitrarily chosen id measures a
   coin flip, not the resolver. Each class contributes one case whose gold is the
   frozenset of every id in it; a hit on any member is correct.

2. A TEMPLATE MAY NOT HAND BACK THE ANSWER.
   Every onURL message starts "Enable", every offURL "Disable", onClickURL "View"/"Check",
   updateURL "Adjust"/"Increase". A template that reuses the leading verb reproduces the
   message verbatim ("Adjust {s}." against "Adjust Brightness"), and the measurement
   degenerates into string matching. Every template below leads with a verb the catalog
   never opens a message with. test_evaluation.py fails the build if that stops holding.

   Polarity wording ("to enable it") IS still carried mid-sentence, and must be: gate [2]
   reads it to tell an onURL from an offURL, and it is how the supplied articles phrase
   the instruction. It is not a leak — `enable` is in text.UI_VERBS, so it is stripped
   from the candidate's content tokens and contributes nothing to gate [4] coverage.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Tuple

from app import config
from app.text import content

# Types a step can actually be phrased for. `null` (11) and `placeholder` (1) are excluded:
# the placeholder is the resolver's own fallback and the null-typed rows are status
# readouts, not screens a troubleshooting step navigates to.
RESOLVABLE_TYPES = ("onURL", "offURL", "onClickURL", "updateURL")

# Every verb the catalog opens a `message` with, measured over the supplied data.
# A generated step may never lead with one of these — see property 2 above.
CATALOG_LEADING_VERBS = frozenset(
    {"view", "enable", "disable", "adjust", "check", "increase", "switch", "optimize"}
)

# Two phrasings per subject, in the register the supplied articles use.
TEMPLATES: Dict[str, Tuple[str, str]] = {
    "onURL": ("Tap the switch next to {s} to enable it.", "Turn on {s}."),
    "offURL": ("Tap the switch next to {s} to disable it.", "Turn off {s}."),
    "onClickURL": ("Tap {s}.", "Navigate to and open {s}."),
    "updateURL": ("Change {s} to the value you want.", "Set {s} to your preferred value."),
}


@dataclass(frozen=True)
class Case:
    case_id: str
    step: str
    subject: str
    message: str
    original_type: str
    gold: FrozenSet[str]
    class_size: int
    template_index: int

    @property
    def is_ambiguous_class(self) -> bool:
        return self.class_size > 1


def _load_entries() -> List[dict]:
    return json.loads(config.DEEPLINKS_PATH.read_text())["deeplinks"]


def subject_of(message: str) -> str:
    """The catalog message minus its leading verb. 'Enable Auto-Sync' -> 'Auto-Sync'."""
    parts = (message or "").split()
    if parts and parts[0].lower() in CATALOG_LEADING_VERBS:
        parts = parts[1:]
    return " ".join(parts).strip()


@lru_cache(maxsize=1)
def _classes() -> Dict[Tuple[str, str], List[str]]:
    """(message, originalType) -> ids, for resolvable types only."""
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for e in _load_entries():
        if e.get("originalType") in RESOLVABLE_TYPES:
            groups[(e["message"], e["originalType"])].append(e["id"])
    return dict(groups)


@lru_cache(maxsize=1)
def build_cases() -> Tuple[Case, ...]:
    """One case per (class, template). Deterministic and ordered."""
    cases: List[Case] = []
    for (message, otype), ids in sorted(_classes().items()):
        subject = subject_of(message)
        # A message that is only a verb, or whose subject carries no content token
        # ('Onurl', 'Offurl'), cannot be phrased into a step that names anything.
        if not subject or not content(subject):
            continue
        gold = frozenset(ids)
        for idx, template in enumerate(TEMPLATES[otype]):
            cases.append(Case(
                case_id=f"{sorted(ids)[0]}::{otype}::{idx}",
                step=template.format(s=subject),
                subject=subject,
                message=message,
                original_type=otype,
                gold=gold,
                class_size=len(ids),
                template_index=idx,
            ))
    return tuple(cases)


def duplicate_message_stats() -> Dict[str, object]:
    """How much of the catalog is not separable by `message` alone."""
    classes = _classes()
    multi = {k: v for k, v in classes.items() if len(v) > 1}
    entries_in_multi = sum(len(v) for v in multi.values())
    total_entries = sum(len(v) for v in classes.values())
    size_hist: Dict[int, int] = defaultdict(int)
    for v in classes.values():
        size_hist[len(v)] += 1
    largest = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:10]
    return {
        "entries_resolvable": total_entries,
        "distinct_classes": len(classes),
        "classes_with_duplicates": len(multi),
        "entries_in_duplicate_classes": entries_in_multi,
        "share_entries_in_duplicate_classes": round(entries_in_multi / total_entries, 4)
        if total_entries else 0.0,
        "class_size_histogram": {str(k): v for k, v in sorted(size_hist.items())},
        "largest_classes": [
            {"message": m, "originalType": t, "n": len(ids)} for (m, t), ids in largest
        ],
    }


def ambiguous_pairs() -> List[Dict[str, object]]:
    """Subjects carrying BOTH an enable and a disable entry.

    These are the cases gate [2] exists for: lexically near-identical, separable only by
    the step's polarity wording. Reported so the ablation's polarity row can be read
    against the number of cases it could possibly affect.
    """
    by_subject: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for (message, otype), ids in _classes().items():
        if otype in ("onURL", "offURL"):
            by_subject[subject_of(message).lower()][otype].extend(ids)
    out = []
    for subject, types in sorted(by_subject.items()):
        if types.get("onURL") and types.get("offURL"):
            out.append({
                "subject": subject,
                "onURL": sorted(types["onURL"]),
                "offURL": sorted(types["offURL"]),
            })
    return out


def stats() -> Dict[str, object]:
    cases = build_cases()
    by_type: Dict[str, int] = defaultdict(int)
    for c in cases:
        by_type[c.original_type] += 1
    ambiguous = sum(1 for c in cases if c.is_ambiguous_class)
    return {
        "n_cases": len(cases),
        "n_classes": len(cases) // len(next(iter(TEMPLATES.values()))) if cases else 0,
        "cases_by_type": dict(sorted(by_type.items())),
        "cases_in_duplicate_classes": ambiguous,
        "n_ambiguous_enable_disable_subjects": len(ambiguous_pairs()),
        "duplicates": duplicate_message_stats(),
    }


if __name__ == "__main__":  # quick inspection: python -m evaluation.synthetic
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    print(json.dumps(stats(), indent=2))
    for c in build_cases()[:6]:
        print(f"  {c.case_id:28s} {c.step!r}  gold={sorted(c.gold)}")
