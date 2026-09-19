"""Deeplink catalog + resolver, gates [0]-[6] (docs/API_CONTRACT.md §5).

The masked URI is never indexed and never matched on (guide §7.4 / directive 3). The
BM25 document for every catalog entry is exactly `message + description + qna_description`.

Resolution has three legal outcomes and no fourth:
    exact catalog entry  |  bixby://dummy_positive  |  null
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import yaml
from rank_bm25 import BM25Okapi

from .. import config
from ..text import content, contains_any, coverage, tokens, word_count

# ---------------------------------------------------------------- intent parsing
_OFF = re.compile(r"\b(turn(?:ing)? off|switch(?:ing)? off|disable|deactivate|toggle off|uncheck)\b", re.I)
_ON = re.compile(r"\b(turn(?:ing)? on|switch(?:ing)? on|enable|activate|toggle on)\b", re.I)
_UPDATE = re.compile(r"\b(set|adjust|change|increase|decrease|drag|slide)\b", re.I)
_NAV = re.compile(r"\b(navigate|go to|open|tap|touch|select|choose|check|view|find)\b", re.I)

# Gate [2]: which catalog originalType may satisfy which parsed intent.
ALLOWED_TYPES: Dict[str, set[str]] = {
    "ON": {"onURL"},
    "OFF": {"offURL"},
    "UPDATE": {"updateURL", "onClickURL"},
    "VIEW": {"onClickURL"},
}

# A step is eligible for deeplink resolution only if it INSTRUCTS a UI interaction.
# Advisory or preparatory prose ("Back up your personal data before you continue.") names
# no screen, so it must never be resolved. Measured failure this prevents: that exact step
# matched "View Personal data intelligence" at coverage 0.67 and was emitted as the
# factory-reset deeplink.
_UI_INSTRUCTION = re.compile(
    r"\b(tap|touch|press|click|select|choose|open|opens|navigate|go to|swipe|scroll|"
    r"toggle|switch|enable|disable|turn on|turn off|turn off|set|adjust|slide|drag|check|view)\b",
    re.I,
)

_TAP_TARGET = re.compile(
    r"\b(?:tap|select|choose|open|opens)\s+(?:on\s+|the\s+|and\s+open\s+)?([A-Za-z][A-Za-z0-9 &'/-]{1,40}?)"
    r"\s*(?:\.|,|;|$|to\b|and\b|when\b|if\b)",
    re.I,
)


def parse_intent(step: str) -> str:
    """Polarity/type intent of a step. Order matters: OFF before ON ('turn off' contains
    neither 'on' as a word nor 'enable', but 'switch on'/'switch off' overlap)."""
    if _OFF.search(step):
        return "OFF"
    if _ON.search(step):
        return "ON"
    if _UPDATE.search(step) and not _NAV.search(step):
        return "UPDATE"
    return "VIEW"


# ---------------------------------------------------------------- lexicons
@lru_cache(maxsize=1)
def load_lexicons() -> Dict[str, List[str]]:
    with open(config.LEXICON_DIR / "action_lexicons.yaml") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------- results
@dataclass
class CandidateTrace:
    catalog_id: str
    message: str
    original_type: Optional[str]
    bm25: float
    coverage: float
    verdict: str  # eligible | reject:polarity | reject:scope(q) | reject:concept | reject:margin


@dataclass
class Resolution:
    decision: str  # "exact" | "dummy_positive" | "none"
    reason: str
    deeplink: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    catalog_id: Optional[str] = None
    trace: List[CandidateTrace] = field(default_factory=list)

    @property
    def is_exact(self) -> bool:
        return self.decision == "exact"


# ---------------------------------------------------------------- catalog
class DeeplinkCatalog:
    def __init__(self, path=None):
        raw = json.loads((path or config.DEEPLINKS_PATH).read_text())
        self.entries: List[Dict[str, Any]] = raw["deeplinks"]
        self.by_uri: Dict[str, Dict[str, Any]] = {e["deeplink"]: e for e in self.entries}
        self.by_id: Dict[str, Dict[str, Any]] = {e["id"]: e for e in self.entries}
        # Gate [1] corpus: descriptive metadata only. The URI is absent by construction.
        self._docs = [
            f"{e['message']} {e['description']} {e.get('qna_description') or ''}"
            for e in self.entries
        ]
        self._bm25 = BM25Okapi([tokens(d) for d in self._docs])
        self._lex = load_lexicons()

    def __len__(self) -> int:
        return len(self.entries)

    # -------------------------------------------------- gate [6]
    def verify_identity(self, emitted: Dict[str, Any]) -> None:
        """Assert an emitted deeplink is the catalog's, field for field (guide §4.2.2)."""
        uri = emitted.get("deeplink")
        entry = self.by_uri.get(uri)
        if entry is None:
            raise ValueError(f"deeplink not in catalog: {uri!r}")
        if uri == config.DUMMY_DEEPLINK:
            return  # prose is authored for DL-DUMMY by design (catalog _readme)
        for fld in ("description", "message", "originalType"):
            if emitted.get(fld) != entry.get(fld):
                raise ValueError(f"deeplink field drift on {uri}: {fld}")

    def _emit(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Copy catalog fields verbatim. Nothing here is authored or reworded."""
        return {
            "deeplink": entry["deeplink"],
            "description": entry["description"],
            "message": entry["message"],
            "originalType": entry["originalType"],
        }

    @staticmethod
    def _emit_validation(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        v = entry.get("validation")
        return dict(v) if v else None

    # -------------------------------------------------- gates [1]-[5], one step
    def resolve_step(self, step: str) -> Resolution:
        intent = parse_intent(step)
        allowed = ALLOWED_TYPES[intent]
        qualifiers = self._lex["scope_qualifiers"]

        scores = self._bm25.get_scores(tokens(step))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[: config.CANDIDATE_POOL]
        peak = max((scores[i] for i in top), default=0.0) or 1.0

        trace: List[CandidateTrace] = []
        survivors: List[tuple[float, Dict[str, Any]]] = []

        for i in top:
            e = self.entries[i]
            norm = scores[i] / peak
            cov = coverage(e["message"], step)

            if e["deeplink"] == config.DUMMY_DEEPLINK:
                verdict = "reject:placeholder"          # never selected by retrieval
            elif e["originalType"] not in allowed:
                verdict = "reject:polarity"             # gate [2]
            else:
                blob = f"{e['description']} {e['message']}"
                hit = contains_any(blob, qualifiers)
                if hit and not contains_any(step, [hit]):
                    verdict = f"reject:scope({hit})"    # gate [3]
                elif cov < config.CONCEPT_COVERAGE_MIN:
                    verdict = "reject:concept"          # gate [4]
                else:
                    verdict = "eligible"
                    survivors.append((norm, e))
            trace.append(CandidateTrace(e["id"], e["message"], e["originalType"], round(norm, 4),
                                        round(cov, 4), verdict))

        if not survivors:
            return Resolution("none", "no_candidate_survived_gates", trace=trace)

        # Gate [5]. A single survivor needs no comparison (directive 7).
        if len(survivors) > 1:
            gap = survivors[0][0] - survivors[1][0]
            if gap < config.MARGIN_DELTA:
                for t in trace:
                    if t.catalog_id == survivors[0][1]["id"]:
                        t.verdict = "reject:margin"
                return Resolution("none", f"margin_{gap:.3f}_below_{config.MARGIN_DELTA}", trace=trace)

        entry = survivors[0][1]
        emitted = self._emit(entry)
        self.verify_identity(emitted)  # gate [6]
        return Resolution("exact", f"intent={intent}", emitted,
                          self._emit_validation(entry), entry["id"], trace)

    # -------------------------------------------------- gate [0] + group resolution
    def resolve_step_group(self, steps: Sequence[str], category: str, physical: bool) -> Resolution:
        """Resolve one stepGroup to at most one deeplink.

        Gate [0] short-circuits before any retrieval runs:
          - category "manual" may never carry an actionable deeplink (guide §4.1)
          - a hardware-button sequence is not a Settings screen, whatever its category
        """
        if category == "manual":
            return Resolution("none", "manual_category_forbids_deeplink")
        if physical:
            return Resolution("none", "physical_interaction_not_a_settings_screen")

        # Terminal steps are the most specific; walk backwards and take the first exact
        # hit. Advisory prose is skipped entirely — it instructs no UI interaction.
        last: Resolution = Resolution("none", "no_ui_instruction_in_step_group")
        for step in reversed(list(steps)):
            if not _UI_INSTRUCTION.search(step):
                continue
            last = self.resolve_step(step)
            if last.is_exact:
                return last

        if self._opens_settings_screen(steps):
            return self._dummy(steps, last.trace)
        return Resolution("none", "no_settings_screen_in_step_group", trace=last.trace)

    # -------------------------------------------------- dummy_positive (catalog _readme)
    def _opens_settings_screen(self, steps: Sequence[str]) -> bool:
        joined = " ".join(steps)
        if contains_any(joined, ["settings"]):
            return True
        return bool(self._named_targets(steps))

    def _named_targets(self, steps: Sequence[str]) -> List[str]:
        """Screen names the steps actually navigate to, in order of appearance.

        Genericity is tested on the WHOLE target, not as a substring: 'Reset' is a
        generic confirmation target, but 'Factory data reset' is the screen we want and
        must not be discarded just because it contains the word.
        """
        generic = {str(g).strip().lower() for g in self._lex["generic_targets"]}
        out: List[str] = []
        for s in steps:
            for m in _TAP_TARGET.finditer(s):
                tgt = m.group(1).strip().rstrip(".,;")
                if not tgt or tgt.lower() in generic or not content(tgt):
                    continue
                if tgt not in out:
                    out.append(tgt)
        return out

    def _dummy(self, steps: Sequence[str], trace: List[CandidateTrace]) -> Resolution:
        """DL-DUMMY is the one entry whose prose we author. Its own qna_description sets
        the rule: 5-7 words naming the concrete screen taken from the steps."""
        # Navigation runs Settings -> section -> screen, so the DESTINATION is the last
        # named target and the top-level section is the first. (Selecting by length ties
        # on equal-length names and picks the wrong one.)
        targets = self._named_targets(steps)
        target = targets[-1] if targets else "the requested"
        parent = targets[0] if len(targets) > 1 else None

        description = self._fit(f"Opens the {target} settings screen")
        candidates = [f"Open {target} under {parent}"] if parent else []
        candidates += [f"Open {target} in device Settings", f"Open the {target} settings screen"]
        message = next((self._fit(c) for c in candidates if self._fits(c)), self._fit(candidates[-1]))

        entry = self.by_uri[config.DUMMY_DEEPLINK]
        emitted = {"deeplink": config.DUMMY_DEEPLINK, "description": description,
                   "message": message, "originalType": entry["originalType"]}
        self.verify_identity(emitted)
        return Resolution("dummy_positive", "settings_screen_absent_from_catalog", emitted,
                          None, entry["id"], trace)

    @staticmethod
    def _fits(s: str) -> bool:
        return config.DESCRIPTION_MIN_WORDS <= word_count(s) <= config.DESCRIPTION_MAX_WORDS

    @staticmethod
    def _fit(s: str) -> str:
        """Clamp authored prose into the 5-7 word window (guide §4.1, DL-DUMMY rule)."""
        words = s.split()
        if len(words) > config.DESCRIPTION_MAX_WORDS:
            words = words[: config.DESCRIPTION_MAX_WORDS]
        while len(words) < config.DESCRIPTION_MIN_WORDS:
            words.append("screen")
        return " ".join(words)


@lru_cache(maxsize=1)
def get_catalog() -> DeeplinkCatalog:
    return DeeplinkCatalog()
