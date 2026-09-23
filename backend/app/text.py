"""Tokenization and content-token policy — shared by retrieval and every gate.

WHY THIS IS ONE MODULE (directive 6): the target-concept gate compares a candidate's
own subject against a step's tokens. If retrieval and the gate tokenized differently,
coverage would be computed against a vocabulary the retriever never saw, and the gate
would silently reject correct candidates.

POLICY
------
tokens(s)   -> lowercased alphanumeric runs. Used for BM25 and for the RHS of coverage.
content(s)  -> tokens(s) minus two closed classes:
                 UI_VERBS   generic interaction verbs and catalog message verbs
                 STOPWORDS  function words
              Used for the LHS of coverage — i.e. for deciding what a candidate is
              actually ABOUT.

The direction matters and is the whole point of the gate:

    coverage = |content(candidate.message) & tokens(step)| / |content(candidate.message)|

Stripping UI_VERBS from the LHS is what stops a candidate passing on a shared generic
word. Measured example: the step "Check for software updates on your device." against
candidate "View Check Samsung Care+ subscription" shares the token "check", but "check"
is a UI verb, so the candidate's content set is {samsung, care, subscription} and
coverage is 0.00 -> rejected. Without the strip it would be 1/4 and, worse, that
candidate ranked #1 on BM25 and was ACCEPTED before this gate existed.

Note UI_VERBS is deliberately a closed, domain-general list: it contains interaction
verbs and the six verbs the catalog uses to open its `message` strings (View, Enable,
Disable, Adjust, Check, Increase, Diagnose, Switch). It contains no product name, no
setting name, no complaint text.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Hyphens and underscores are dropped BEFORE splitting, so a compound name tokenises the
# same however it happens to be written.
#
# REGRESSION this fixes: "Wi-Fi" tokenised to ["wi", "fi"] and "WiFi" to ["wifi"], two
# sets that never intersect. The step "Navigate to Settings, tap Connections, and then tap
# Wi-Fi." ranked DL-0308 "View WiFi Settings" at BM25 0.77 and then killed it at gate [4]
# with coverage 0.00, because the candidate's only content token was "wifi" and the step
# contained no such token. 29 catalog entries carry a compound or hyphenated name
# (WiFi, Bluetooth, SmartThings, eSIM, Auto-Sync, ...), and every one was unreachable from
# the spelling an article happens to use.
#
# This is applied inside the shared tokeniser, so BOTH sides get it: the BM25 index, the
# candidate's message and the step text. Normalising one side only would just move the
# mismatch.
_JOINERS = re.compile(r"[-_]")

# Generic interaction verbs + the verbs every catalog `message` starts with.
UI_VERBS: Set[str] = {
    "tap", "click", "press", "hold", "touch", "select", "choose", "open", "opens",
    "navigate", "go", "goes", "swipe", "scroll", "check", "checks", "view", "views",
    "enable", "enables", "disable", "disables", "turn", "switch", "switches",
    "adjust", "adjusts", "set", "sets", "update", "updates", "increase", "increases",
    "decrease", "decreases", "diagnose", "diagnoses", "retrieve", "retrieves",
    "use", "uses", "using", "via", "device", "devices", "settings", "setting",
    "page", "screen",
}

STOPWORDS: Set[str] = {
    "the", "a", "an", "to", "on", "off", "in", "of", "for", "your", "you", "and",
    "or", "it", "its", "is", "are", "be", "with", "next", "at", "by", "from",
    "then", "this", "that", "into", "onto", "again", "any", "all", "if", "when",
    "while", "so", "as", "up", "down", "out", "not", "no", "do", "does",
}

_NOISE = UI_VERBS | STOPWORDS


def tokens(s: str | None) -> List[str]:
    """Lowercased alphanumeric runs, with hyphens and underscores closed up first.

    No stemming — the catalog is already normalized. The joiner strip is what makes
    "Wi-Fi", "WiFi" and "wi_fi" a single token; see _JOINERS.
    """
    return _TOKEN_RE.findall(_JOINERS.sub("", (s or "").lower()))


def content(s: str | None) -> List[str]:
    """Tokens carrying subject meaning: UI verbs and function words removed."""
    return [t for t in tokens(s) if t not in _NOISE]


# The complete set of verbs the catalog opens a `message` with, measured over the 578
# supplied entries: View 243, Enable 140, Disable 137, Adjust 30, Check 9, Increase 5,
# Switch 1, Optimize 1. Every one is a UI verb except `optimize`, so this is UI_VERBS
# plus that single outlier rather than a second independent policy.
CATALOG_LEADING_VERBS: Set[str] = {
    "view", "enable", "disable", "adjust", "check", "increase", "switch", "optimize",
}


def subject_of(message: str) -> str:
    """A catalog message minus its leading verb: 'Enable Auto-Sync' -> 'Auto-Sync'.

    This is what the entry is ABOUT, with the instruction verb removed. Used by gate [2]
    to subtract a candidate's own name from a step before reading the step's polarity,
    and by the evaluation gold set to phrase a step from an entry. Both must strip
    identically or the measurement stops describing the resolver.
    """
    parts = (message or "").split()
    if parts and parts[0].lower() in CATALOG_LEADING_VERBS:
        parts = parts[1:]
    return " ".join(parts).strip()


def coverage(candidate_message: str, step_text: str) -> float:
    """Fraction of the CANDIDATE's own subject present in the step. See module docstring.

    Returns 0.0 when the candidate has no content tokens at all — a message made only
    of generic verbs tells us nothing about what it targets, so it may not be selected.
    """
    subject = content(candidate_message)
    if not subject:
        return 0.0
    step = set(tokens(step_text))
    return sum(1 for t in subject if t in step) / len(subject)


def word_count(s: str) -> int:
    return len([w for w in s.split() if w.strip()])


def to_title_case(s: str) -> str:
    """Title Case per guide §4.1 (`actionName`), defined once and used by both the
    repairer and the validator.

    Deliberately NOT str.title(), which capitalises after every non-alphabetic character:
    str.title() turns 'Pop-up' into 'Pop-Up' and "Samsung's" into "Samsung'S". Splitting
    on whitespace only gives the conventional reading.
    """
    return " ".join(w.capitalize() for w in s.split())


def is_title_case(s: str) -> bool:
    return s == to_title_case(s)


def contains_any(text: str, phrases: Iterable[str]) -> str | None:
    """Word-boundary phrase match. Returns the matching phrase, or None.

    Word boundaries matter: 'contact' must not fire on 'connect', and 'reset' must not
    fire on 'preset'.
    """
    low = (text or "").lower()
    for p in phrases:
        # str() guards against YAML coercing bare tokens (yes/no/on/off) to booleans.
        p = str(p)
        if re.search(rf"\b{re.escape(p.lower())}\b", low):
            return p
    return None
