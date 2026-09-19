"""Offline deterministic provider. No API key, no network — CI runs entirely on this.

It is NOT a canned answer: it is a real structural parser over whatever evidence it is
handed (directive 17 — it contains no fixture string). Quality is below the LLM's, which
is the point: if the deterministic half of the pipeline is correct, the stub still
produces schema-valid, gate-passing output on any article, and every test below the
extraction boundary stays hermetic.
"""
from __future__ import annotations

import re
from typing import List

from ..ir import ExtractedAction, ExtractedStep, ExtractedStepGroup, Extraction
from ..text import content
from .base import LLMProvider

_HEADING = re.compile(r"^\s{0,3}#{1,4}\s*(?:\d+[.)]\s*)?(.+?)\s*$", re.MULTILINE)
_SENT = re.compile(r"(?<=[.!?])\s+")
# A step is an instruction: starts with a bare verb, or is an imperative clause.
_IMPERATIVE = re.compile(
    r"^(tap|press|touch|select|choose|open|navigate|go|swipe|scroll|check|turn|switch|"
    r"enable|disable|remove|wipe|clean|connect|insert|eject|shine|contact|back up|"
    r"restart|reboot|hold|try|use|avoid|adjust|set|confirm|inspect|examine)\b",
    re.IGNORECASE,
)
MAX_STEPS_PER_ACTION = 8
MAX_ACTIONS = 12


def _title_case(s: str) -> str:
    return " ".join(w.capitalize() if w.islower() or w.isupper() else w for w in s.split())


def _sentences(block: str) -> List[tuple[str, int, int]]:
    out, cursor = [], 0
    for part in _SENT.split(block):
        part_clean = part.strip()
        if not part_clean:
            continue
        idx = block.find(part_clean, cursor)
        if idx < 0:
            idx = cursor
        out.append((part_clean, idx, idx + len(part_clean)))
        cursor = idx + len(part_clean)
    return out


def _paraphrase(query: str, n: int = 9) -> List[str]:
    """Deterministic register variations. The real provider writes better ones; these
    only need to be distinct and register-varied to exercise the cache seeding."""
    core = " ".join(content(query)[:8]) or query.lower()
    templates = [
        "{q}",
        "{c}",
        "problem: {c}",
        "why does this happen - {c}",
        "having trouble with {c}",
        "{c} issue on my device",
        "need help, {c}",
        "this is frustrating, {c}",
        "how do i fix {c}",
        "device fault: {c}",
    ]
    seen, out = set(), []
    for t in templates:
        v = t.format(q=query.strip(), c=core).strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
        if len(out) >= n:
            break
    return out


class StubProvider(LLMProvider):
    name = "stub"
    model = "offline-structural-parser"

    def extract(self, query: str, evidence: str) -> Extraction:
        headings = list(_HEADING.finditer(evidence))
        actions: List[ExtractedAction] = []

        for i, m in enumerate(headings):
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(evidence)
            block = evidence[start:end]
            heading = m.group(1).strip()
            if not heading or len(heading) > 90:
                continue

            steps: List[ExtractedStep] = []
            for sent, s0, s1 in _sentences(block):
                if _IMPERATIVE.match(sent) and 8 <= len(sent) <= 200:
                    steps.append(ExtractedStep(text=sent, source_span=(start + s0, start + s1)))
                if len(steps) >= MAX_STEPS_PER_ACTION:
                    break
            if not steps:
                continue

            subject = " ".join(content(heading)[:2]) or "the reported"
            actions.append(
                ExtractedAction(
                    action_name=_title_case(" ".join(heading.split()[:4])),
                    description=f"It will address {subject} on your device",
                    category_hint=None,
                    step_groups=[ExtractedStepGroup(steps=steps)],
                    source_span=(m.start(), end),
                )
            )
            if len(actions) >= MAX_ACTIONS:
                break

        # Topic and title come from the EVIDENCE, not the complaint: the article names the
        # issue in support vocabulary, whereas the raw complaint is colloquial and would
        # yield a topic like "My Galaxy".
        headline = evidence.split("\n", 1)[0] if evidence else ""
        subject = content(headline) or content(query)
        topic = " ".join(subject[:2]) or "Device"
        return Extraction(
            goal_topic=_title_case(topic),
            title=(" ".join(subject[:3]) or "device issue").capitalize(),
            actions=actions,
            query_variations=_paraphrase(query),
        )
