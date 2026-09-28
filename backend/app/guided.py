"""Guided mode: walk the VALIDATED plan one action at a time.

NOT PART OF SAMSUNG'S CONTRACT. Guide §5 names POST /v1/troubleshoot and GET /health;
everything here is ours, is served under /v1/guided/*, and never changes the graded
response. It reads that response; it does not produce one.

WHY IT EXISTS
-------------
The theme is a *Guided* Troubleshooting Engine, and a list is not guidance. A customer
handed eight actions at once does the easy ones, skips to a factory reset, or gives up
and calls. So this layer presents one action at a time, asks whether it fixed the
problem, and stops the moment it did.

Three properties are enforced by the server, not left to the UI:

  1. ORDER. Actions are presented in the order the validated plan already has: the five
     disruption tiers with `critical` last (contract §6). This module never reorders.
  2. THE SAFETY GATE. A critical action is withheld until the customer explicitly
     confirms. Answering it without confirming is refused with 409. The warning shown at
     the gate is QUOTED from the supplied article, never written here: a sentence is
     eligible only if it contains a data-loss term (data/lexicons, `data_loss_warning`)
     AND sits within GUIDED_WARNING_WINDOW_CHARS of that action in the article. If the
     article has no such sentence, the gate says so rather than inventing one.
  3. THE HANDOFF. When the plan is exhausted, the customer asks for a person, or the
     article cannot answer the complaint at all, the session ends in an agent handoff:
     what was tried, what each outcome was, and what is left. It is assembled from the
     session record with no model call, so it cannot misreport what happened.

WHAT IT DOES NOT DO
-------------------
No learning. Outcomes are recorded and counted, but nothing here changes a plan or its
order based on them: Phase 0 rejected adaptive re-planning because it cannot be measured
honestly without real users, and that is still true.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .pipeline import spans
from .pipeline.deeplinks import load_lexicons
from .pipeline.validate import URL_PATTERN
from .text import contains_any

# Terminal and non-terminal states. `awaiting_confirmation` is only ever entered for a
# critical action, and only an explicit confirm leaves it towards `active`.
ACTIVE = "active"
AWAITING_CONFIRMATION = "awaiting_confirmation"
RESOLVED = "resolved"
ESCALATED = "escalated"
NO_PLAN = "no_plan"
TERMINAL = frozenset({RESOLVED, ESCALATED, NO_PLAN})

# What a customer can report about the action in front of them.
FIXED = "fixed"
NOT_FIXED = "not_fixed"
COULD_NOT_DO = "could_not_do"
DECLINED = "declined"            # recorded by the safety gate, not by /answer
OUTCOMES = frozenset({FIXED, NOT_FIXED, COULD_NOT_DO})

# Why a session ended in a handoff.
REASON_EXHAUSTED = "plan_exhausted"
REASON_REQUESTED = "customer_requested"
REASON_NO_PLAN = "article_cannot_answer"


class GuidedError(Exception):
    """An illegal transition. The API maps this to 409 Conflict."""


# ----------------------------------------------------------------- safety gate
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def _sentences(article: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group().strip())
            for m in _SENTENCE.finditer(article or "") if m.group().strip()]


def _anchors(action: Dict[str, Any], article: str) -> List[Tuple[int, int]]:
    """Where this action is supported in the article: its name and each of its steps,
    located by the same content-overlap routine that grounds the plan's spans."""
    texts = [action.get("actionName") or ""]
    for group in action.get("stepGroups") or []:
        texts.extend(group.get("steps") or [])
    found = []
    for text in texts:
        start, end, _provenance, _conf = spans.locate(text, article)
        if start is not None:
            found.append((start, end))
    return found


def _action_text(action: Dict[str, Any]) -> str:
    parts = [action.get("actionName") or ""]
    for group in action.get("stepGroups") or []:
        parts.extend(group.get("steps") or [])
    return " ".join(parts)


def is_support_step(action: Dict[str, Any]) -> bool:
    """A 'contact support / service centre' action: the plan's own escalation point.

    The ordering tiers deliberately place service escalation (tier 3) BEFORE critical
    actions (tier 4), so the plan already says "talk to Samsung before you reset". Guided
    mode honours that by offering the handoff at this step, instead of letting a customer
    answer 'did not help' to "Contact Support" and walk on into a factory reset alone.
    Same lexicon ordering.tier_of reads, so the two can never disagree.
    """
    return contains_any(_action_text(action),
                        load_lexicons().get("service_escalation") or []) is not None


def is_destructive(action: Dict[str, Any]) -> bool:
    """Does this action destroy data? Only these get a data-loss quote at the gate."""
    return contains_any(_action_text(action),
                        load_lexicons().get("destructive_operation") or []) is not None


def safety_notice(action: Dict[str, Any], article: str) -> Dict[str, Any]:
    """The warning shown before a critical action, quoted from the supplied article.

    Every critical action is gated. Only a DESTRUCTIVE one (lexicon
    `destructive_operation`) is eligible for a data-loss quote, because a restart does not
    erase anything and showing it the reset warning that sits beside it in the article is
    a false statement. A quote must contain a `data_loss_warning` term, sit within
    GUIDED_WARNING_WINDOW_CHARS of the action in the article, and not be one of the
    action's own instructions.

    Returns {"quotes": [...], "source": "article"} or {"quotes": [], "source": "none"}.
    Never writes warning text of its own: the absence of a quote is reported as an absence.
    """
    lex = load_lexicons()
    terms = lex.get("data_loss_warning") or []
    destructive = is_destructive(action)
    anchors = _anchors(action, article)
    base = {"quotes": [], "source": "none", "located": bool(anchors),
            "destructive": destructive}
    if not destructive or not anchors or not terms:
        return base

    own_steps = {s.strip().lower().rstrip(".")
                 for g in action.get("stepGroups") or [] for s in g.get("steps") or []}
    window = config.GUIDED_WARNING_WINDOW_CHARS
    scored = []
    for start, end, sentence in _sentences(article):
        if contains_any(sentence, terms) is None:
            continue
        if sentence.lower().rstrip(".") in own_steps:
            continue
        distance = min(max(0, a_start - end, start - a_end) for a_start, a_end in anchors)
        if distance <= window:
            scored.append((distance, start, sentence))
    scored.sort()
    quotes, seen = [], set()
    for _distance, _start, sentence in scored:
        clean = URL_PATTERN.sub("", sentence).strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            quotes.append(clean)
        if len(quotes) >= config.GUIDED_MAX_WARNINGS:
            break
    return {**base, "quotes": quotes, "source": "article" if quotes else "none"}


# ----------------------------------------------------------------- session
@dataclass
class Attempt:
    index: int
    action_name: str
    category: str
    outcome: str
    at: float


@dataclass
class GuidedSession:
    session_id: str
    query: str
    device: Optional[str]
    domain: Optional[str]
    symptoms: List[str]
    article_title: Optional[str]
    article_text: str
    actions: List[Dict[str, Any]]
    cache_hit: bool
    fallback: Optional[str]
    status: str = ACTIVE
    cursor: int = 0
    confirmed: set = field(default_factory=set)
    attempts: List[Attempt] = field(default_factory=list)
    resolved_by: Optional[int] = None
    end_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---- transitions
    def _enter_current(self) -> None:
        """Move onto the action at `cursor`, or end the session if there is none."""
        if self.cursor >= len(self.actions):
            self.status, self.end_reason = ESCALATED, REASON_EXHAUSTED
            return
        action = self.actions[self.cursor]
        if action.get("category") == "critical" and self.cursor not in self.confirmed:
            self.status = AWAITING_CONFIRMATION
        else:
            self.status = ACTIVE

    def _record(self, outcome: str) -> None:
        action = self.actions[self.cursor]
        self.attempts.append(Attempt(self.cursor, action.get("actionName") or "",
                                     action.get("category") or "", outcome, time.time()))
        self.updated_at = time.time()

    def answer(self, outcome: str) -> None:
        if outcome not in OUTCOMES:
            raise GuidedError(f"outcome must be one of {sorted(OUTCOMES)}")
        if self.status == AWAITING_CONFIRMATION:
            raise GuidedError("this is a critical step: confirm or decline it first")
        if self.status != ACTIVE:
            raise GuidedError(f"session is {self.status}; nothing to answer")
        self._record(outcome)
        if outcome == FIXED:
            self.status, self.resolved_by = RESOLVED, self.cursor
            return
        self.cursor += 1
        self._enter_current()

    def confirm(self, proceed: bool) -> None:
        if self.status != AWAITING_CONFIRMATION:
            raise GuidedError("there is no critical step waiting for confirmation")
        if proceed:
            self.confirmed.add(self.cursor)
            self.status = ACTIVE
            self.updated_at = time.time()
            return
        self._record(DECLINED)
        self.cursor += 1
        self._enter_current()

    def escalate(self) -> None:
        if self.status in TERMINAL:
            raise GuidedError(f"session is already {self.status}")
        self.status, self.end_reason = ESCALATED, REASON_REQUESTED
        self.updated_at = time.time()

    # ---- views
    def current(self) -> Optional[Dict[str, Any]]:
        if self.status not in (ACTIVE, AWAITING_CONFIRMATION):
            return None
        action = self.actions[self.cursor]
        view: Dict[str, Any] = {"index": self.cursor, "action": action,
                                "support_step": is_support_step(action)}
        if action.get("category") == "critical":
            view["safety_gate"] = {
                "confirmed": self.cursor in self.confirmed,
                **safety_notice(action, self.article_text),
            }
        return view

    def view(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "session_id": self.session_id,
            "status": self.status,
            "progress": {"position": min(self.cursor + 1, len(self.actions)),
                         "total": len(self.actions)},
            "current": self.current(),
            "attempts": [a.__dict__ for a in self.attempts],
            "resolved_by": self.resolved_by,
            "cache_hit": self.cache_hit,
        }
        if self.status in (ESCALATED, NO_PLAN):
            out["handoff"] = build_handoff(self)
        return out


# ----------------------------------------------------------------- handoff
_OUTCOME_LABEL = {
    FIXED: "fixed the problem",
    NOT_FIXED: "did not help",
    COULD_NOT_DO: "customer could not complete it",
    DECLINED: "customer declined (critical step)",
}
_REASON_LABEL = {
    REASON_EXHAUSTED: "every step in the plan was tried",
    REASON_REQUESTED: "the customer asked for an agent",
    REASON_NO_PLAN: "the supplied article does not answer this complaint",
}


def _clean(text: Optional[str]) -> str:
    """Handoff text is shown to an agent and may be pasted anywhere: no URLs, ever."""
    return re.sub(r"\s+", " ", URL_PATTERN.sub("", text or "")).strip()


def build_handoff(session: GuidedSession) -> Dict[str, Any]:
    """What an agent needs so the customer is not asked to repeat anything.

    Assembled from the session record alone. No model call: a summary that could
    misreport what the customer already tried would cost more agent time than it saves.
    """
    tried = [{"step": a.index + 1, "action": _clean(a.action_name), "category": a.category,
              "outcome": a.outcome, "outcome_label": _OUTCOME_LABEL.get(a.outcome, a.outcome)}
             for a in session.attempts]
    attempted = {a.index for a in session.attempts}
    remaining = [{"step": i + 1, "action": _clean(act.get("actionName")),
                  "category": act.get("category"),
                  "has_settings_link": any((g.get("actionableDeeplink") or {}).get("deeplink")
                                           for g in act.get("stepGroups") or [])}
                 for i, act in enumerate(session.actions) if i not in attempted]
    reason = session.end_reason or (REASON_NO_PLAN if session.status == NO_PLAN else None)
    total = len(session.actions)

    declined = sum(1 for t in tried if t["outcome"] == DECLINED)
    if session.status == NO_PLAN:
        headline = ("No plan: the supplied article does not cover this complaint. "
                    "Nothing was attempted.")
    else:
        # "Tried" must not include a step the customer refused: an agent reading "tried
        # 10 of 10" would assume the factory reset had been done.
        done = len(tried) - declined
        headline = (f"Customer tried {done} of {total} step{'' if total == 1 else 's'}"
                    + (f" from '{_clean(session.article_title)}'" if session.article_title else "")
                    + (f" and declined {declined}" if declined else "")
                    + ". None resolved the issue.")

    lines = [
        "AGENT HANDOFF",
        f"Complaint: {_clean(session.query)}",
        f"Device: {session.device or 'not stated'} | Area: {session.domain or 'not detected'}",
    ]
    if session.symptoms:
        lines.append(f"Symptoms: {', '.join(_clean(s) for s in session.symptoms)}")
    if session.article_title:
        lines.append(f"Knowledge article: {_clean(session.article_title)}")
    lines.append(f"Why handed off: {_REASON_LABEL.get(reason, reason or 'unknown')}")
    lines.append(headline)
    if tried:
        lines.append("Already tried:")
        lines.extend(f"  {t['step']}. {t['action']} - {t['outcome_label']}" for t in tried)
    if remaining:
        lines.append("Not yet tried:")
        lines.extend(f"  {r['step']}. {r['action']} ({r['category']})" for r in remaining)

    return {
        "reason": reason,
        "reason_label": _REASON_LABEL.get(reason, reason),
        "headline": headline,
        "complaint": _clean(session.query),
        "device": session.device,
        "domain": session.domain,
        "symptoms": [_clean(s) for s in session.symptoms],
        "article_title": _clean(session.article_title) or None,
        "tried": tried,
        "declined": declined,
        "not_tried": remaining,
        "duration_s": round(session.updated_at - session.created_at, 1),
        "text": "\n".join(lines),
    }


# ----------------------------------------------------------------- store
class SessionStore:
    """Bounded, thread-safe, in-process. Oldest session is dropped first when full, and
    any session idle longer than the TTL is dropped on the next write."""

    def __init__(self, max_sessions: Optional[int] = None, ttl_s: Optional[int] = None):
        self._max = max_sessions or config.GUIDED_MAX_SESSIONS
        self._ttl = ttl_s or config.GUIDED_SESSION_TTL_S
        self._sessions: Dict[str, GuidedSession] = {}
        self._lock = threading.RLock()
        self.counters: Dict[str, int] = {"started": 0, RESOLVED: 0, ESCALATED: 0,
                                         NO_PLAN: 0, "steps_to_resolution_total": 0}

    def _prune(self) -> None:
        cutoff = time.time() - self._ttl
        for sid in [s for s, v in self._sessions.items() if v.updated_at < cutoff]:
            del self._sessions[sid]
        while len(self._sessions) >= self._max:
            oldest = min(self._sessions.values(), key=lambda v: v.updated_at)
            del self._sessions[oldest.session_id]

    def start(self, query: str, envelope: Dict[str, Any], enrichment: Dict[str, Any],
              article_title: Optional[str], article_text: str) -> GuidedSession:
        contexts = (envelope.get("response") or {}).get("contexts") or []
        actions = list(contexts[0].get("actions") or []) if contexts else []
        meta = envelope.get("meta") or {}
        session = GuidedSession(
            session_id=uuid.uuid4().hex,
            query=query,
            device=enrichment.get("device"),
            domain=enrichment.get("domain"),
            symptoms=list(enrichment.get("symptoms") or []),
            article_title=article_title,
            article_text=article_text or "",
            actions=actions,
            cache_hit=bool(meta.get("cache_hit")),
            fallback=meta.get("fallback"),
        )
        if not actions:
            session.status, session.end_reason = NO_PLAN, REASON_NO_PLAN
        else:
            session._enter_current()                          # noqa: SLF001
        with self._lock:
            self._prune()
            self._sessions[session.session_id] = session
            self.counters["started"] += 1
            if session.status == NO_PLAN:
                self.counters[NO_PLAN] += 1
        return session

    def get(self, session_id: str) -> GuidedSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def apply(self, session_id: str, verb: str, **kwargs) -> GuidedSession:
        """Run one transition under the store lock and count terminal outcomes once."""
        with self._lock:
            session = self.get(session_id)
            before = session.status
            getattr(session, verb)(**kwargs)
            if session.status != before and session.status in TERMINAL:
                self.counters[session.status] += 1
                if session.status == RESOLVED and session.resolved_by is not None:
                    self.counters["steps_to_resolution_total"] += session.resolved_by + 1
            return session

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            resolved = self.counters[RESOLVED]
            return {
                **self.counters,
                "live_sessions": len(self._sessions),
                "mean_steps_to_resolution": (
                    round(self.counters["steps_to_resolution_total"] / resolved, 2)
                    if resolved else None),
                "note": "counts of sessions run on this server process, not a measured "
                        "resolution rate",
            }


_STORE: Optional[SessionStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> SessionStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = SessionStore()
    return _STORE


def reset_store() -> None:
    """Tests only."""
    global _STORE
    with _STORE_LOCK:
        _STORE = None
