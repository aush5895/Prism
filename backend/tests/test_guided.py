"""Guided mode (app/guided.py) — non-spec, walks the validated plan one action at a time.

What is asserted is the CONTRACT the feature makes to a customer and to an agent:
  - it never reorders the validated plan, so `critical` stays last
  - a critical action cannot be answered until it has been explicitly confirmed
  - a warning at the gate is QUOTED from the supplied article, never written by us, and
    only a destructive action is shown a data-loss warning
  - the handoff reports exactly what happened, and carries no URL
  - the graded response is untouched: /v1/troubleshoot is byte-identical with or without
    guided mode existing
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import guided, main
from app.contracts import TroubleshootRequest
from app.guided import (ACTIVE, AWAITING_CONFIRMATION, COULD_NOT_DO, DECLINED, ESCALATED,
                        FIXED, NO_PLAN, NOT_FIXED, RESOLVED, GuidedError, SessionStore)
from app.llm.replay import ReplayProvider
from app.main import app, run_pipeline
from app.pipeline.ground import normalize_siis
from app.pipeline.validate import URL_PATTERN

from conftest import FIXTURES


# ----------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def fresh_store():
    guided.reset_store()
    yield
    guided.reset_store()


@pytest.fixture
def row21_request(row21) -> TroubleshootRequest:
    return TroubleshootRequest(query=row21["original_query"],
                               siis_response=row21["siis_response"])


@pytest.fixture
def row21_envelope(row21_request) -> dict:
    """The REAL recorded Gemini extraction for row_21, through the real pipeline."""
    return run_pipeline(row21_request,
                        provider=ReplayProvider(FIXTURES / "extraction_row21.json"),
                        use_cache=False).model_dump(exclude_none=False)


@pytest.fixture
def row21_article(row21) -> str:
    return normalize_siis(row21["siis_response"]).text


def _open(envelope: dict, article: str = "", query: str = "My Galaxy S22 touch is laggy",
          store: SessionStore | None = None):
    store = store or SessionStore()
    session = store.start(query=query, envelope=envelope,
                          enrichment={"device": "Galaxy S22", "domain": "display",
                                      "symptoms": ["laggy touch"]},
                          article_title="Touchscreen issues", article_text=article)
    return store, session


def _plan(*categories: str) -> dict:
    """A validated-shape response with one action per category, in the given order."""
    return {"response": {"contexts": [{"goal": "g", "title": "t", "score": 0.5, "actions": [
        {"actionName": f"Action {i}", "description": "d", "category": c,
         "stepGroups": [{"steps": [f"Do thing {i}."], "actionableDeeplink": None,
                         "validationDeeplink": None}]}
        for i, c in enumerate(categories)]}]}, "meta": {"cache_hit": False}}


def _walk_to_status(store, session, status):
    """Answer NOT_FIXED / confirm until the session reaches `status`."""
    for _ in range(50):
        if session.status == status:
            return
        if session.status == AWAITING_CONFIRMATION:
            store.apply(session.session_id, "confirm", proceed=True)
        elif session.status == ACTIVE:
            store.apply(session.session_id, "answer", outcome=NOT_FIXED)
        else:
            break
    assert session.status == status


# ----------------------------------------------------------------- order
def test_the_walk_follows_the_validated_plan_exactly(row21_envelope, row21_article):
    """Guided mode reads the plan; it never reorders it. Contract §6 puts critical last,
    and that must survive being walked one step at a time."""
    store, session = _open(row21_envelope, row21_article)
    planned = [a["actionName"] for a in
               row21_envelope["response"]["contexts"][0]["actions"]]
    seen = []
    for _ in range(len(planned) + 5):
        current = session.current()
        if current is None:
            break
        seen.append(current["action"]["actionName"])
        if session.status == AWAITING_CONFIRMATION:
            store.apply(session.session_id, "confirm", proceed=True)
        store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    assert seen == planned
    categories = [a["category"] for a in row21_envelope["response"]["contexts"][0]["actions"]]
    first_critical = categories.index("critical")
    assert all(c == "critical" for c in categories[first_critical:])


# ----------------------------------------------------------------- the safety gate
def test_a_critical_action_cannot_be_answered_before_it_is_confirmed():
    store, session = _open(_plan("auto", "critical"))
    store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    assert session.status == AWAITING_CONFIRMATION
    with pytest.raises(GuidedError):
        store.apply(session.session_id, "answer", outcome=FIXED)
    store.apply(session.session_id, "confirm", proceed=True)
    assert session.status == ACTIVE
    store.apply(session.session_id, "answer", outcome=FIXED)
    assert session.status == RESOLVED


def test_declining_a_critical_step_is_recorded_not_skipped_silently():
    store, session = _open(_plan("critical", "critical"))
    store.apply(session.session_id, "confirm", proceed=False)
    assert session.attempts[0].outcome == DECLINED
    assert session.status == AWAITING_CONFIRMATION, "the next critical step is gated too"
    store.apply(session.session_id, "confirm", proceed=False)
    assert session.status == ESCALATED
    handoff = session.view()["handoff"]
    assert [t["outcome"] for t in handoff["tried"]] == [DECLINED, DECLINED]


def test_confirm_is_refused_when_nothing_is_waiting():
    store, session = _open(_plan("auto"))
    with pytest.raises(GuidedError):
        store.apply(session.session_id, "confirm", proceed=True)


def test_every_gate_quote_is_verbatim_from_the_supplied_article(row21_envelope, row21_article):
    """We never write warning text. Anything shown at the gate is a substring of
    Samsung's own article."""
    store, session = _open(row21_envelope, row21_article)
    quotes = []
    for action in row21_envelope["response"]["contexts"][0]["actions"]:
        if action["category"] == "critical":
            quotes += guided.safety_notice(action, row21_article)["quotes"]
    assert quotes, "row_21 has a factory reset with a warning beside it"
    for quote in quotes:
        assert quote in row21_article, f"not in the article: {quote!r}"


def test_the_factory_reset_gate_quotes_samsungs_data_loss_warning(row21_envelope,
                                                                   row21_article):
    reset = next(a for a in row21_envelope["response"]["contexts"][0]["actions"]
                 if "reset" in a["actionName"].lower())
    notice = guided.safety_notice(reset, row21_article)
    assert notice["destructive"] is True
    assert notice["source"] == "article"
    assert any("back up" in q.lower() or "erase" in q.lower() for q in notice["quotes"])


def test_a_restart_is_gated_but_never_shown_a_data_loss_warning(row21_envelope,
                                                                 row21_article):
    """REGRESSION. With warnings quoted for every critical action, row_21's restart and
    safe-mode steps were shown the factory-reset warning that sits beside them in the
    article, and a restart step was shown a sentence about wiping the screen with a
    cloth. A restart erases nothing; saying otherwise teaches the customer to click
    through the gate."""
    actions = row21_envelope["response"]["contexts"][0]["actions"]
    non_destructive = [a for a in actions if a["category"] == "critical"
                       and not guided.is_destructive(a)]
    assert non_destructive, "row_21 has a restart / safe mode step"
    for action in non_destructive:
        notice = guided.safety_notice(action, row21_article)
        assert notice["quotes"] == [], (action["actionName"], notice["quotes"])


def test_a_gate_quote_is_never_one_of_the_actions_own_instructions(row21_envelope,
                                                                     row21_article):
    """REGRESSION. 'Tap Delete all.' was quoted as the reset's warning. It is the button
    the customer presses, not a warning about pressing it."""
    for action in row21_envelope["response"]["contexts"][0]["actions"]:
        own = {s.strip().lower().rstrip(".") for g in action["stepGroups"]
               for s in g["steps"]}
        for quote in guided.safety_notice(action, row21_article)["quotes"]:
            assert quote.lower().rstrip(".") not in own


def test_no_warning_is_invented_when_the_article_has_none():
    action = {"actionName": "Factory Data Reset", "category": "critical",
              "stepGroups": [{"steps": ["Tap Factory data reset."]}]}
    notice = guided.safety_notice(action, "Open Settings. Tap Factory data reset. Done.")
    assert notice["destructive"] is True
    assert notice["quotes"] == [] and notice["source"] == "none"


# ----------------------------------------------------------------- endings
def test_fixed_ends_the_session_and_records_which_step_did_it():
    store, session = _open(_plan("manual", "auto", "auto"))
    store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    store.apply(session.session_id, "answer", outcome=FIXED)
    assert session.status == RESOLVED and session.resolved_by == 1
    with pytest.raises(GuidedError):
        store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    assert store.snapshot()["mean_steps_to_resolution"] == 2.0


def test_running_out_of_steps_hands_off_with_the_full_record():
    store, session = _open(_plan("manual", "auto"))
    store.apply(session.session_id, "answer", outcome=COULD_NOT_DO)
    store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    view = session.view()
    assert view["status"] == ESCALATED
    handoff = view["handoff"]
    assert handoff["reason"] == guided.REASON_EXHAUSTED
    assert [t["outcome"] for t in handoff["tried"]] == [COULD_NOT_DO, NOT_FIXED]
    assert handoff["not_tried"] == []
    assert "2 of 2 steps" in handoff["headline"]


def test_asking_for_an_agent_lists_what_is_left():
    store, session = _open(_plan("manual", "auto", "critical"))
    store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    store.apply(session.session_id, "escalate")
    handoff = session.view()["handoff"]
    assert handoff["reason"] == guided.REASON_REQUESTED
    assert [r["step"] for r in handoff["not_tried"]] == [2, 3]
    assert "Not yet tried:" in handoff["text"]
    with pytest.raises(GuidedError):
        store.apply(session.session_id, "escalate")


def test_an_empty_plan_hands_off_immediately():
    """The pipeline returned no plan (a `no_match` or `no_siis_context` fallback). The
    customer goes straight to a person, and the agent is told nothing was attempted.

    Not row_1: with the live provider row_1's mismatched email-server article still yields
    a five-action plan, and no measured signal separates that pairing from a correct one
    (LIMITATIONS.md). This covers the fallback paths that do produce an empty plan."""
    store, session = _open({"response": {"contexts": []},
                            "meta": {"cache_hit": False, "fallback": "no_match"}})
    view = session.view()
    assert view["status"] == NO_PLAN
    assert view["current"] is None
    assert view["handoff"]["reason"] == guided.REASON_NO_PLAN
    assert view["handoff"]["tried"] == []


def test_the_handoff_never_carries_a_url():
    store, session = _open(_plan("auto"),
                           query="see https://example.com/help my screen [x](http://y) lags")
    store.apply(session.session_id, "escalate")
    text = json.dumps(session.view()["handoff"])
    assert not URL_PATTERN.search(text)


def test_the_store_is_bounded():
    store = SessionStore(max_sessions=3, ttl_s=3600)
    ids = [_open(_plan("auto"), store=store)[1].session_id for _ in range(5)]
    assert len(store._sessions) <= 3                              # noqa: SLF001
    with pytest.raises(KeyError):
        store.get(ids[0])


# ----------------------------------------------------------------- the API
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(main._state, "provider",                  # noqa: SLF001
                        ReplayProvider(FIXTURES / "extraction_row21.json"))
    monkeypatch.setattr(main.config, "CACHE_ENABLED", False)
    return TestClient(app)


def test_api_walk_with_the_gate_enforced(client, row21):
    body = {"query": row21["original_query"], "siis_response": row21["siis_response"]}
    started = client.post("/v1/guided/start", json=body)
    assert started.status_code == 200
    session = started.json()["session"]
    sid = session["session_id"]
    assert session["progress"]["total"] >= 3

    while session["status"] == "active":
        session = client.post(f"/v1/guided/{sid}/answer",
                              json={"outcome": "not_fixed"}).json()
    assert session["status"] == "awaiting_confirmation"
    assert session["current"]["safety_gate"]["confirmed"] is False

    refused = client.post(f"/v1/guided/{sid}/answer", json={"outcome": "fixed"})
    assert refused.status_code == 409

    session = client.post(f"/v1/guided/{sid}/confirm", json={"proceed": True}).json()
    assert session["status"] == "active"
    done = client.post(f"/v1/guided/{sid}/answer", json={"outcome": "fixed"}).json()
    assert done["status"] == "resolved"

    assert client.get("/metrics").json()["guided"]["resolved"] == 1


def test_api_rejects_unknown_sessions_and_bad_outcomes(client, row21):
    assert client.get("/v1/guided/nope").status_code == 404
    body = {"query": row21["original_query"], "siis_response": row21["siis_response"]}
    sid = client.post("/v1/guided/start", json=body).json()["session"]["session_id"]
    assert client.post(f"/v1/guided/{sid}/answer",
                       json={"outcome": "maybe"}).status_code == 422


def test_guided_mode_does_not_change_the_graded_response(client, row21):
    """The envelope a guided session walks is the one /v1/troubleshoot returns."""
    body = {"query": row21["original_query"], "siis_response": row21["siis_response"]}
    graded = client.post("/v1/troubleshoot", json=body).json()
    walked = client.post("/v1/guided/start", json=body).json()["envelope"]
    assert walked["response"] == graded["response"]
    assert walked["query_variations"] == graded["query_variations"]


# ----------------------------------------------------------------- walk-through regressions
def test_a_declined_step_is_not_reported_as_tried():
    """REGRESSION. Walking row_21 and declining the factory reset produced the headline
    'Customer tried 10 of 10 steps'. An agent reading that would assume the reset had
    already been done and skip the one step left."""
    store, session = _open(_plan("auto", "critical"))
    store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    store.apply(session.session_id, "confirm", proceed=False)
    handoff = session.view()["handoff"]
    assert handoff["declined"] == 1
    assert "tried 1 of 2 steps" in handoff["headline"]
    assert "declined 1" in handoff["headline"]


def test_the_plans_support_step_is_offered_as_the_handoff_point(row21_envelope,
                                                                  row21_article):
    """REGRESSION. Walking row_21, 'Contact Support' could only be answered 'did not help',
    after which the customer walked on alone into restart, safe mode and factory reset.
    The tiers put support BEFORE critical on purpose; the view must mark it so the UI can
    offer the agent there."""
    store, session = _open(row21_envelope, row21_article)
    flagged = []
    while session.current() is not None:
        current = session.current()
        if current["support_step"]:
            flagged.append(current["action"]["actionName"])
        if session.status == AWAITING_CONFIRMATION:
            store.apply(session.session_id, "confirm", proceed=True)
        store.apply(session.session_id, "answer", outcome=NOT_FIXED)
    assert flagged, "row_21's plan contains a support step"
    actions = row21_envelope["response"]["contexts"][0]["actions"]
    names = [a["actionName"] for a in actions]
    first_critical = [a["category"] for a in actions].index("critical")
    assert all(names.index(n) < first_critical for n in flagged)
