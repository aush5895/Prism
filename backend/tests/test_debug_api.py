"""The non-spec debug surface the demo UI runs on.

Two things are load-bearing here. The resolver trace must never leak into the graded
`response` object (guide §4.2.4 forbids extra fields in the schema-validated payload),
and the source spans the UI highlights must actually support the step they are attached
to — a confident highlight over unrelated text would turn the "we did not hallucinate"
panel into a demonstration of the opposite.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.contracts import TroubleshootRequest
from app.llm.replay import ReplayProvider
from app.main import app, run_pipeline
from app.pipeline.deeplinks import get_catalog
from app.pipeline.spans import locate
from app.schema_samsung import ContextDeeplinkResponse
from app.text import content, tokens

from conftest import FIXTURES


@pytest.fixture
def row21(repo_root):
    rows = json.loads((repo_root / "data" / "siis_responses.json").read_text(encoding="utf-8"))
    return next(r for r in rows["responses"] if r["id"] == "row_21")


@pytest.fixture
def debug_run(row21):
    debug: dict = {}
    envelope = run_pipeline(
        TroubleshootRequest(query=row21["original_query"],
                            siis_response=row21["siis_response"]),
        provider=ReplayProvider(FIXTURES / "extraction_row21.json"),
        debug_sink=debug, use_cache=False)
    return envelope, debug


# ----------------------------------------------------------------- isolation
def test_debug_data_never_enters_the_graded_response(debug_run):
    """The trace is a SIBLING of `response`, never a field inside it."""
    envelope, debug = debug_run
    assert debug["resolutions"], "the trace should exist"

    blob = json.dumps(envelope.response)
    for leaked in ("resolutions", "rejected", "bm25", "verdict", "source_span",
                   "provenance", "canonical"):
        assert leaked not in blob, f"{leaked!r} leaked into the graded response"

    # and it still satisfies Samsung's unmodified schema
    ContextDeeplinkResponse(**envelope.response)


def test_running_without_a_debug_sink_changes_nothing(row21):
    """The graded path must be byte-identical whether or not anyone is watching."""
    provider = ReplayProvider(FIXTURES / "extraction_row21.json")
    request = TroubleshootRequest(query=row21["original_query"],
                                  siis_response=row21["siis_response"])
    plain = run_pipeline(request, provider=provider, use_cache=False)
    sink: dict = {}
    watched = run_pipeline(request, provider=provider, use_cache=False, debug_sink=sink)
    assert json.dumps(plain.response, sort_keys=True) == \
        json.dumps(watched.response, sort_keys=True)
    assert sink


# ----------------------------------------------------------------- payload shape
def test_debug_payload_carries_what_the_ui_panels_need(debug_run):
    _envelope, debug = debug_run
    assert set(debug) >= {"enrichment", "evidence", "resolutions", "spans", "score_terms"}
    assert debug["enrichment"]["device"] and debug["enrichment"]["canonical"]
    assert debug["evidence"]["text"]
    trace = debug["resolutions"][0]
    assert {"action", "decision", "reason", "rejected"} <= set(trace)


def test_every_rejected_candidate_carries_a_verdict_and_its_intent(debug_run):
    """Panel 3 has to explain WHY a candidate lost, not merely that it did."""
    _envelope, debug = debug_run
    rejects = [r for res in debug["resolutions"] for r in res["rejected"]]
    assert rejects
    for reject in rejects:
        assert reject["verdict"].startswith("reject:")
        assert reject["intent"] in {"ON", "OFF", "VIEW", "UPDATE"}
        assert reject["message"] and reject["catalog_id"]


def test_catalog_ids_prove_every_emitted_uri_came_from_the_catalog(debug_run):
    """The demo's proof panel claims each URI was copied verbatim from a real entry.

    The graded response cannot carry a catalog id (schema.py has no field for one), so
    the id is exposed on the debug sibling, keyed by URI. Keyed by URI and not by
    position so it survives a cache hit, where no resolver trace exists at all.
    """
    envelope, debug = debug_run
    ids = debug["catalog_ids"]
    assert ids, "row_21 emits deeplinks, so ids must be present"

    catalog = get_catalog()
    emitted = [g["actionableDeeplink"] for a in envelope.response["contexts"][0]["actions"]
               for g in a["stepGroups"] if g["actionableDeeplink"]]
    for deeplink in emitted:
        uri = deeplink["deeplink"]
        assert uri in ids, f"{uri} emitted without a catalog id"
        entry = catalog.by_id[ids[uri]]
        assert entry["deeplink"] == uri
        # the placeholder's prose is authored per the catalog's own rule; everything
        # else must match the catalog field for field
        if uri != "bixby://dummy_positive":
            assert entry["message"] == deeplink["message"]
            assert entry["originalType"] == deeplink["originalType"]


def test_catalog_ids_survive_a_cache_hit(row21):
    """A cached plan never runs the resolver, so the trace is gone -- but the proof panel
    must still be able to name the entry behind each URI."""
    from app.pipeline.cache import reset_cache

    reset_cache()
    request = TroubleshootRequest(query=row21["original_query"],
                                  siis_response=row21["siis_response"])
    provider = ReplayProvider(FIXTURES / "extraction_row21.json")

    cold: dict = {}
    run_pipeline(request, provider=provider, debug_sink=cold)
    warm: dict = {}
    envelope = run_pipeline(request, provider=provider, debug_sink=warm)

    assert envelope.meta.cache_hit is True
    assert "resolutions" not in warm, "a cache hit runs no resolver"
    assert warm["catalog_ids"] == cold["catalog_ids"]
    reset_cache()


# ----------------------------------------------------------------- span honesty
def test_located_spans_actually_support_their_step(debug_run):
    """REGRESSION. The recorded extraction's own character offsets are guesses: measured
    on this fixture, 0 of 29 model-claimed spans quoted text with any overlap with their
    step. The suite did not catch it because the only span assertion checks BOUNDS.

    Every span the UI highlights must therefore contain the step's subject vocabulary.
    """
    envelope, debug = debug_run
    evidence = debug["evidence"]["text"]
    actions = envelope.response["contexts"][0]["actions"]

    checked = 0
    for span in debug["spans"]:
        if span["start"] is None:
            assert span["provenance"] == "unlocated"
            continue
        assert 0 <= span["start"] < span["end"] <= len(evidence)
        step = (actions[span["action_index"]]["stepGroups"][span["group_index"]]
                ["steps"][span["step_index"]])
        subject = set(content(step))
        if not subject:
            continue
        quoted = set(tokens(evidence[span["start"]:span["end"]]))
        overlap = sum(1 for t in subject if t in quoted) / len(subject)
        assert overlap >= 0.5, f"span does not support {step!r}: {evidence[span['start']:span['end']]!r}"
        checked += 1
    assert checked >= 10, "expected most steps to be locatable"


def test_a_wrong_model_span_is_relocated_not_trusted():
    """A claimed offset that quotes unrelated text must lose to the real location.

    Relocation now happens in the PIPELINE (app/pipeline/spans.py), not only on the debug
    path, because span_coverage is scored from it. See test_spans.py for the full set.
    """
    evidence = ("Touchscreen issues. If your screen protector is peeling, please remove it. "
                "A damaged charger might not supply enough power.")
    start, end, provenance, _confidence = locate(
        "Remove the peeling screen protector.", evidence, (0, 12))
    assert provenance == "located"
    assert "protector" in evidence[start:end]


def test_a_step_that_cannot_be_located_returns_no_span():
    """An honest blank beats a confident highlight over unrelated text."""
    evidence = "Touchscreen issues. Go to Settings and tap Display."
    start, end, provenance, _confidence = locate(
        "Replace the refrigerator water filter cartridge.", evidence, None)
    assert start is None and end is None
    assert provenance == "unlocated"


# ----------------------------------------------------------------- http surface
def test_samples_endpoint_serves_the_supplied_rows():
    with TestClient(app) as client:
        payload = client.get("/v1/samples").json()
    assert len(payload["samples"]) == 20
    assert all({"id", "query", "siis_response"} <= set(s) for s in payload["samples"])


def test_debug_endpoint_returns_envelope_and_debug_as_siblings(row21):
    with TestClient(app) as client:
        response = client.post("/v1/troubleshoot/debug",
                               json={"query": row21["original_query"],
                                     "siis_response": row21["siis_response"]})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"envelope", "debug"}
    assert "debug" not in payload["envelope"]
    assert "resolutions" not in json.dumps(payload["envelope"]["response"])


def test_cors_allows_the_vite_dev_origin():
    with TestClient(app) as client:
        response = client.options(
            "/v1/troubleshoot/debug",
            headers={"Origin": "http://localhost:5173",
                     "Access-Control-Request-Method": "POST"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
