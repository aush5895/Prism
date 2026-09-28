"""FastAPI service. Endpoints per Theme 2 guide §5: POST /v1/troubleshoot, GET /health.
GET /metrics is ours and is marked non-spec in the README.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Literal, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config, guided
from .contracts import (FALLBACK_NO_MATCH, FALLBACK_NO_SIIS_CONTEXT,
                        FALLBACK_SCHEMA_REPAIR_EXHAUSTED, Meta, TroubleshootEnvelope,
                        TroubleshootRequest)
from .llm import get_provider
from .pipeline import article_fit, assemble, ground, validate
from .pipeline.cache import evidence_key, get_cache
from .pipeline.deeplinks import get_catalog
from .pipeline.enrich import enrich
from .telemetry import METRICS, StageTimer

log = logging.getLogger("prism")

_state: Dict[str, Any] = {"ready": False, "provider": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """/health stays 503 until the catalog and indexes are built — guide §5 requires 200
    only when caching, model connections and indexes are fully initialized. Warming here
    also means cold-start cost is paid at boot, not inside the first request's latency."""
    get_catalog()
    ground._reference_corpus()  # noqa: SLF001 - warm the fallback index at boot
    if config.CACHE_ENABLED:
        # Guide §5 requires /health 200 only once caching is initialised, and loading an
        # embedding model inside the first request would land on that request's latency.
        try:
            get_cache().embedder
        except Exception as exc:  # pragma: no cover - cache is optional, never fatal
            log.warning("cache embedder unavailable (%s); running without the fast path", exc)
    try:
        _state["provider"] = get_provider()
    except Exception as exc:  # no key configured, or provider package missing
        log.warning("provider %s unavailable (%s); falling back to offline stub",
                    config.LLM_PROVIDER, exc)
        _state["provider"] = get_provider("stub")
    _state["ready"] = True
    yield


app = FastAPI(title="Smart Guided Troubleshooting Engine", version="0.1.0", lifespan=lifespan)

# The demo UI is served by Vite on another port, so the browser treats it as a different
# origin. Scoped to localhost dev origins and overridable by env; this is a development
# affordance, not an invitation for the deployed service to accept any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv(
        "PRISM_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    if not _state["ready"]:
        return JSONResponse({"status": "initializing"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Not part of Samsung's contract. Operational telemetry only."""
    snapshot = METRICS.snapshot()
    if config.CACHE_ENABLED:
        snapshot["cache"] = get_cache().snapshot()
    snapshot["guided"] = guided.get_store().snapshot()
    return snapshot


def run_pipeline(req: TroubleshootRequest, provider=None, use_cache: bool | None = None,
                 debug_sink: Dict[str, Any] | None = None) -> TroubleshootEnvelope:
    """The vertical slice, end to end. Importable so tests never need a running server.

    `debug_sink`, when supplied, is filled in place with the internals the demo UI needs:
    enrichment slots, the grounding evidence, each emitted step's source span, and the
    resolver's accept/reject trace. It is a SIBLING of the graded response and never a
    field inside it -- guide 4.2.4 forbids extra keys in the schema-validated object.
    Passing nothing changes no behaviour, so the graded path is untouched.
    """
    timer = StageTimer()
    provider = provider or _state.get("provider") or get_provider("stub")
    catalog = get_catalog()
    meta = Meta(model=f"{provider.name}:{provider.model}")
    caching = config.CACHE_ENABLED if use_cache is None else use_cache

    with timer.stage("enrich"):
        enriched = enrich(req.query)

    # Stage 8 fast path (contract §7). Before grounding and before the LLM: a hit answers
    # from an already-VALIDATED plan, so none of the work below needs to run.
    evidence_id = evidence_key(req.siis_response) if caching else None
    if caching:
        with timer.stage("cache_lookup"):
            lookup = get_cache().lookup(enriched, evidence=evidence_id)
        if lookup.hit and lookup.plan is not None:
            if debug_sink is not None:
                debug_sink["enrichment"] = enriched.model_dump()
                debug_sink["cache"] = {
                    "tier": lookup.tier, "similarity": lookup.similarity,
                    "matched_key": lookup.matched_key,
                    "source_query": lookup.source_query,
                }
                debug_sink["catalog_ids"] = _catalog_ids_for(lookup.plan.response, catalog)
                _replay_grounding(debug_sink, lookup.plan, get_cache())
            meta.cache_hit = True
            meta.cost_usd = 0.0
            # The stored fit judged the COLD run's complaint. It is replayed only when this
            # request is those same words (a refresh, or guided mode starting from the
            # plan just shown). Otherwise it judged another customer: sharing a plan does
            # not mean sharing problems ("screen flickers" and "screen flickers and the
            # battery drains" can land on one plan), and no model read this complaint, so
            # no verdict: unknown. The debug view still shows the cold run's fit, labelled.
            cached_fit = (lookup.plan.grounding or {}).get("article_fit")
            if cached_fit and _same_words(req.query, lookup.plan.source_query):
                meta.article_fit = {**cached_fit, "from_cache": True}
            else:
                meta.article_fit = article_fit.unknown("cache_hit")
            meta.model = lookup.plan.model or meta.model
            return _finish(req, list(lookup.plan.query_variations), lookup.plan.response,
                           meta, timer)

    with timer.stage("ground"):
        evidence = ground.ground(req.query, req.siis_response)

    if not evidence.text.strip():
        meta.fallback = FALLBACK_NO_SIIS_CONTEXT
        return _finish(req, [], {"contexts": []}, meta, timer)

    with timer.stage("extract"):
        extraction = provider.extract(req.query, evidence.text)

    with timer.stage("alignment"):
        alignment = ground.evidence_alignment(req.query, evidence)

    with timer.stage("article_fit"):
        # The fit is not graded, so nothing in it may cost the customer the graded plan:
        # any failure degrades it to "unknown" (review found an IndexError that became a
        # 500). It also carries article text outside `response`, so it gets the same
        # zero-URL check; _find already refuses a URL, and this fails closed.
        try:
            fit = article_fit.assess(extraction, evidence.text)
            validate.assert_no_urls(fit)
        except validate.ValidationError as exc:
            log.error("article fit carried a URL; reported as unknown: %s", exc)
            fit = article_fit.unknown("url_in_evidence")
        except Exception:  # noqa: BLE001 - deliberate: never fail the graded path
            log.exception("article fit failed; reported as unknown")
            fit = article_fit.unknown("fit_error")
    meta.article_fit = fit

    with timer.stage("resolve_and_order"):
        response, dbg = assemble.build_response(extraction, enriched, evidence, catalog, alignment)

    with timer.stage("validate"):
        response = validate.scrub_urls(response)
        if not response.get("contexts") or not response["contexts"][0]["actions"]:
            meta.fallback = FALLBACK_NO_MATCH
            response = {"contexts": []}
        else:
            try:
                validate.assert_rules(response)
                validate.validate_response(response)
            except Exception as exc:
                # D1 fails closed. The targeted repair loop lands on D3 (contract §3.3).
                log.error("schema/rule validation failed: %s", exc)
                meta.fallback = FALLBACK_SCHEMA_REPAIR_EXHAUSTED
                meta.warnings.append(str(exc))
                response = {"contexts": []}
        validate.assert_no_urls(response)

    # Name the WINNER as well as the losers: panel 3 exists to show why one candidate beat
    # the others, which is unreadable if only the rejects are named.
    for entry in dbg.resolutions:
        won = catalog.by_id.get(entry.get("catalog_id") or "")
        entry["accepted_message"] = won.get("message") if won else None
        entry["accepted_type"] = won.get("originalType") if won else None
    # Spans come from the pipeline: assemble.build_response relocates and verifies them
    # before scoring, so the UI and the confidence score read the same set rather than two
    # independent guesses. Kept with the cached plan so a hit can show them too.
    grounding = {
        "evidence": {"title": evidence.title, "source": evidence.source,
                     "source_id": evidence.source_id},
        "alignment": alignment,
        "resolutions": dbg.resolutions,
        "score_terms": {"span_coverage": dbg.span_coverage,
                        "deeplink_precision": dbg.deeplink_precision,
                        "evidence_alignment": dbg.evidence_alignment},
        "spans": dbg.spans,
        "article_fit": fit,
    }

    if debug_sink is not None:
        debug_sink["enrichment"] = enriched.model_dump()
        debug_sink["evidence"] = {"text": evidence.text, **grounding["evidence"]}
        debug_sink["alignment"] = alignment
        debug_sink["cache"] = {"tier": "miss", "similarity": None,
                               "matched_key": None, "source_query": None}
        debug_sink["resolutions"] = dbg.resolutions
        debug_sink["score_terms"] = grounding["score_terms"]
        debug_sink["spans"] = dbg.spans
        debug_sink["article_fit"] = fit
        debug_sink["catalog_ids"] = _catalog_ids_for(response, catalog)

    variations = list(extraction.query_variations)[: config.VARIATIONS_MAX]
    usage = provider.last_usage()
    meta.cost_usd = round(usage.get("cost_usd", 0.0), 6)

    # Seed the cache only now, once the plan has passed validation. One cold query stores
    # ~10 vectors (canonical + every variation), so the next user's differently-worded
    # complaint lands on this validated plan without an LLM call. Nothing that fell back
    # is ever stored - store_if_valid enforces that.
    if caching:
        with timer.stage("cache_store"):
            get_cache().store_if_valid(enriched, response, variations, meta.fallback,
                                       model=meta.model, evidence=evidence_id,
                                       grounding=grounding, article_text=evidence.text)

    return _finish(req, variations, response, meta, timer)


def _same_words(a: Optional[str], b: Optional[str]) -> bool:
    """The same complaint text, ignoring case and whitespace only."""
    squash = lambda t: " ".join((t or "").lower().split())  # noqa: E731
    return bool(a and b) and squash(a) == squash(b)


def _replay_grounding(debug_sink: Dict[str, Any], plan, cache) -> None:
    """On a cache hit, show how the plan was grounded when it was built.

    LIMITATIONS.md used to record that a hit left the grounding view blank, exactly when
    the system is fastest. The spans are replayed against the EXACT text they were
    measured on, stored with the plan, never against the incoming request's copy: two
    copies of one article can share an evidence key while differing in whitespace.
    Labelled as coming from the cold run, never as re-measured.
    """
    grounding = plan.grounding
    text = cache.article(plan)
    if not grounding or text is None:
        return
    debug_sink["evidence"] = {"text": text, **grounding["evidence"]}
    debug_sink["alignment"] = grounding["alignment"]
    debug_sink["resolutions"] = grounding["resolutions"]
    debug_sink["score_terms"] = grounding["score_terms"]
    debug_sink["spans"] = grounding["spans"]
    # The cold run's fit, for the cold run's complaint: debug only, never the customer's.
    debug_sink["article_fit"] = grounding.get("article_fit")
    debug_sink["grounding_from"] = {"cold_run_query": plan.source_query,
                                    "stored_at": plan.stored_at}


def _catalog_ids_for(response: Dict[str, Any], catalog) -> Dict[str, str]:
    """Map every emitted URI to the catalog entry it was copied from.

    The graded response cannot carry a catalog id -- schema.py has no field for one -- but
    the demo's proof panel needs it to show the URI was taken verbatim from a real entry
    rather than assembled. Keyed by URI rather than by position so it survives a CACHE
    HIT, where no resolver trace exists because the resolver never ran.
    """
    ids: Dict[str, str] = {}
    for context in response.get("contexts", []):
        for action in context.get("actions", []):
            for group in action.get("stepGroups", []):
                deeplink = group.get("actionableDeeplink")
                uri = (deeplink or {}).get("deeplink")
                entry = catalog.by_uri.get(uri) if uri else None
                if entry:
                    ids[uri] = entry["id"]
    return ids


def _finish(req, variations, response, meta: Meta, timer: StageTimer) -> TroubleshootEnvelope:
    meta.stage_latency_ms = timer.stages
    meta.latency_ms = timer.total_ms
    METRICS.record("/v1/troubleshoot", meta.latency_ms, timer.stages,
                   meta.cache_hit, meta.cost_usd or 0.0)
    return TroubleshootEnvelope(query=req.query, query_variations=variations,
                                response=response, meta=meta)


# --------------------------------------------------------------------------- non-spec
# Everything below this line is OURS, for the demo UI. It is not part of Samsung's
# contract (guide 5 names only POST /v1/troubleshoot and GET /health), and none of it
# changes the graded response.


@app.get("/v1/samples")
def samples() -> Dict[str, Any]:
    """The supplied SIIS rows, so the demo UI can offer a complaint to run without a
    human pasting a 5 KB article into a textarea. Reference DATA, read from data/."""
    rows = json.loads(config.SIIS_PATH.read_text(encoding="utf-8"))["responses"]
    # `device` is enrich.py's own parse of the complaint, carried so the entry screen can
    # show the detected model without a JS reimplementation of _DEVICE. It is the same
    # value stage [0] computes at request time; nothing downstream reads it from here.
    return {"samples": [{"id": r["id"], "query": r["original_query"],
                         "device": enrich(r["original_query"]).device,
                         "siis_response": r["siis_response"]} for r in rows]}


@app.post("/v1/troubleshoot/debug")
def troubleshoot_debug(req: TroubleshootRequest) -> Dict[str, Any]:
    """The same pipeline, plus the internals the UI visualises.

    `envelope` is byte-identical to what POST /v1/troubleshoot returns; `debug` is a
    SIBLING key holding enrichment slots, the grounding evidence, per-step source spans
    and the resolver's accept/reject trace. Nothing here is nested inside the
    schema-validated `response` object.
    """
    debug: Dict[str, Any] = {}
    try:
        envelope = run_pipeline(req, debug_sink=debug)
    except validate.ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"response_refused: {exc}") from exc
    return {"envelope": envelope.model_dump(exclude_none=False), "debug": debug}


@app.post("/v1/troubleshoot")
def troubleshoot(req: TroubleshootRequest) -> Dict[str, Any]:
    try:
        return run_pipeline(req).model_dump(exclude_none=False)
    except validate.ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"response_refused: {exc}") from exc


# --------------------------------------------------------------------------- guided mode
# Non-spec. Walks the validated plan one action at a time; see app/guided.py. Every
# session starts from run_pipeline, so the plan it walks is exactly the graded response,
# including a cache hit.


class GuidedAnswer(BaseModel):
    model_config = {"extra": "forbid"}
    outcome: Literal["fixed", "not_fixed", "could_not_do", "skipped"]


class GuidedConfirm(BaseModel):
    model_config = {"extra": "forbid"}
    proceed: bool


def _guided_call(session_id: str, verb: str, **kwargs) -> Dict[str, Any]:
    try:
        return guided.get_store().apply_view(session_id, verb, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown or expired session") from exc
    except guided.GuidedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/guided/start")
def guided_start(req: TroubleshootRequest) -> Dict[str, Any]:
    """Run the normal pipeline, then open a session over its validated plan."""
    try:
        envelope = run_pipeline(req).model_dump(exclude_none=False)
    except validate.ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"response_refused: {exc}") from exc
    # The SAME evidence the pipeline grounded on: the supplied article, or the fallback
    # index's when none was supplied. Reading only the supplied article left the safety
    # gate with no text to quote whenever a request came without one.
    article = ground.ground(req.query, req.siis_response)
    store = guided.get_store()
    session = store.start(
        query=req.query,
        envelope=envelope,
        enrichment=enrich(req.query).model_dump(),
        article_title=article.title if article else None,
        article_text=article.text if article else "",
    )
    # Catalog ids ride beside the envelope, as they do on /v1/troubleshoot/debug, so the
    # guided screen's proof panel can name the entry behind each link.
    return {"session": store.view(session.session_id), "envelope": envelope,
            "catalog_ids": _catalog_ids_for(envelope["response"], get_catalog())}


@app.get("/v1/guided/{session_id}")
def guided_get(session_id: str) -> Dict[str, Any]:
    try:
        return guided.get_store().view(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown or expired session") from exc


@app.post("/v1/guided/{session_id}/answer")
def guided_answer(session_id: str, body: GuidedAnswer) -> Dict[str, Any]:
    """Did the action in front of the customer fix it? Refused (409) on a critical action
    that has not been confirmed, and on a session that has already ended."""
    return _guided_call(session_id, "answer", outcome=body.outcome)


@app.post("/v1/guided/{session_id}/confirm")
def guided_confirm(session_id: str, body: GuidedConfirm) -> Dict[str, Any]:
    """The safety gate. `proceed: false` records the critical step as declined and moves
    on; it is never silently skipped."""
    return _guided_call(session_id, "confirm", proceed=body.proceed)


@app.post("/v1/guided/{session_id}/escalate")
def guided_escalate(session_id: str) -> Dict[str, Any]:
    """The customer wants a person. Ends the session with an agent handoff."""
    return _guided_call(session_id, "escalate")
