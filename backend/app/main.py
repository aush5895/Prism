"""FastAPI service. Endpoints per Theme 2 guide §5: POST /v1/troubleshoot, GET /health.
GET /metrics is ours and is marked non-spec in the README.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .contracts import (FALLBACK_NO_MATCH, FALLBACK_NO_SIIS_CONTEXT,
                        FALLBACK_SCHEMA_REPAIR_EXHAUSTED, Meta, TroubleshootEnvelope,
                        TroubleshootRequest)
from .llm import get_provider
from .pipeline import assemble, ground, validate
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
            meta.cache_hit = True
            meta.cost_usd = 0.0
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

    if debug_sink is not None:
        debug_sink["enrichment"] = enriched.model_dump()
        debug_sink["evidence"] = {"text": evidence.text, "title": evidence.title,
                                  "source": evidence.source, "source_id": evidence.source_id}
        debug_sink["alignment"] = alignment
        debug_sink["cache"] = {"tier": "miss", "similarity": None,
                               "matched_key": None, "source_query": None}
        # Name the WINNER as well as the losers: panel 3 exists to show why one
        # candidate beat the others, which is unreadable if only the rejects are named.
        for entry in dbg.resolutions:
            won = catalog.by_id.get(entry.get("catalog_id") or "")
            entry["accepted_message"] = won.get("message") if won else None
            entry["accepted_type"] = won.get("originalType") if won else None
        debug_sink["resolutions"] = dbg.resolutions
        debug_sink["score_terms"] = {"span_coverage": dbg.span_coverage,
                                     "deeplink_precision": dbg.deeplink_precision,
                                     "evidence_alignment": dbg.evidence_alignment}
        # Spans come from the pipeline now: assemble.build_response relocates and
        # verifies them before scoring, so the UI and the confidence score read the
        # same set rather than two independent guesses.
        debug_sink["spans"] = dbg.spans

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
                                       model=meta.model, evidence=evidence_id)

    return _finish(req, variations, response, meta, timer)


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
    return {"samples": [{"id": r["id"], "query": r["original_query"],
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
