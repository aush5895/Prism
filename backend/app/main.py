"""FastAPI service. Endpoints per Theme 2 guide §5: POST /v1/troubleshoot, GET /health.
GET /metrics is ours and is marked non-spec in the README.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import config
from .contracts import (FALLBACK_NO_MATCH, FALLBACK_NO_SIIS_CONTEXT,
                        FALLBACK_SCHEMA_REPAIR_EXHAUSTED, Meta, TroubleshootEnvelope,
                        TroubleshootRequest)
from .llm import get_provider
from .pipeline import assemble, ground, validate
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
    try:
        _state["provider"] = get_provider()
    except Exception as exc:  # no key configured, or provider package missing
        log.warning("provider %s unavailable (%s); falling back to offline stub",
                    config.LLM_PROVIDER, exc)
        _state["provider"] = get_provider("stub")
    _state["ready"] = True
    yield


app = FastAPI(title="Smart Guided Troubleshooting Engine", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not _state["ready"]:
        return JSONResponse({"status": "initializing"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Not part of Samsung's contract. Operational telemetry only."""
    return METRICS.snapshot()


def run_pipeline(req: TroubleshootRequest, provider=None) -> TroubleshootEnvelope:
    """The vertical slice, end to end. Importable so tests never need a running server."""
    timer = StageTimer()
    provider = provider or _state.get("provider") or get_provider("stub")
    catalog = get_catalog()
    meta = Meta(model=f"{provider.name}:{provider.model}")

    with timer.stage("enrich"):
        enriched = enrich(req.query)

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

    variations = list(extraction.query_variations)[: config.VARIATIONS_MAX]
    usage = provider.last_usage()
    meta.cost_usd = round(usage.get("cost_usd", 0.0), 6)
    return _finish(req, variations, response, meta, timer)


def _finish(req, variations, response, meta: Meta, timer: StageTimer) -> TroubleshootEnvelope:
    meta.stage_latency_ms = timer.stages
    meta.latency_ms = timer.total_ms
    METRICS.record("/v1/troubleshoot", meta.latency_ms, timer.stages,
                   meta.cache_hit, meta.cost_usd or 0.0)
    return TroubleshootEnvelope(query=req.query, query_variations=variations,
                                response=response, meta=meta)


@app.post("/v1/troubleshoot")
def troubleshoot(req: TroubleshootRequest) -> Dict[str, Any]:
    try:
        return run_pipeline(req).model_dump(exclude_none=False)
    except validate.ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"response_refused: {exc}") from exc
