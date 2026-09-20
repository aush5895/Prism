"""Gemini provider. Default per directive. Key is read from the environment, never stored.

google-genai is imported lazily so the package stays an optional dependency: CI installs
nothing and runs on StubProvider/ReplayProvider.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict

from .. import config
from ..ir import Extraction
from .base import EXTRACTION_JSON_SCHEMA, SYSTEM_PROMPT, USER_TEMPLATE, LLMProvider

# Public per-1M-token pricing, overridable without a code change. Used only to populate
# meta.cost_usd; it is an estimate and metrics.md labels it as such.
# Verified against https://ai.google.dev/gemini-api/docs/pricing on 2026-09-21. The
# 2.0/2.5 lines were retired for new API keys by Google in favour of the 3.x line
# (confirmed live: the API 404s with "no longer available to new users" and names the
# 3.x replacement) — kept here only so a recording made against an older key still
# prices correctly.
_PRICE_PER_MTOK: Dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.6-flash": (0.75, 3.75),
}

# Transient overload (503) observed live against the 3.x line during rollout spikes;
# google-genai's own tenacity retry exhausts in well under a second, so add a slower
# app-level retry on top rather than letting one blip fail the whole extraction.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _repair_spans(payload: Dict[str, Any]) -> None:
    """`ir.py` requires source_span as exactly [start, end]. response_json_schema now
    constrains this too, but a live run still produced a 4-element span for an action
    covering two disjoint regions of the article (an enable path and a disable path
    described apart in the text) before that schema fix landed — collapse any
    off-length span to its [min, max] envelope instead of hard-failing the whole
    extraction on one field."""

    def fix(span: Any) -> Any:
        if not isinstance(span, list) or len(span) == 2:
            return span
        ints = [x for x in span if isinstance(x, int)]
        return [min(ints), max(ints)] if len(ints) >= 2 else None

    for action in payload.get("actions", []) or []:
        if "source_span" in action:
            action["source_span"] = fix(action["source_span"])
        for group in action.get("step_groups", []) or []:
            for step in group.get("steps", []) or []:
                if "source_span" in step:
                    step["source_span"] = fix(step["source_span"])


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str | None = None):
        self.model = model or config.LLM_MODEL
        api_key = os.getenv(config.GEMINI_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"{config.GEMINI_API_KEY_ENV} is not set. Export it, or run with "
                f"LLM_PROVIDER=stub for offline operation."
            )
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install google-genai to use the gemini provider") from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._usage: Dict[str, float] = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "cost_usd": 0.0}

    def extract(self, query: str, evidence: str) -> Extraction:
        from google.genai import errors, types  # type: ignore

        resp = None
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=USER_TEMPLATE.format(query=query, evidence=evidence),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=config.LLM_TEMPERATURE,
                        seed=0,  # temperature=0 alone was observed live to still vary
                        # run-to-run on action grouping; a fixed seed tightens that for
                        # the replay-fixture use case, though Google does not guarantee
                        # bit-exact determinism even with both set.
                        response_mime_type="application/json",
                        response_json_schema=EXTRACTION_JSON_SCHEMA,
                    ),
                )
                break
            except errors.ServerError as exc:  # transient 503/UNAVAILABLE, observed live
                last_exc = exc
                if attempt + 1 < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
        if resp is None:
            assert last_exc is not None
            raise last_exc

        self._record_usage(resp)
        payload = self._parse(resp.text)
        _repair_spans(payload)
        return Extraction(**payload)

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        """Guide §4.2.4 forbids markdown wrapping. Strip it defensively anyway — a
        provider that ignores response_mime_type must not take the request down."""
        return json.loads(_FENCE.sub("", (raw or "").strip()))

    def _record_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage_metadata", None)
        pin = float(getattr(usage, "prompt_token_count", 0) or 0)
        # Gemini 3.x models may think before answering; thoughts_token_count is billed
        # as output but excluded from candidates_token_count, so it was silently
        # dropped from cost_usd until this fix (observed live: 83 thinking tokens on a
        # trivial prompt against gemini-3.6-flash, 0 against gemini-3.1-flash-lite).
        pout = float(getattr(usage, "candidates_token_count", 0) or 0)
        pthink = float(getattr(usage, "thoughts_token_count", 0) or 0)
        rate_in, rate_out = _PRICE_PER_MTOK.get(self.model, (0.0, 0.0))
        self._usage = {
            "prompt_tokens": pin,
            "completion_tokens": pout + pthink,
            "cost_usd": (pin * rate_in + (pout + pthink) * rate_out) / 1_000_000,
        }

    def last_usage(self) -> Dict[str, float]:
        return dict(self._usage)
