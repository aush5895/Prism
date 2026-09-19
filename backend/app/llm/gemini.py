"""Gemini provider. Default per directive. Key is read from the environment, never stored.

google-genai is imported lazily so the package stays an optional dependency: CI installs
nothing and runs on StubProvider/ReplayProvider.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from .. import config
from ..ir import Extraction
from .base import EXTRACTION_JSON_SCHEMA, SYSTEM_PROMPT, USER_TEMPLATE, LLMProvider

# Public per-1M-token pricing, overridable without a code change. Used only to populate
# meta.cost_usd; it is an estimate and metrics.md labels it as such.
_PRICE_PER_MTOK: Dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


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
        from google.genai import types  # type: ignore

        resp = self._client.models.generate_content(
            model=self.model,
            contents=USER_TEMPLATE.format(query=query, evidence=evidence),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=config.LLM_TEMPERATURE,
                response_mime_type="application/json",
                response_json_schema=EXTRACTION_JSON_SCHEMA,
            ),
        )
        self._record_usage(resp)
        return Extraction(**self._parse(resp.text))

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        """Guide §4.2.4 forbids markdown wrapping. Strip it defensively anyway — a
        provider that ignores response_mime_type must not take the request down."""
        return json.loads(_FENCE.sub("", (raw or "").strip()))

    def _record_usage(self, resp: Any) -> None:
        usage = getattr(resp, "usage_metadata", None)
        pin = float(getattr(usage, "prompt_token_count", 0) or 0)
        pout = float(getattr(usage, "candidates_token_count", 0) or 0)
        rate_in, rate_out = _PRICE_PER_MTOK.get(self.model, (0.0, 0.0))
        self._usage = {
            "prompt_tokens": pin,
            "completion_tokens": pout,
            "cost_usd": (pin * rate_in + pout * rate_out) / 1_000_000,
        }

    def last_usage(self) -> Dict[str, float]:
        return dict(self._usage)
