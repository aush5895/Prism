"""Public API contract models — request, meta and envelope (docs/API_CONTRACT.md §2, §3).

The envelope deliberately keeps `query_variations` and `meta` as SIBLINGS of `response`.
`response` alone is validated against Samsung's unmodified ContextDeeplinkResponse, so
the graded object is exactly the shape schema.py defines (contract §3.1, ambiguity A2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

MAX_QUERY_CHARS = 2000


class SiisResponseObject(BaseModel):
    """Kit fixtures ship {title, content}; the API contract (C1) types it as a string.
    We accept both and normalize (contract §2)."""
    title: Optional[str] = None
    content: str


class TroubleshootRequest(BaseModel):
    """POST /v1/troubleshoot request body. Unknown fields are rejected, not ignored."""
    model_config = {"extra": "forbid"}

    query: str = Field(..., min_length=1)
    siis_response: Optional[Union[str, SiisResponseObject]] = None

    @field_validator("query")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class Meta(BaseModel):
    """Runtime telemetry. NEVER authored, defaulted to a flattering value, or copied
    from a previous run (contract §3.1, directive 10)."""
    latency_ms: Optional[float] = None
    cache_hit: bool = False
    model: Optional[str] = None
    cost_usd: Optional[float] = None
    fallback: Optional[str] = None
    stage_latency_ms: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class TroubleshootEnvelope(BaseModel):
    query: str
    query_variations: List[str] = Field(default_factory=list)
    response: Dict[str, Any]
    meta: Meta


# Fallback reason codes (contract §3.3)
FALLBACK_NO_MATCH = "no_match"
FALLBACK_NO_SIIS_CONTEXT = "no_siis_context"
FALLBACK_SCHEMA_REPAIR_EXHAUSTED = "schema_repair_exhausted"
