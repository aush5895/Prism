"""Intermediate representation — the ONLY thing the LLM is allowed to produce.

Note what is absent: no deeplink field, anywhere. The LLM structurally cannot name a
URI (directive 4 / contract §9 row 4). Deeplinks are attached downstream by the
deterministic resolver, from the catalog.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

CategoryHint = Literal["auto", "manual", "critical"]


class ExtractedStep(BaseModel):
    text: str
    # [start, end) char offsets into the grounding evidence. Traceability is graded:
    # span_coverage is 40% of the confidence score (contract §3.4).
    source_span: Optional[Tuple[int, int]] = None


class ExtractedStepGroup(BaseModel):
    """One distinct operation on one screen (contract §7 stage 3, ambiguity A5)."""
    steps: List[ExtractedStep]


class ExtractedAction(BaseModel):
    action_name: str
    description: str
    category_hint: Optional[CategoryHint] = None  # advisory; rules override (contract §6.2)
    step_groups: List[ExtractedStepGroup]
    source_span: Optional[Tuple[int, int]] = None


class Extraction(BaseModel):
    goal_topic: str
    title: str
    actions: List[ExtractedAction] = Field(default_factory=list)
    query_variations: List[str] = Field(default_factory=list)


class EnrichedQuery(BaseModel):
    raw: str
    canonical: str
    device: Optional[str] = None
    domain: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    cache_key: str


class Evidence(BaseModel):
    text: str
    title: Optional[str] = None
    source: Literal["request", "fallback_index", "cache", "none"] = "none"
    source_id: Optional[str] = None
