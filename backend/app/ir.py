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


class ComplaintIssue(BaseModel):
    """One problem the customer described, and whether the article addresses it.

    `covered` is the model's CLAIM. It is only believed once pipeline/article_fit.py has
    found `evidence` in the article: the model can say "yes", but it has to quote the
    sentence that proves it, and the quote has to be there.
    """
    issue: str
    covered: bool = False
    evidence: str = ""


class Extraction(BaseModel):
    goal_topic: str
    title: str
    actions: List[ExtractedAction] = Field(default_factory=list)
    query_variations: List[str] = Field(default_factory=list)
    # Empty from providers that cannot judge coverage (the offline stub, recordings made
    # before this field existed). Empty means "unknown", never "covered".
    complaint_issues: List[ComplaintIssue] = Field(default_factory=list)


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
