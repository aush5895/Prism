"""[G] Grounding (contract §2, §7 stage 2; citations C1-C4).

`siis_response` is SUPPLIED with the request. When it is absent the guide's own order
applies (C2): cache first, then a fallback index, then give up with `no_siis_context`.
The cache leg lands on D3; this module already exposes the seam for it.

Also owns `evidence_alignment` — the third term of the confidence score (contract §3.4).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from .. import config
from ..contracts import SiisResponseObject
from ..ir import Evidence
from ..text import tokens


@lru_cache(maxsize=1)
def _reference_corpus() -> Tuple[BM25Okapi, List[dict], List[str]]:
    """BM25 over the supplied SIIS rows.

    Two jobs: (a) the C2 fallback when no evidence is supplied, (b) the IDF statistics
    that make `evidence_alignment` comparable across requests. This is reference DATA,
    loaded from data/ — no article text lives in code (directive 17).
    """
    raw = json.loads(config.SIIS_PATH.read_text())
    rows = raw["responses"]
    # Must match normalize_siis() byte for byte, so supplied evidence is recognised as a
    # corpus document rather than appended as a near-duplicate — which would split the
    # retrieval mass across two copies and depress evidence_alignment.
    docs = [f"{r['siis_response']['title']}\n{r['siis_response']['content']}" for r in rows]
    return BM25Okapi([tokens(d) for d in docs]), rows, docs


def normalize_siis(value) -> Optional[Evidence]:
    """Accept both the API contract's string form and the fixture's {title, content}."""
    if value is None:
        return None
    if isinstance(value, SiisResponseObject):
        text = f"{value.title}\n{value.content}" if value.title else value.content
        return Evidence(text=text, title=value.title, source="request")
    if isinstance(value, dict):
        title = value.get("title")
        body = value.get("content", "")
        return Evidence(text=f"{title}\n{body}" if title else body, title=title, source="request")
    text = str(value).strip()
    return Evidence(text=text, source="request") if text else None


def fallback_lookup(query: str) -> Optional[Evidence]:
    """C2's documented fallback index. Returns the best-matching reference article."""
    bm25, rows, docs = _reference_corpus()
    scores = bm25.get_scores(tokens(query))
    if not len(scores) or max(scores) <= 0:
        return None
    i = int(max(range(len(scores)), key=lambda j: scores[j]))
    row = rows[i]
    return Evidence(
        text=docs[i],
        title=row["siis_response"]["title"],
        source="fallback_index",
        source_id=row["id"],
    )


def evidence_alignment(query: str, evidence: Evidence) -> float:
    """Share of the query's top-3 retrieval mass that the grounding evidence carries.

    Computed against the reference corpus so IDF is stable. If the evidence is not one of
    the reference rows it is appended as an extra document before scoring, so supplied and
    fallback evidence are measured on the same scale. Range [0, 1].
    """
    bm25, _rows, docs = _reference_corpus()
    q = tokens(query)
    if evidence.text in docs:
        scores = list(bm25.get_scores(q))
        idx = docs.index(evidence.text)
    else:
        extended = BM25Okapi([tokens(d) for d in docs + [evidence.text]])
        scores = list(extended.get_scores(q))
        idx = len(docs)
    top3 = sum(sorted(scores, reverse=True)[:3])
    return float(scores[idx] / top3) if top3 > 0 else 0.0


def ground(query: str, siis_response) -> Evidence:
    ev = normalize_siis(siis_response)
    if ev is not None and ev.text.strip():
        return ev  # C1: supplied evidence is used directly, never second-guessed
    ev = fallback_lookup(query)
    return ev if ev is not None else Evidence(text="", source="none")
