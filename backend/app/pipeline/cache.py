"""Two-tier semantic cache (contract §7 stage 8; Theme 2 guide §6.2, §6.3).

    L0   exact hash of the canonical query from enrich.py
    L1   embedding cosine over every stored key

WHY SEEDING IS THE MECHANISM, NOT A DETAIL
------------------------------------------
On a cold miss the plan is stored under the canonical query AND under all 8-10
`query_variations` the extractor produced. One cold query therefore seeds ~10 vectors,
and the NEXT user's differently-worded complaint lands on an already-validated plan
without an LLM call. This is what guide §4.1's paraphrase requirement is actually for:
the variations are cache seed material, not decoration.

Only plans that VALIDATED are ever stored. A fallback, a schema failure or an empty plan
is never cached — a cache that remembers failures serves them faster.

PRECISION OVER RECALL
---------------------
A hit must clear ALL THREE tests:
  1. the EVIDENCE key: the supplied article must be the one the plan was built from
  2. cosine >= config.CACHE_SIMILARITY_MIN
  3. the slot guard: cached device and domain must equal the incoming query's

Test 1 was added after measurement, and it is the one that matters most. See
`evidence_key` below: with a query-only key the held-out false positive rate was 22% and
no similarity threshold in a 0.40-0.90 sweep brought it to zero, because five of the
supplied rows are black-screen complaints with five different articles and a paraphrase
like "how to fix black screen" contains nothing that could separate them.

Guard failure is a MISS, not a downgrade: the full pipeline runs. This matters because
the two failures are not symmetric. A miss costs one LLM call. A false positive answers
a Galaxy S22 display complaint with a Galaxy Watch battery plan, confidently and in
40 ms, and nothing downstream can detect it — the plan is internally valid, it is just
about the wrong problem. Every hit and every guard rejection is logged with its
similarity, matched key and source query so false positives are auditable after the fact.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..ir import EnrichedQuery
from .embeddings import Embedder, get_embedder

log = logging.getLogger("prism.cache")


def evidence_key(siis_response: Any) -> Optional[str]:
    """Identity of the supplied reference article.

    THIS IS PART OF THE CACHE KEY, not an extra. `siis_response` is supplied WITH the
    request (frozen decision 2), so the request's full input is (query, article) and the
    plan is a function of BOTH. Keying on the query alone is not a weaker cache, it is an
    incorrect one.

    Measured on the supplied kit: five of the twenty rows are black/blank-screen
    complaints with five DIFFERENT articles and five different correct plans. A
    paraphrase like "how to fix black screen" carries no device and no distinguishing
    detail, so nothing in the query can separate them — but the article can, and it is
    right there in the request. Before this was part of the key the held-out false
    positive rate was 22%; no similarity threshold fixed it, because the information
    needed was not in the text being compared.
    """
    if siis_response is None:
        return None
    if isinstance(siis_response, str):
        text = siis_response
    else:
        title = getattr(siis_response, "title", None) or ""
        content = getattr(siis_response, "content", None)
        if content is None and isinstance(siis_response, dict):
            title = siis_response.get("title") or ""
            content = siis_response.get("content") or ""
        text = f"{title}\n{content or ''}"
    text = " ".join(text.split())
    if not text:
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class CachedPlan:
    """A validated plan plus the slots and the EVIDENCE it was built for."""
    response: Dict[str, Any]
    query_variations: List[str]
    source_query: str
    device: Optional[str]
    domain: Optional[str]
    model: Optional[str]
    evidence_key: Optional[str] = None
    stored_at: float = field(default_factory=time.time)


@dataclass
class CacheLookup:
    """The outcome of one lookup. `tier` is the headline; the rest is the audit trail."""
    hit: bool
    tier: str                       # "L0" | "L1" | "miss"
    plan: Optional[CachedPlan] = None
    similarity: Optional[float] = None
    matched_key: Optional[str] = None
    source_query: Optional[str] = None
    guard_rejected: bool = False
    guard_reason: Optional[str] = None
    lookup_ms: float = 0.0


class SemanticCache:
    """In-process, thread-safe, bounded. No Redis (Phase 0 §7): a single uvicorn process
    serves this API, so a network round trip to another process would cost more than the
    lookup it replaces."""

    def __init__(self, embedder: Optional[Embedder] = None,
                 similarity_min: Optional[float] = None,
                 slot_guard: Optional[bool] = None,
                 max_entries: Optional[int] = None):
        self._embedder = embedder
        self._similarity_min = (config.CACHE_SIMILARITY_MIN if similarity_min is None
                                else similarity_min)
        self._slot_guard = config.CACHE_SLOT_GUARD if slot_guard is None else slot_guard
        self._max_entries = max_entries or config.CACHE_MAX_ENTRIES
        self._lock = threading.RLock()

        self._l0: Dict[str, CachedPlan] = {}          # canonical query -> plan
        self._keys: List[str] = []                    # L1 key text, row-aligned with...
        self._plans: List[CachedPlan] = []            # ...the plan each key points at
        self._matrix: Optional[np.ndarray] = None     # (n, d) normalised, or None
        self._encoded = 0                             # rows of _keys already in _matrix
        self._embedder_failed = False                 # no backend; L1 off, L0 still on

        self.stats: Dict[str, int] = {
            "lookups": 0, "l0_hits": 0, "l1_hits": 0, "misses": 0,
            "guard_rejections": 0, "stores": 0, "keys": 0,
        }
        self.hit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ properties
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _embedder_or_none(self) -> Optional[Embedder]:
        """L1 is an optimisation, not a correctness requirement. If no embedding backend
        can be loaded, L0 keeps working and L1 degrades to 'no match' — the request takes
        the full pipeline and gets a correct answer slowly, which is the right direction
        to fail. Logged once, not per lookup."""
        if self._embedder is not None:
            return self._embedder
        if self._embedder_failed:
            return None
        try:
            self._embedder = get_embedder()
            return self._embedder
        except Exception as exc:
            self._embedder_failed = True
            log.warning("no embedding backend; L1 semantic cache disabled, L0 still active (%s)",
                        exc)
            return None

    @property
    def similarity_min(self) -> float:
        return self._similarity_min

    @similarity_min.setter
    def similarity_min(self, value: float) -> None:
        self._similarity_min = value

    def __len__(self) -> int:
        return len(self._plans)

    # ------------------------------------------------------------------ storing
    def store(self, enriched: EnrichedQuery, response: Dict[str, Any],
              query_variations: Sequence[str], model: Optional[str] = None,
              evidence: Optional[str] = None) -> int:
        """Store a VALIDATED plan under its canonical query and every variation.

        Callers must not call this for a fallback or an empty plan; `store_if_valid`
        makes that check explicit at the call site.
        """
        plan = CachedPlan(
            response=response,
            query_variations=list(query_variations),
            source_query=enriched.raw,
            device=enriched.device,
            domain=enriched.domain,
            model=model,
            evidence_key=evidence,
        )
        seeds = self._seed_keys(enriched, query_variations)
        with self._lock:
            self._l0[enriched.canonical] = plan
            added = 0
            existing = set(self._keys)
            for key in seeds:
                if key in existing:
                    continue
                self._keys.append(key)
                self._plans.append(plan)
                existing.add(key)
                added += 1
            self.stats["stores"] += 1
            self.stats["keys"] = len(self._keys)
            self._evict_if_needed()
        log.info("cache store: %d new keys from %r (device=%s domain=%s)",
                 added, enriched.raw[:60], enriched.device, enriched.domain)
        return added

    def store_if_valid(self, enriched: EnrichedQuery, response: Dict[str, Any],
                       query_variations: Sequence[str], fallback: Optional[str],
                       model: Optional[str] = None, evidence: Optional[str] = None) -> int:
        """The only storing path the pipeline uses. A cache that remembers failures
        serves them faster, so nothing without a real plan is ever written."""
        if fallback:
            return 0
        contexts = (response or {}).get("contexts") or []
        if not contexts or not contexts[0].get("actions"):
            return 0
        return self.store(enriched, response, query_variations, model=model,
                          evidence=evidence)

    @staticmethod
    def _seed_keys(enriched: EnrichedQuery, variations: Sequence[str]) -> List[str]:
        """Canonical query first, then the variations. Deduplicated, order stable."""
        seen, out = set(), []
        for text in [enriched.canonical, enriched.raw, *variations]:
            key = (text or "").strip()
            if not key or key.lower() in seen:
                continue
            seen.add(key.lower())
            out.append(key)
        return out

    def _evict_if_needed(self) -> None:
        """Oldest-first eviction on key count. Called under the lock."""
        if len(self._keys) <= self._max_entries:
            return
        overflow = len(self._keys) - self._max_entries
        self._keys = self._keys[overflow:]
        self._plans = self._plans[overflow:]
        # Eviction removes rows from the FRONT, so the append offset no longer lines up
        # and the matrix has to be rebuilt. Eviction is rare; a store is not.
        self._matrix = None
        self._encoded = 0
        live = {id(p) for p in self._plans}
        self._l0 = {k: v for k, v in self._l0.items() if id(v) in live}
        self.stats["keys"] = len(self._keys)

    # ------------------------------------------------------------------ lookup
    def lookup(self, enriched: EnrichedQuery,
               evidence: Optional[str] = None) -> CacheLookup:
        started = time.perf_counter()
        with self._lock:
            self.stats["lookups"] += 1

            # ---- L0: exact canonical match.
            plan = self._l0.get(enriched.canonical)
            if plan is not None:
                ok, reason = self._guard(plan, enriched, evidence)
                if ok:
                    self.stats["l0_hits"] += 1
                    result = CacheLookup(True, "L0", plan, 1.0, enriched.canonical,
                                         plan.source_query)
                    self._record(result, enriched)
                    result.lookup_ms = (time.perf_counter() - started) * 1000.0
                    return result
                self.stats["guard_rejections"] += 1
                self.stats["misses"] += 1
                result = CacheLookup(False, "miss", None, 1.0, enriched.canonical,
                                     plan.source_query, True, reason)
                self._record(result, enriched)
                result.lookup_ms = (time.perf_counter() - started) * 1000.0
                return result

            # ---- L1: cosine over every stored key.
            best_index, best_score = self._best_match(enriched)
            if best_index is not None and best_score >= self._similarity_min:
                plan = self._plans[best_index]
                ok, reason = self._guard(plan, enriched, evidence)
                if ok:
                    self.stats["l1_hits"] += 1
                    result = CacheLookup(True, "L1", plan, best_score,
                                         self._keys[best_index], plan.source_query)
                    self._record(result, enriched)
                    result.lookup_ms = (time.perf_counter() - started) * 1000.0
                    return result
                self.stats["guard_rejections"] += 1
                self.stats["misses"] += 1
                result = CacheLookup(False, "miss", None, best_score,
                                     self._keys[best_index], plan.source_query, True, reason)
                self._record(result, enriched)
                result.lookup_ms = (time.perf_counter() - started) * 1000.0
                return result

            self.stats["misses"] += 1
            result = CacheLookup(False, "miss", None, best_score)
            result.lookup_ms = (time.perf_counter() - started) * 1000.0
            return result

    def _best_match(self, enriched: EnrichedQuery) -> Tuple[Optional[int], Optional[float]]:
        """Highest cosine against the stored keys. One matmul; no index."""
        if not self._keys:
            return None, None
        matrix = self._ensure_matrix()
        if matrix is None or matrix.shape[0] == 0:
            return None, None
        embedder = self._embedder_or_none()
        if embedder is None:
            return None, None
        probe_text = enriched.raw or enriched.canonical
        probe = embedder.encode([probe_text])
        if probe.shape[0] == 0 or probe.shape[1] != matrix.shape[1]:
            return None, None
        scores = matrix @ probe[0]
        best = int(np.argmax(scores))
        return best, float(scores[best])

    def _ensure_matrix(self) -> Optional[np.ndarray]:
        """Bring the L1 matrix up to date. Called under the lock.

        APPENDS new rows rather than rebuilding, which is the difference between a fast
        path and a slow one. Measured before this: a lookup immediately after a store cost
        76.9 ms at 102 keys, 463.8 ms at 552 and 1086.8 ms at 1272 -- linear in the whole
        store, because every key was re-encoded whenever one was added. The published p95
        came from a seed-then-probe benchmark that never interleaves stores, so it never
        saw this; a live service writes on every cold miss and pays it on the next request.

        Appending is EXACT, not an approximation, for a corpus-independent embedder:
        encoding a string gives the same vector whatever else is stored. TF-IDF is the
        exception -- its IDF weights shift as the corpus grows -- so that backend is
        refitted and rebuilt wholesale, and says so via `corpus_dependent`.
        """
        if not self._keys:
            self._matrix, self._encoded = None, 0
            return None
        if self._matrix is not None and self._encoded == len(self._keys):
            return self._matrix

        embedder = self._embedder_or_none()
        if embedder is None:
            return None

        if embedder.corpus_dependent:
            embedder.fit(self._keys)
            self._matrix = embedder.encode(self._keys)
            self._encoded = len(self._keys)
            return self._matrix

        fresh = embedder.encode(self._keys[self._encoded:])
        self._matrix = (fresh if self._matrix is None or self._encoded == 0
                        else np.vstack((self._matrix, fresh)))
        self._encoded = len(self._keys)
        return self._matrix

    # ------------------------------------------------------------------ slot guard
    def _guard(self, plan: CachedPlan, enriched: EnrichedQuery,
               evidence: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Evidence identity first, then device and domain. Precision over recall.

        The EVIDENCE test is strict equality and is not optional, because the plan is a
        function of the article as much as of the query — see `evidence_key`. The device
        and domain tests treat `None` as compatible: enrichment simply did not detect a
        slot, which is not proof of a conflict, whereas two different detected values are.
        """
        if plan.evidence_key != evidence:
            return False, (f"evidence {str(evidence)[:8]!r} != cached "
                           f"{str(plan.evidence_key)[:8]!r}")
        if not self._slot_guard:
            return True, None
        if plan.device and enriched.device and \
                plan.device.strip().lower() != enriched.device.strip().lower():
            return False, f"device {enriched.device!r} != cached {plan.device!r}"
        if plan.domain and enriched.domain and plan.domain != enriched.domain:
            return False, f"domain {enriched.domain!r} != cached {plan.domain!r}"
        return True, None

    # ------------------------------------------------------------------ audit
    def _record(self, result: CacheLookup, enriched: EnrichedQuery) -> None:
        """Every hit and every guard rejection, so a false positive can be found after
        the fact rather than argued about."""
        entry = {
            "incoming_query": enriched.raw,
            "canonical": enriched.canonical,
            "tier": result.tier,
            "hit": result.hit,
            "similarity": round(result.similarity, 4) if result.similarity is not None else None,
            "matched_key": result.matched_key,
            "source_query": result.source_query,
            "guard_rejected": result.guard_rejected,
            "guard_reason": result.guard_reason,
            "device": enriched.device,
            "domain": enriched.domain,
            "evidence_key": (result.plan.evidence_key if result.plan else None),
        }
        self.hit_log.append(entry)
        if len(self.hit_log) > config.CACHE_HIT_LOG_MAX:
            del self.hit_log[: len(self.hit_log) - config.CACHE_HIT_LOG_MAX]
        if result.guard_rejected:
            log.info("cache guard rejection: %s (sim %.4f, key %r)",
                     result.guard_reason, result.similarity or 0.0, result.matched_key)
        elif result.hit:
            log.info("cache %s hit: sim %.4f, key %r, source %r", result.tier,
                     result.similarity or 0.0, result.matched_key, result.source_query)

    # ------------------------------------------------------------------ admin
    def clear(self) -> None:
        with self._lock:
            self._l0.clear()
            self._keys.clear()
            self._plans.clear()
            self._matrix = None
            self._encoded = 0
            self.hit_log.clear()
            for key in self.stats:
                self.stats[key] = 0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = self.stats["lookups"] or 1
            return {
                **self.stats,
                "entries": len(self._plans),
                "distinct_plans": len({id(p) for p in self._plans}),
                "hit_rate_pct": round(100.0 * (self.stats["l0_hits"] + self.stats["l1_hits"])
                                      / total, 1),
                "similarity_min": self._similarity_min,
                "slot_guard": self._slot_guard,
                "embedder": self._embedder.name if self._embedder else (
                    "unavailable" if self._embedder_failed else "unloaded"),
            }


_CACHE: Optional[SemanticCache] = None
_CACHE_LOCK = threading.Lock()


def get_cache() -> SemanticCache:
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = SemanticCache()
    return _CACHE


def reset_cache() -> None:
    """Drop the process-wide cache. Tests and the eval harness only."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None
