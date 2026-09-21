"""Semantic cache behaviour (contract §7 stage 8; guide §6.2, §6.3).

The cache is the one component that can return a confidently wrong answer fast, so these
tests are weighted towards what must NOT happen: no unvalidated plan is ever stored, no
plan crosses a device or domain boundary, and a hit is byte-identical to what the cold
path would have produced rather than merely similar to it.

L1 similarity is exercised through a deterministic fake embedder. The real backend is
either a downloaded transformer or a TF-IDF fit over whatever happens to be in the store,
and neither gives a test a stable number to assert on; the fake makes cosine exactly
predictable so the THRESHOLD logic is what is under test, not the model.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import pytest

from app import config
from app.contracts import TroubleshootRequest
from app.ir import EnrichedQuery
from app.llm.replay import ReplayProvider
from app.main import run_pipeline
from app.pipeline.cache import (SemanticCache, evidence_key, get_cache,
                                reset_cache)
from app.pipeline.embeddings import Embedder, get_embedder
from app.pipeline.enrich import enrich

from conftest import FIXTURES

VOCAB = ["screen", "touch", "lag", "battery", "drain", "camera", "blurry", "sound"]


class FakeEmbedder(Embedder):
    """Bag-of-words over a fixed basis, L2-normalised. Cosine is then exactly the
    overlap of the vocabulary words two strings share, which a test can reason about."""

    name = "fake-bow"
    dim = len(VOCAB)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            low = (text or "").lower()
            rows.append([1.0 if word in low else 0.0 for word in VOCAB])
        return self._normalise(np.array(rows, dtype=np.float32))


def _plan(action_name: str = "Adjust Touch Sensitivity") -> dict:
    """A minimal VALID-shaped response. Not a schema fixture — the cache never inspects
    anything below `contexts[0].actions`."""
    return {"contexts": [{
        "goal": "Follow these steps to perform this Touchscreen Troubleshooting",
        "title": "Touchscreen response issues",
        "score": 0.8,
        "actions": [{
            "actionName": action_name,
            "description": "It will improve touch response on screen",
            "category": "auto",
            "stepGroups": [{"steps": ["Tap Display."], "actionableDeeplink": None,
                            "validationDeeplink": None}],
        }],
    }]}


@pytest.fixture
def cache() -> SemanticCache:
    return SemanticCache(embedder=FakeEmbedder(), similarity_min=0.9)


# ----------------------------------------------------------------- cold start
def test_cold_cache_reports_a_miss_rather_than_crashing(cache):
    """An empty store has no matrix to multiply against. That must be a miss, not an
    exception on the very first request the service ever serves."""
    result = cache.lookup(enrich("My Galaxy S22 screen is laggy"))
    assert result.hit is False
    assert result.tier == "miss"
    assert result.plan is None
    assert len(cache) == 0


def test_cold_start_through_the_real_pipeline_does_not_crash():
    """The same property end to end, with whichever embedding backend this machine has."""
    reset_cache()
    env = run_pipeline(
        TroubleshootRequest(query="My Galaxy S22 screen is laggy and slow to respond",
                            siis_response="Touchscreen issues\nGo to Settings, tap Display."),
        provider=ReplayProvider(FIXTURES / "extraction_row21.json"))
    assert env.meta.cache_hit is False


def test_cache_without_an_embedding_backend_still_serves_l0():
    """L1 is an optimisation. With no backend at all the exact tier must keep working and
    the semantic tier must degrade to 'no match', never to an exception."""
    broken = SemanticCache(embedder=None)
    broken._embedder_failed = True  # simulate "no backend could be loaded"

    enriched = enrich("My Galaxy S22 screen is laggy")
    broken.store(enriched, _plan(), ["s22 screen lag"])

    assert broken.lookup(enriched).tier == "L0"                  # exact still works
    assert broken.lookup(enrich("totally different complaint")).hit is False  # L1 off, no crash


# ----------------------------------------------------------------- storing policy
@pytest.mark.parametrize("response,fallback,reason", [
    ({"contexts": []}, "no_match", "fallback set"),
    ({"contexts": []}, None, "no contexts"),
    ({"contexts": [{"actions": []}]}, None, "no actions"),
    (_plan(), "schema_repair_exhausted", "fallback set despite a plan"),
])
def test_only_validated_plans_are_ever_stored(cache, response, fallback, reason):
    """A cache that remembers failures serves them faster."""
    stored = cache.store_if_valid(enrich("My Galaxy S22 screen is laggy"), response,
                                  ["a", "b"], fallback)
    assert stored == 0, reason
    assert len(cache) == 0


def test_a_validated_plan_is_stored_under_every_variation(cache):
    """SEEDING is the mechanism: one cold query must seed the canonical form plus every
    paraphrase, which is what guide §4.1's 8-10 variations are actually for."""
    variations = [f"paraphrase number {i} about screen lag" for i in range(9)]
    added = cache.store_if_valid(enrich("My Galaxy S22 screen is laggy"), _plan(),
                                 variations, None)
    assert added >= len(variations)
    assert len(cache) == added


# ----------------------------------------------------------------- hit semantics
def test_l0_hit_is_exact_on_the_canonical_query(cache):
    enriched = enrich("My Galaxy S22 screen is laggy")
    cache.store(enriched, _plan(), [])
    result = cache.lookup(enrich("My Galaxy S22 screen is laggy"))
    assert result.hit and result.tier == "L0" and result.similarity == 1.0


def test_l1_hit_requires_clearing_the_threshold(cache):
    """Below the threshold is a miss. The fake embedder makes the cosine exact: the
    stored key shares no vocabulary word with the probe, so similarity is 0."""
    cache.store(enrich("My Galaxy S22 screen touch is laggy"), _plan(),
                ["screen touch lag"])
    assert cache.lookup(enrich("My camera photos look blurry")).hit is False


def test_l1_hit_on_a_paraphrase_that_clears_the_threshold(cache):
    cache.store(enrich("My Galaxy S22 screen touch is laggy"), _plan(),
                ["screen touch lag"])
    result = cache.lookup(enrich("Galaxy S22 touch and screen lag when I tap"))
    assert result.hit and result.tier == "L1"
    assert result.similarity >= cache.similarity_min
    assert result.matched_key and result.source_query


# ----------------------------------------------------------------- evidence identity
def test_a_different_article_is_never_served_from_cache(cache):
    """REGRESSION, and the most important test in this file.

    The plan is a function of (query, article), because `siis_response` arrives WITH the
    request. Keying on the query alone measured a 22% held-out false positive rate, and
    no similarity threshold from 0.40 to 0.90 brought it to zero: five of the supplied
    rows are black-screen complaints with five different articles, and a paraphrase like
    "how to fix black screen" contains nothing that could separate them. The article can,
    and it is already in the request.
    """
    enriched = enrich("My Galaxy S22 screen is completely black")
    cache.store(enriched, _plan("First Article Action"), ["screen black"],
                evidence=evidence_key("Article A: screen is cracked"))

    # identical wording, different supplied article -> must NOT hit
    result = cache.lookup(enriched, evidence=evidence_key("Article B: battery is dead"))
    assert result.hit is False
    assert result.guard_rejected is True
    assert "evidence" in (result.guard_reason or "")

    # same article -> hits
    same = cache.lookup(enriched, evidence=evidence_key("Article A: screen is cracked"))
    assert same.hit is True


def test_evidence_key_is_stable_across_equivalent_shapes():
    """The kit ships {title, content}; the contract types it as a string. Both must key
    identically or the same request would miss its own cached plan."""
    from app.contracts import SiisResponseObject

    as_object = SiisResponseObject(title="Touchscreen issues", content="Go to Settings.")
    as_string = "Touchscreen issues\nGo to Settings."
    assert evidence_key(as_object) == evidence_key(as_string)
    assert evidence_key(as_string) == evidence_key("Touchscreen issues    Go to Settings.")
    assert evidence_key(None) is None
    assert evidence_key("") is None


def test_same_article_different_wording_still_hits(cache):
    """The evidence test must not defeat the point of the cache: a reworded complaint
    about the SAME article is exactly the case the fast path exists for."""
    article = evidence_key("Touchscreen issues article body")
    cache.store(enrich("My Galaxy S22 screen touch is laggy"), _plan(),
                ["screen touch lag"], evidence=article)
    result = cache.lookup(enrich("Galaxy S22 touch and screen lag when I tap"),
                          evidence=article)
    assert result.hit and result.tier == "L1"


# ----------------------------------------------------------------- slot guard
def test_guard_rejects_a_different_device_and_reports_a_miss(cache):
    """Precision over recall: a near-identical complaint about another handset must not
    inherit this plan. Guard failure is a MISS, so the full pipeline runs."""
    cache.store(enrich("My Galaxy S22 screen touch is laggy"), _plan(),
                ["screen touch lag"])
    result = cache.lookup(enrich("My Galaxy S24 screen touch is laggy"))
    assert result.hit is False
    assert result.guard_rejected is True
    assert "device" in (result.guard_reason or "")
    assert cache.stats["guard_rejections"] == 1


def test_guard_rejects_a_different_domain(cache):
    cache.store(enrich("Galaxy S22 screen touch is laggy"), _plan(), ["screen touch lag"])
    result = cache.lookup(enrich("Galaxy S22 battery drain is fast"))
    assert result.hit is False


def test_guard_rejection_is_logged_for_audit(cache):
    """A false positive has to be findable after the fact, not argued about."""
    cache.store(enrich("My Galaxy S22 screen touch is laggy"), _plan(), ["screen touch lag"])
    cache.lookup(enrich("My Galaxy S24 screen touch is laggy"))
    entry = cache.hit_log[-1]
    assert entry["guard_rejected"] is True
    assert entry["similarity"] is not None
    assert entry["matched_key"] and entry["source_query"]
    assert entry["incoming_query"]


def test_guard_can_be_disabled_for_measurement_only():
    """The ablation needs to measure what the guard is worth; production never does this."""
    unguarded = SemanticCache(embedder=FakeEmbedder(), similarity_min=0.9, slot_guard=False)
    unguarded.store(enrich("My Galaxy S22 screen touch is laggy"), _plan(), ["screen touch lag"])
    assert unguarded.lookup(enrich("My Galaxy S24 screen touch is laggy")).hit is True


# ----------------------------------------------------------------- end to end
def test_cache_hit_returns_a_byte_identical_plan_and_sets_meta():
    """The headline guarantee: a hit is not merely similar to the cold answer, it IS the
    cold answer, and the envelope says so."""
    reset_cache()
    request = TroubleshootRequest(
        query="My Galaxy S22 screen inputs are delayed and touch is laggy",
        siis_response=json.loads(
            (FIXTURES.parent.parent.parent / "data" / "siis_responses.json").read_text()
        )["responses"][0]["siis_response"])
    provider = ReplayProvider(FIXTURES / "extraction_row21.json")

    cold = run_pipeline(request, provider=provider)
    assert cold.meta.cache_hit is False
    assert cold.response.get("contexts"), "cold path must produce a plan to cache"

    warm = run_pipeline(request, provider=provider)
    assert warm.meta.cache_hit is True
    assert json.dumps(warm.response, sort_keys=True) == json.dumps(cold.response, sort_keys=True)
    assert warm.meta.cost_usd == 0.0
    assert warm.meta.latency_ms is not None and warm.meta.latency_ms > 0


def test_guard_failure_falls_through_to_the_full_pipeline():
    """A guard rejection must not degrade the answer — it must produce a real one."""
    reset_cache()
    rows = json.loads(
        (FIXTURES.parent.parent.parent / "data" / "siis_responses.json").read_text())["responses"]
    siis = rows[0]["siis_response"]
    provider = ReplayProvider(FIXTURES / "extraction_row21.json")

    first = run_pipeline(TroubleshootRequest(
        query="My Galaxy S22 screen inputs are delayed and touch is laggy",
        siis_response=siis), provider=provider)
    assert first.response.get("contexts")

    # Same complaint, different handset: the guard must refuse the cached plan.
    other = run_pipeline(TroubleshootRequest(
        query="My Galaxy S24 Ultra screen inputs are delayed and touch is laggy",
        siis_response=siis), provider=provider)
    assert other.meta.cache_hit is False
    assert other.response.get("contexts"), "the full pipeline must still answer"


def test_cache_can_be_disabled_entirely():
    reset_cache()
    request = TroubleshootRequest(query="My Galaxy S22 screen is laggy",
                                  siis_response="Touchscreen issues\nGo to Settings.")
    provider = ReplayProvider(FIXTURES / "extraction_row21.json")
    run_pipeline(request, provider=provider, use_cache=False)
    assert run_pipeline(request, provider=provider, use_cache=False).meta.cache_hit is False


# ----------------------------------------------------------------- incremental matrix
def test_appending_a_key_gives_the_same_matrix_as_rebuilding():
    """The L1 matrix appends new rows instead of re-encoding every key.

    That is only legitimate because a transformer embedding is corpus-independent, so the
    appended matrix must be bit-for-bit what a full rebuild would produce. If this drifts,
    the cache is silently comparing against stale vectors.
    """
    incremental = SemanticCache(embedder=FakeEmbedder(), similarity_min=0.9)
    for i, text in enumerate(["screen touch lag", "battery drain fast", "camera blurry"]):
        incremental.store(enrich(f"Galaxy S22 {text}"), _plan(), [text])
        incremental.lookup(enrich("Galaxy S22 screen"))      # forces an encode each time

    rebuilt = SemanticCache(embedder=FakeEmbedder(), similarity_min=0.9)
    for text in ["screen touch lag", "battery drain fast", "camera blurry"]:
        rebuilt.store(enrich(f"Galaxy S22 {text}"), _plan(), [text])
    rebuilt.lookup(enrich("Galaxy S22 screen"))              # one encode at the end

    assert incremental._keys == rebuilt._keys
    assert np.allclose(incremental._ensure_matrix(), rebuilt._ensure_matrix())


def test_only_the_new_keys_are_encoded_after_a_store():
    """The property that makes store-then-hit constant time instead of linear in the
    whole store. Measured before the change: 1086.8 ms at 1272 keys; after: 19.6 ms."""
    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            self.encoded = []

        def encode(self, texts):
            self.encoded.append(len(texts))
            return super().encode(texts)

    embedder = CountingEmbedder()
    cache = SemanticCache(embedder=embedder, similarity_min=0.9)
    cache.store(enrich("Galaxy S22 screen touch lag"), _plan(), [f"seed {i}" for i in range(9)])
    cache.lookup(enrich("Galaxy S22 screen"))
    embedder.encoded.clear()

    cache.store(enrich("Galaxy S22 battery drain"), _plan(), ["one more key"])
    cache.lookup(enrich("Galaxy S22 screen"))

    key_encodes = [n for n in embedder.encoded if n > 1]
    assert key_encodes, "expected the new keys to be encoded"
    assert max(key_encodes) <= 3, f"re-encoded the whole store: {embedder.encoded}"


def test_a_corpus_dependent_embedder_still_rebuilds():
    """TF-IDF's IDF weights shift as the corpus grows, so its existing rows go stale and
    appending would be wrong. It must declare that and be rebuilt."""
    from app.pipeline.embeddings import TfidfSvdEmbedder

    assert TfidfSvdEmbedder.corpus_dependent is True
    assert FakeEmbedder.corpus_dependent is False

    cache = SemanticCache(embedder=TfidfSvdEmbedder(), similarity_min=0.1)
    cache.store(enrich("Galaxy S22 screen touch lag"), _plan(), ["screen lag"])
    cache.lookup(enrich("Galaxy S22 screen"))
    cache.store(enrich("Galaxy S22 battery drain"), _plan(), ["battery drain"])
    matrix = cache._ensure_matrix()
    assert matrix.shape[0] == len(cache._keys), "every key must be present after a rebuild"


def test_eviction_rebuilds_rather_than_appending():
    """Eviction drops rows from the FRONT, so the append offset stops lining up."""
    cache = SemanticCache(embedder=FakeEmbedder(), similarity_min=0.9, max_entries=4)
    for i in range(4):
        cache.store(enrich(f"Galaxy S2{i} screen issue {i}"), _plan(), [f"key {i} screen"])
        cache.lookup(enrich("Galaxy S20 screen"))
    assert len(cache._keys) <= 4
    matrix = cache._ensure_matrix()
    assert matrix.shape[0] == len(cache._keys)


# ----------------------------------------------------------------- backend selection
def test_an_embedding_backend_is_available_and_normalised():
    """Whichever backend this machine resolves to, vectors must be unit length or cosine
    is not cosine."""
    embedder = get_embedder()
    vectors = embedder.encode(["screen is laggy", "touch does not respond"])
    assert vectors.shape[0] == 2
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_snapshot_reports_which_backend_and_threshold_ran(cache):
    cache.store(enrich("My Galaxy S22 screen is laggy"), _plan(), ["screen lag"])
    cache.lookup(enrich("My Galaxy S22 screen is laggy"))
    snap = cache.snapshot()
    assert snap["l0_hits"] == 1
    assert snap["embedder"] == "fake-bow"
    assert snap["similarity_min"] == 0.9
    assert snap["slot_guard"] is True
