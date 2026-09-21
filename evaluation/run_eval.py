"""Measurement harness: regenerates evaluation/report.json and docs/metrics.md.

    python -m evaluation.run_eval                 # offline stub, no API key
    python -m evaluation.run_eval --provider gemini

Four measurements, in the order metrics.md presents them:

  COMPLIANCE   all 20 supplied SIIS rows through the real run_pipeline — schema validity,
               rule compliance, URL leaks, catalog identity of every emitted URI, latency.
  RESOLUTION   every synthetic gold case through catalog.resolve_step. The headline split
               is WRONG vs ABSTAINED, not one accuracy number: an abstention degrades to
               dummy_positive and still opens a Settings screen, while a wrong deeplink
               sends the user to the wrong one. Only the second is a defect that reaches
               the user as a falsehood.
  ABLATION     the same set with gates switched on cumulatively, so each gate's
               contribution is measured rather than asserted.
  MARGIN SWEEP MARGIN_DELTA fitted on a seeded random half and reported on the held-out
               half. Fitting and reporting on the same set is fitting on the test set.

EVERY number in docs/metrics.md is rendered from report.json by render_metrics(); nothing
in that file is typed by hand. backend/tests/test_evaluation.py enforces this.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config  # noqa: E402
from app.contracts import TroubleshootRequest  # noqa: E402
from app.llm import get_provider  # noqa: E402
from app.llm.base import LLMProvider  # noqa: E402
from app.llm.replay import ReplayProvider  # noqa: E402
from app.main import run_pipeline  # noqa: E402
from app.pipeline import validate  # noqa: E402
from app.pipeline.cache import (SemanticCache, evidence_key, get_cache,  # noqa: E402
                                reset_cache)
from app.pipeline.embeddings import get_embedder  # noqa: E402
from app.pipeline.enrich import enrich  # noqa: E402
from app.pipeline.deeplinks import (GATE_CONCEPT, GATE_MARGIN, GATE_POLARITY, GATE_SCOPE,
                                    GATES_ALL, GATES_NONE, get_catalog)  # noqa: E402

from evaluation import synthetic  # noqa: E402

REPORT_PATH = ROOT / "evaluation" / "report.json"
METRICS_PATH = ROOT / "docs" / "metrics.md"

SPLIT_SEED = 20260921
MARGIN_SWEEP = (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16)
# Declared BEFORE looking at any held-out number: precision is the constraint (a wrong
# deeplink is the defect this project exists to prevent), accuracy is what we maximise
# under it. Stated up front so the choice cannot be reverse-fitted to a nicer answer.
PRECISION_FLOOR = 98.0
# A held-out difference smaller than this many cases is not evidence. Half the gold set is
# ~425 cases, so one case is ~0.24 percentage points: two thresholds separated by a couple
# of cases are indistinguishable, and swapping a shipped value on that basis is noise
# chasing. Declared with the selection rule, before any held-out number is read.
MATERIAL_DELTA_CASES = 5

ABLATION_STAGES = (
    ("V0", "no gates", GATES_NONE),
    ("V1", "+ polarity", frozenset({GATE_POLARITY})),
    ("V2", "+ scope", frozenset({GATE_POLARITY, GATE_SCOPE})),
    ("V3", "+ target concept", frozenset({GATE_POLARITY, GATE_SCOPE, GATE_CONCEPT})),
    ("V4", "+ margin (shipped)", GATES_ALL),
)


def pct(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


# ----------------------------------------------------------------- resolution scoring
def score_cases(cases: Sequence[synthetic.Case], gates=None) -> Dict[str, Any]:
    """Run each case through gate [1]-[5] resolution and bucket the outcome.

    correct   an exact match on ANY member of the gold equivalence class
    wrong     an exact match on something outside it  -> user sent to the wrong screen
    abstained no exact match                          -> degrades to dummy_positive
    """
    catalog = get_catalog()
    correct = wrong = abstained = 0
    wrong_examples: List[Dict[str, Any]] = []

    for case in cases:
        res = catalog.resolve_step(case.step, gates=gates)
        if not res.is_exact:
            abstained += 1
        elif res.catalog_id in case.gold:
            correct += 1
        else:
            wrong += 1
            if len(wrong_examples) < 8:
                got = catalog.by_id.get(res.catalog_id, {})
                wrong_examples.append({
                    "step": case.step,
                    "gold_ids": sorted(case.gold),
                    "gold_message": case.message,
                    "gold_type": case.original_type,
                    "got_id": res.catalog_id,
                    "got_message": got.get("message"),
                    "got_type": got.get("originalType"),
                })

    total = len(cases)
    decided = correct + wrong
    return {
        "n_cases": total,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "accuracy_at_1_pct": pct(correct, total),
        "wrong_pct": pct(wrong, total),
        "abstained_pct": pct(abstained, total),
        "precision_pct": pct(correct, decided),
        "wrong_examples": wrong_examples,
    }


# ----------------------------------------------------------------- compliance
def _iter_groups(response: Dict[str, Any]):
    for ctx in response.get("contexts", []):
        for action in ctx.get("actions", []):
            for group in action.get("stepGroups", []):
                yield action, group


def run_compliance(provider_name: str, rate_limit_rpm: int = 0) -> Dict[str, Any]:
    """All 20 supplied rows through the production pipeline.

    `rate_limit_rpm` paces requests for live providers. Gemini's free tier allows 15
    generate_content calls per minute per model, and a 20-row run sits close enough to
    that ceiling to trip it intermittently (observed: 429 RESOURCE_EXHAUSTED mid-run).
    Pacing belongs here rather than in the provider: it is a property of how hard this
    harness hammers the API, not of how the service answers one request.
    """
    catalog = get_catalog()
    rows = json.loads((ROOT / "data" / "siis_responses.json").read_text(encoding="utf-8"))["responses"]
    min_interval_s = 60.0 / rate_limit_rpm if rate_limit_rpm > 0 else 0.0
    next_slot = 0.0

    if provider_name == "replay":
        provider = ReplayProvider(ROOT / "backend" / "tests" / "fixtures" / "extraction_row21.json")
    else:
        provider = get_provider(provider_name)

    per_row: List[Dict[str, Any]] = []
    plans: List[Dict[str, Any]] = []   # validated plans, reused by the cache evaluation
    latencies: List[float] = []
    url_leaks = catalog_invalid = 0
    auto_groups = auto_groups_with_deeplink = 0
    total_cost = 0.0

    for row in rows:
        req = TroubleshootRequest(query=row["original_query"], siis_response=row["siis_response"])
        if min_interval_s:
            wait = next_slot - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            next_slot = time.monotonic() + min_interval_s
        t0 = time.perf_counter()
        # use_cache=False: compliance measures the COLD pipeline. With the cache on,
        # row N could be answered from row M's plan and this would stop being a
        # measurement of the pipeline at all.
        env = run_pipeline(req, provider=provider, use_cache=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        total_cost += env.meta.cost_usd or 0.0

        response = env.response
        n_actions = sum(len(c.get("actions", [])) for c in response.get("contexts", []))
        resolved = dummy = null = 0
        row_invalid: List[str] = []

        for action, group in _iter_groups(response):
            dl = group.get("actionableDeeplink")
            if action["category"] == "auto":
                auto_groups += 1
                if dl:
                    auto_groups_with_deeplink += 1
            if dl is None:
                null += 1
            elif dl["deeplink"] == config.DUMMY_DEEPLINK:
                dummy += 1
            else:
                resolved += 1
            if dl:
                try:
                    catalog.verify_identity(dl)
                except ValueError as exc:
                    row_invalid.append(str(exc))

        catalog_invalid += len(row_invalid)

        try:
            validate.assert_no_urls(response)
            leaked = False
        except validate.ValidationError:
            leaked = True
            url_leaks += 1

        try:
            validate.assert_rules(response)
            rule_ok, rule_error = True, None
        except Exception as exc:
            rule_ok, rule_error = False, str(exc)

        try:
            validate.validate_response(response)
            schema_ok, schema_error = True, None
        except Exception as exc:
            schema_ok, schema_error = False, str(exc)

        if not env.meta.fallback and response.get("contexts"):
            plans.append({
                "row_id": row["id"],
                "query": row["original_query"],
                "siis_response": row["siis_response"],
                "response": response,
                "variations": list(env.query_variations),
            })

        per_row.append({
            "id": row["id"],
            "actions": n_actions,
            "resolved": resolved,
            "dummy_positive": dummy,
            "null": null,
            "schema_valid": schema_ok,
            "schema_error": schema_error,
            "rule_compliant": rule_ok,
            "rule_error": rule_error,
            "url_leak": leaked,
            "catalog_invalid": len(row_invalid),
            "fallback": env.meta.fallback,
            "latency_ms": round(elapsed_ms, 1),
        })

    n = len(per_row)
    with_plan = [r for r in per_row if r["actions"] > 0]
    ordered = sorted(latencies)
    return {
        "provider": f"{provider.name}:{provider.model}",
        "n_rows": n,
        "rows_with_plan": len(with_plan),
        "rows_fallback": sum(1 for r in per_row if r["fallback"]),
        "schema_valid_pct": pct(sum(r["schema_valid"] for r in per_row), n),
        "rule_compliant_pct": pct(sum(r["rule_compliant"] for r in per_row), n),
        "url_leaks": url_leaks,
        "catalog_invalid_deeplinks": catalog_invalid,
        "total_actions": sum(r["actions"] for r in per_row),
        "total_resolved": sum(r["resolved"] for r in per_row),
        "total_dummy_positive": sum(r["dummy_positive"] for r in per_row),
        "total_null": sum(r["null"] for r in per_row),
        "auto_step_groups": auto_groups,
        "auto_step_groups_with_deeplink": auto_groups_with_deeplink,
        "auto_step_groups_with_deeplink_pct": pct(auto_groups_with_deeplink, auto_groups),
        "latency_p50_ms": round(statistics.median(ordered), 1) if ordered else 0.0,
        "latency_p95_ms": round(ordered[max(0, int(round(0.95 * (len(ordered) - 1))))], 1)
        if ordered else 0.0,
        "latency_mean_ms": round(statistics.fmean(ordered), 1) if ordered else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "per_row": per_row,
        "_plans": plans,   # stripped before the report is written; see build_report
    }


# ----------------------------------------------------------------- ablation & sweep
def run_ablation(cases: Sequence[synthetic.Case]) -> List[Dict[str, Any]]:
    out = []
    for label, description, gates in ABLATION_STAGES:
        scored = score_cases(cases, gates=gates)
        scored.pop("wrong_examples", None)
        out.append({"stage": label, "gates": description, **scored})
    return out


class _margin:
    """Temporarily set config.MARGIN_DELTA. The resolver reads it per call, so this is
    enough to sweep it without touching production code."""

    def __init__(self, value: float):
        self.value = value

    def __enter__(self):
        self._previous = config.MARGIN_DELTA
        config.MARGIN_DELTA = self.value

    def __exit__(self, *exc):
        config.MARGIN_DELTA = self._previous


def run_margin_sweep(cases: Sequence[synthetic.Case]) -> Dict[str, Any]:
    """Fit on a seeded random half; report on the held-out half.

    The full sweep is shown for the FIT half only. Held-out numbers are reported for
    exactly two deltas — the one the fit half selected and the shipped 0.08 — because
    scanning the held-out half for the best value would make it a second fit set.
    """
    shuffled = list(cases)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    fit, holdout = shuffled[:midpoint], shuffled[midpoint:]

    fit_rows = []
    for delta in MARGIN_SWEEP:
        with _margin(delta):
            scored = score_cases(fit)
        scored.pop("wrong_examples", None)
        fit_rows.append({"margin_delta": delta, **scored})

    eligible = [r for r in fit_rows if r["precision_pct"] >= PRECISION_FLOOR]
    pool = eligible or fit_rows
    selected = max(pool, key=lambda r: (r["accuracy_at_1_pct"], r["precision_pct"]))

    holdout_rows = {}
    for delta in sorted({selected["margin_delta"], config.MARGIN_DELTA}):
        with _margin(delta):
            scored = score_cases(holdout)
        scored.pop("wrong_examples", None)
        holdout_rows[f"{delta}"] = {"margin_delta": delta, **scored}

    shipped_delta = config.MARGIN_DELTA
    selected_delta = selected["margin_delta"]
    shipped_row = holdout_rows[f"{shipped_delta}"]
    selected_row = holdout_rows[f"{selected_delta}"]

    # Verdict is COMPUTED from the held-out halves, not asserted. A change is only
    # justified if it wins by more than MATERIAL_DELTA_CASES correct answers AND does not
    # increase wrong answers, which are the defect this project exists to prevent.
    gained_correct = selected_row["correct"] - shipped_row["correct"]
    extra_wrong = selected_row["wrong"] - shipped_row["wrong"]
    change = (selected_delta != shipped_delta
              and gained_correct > MATERIAL_DELTA_CASES
              and extra_wrong <= 0)
    if selected_delta == shipped_delta:
        rationale = (f"The fit half selected the shipped δ = {shipped_delta}; nothing to "
                     f"decide.")
    elif change:
        rationale = (f"δ = {selected_delta} wins {gained_correct} more correct answers on "
                     f"the held-out half with {abs(extra_wrong)} fewer wrong, both beyond "
                     f"the {MATERIAL_DELTA_CASES}-case materiality bar. Change.")
    else:
        # Name the condition that actually bound, rather than asserting both.
        reasons = []
        if gained_correct <= MATERIAL_DELTA_CASES:
            reasons.append(
                f"it is worth only {gained_correct} more correct answers on the held-out "
                f"half, inside the {MATERIAL_DELTA_CASES}-case materiality bar")
        if extra_wrong > 0:
            reasons.append(
                f"it produces {extra_wrong} more wrong deeplinks on the held-out half "
                f"({selected_row['wrong_pct']}% against {shipped_row['wrong_pct']}%), and a "
                f"wrong deeplink is the error this project weights hardest")
        rationale = (f"δ = {selected_delta} is not adopted because "
                     + "; and ".join(reasons) + f". Keep δ = {shipped_delta}.")

    return {
        "seed": SPLIT_SEED,
        "n_fit": len(fit),
        "n_holdout": len(holdout),
        "selection_rule": f"max accuracy@1 subject to precision >= {PRECISION_FLOOR}%",
        "precision_floor_pct": PRECISION_FLOOR,
        "materiality_bar_cases": MATERIAL_DELTA_CASES,
        "shipped_margin_delta": shipped_delta,
        "selected_margin_delta": selected_delta,
        "selection_met_floor": bool(eligible),
        "holdout_gained_correct": gained_correct,
        "holdout_extra_wrong": extra_wrong,
        "recommend_change": change,
        "verdict": rationale,
        "fit_sweep": fit_rows,
        "holdout": holdout_rows,
    }


def run_scope_field_comparison(cases: Sequence[synthetic.Case]) -> Dict[str, Any]:
    """Gate [3] reading `message` only (shipped) vs `message + description` (previous).

    42 of the 578 entries carry a qualifier in `description` that never appears in
    `message`, so the wider read rejects candidates a step legitimately names.
    """
    out = {}
    for label, fields in (("message_only_shipped", ("message",)),
                          ("message_and_description_previous", ("message", "description"))):
        previous = config.SCOPE_FIELDS
        config.SCOPE_FIELDS = fields
        try:
            scored = score_cases(cases)
        finally:
            config.SCOPE_FIELDS = previous
        scored.pop("wrong_examples", None)
        out[label] = {"scope_fields": list(fields), **scored}
    return out


# ----------------------------------------------------------------- cache (D3b)
# Declared BEFORE any held-out number is read. A false positive answers the wrong
# complaint confidently and fast, and nothing downstream can detect it; a miss only costs
# one LLM call. So the threshold is chosen for zero false positives first and hit rate
# second, never the reverse.
CACHE_THRESHOLD_SWEEP = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
CACHE_SEED_VARIATIONS = 5          # variations 1-5 seed the cache
CACHE_TARGET_HIT_RATE = 80.0       # Theme 2 guide 6.3
CACHE_TARGET_P95_MS = 300.0        # Theme 2 guide 6.2


class _ExplodingProvider(LLMProvider):
    """Proves a fast-path request never reaches the LLM. If extraction is called at all,
    the run was not a cache hit and the latency measured would be meaningless."""

    name = "exploding"
    model = "never-called"

    def extract(self, query: str, evidence: str):  # noqa: ARG002
        raise AssertionError("LLM called on what should have been a cache hit")


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, int(round(q * (len(ordered) - 1))))], 1)


def _seed_cache(cache: SemanticCache, plans: Sequence[Dict[str, Any]]) -> int:
    """Store each row's plan under its canonical query plus variations 1-5 ONLY.

    Variations 6-10 are never seen by the cache; they are the held-out paraphrases the
    hit rate is measured on. Seeding with all ten and then querying with those same ten
    would measure nothing but a dictionary lookup.
    """
    for plan in plans:
        cache.store(enrich(plan["query"]), plan["response"],
                    plan["variations"][:CACHE_SEED_VARIATIONS],
                    evidence=evidence_key(plan["siis_response"]))
    return len(plans)


def _probe_cache(cache: SemanticCache, plans: Sequence[Dict[str, Any]],
                 probe_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Query held-out paraphrases and bucket every outcome.

    A hit is CORRECT only if it returns the plan belonging to the row the paraphrase came
    from. A hit on another row's plan is a FALSE POSITIVE and is worse than a miss: the
    user gets a confident, internally valid plan for somebody else's problem.
    """
    by_source = {p["query"]: p["row_id"] for p in plans}
    # Rows are grouped by ARTICLE, not by id. The supplied kit has 20 rows but only 11
    # distinct articles: six rows share one. Two rows with the same article are the same
    # cache-equivalence class, because the plan is grounded in that article — scoring a
    # hit on a sibling as a false positive would repeat exactly the mistake the D2 gold
    # set makes when it scores a shared catalog message against one arbitrary id.
    article_of = {p["row_id"]: evidence_key(p["siis_response"]) for p in plans}
    hits = correct = same_article = cross_article = misses = guard_rejections = 0
    l0_ms: List[float] = []
    l1_ms: List[float] = []
    false_positive_examples: List[Dict[str, Any]] = []

    for row in probe_rows:
        for probe in row["variations"][CACHE_SEED_VARIATIONS:]:
            if not probe or not probe.strip():
                continue
            started = time.perf_counter()
            # A real request carries its article; the probe must too, or the evidence
            # test would be measured against something no caller ever sends.
            result = cache.lookup(enrich(probe),
                                  evidence=evidence_key(row["siis_response"]))
            elapsed = (time.perf_counter() - started) * 1000.0

            if not result.hit:
                misses += 1
                if result.guard_rejected:
                    guard_rejections += 1
                continue

            hits += 1
            (l0_ms if result.tier == "L0" else l1_ms).append(elapsed)
            landed_on = by_source.get(result.source_query or "")
            if landed_on == row["row_id"]:
                correct += 1
            elif article_of.get(landed_on) == article_of.get(row["row_id"]):
                # Same article, sibling complaint. Not a falsehood: the plan is grounded
                # in the very article this request supplied. Counted separately because
                # it is not identical to what a cold run for this exact query would give.
                same_article += 1
            else:
                cross_article += 1
                if len(false_positive_examples) < 8:
                    false_positive_examples.append({
                        "probe": probe,
                        "expected_row": row["row_id"],
                        "got_row": landed_on,
                        "similarity": round(result.similarity or 0.0, 4),
                        "matched_key": result.matched_key,
                        "source_query": result.source_query,
                    })

    probes = hits + misses
    return {
        "probes": probes,
        "hits": hits,
        "correct_hits": correct,
        "same_article_hits": same_article,
        "false_positives": cross_article,     # cross-article only: the real defect
        "misses": misses,
        "guard_rejections": guard_rejections,
        "hit_rate_pct": pct(hits, probes),
        "correct_hit_rate_pct": pct(correct, probes),
        "same_article_hit_rate_pct": pct(same_article, probes),
        "grounded_hit_rate_pct": pct(correct + same_article, probes),
        "false_positive_rate_pct": pct(cross_article, probes),
        "l0_hits": len(l0_ms),
        "l1_hits": len(l1_ms),
        "lookup_l0_p50_ms": _percentile(l0_ms, 0.50),
        "lookup_l0_p95_ms": _percentile(l0_ms, 0.95),
        "lookup_l1_p50_ms": _percentile(l1_ms, 0.50),
        "lookup_l1_p95_ms": _percentile(l1_ms, 0.95),
        "false_positive_examples": false_positive_examples,
    }


def _fast_path_latency(plans: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """End-to-end run_pipeline latency on requests that hit, measured through the real
    entry point rather than by timing the lookup in isolation.

    The provider raises if called, so a recorded sample is proof the LLM was skipped.
    """
    reset_cache()
    cache = get_cache()
    cache.similarity_min = threshold
    _seed_cache(cache, plans)

    l0: List[float] = []
    l1: List[float] = []
    for plan in plans:
        probes = [(plan["query"], l0)]
        probes += [(v, l1) for v in plan["variations"][CACHE_SEED_VARIATIONS:]]
        for probe, bucket in probes:
            if not probe or not probe.strip():
                continue
            request = TroubleshootRequest(query=probe, siis_response=plan["siis_response"])
            started = time.perf_counter()
            try:
                env = run_pipeline(request, provider=_ExplodingProvider())
            except AssertionError:
                continue  # a miss: it reached the LLM, so it is not a fast-path sample
            elapsed = (time.perf_counter() - started) * 1000.0
            if env.meta.cache_hit:
                bucket.append(elapsed)
    reset_cache()
    return {
        "l0_samples": len(l0),
        "l1_samples": len(l1),
        "l0_p50_ms": _percentile(l0, 0.50),
        "l0_p95_ms": _percentile(l0, 0.95),
        "l1_p50_ms": _percentile(l1, 0.50),
        "l1_p95_ms": _percentile(l1, 0.95),
        "all_hits_p50_ms": _percentile(l0 + l1, 0.50),
        "all_hits_p95_ms": _percentile(l0 + l1, 0.95),
    }


def run_cache_eval(plans: Sequence[Dict[str, Any]], cold_latencies: Sequence[float]
                   ) -> Dict[str, Any]:
    """Seed / probe split, threshold sweep, and the two numbers Samsung grades."""
    if len(plans) < 4:
        return {"status": f"not enough validated plans to measure ({len(plans)})"}

    embedder_name = get_embedder().name

    # Rows are split for THRESHOLD FITTING. The cache is seeded with every row in both
    # phases, because that is what a production cache holds and it is also the harder
    # test for false positives; only the PROBES differ between fit and held-out.
    ordered = sorted(plans, key=lambda p: p["row_id"])
    shuffled = list(ordered)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    fit_rows, holdout_rows = shuffled[:midpoint], shuffled[midpoint:]

    sweep_rows = []
    for threshold in CACHE_THRESHOLD_SWEEP:
        cache = SemanticCache(similarity_min=threshold)
        _seed_cache(cache, ordered)
        scored = _probe_cache(cache, ordered, fit_rows)
        scored.pop("false_positive_examples", None)
        sweep_rows.append({"threshold": threshold, **scored})

    clean = [r for r in sweep_rows if r["false_positives"] == 0]
    pool = clean or sweep_rows
    selected = max(pool, key=lambda r: (r["hit_rate_pct"], -r["threshold"]))

    holdout_results = {}
    for threshold in sorted({selected["threshold"], config.CACHE_SIMILARITY_MIN}):
        cache = SemanticCache(similarity_min=threshold)
        _seed_cache(cache, ordered)
        holdout_results[f"{threshold}"] = {
            "threshold": threshold, **_probe_cache(cache, ordered, holdout_rows)}

    shipped = holdout_results[f"{config.CACHE_SIMILARITY_MIN}"]
    latency = _fast_path_latency(ordered, config.CACHE_SIMILARITY_MIN)

    # What the slot guard is worth, measured rather than asserted.
    unguarded_cache = SemanticCache(similarity_min=config.CACHE_SIMILARITY_MIN,
                                    slot_guard=False)
    _seed_cache(unguarded_cache, ordered)
    unguarded = _probe_cache(unguarded_cache, ordered, holdout_rows)
    unguarded.pop("false_positive_examples", None)

    distinct_articles = len({evidence_key(p["siis_response"]) for p in ordered})

    # Threshold verdict. The sweep shows zero cross-article false positives at EVERY
    # threshold, because the evidence key — not the threshold — is what prevents them.
    # That makes the sweep blind to the failure mode the threshold actually guards:
    # a within-article mismatch, where one article covers several distinct symptoms and a
    # battery question lands on a screen plan. Eleven articles barely exercise that, so
    # taking the lowest-scoring-best threshold would be tuning a safety margin against a
    # measurement that cannot see what it is for.
    all_clean = all(r["false_positives"] == 0 for r in sweep_rows)
    shipped_row = holdout_results[f"{config.CACHE_SIMILARITY_MIN}"]
    selected_row = holdout_results[f"{selected['threshold']}"]
    if selected["threshold"] == config.CACHE_SIMILARITY_MIN:
        verdict = (f"The fit half selected the shipped threshold "
                   f"{config.CACHE_SIMILARITY_MIN}; nothing to decide.")
    elif all_clean:
        verdict = (
            f"The fit half selected {selected['threshold']}, worth "
            f"{selected_row['hit_rate_pct'] - shipped_row['hit_rate_pct']:+.1f} points of "
            f"hit rate on the held-out half. It is NOT adopted. Cross-article false "
            f"positives are zero at every threshold in the sweep, so the sweep is "
            f"measuring the evidence key rather than the threshold. What the threshold "
            f"guards is a within-article mismatch, and {distinct_articles} articles "
            f"cannot exercise that. The shipped {config.CACHE_SIMILARITY_MIN} is kept as "
            f"a deliberate safety margin; this is a judgement, not a number the data "
            f"forced.")
    else:
        verdict = (f"The fit half selected {selected['threshold']} under the declared "
                   f"rule; shipped is {config.CACHE_SIMILARITY_MIN}.")

    return {
        "status": "measured",
        "embedder": embedder_name,
        "distinct_articles": distinct_articles,
        "seed_variations": CACHE_SEED_VARIATIONS,
        "protocol": (f"seed the cache with each row's canonical query plus variations "
                     f"1-{CACHE_SEED_VARIATIONS}; probe with variations "
                     f"{CACHE_SEED_VARIATIONS + 1}-10, which the cache has never seen"),
        "selection_rule": "zero false positives first, then maximum hit rate",
        "seed_rows": len(ordered),
        "n_fit_rows": len(fit_rows),
        "n_holdout_rows": len(holdout_rows),
        "shipped_threshold": config.CACHE_SIMILARITY_MIN,
        "selected_threshold": selected["threshold"],
        "selection_had_clean_option": bool(clean),
        "all_thresholds_clean": all_clean,
        "verdict": verdict,
        "fit_sweep": sweep_rows,
        "holdout": holdout_results,
        "holdout_shipped": shipped,
        "unguarded_holdout": unguarded,
        "latency": latency,
        "cold_p50_ms": _percentile(cold_latencies, 0.50),
        "cold_p95_ms": _percentile(cold_latencies, 0.95),
        "llm_calls_avoided": shipped["hits"],
        "llm_calls_without_cache": shipped["probes"],
        "target_hit_rate_pct": CACHE_TARGET_HIT_RATE,
        "target_p95_ms": CACHE_TARGET_P95_MS,
        "meets_hit_rate_target": shipped["hit_rate_pct"] >= CACHE_TARGET_HIT_RATE,
        "meets_latency_target": latency["all_hits_p95_ms"] <= CACHE_TARGET_P95_MS,
        "meets_zero_false_positives": shipped["false_positives"] == 0,
    }


# ----------------------------------------------------------------- report
def build_report(provider_name: str, rate_limit_rpm: int = 0) -> Dict[str, Any]:
    cases = synthetic.build_cases()
    catalog = get_catalog()

    resolution = score_cases(cases, gates=GATES_ALL)
    compliance = run_compliance(provider_name, rate_limit_rpm=rate_limit_rpm)
    # The cache evaluation reuses the plans the compliance run already produced, so
    # measuring the cache costs no extra LLM calls.
    plans = compliance.pop("_plans", [])
    cache_eval = run_cache_eval(plans, [r["latency_ms"] for r in compliance["per_row"]])
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider_requested": provider_name,
        "catalog": {
            "n_entries": len(catalog),
            "n_unique_uris": len({e["deeplink"] for e in catalog.entries}),
        },
        "config": {
            "concept_coverage_min": config.CONCEPT_COVERAGE_MIN,
            "margin_delta": config.MARGIN_DELTA,
            "candidate_pool": config.CANDIDATE_POOL,
            "scope_fields": list(config.SCOPE_FIELDS),
        },
        "gold_set": synthetic.stats(),
        "compliance": compliance,
        "resolution": resolution,
        "ablation": run_ablation(cases),
        "margin_sweep": run_margin_sweep(cases),
        "scope_fields_comparison": run_scope_field_comparison(cases),
        "cache": cache_eval,
    }


# ----------------------------------------------------------------- metrics.md
def render_metrics(report: Dict[str, Any]) -> str:
    """Render docs/metrics.md. Every number comes out of `report`; none is typed here."""
    c = report["compliance"]
    r = report["resolution"]
    g = report["gold_set"]
    sweep = report["margin_sweep"]
    scope = report["scope_fields_comparison"]
    dup = g["duplicates"]

    lines: List[str] = []
    add = lines.append

    add("# Measured results")
    add("")
    add(f"Generated by `python -m evaluation.run_eval` at {report['generated_at_utc']}. ")
    add("Every figure on this page is rendered from `evaluation/report.json`; nothing here "
        "is typed by hand. Re-run the command to regenerate both files.")
    add("")
    add(f"- Extraction provider: `{c['provider']}`")
    add(f"- Catalog: {report['catalog']['n_entries']} entries, "
        f"{report['catalog']['n_unique_uris']} unique URIs")
    add(f"- Gates: concept coverage ≥ {report['config']['concept_coverage_min']}, "
        f"margin δ = {report['config']['margin_delta']}, "
        f"candidate pool = {report['config']['candidate_pool']}, "
        f"scope fields = {report['config']['scope_fields']}")
    add("")

    # ---- 1
    add("## 1. Schema and rule compliance")
    add("")
    add(f"All {c['n_rows']} supplied SIIS rows through `run_pipeline`.")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Rows evaluated | {c['n_rows']} |")
    add(f"| Rows producing a plan | {c['rows_with_plan']} |")
    add(f"| Rows falling back | {c['rows_fallback']} |")
    add(f"| Schema-valid (Samsung `schema.py`, unmodified) | {c['schema_valid_pct']}% |")
    add(f"| Rule-compliant (`validate.assert_rules`) | {c['rule_compliant_pct']}% |")
    add(f"| URL leaks | {c['url_leaks']} |")
    add(f"| Catalog-invalid deeplinks emitted | {c['catalog_invalid_deeplinks']} |")
    add("")

    # ---- 2
    add("## 2. Accuracy")
    add("")
    add(f"Gold set: {g['n_cases']} labelled steps over {g['n_classes']} catalog "
        "equivalence classes, generated from the catalog by `evaluation/synthetic.py`.")
    add(f"{dup['classes_with_duplicates']} classes hold more than one entry "
        f"({dup['entries_in_duplicate_classes']} of {dup['entries_resolvable']} resolvable "
        "entries), so gold is an equivalence class and a hit on any member counts.")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Cases | {r['n_cases']} |")
    add(f"| accuracy@1 | {r['accuracy_at_1_pct']}% |")
    add(f"| **Wrong** (resolved outside the gold class) | **{r['wrong_pct']}%** |")
    add(f"| Abstained (no exact match, degrades to `dummy_positive`) | {r['abstained_pct']}% |")
    add(f"| **Precision** (correct / decided) | **{r['precision_pct']}%** |")
    add("")
    add("Wrong and abstained are reported separately on purpose. An abstention degrades to "
        "`bixby://dummy_positive`, which still opens a Settings screen; a wrong deeplink "
        "sends the user somewhere else entirely. Only the second reaches the user as a "
        "falsehood.")
    add("")
    add("Plan-level deeplink outcomes over the 20 supplied rows:")
    add("")
    add("| Outcome | Count |")
    add("|---|---|")
    add(f"| Actions emitted | {c['total_actions']} |")
    add(f"| Exact catalog deeplinks | {c['total_resolved']} |")
    add(f"| `dummy_positive` | {c['total_dummy_positive']} |")
    add(f"| Null | {c['total_null']} |")
    add(f"| `auto` stepGroups | {c['auto_step_groups']} |")
    add(f"| `auto` stepGroups carrying a deeplink | {c['auto_step_groups_with_deeplink']} "
        f"({c['auto_step_groups_with_deeplink_pct']}%) |")
    add("")
    if r["wrong_examples"]:
        add("Wrong resolutions, up to 8:")
        add("")
        add("| Step | Gold | Got |")
        add("|---|---|---|")
        for w in r["wrong_examples"]:
            add(f"| `{w['step']}` | {w['gold_message']} ({w['gold_type']}) | "
                f"{w['got_message']} ({w['got_type']}, {w['got_id']}) |")
        add("")

    # ---- 3
    add("## 3. Latency")
    add("")
    add("Wall-clock per row, end to end through `run_pipeline`.")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| p50 | {c['latency_p50_ms']} ms |")
    add(f"| p95 | {c['latency_p95_ms']} ms |")
    add(f"| mean | {c['latency_mean_ms']} ms |")
    add("")

    # ---- 4
    add("## 4. Cost and cache")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Extraction provider | `{c['provider']}` |")
    add(f"| Total cost, {c['n_rows']} rows | ${c['total_cost_usd']} |")
    add("")
    add("Cost is an estimate from published per-token rates, not a billed figure. The "
        "offline stub provider costs nothing, so a stub run reports $0.")
    add("")

    cache = report["cache"]
    if cache.get("status") != "measured":
        add(f"Cache: {cache.get('status')}")
        add("")
    else:
        hold = cache["holdout_shipped"]
        lat = cache["latency"]
        add("### Semantic cache")
        add("")
        add(f"Two tiers in process: L0 is an exact hash of the canonical query, L1 is "
            f"embedding cosine over every stored key using `{cache['embedder']}`.")
        add("")
        add(f"**Protocol.** {cache['protocol'].capitalize()}. Seeding the cache with all "
            "ten variations and then querying with those same ten would measure a "
            "dictionary lookup, not generalisation.")
        add("")
        add("| Metric | Value | Target |")
        add("|---|---|---|")
        add(f"| Rows seeded | {cache['seed_rows']} | |")
        add(f"| Held-out paraphrases probed | {hold['probes']} | |")
        add(f"| **Hit rate** | **{hold['hit_rate_pct']}%** | "
            f">= {cache['target_hit_rate_pct']}% |")
        add(f"| ... served this row's own plan | {hold['correct_hit_rate_pct']}% | |")
        add(f"| ... served a sibling row's plan, same article | "
            f"{hold['same_article_hit_rate_pct']}% | |")
        add(f"| **Cross-article false positives** | "
            f"**{hold['false_positive_rate_pct']}%** | 0% |")
        add(f"| Misses | {hold['misses']} | |")
        add(f"| Slot-guard rejections | {hold['guard_rejections']} | |")
        add(f"| L0 / L1 hits | {hold['l0_hits']} / {hold['l1_hits']} | |")
        add(f"| LLM calls avoided | {cache['llm_calls_avoided']} of "
            f"{cache['llm_calls_without_cache']} | |")
        add("")
        add("A false positive is reported separately because it is worse than a miss. A "
            "miss costs one LLM call. A false positive answers the wrong complaint "
            "confidently, in milliseconds, with an internally valid plan that nothing "
            "downstream can detect as wrong.")
        add("")
        add(f"Hits are split three ways because the {cache['seed_rows']} supplied rows "
            f"carry only {cache['distinct_articles']} distinct articles — six rows share "
            "one. A hit on a sibling row with the SAME article is not a falsehood: the "
            "plan served is grounded in the very article the request supplied. It is "
            "counted apart from an exact-row hit only because it was extracted for a "
            "differently-worded sibling complaint. A cross-article hit is the real "
            "defect, and it is what the 0% target refers to.")
        add("")
        add("**Latency**, end to end through `run_pipeline` with a provider that raises "
            "if called, so every sample is proof the LLM was skipped.")
        add("")
        add("| Path | p50 | p95 | Target p95 |")
        add("|---|---|---|---|")
        add(f"| L0 hit (exact) | {lat['l0_p50_ms']} ms | {lat['l0_p95_ms']} ms | "
            f"{cache['target_p95_ms']} ms |")
        add(f"| L1 hit (semantic) | {lat['l1_p50_ms']} ms | {lat['l1_p95_ms']} ms | "
            f"{cache['target_p95_ms']} ms |")
        add(f"| All hits | {lat['all_hits_p50_ms']} ms | {lat['all_hits_p95_ms']} ms | "
            f"{cache['target_p95_ms']} ms |")
        add(f"| Cold miss (full pipeline) | {cache['cold_p50_ms']} ms | "
            f"{cache['cold_p95_ms']} ms | n/a |")
        add("")
        verdicts = [
            f"hit rate {'MET' if cache['meets_hit_rate_target'] else 'NOT MET'}",
            f"p95 {'MET' if cache['meets_latency_target'] else 'NOT MET'}",
            f"zero false positives {'MET' if cache['meets_zero_false_positives'] else 'NOT MET'}",
        ]
        add("Targets: " + "; ".join(verdicts) + ".")
        add("")

        guard = cache["unguarded_holdout"]
        add(f"**What the slot guard is worth.** With the device/domain guard removed, the "
            f"same held-out probes give {guard['hit_rate_pct']}% hit rate and "
            f"{guard['false_positive_rate_pct']}% cross-article false positives, against "
            f"{hold['hit_rate_pct']}% and {hold['false_positive_rate_pct']}% with it.")
        if guard["false_positives"] <= hold["false_positives"]:
            add("")
            add(f"On this corpus the guard therefore costs "
                f"{guard['hit_rate_pct'] - hold['hit_rate_pct']:.1f} points of hit rate "
                "and prevents nothing measurable, because the evidence key already makes "
                "a cross-article hit impossible. Read that as 'not exercised here', not "
                "as 'useless': the guard is what separates two complaints that share an "
                "article but not a device or a domain, and an article covering several "
                "symptoms is exactly where it would earn its place. It is kept for the "
                "same reason gate [3] is kept in section 5 — the measurement is blind to "
                "the case it exists for.")
        add("")
        if hold.get("false_positive_examples"):
            add("False positives, up to 8:")
            add("")
            add("| Probe | Expected | Landed on | Similarity |")
            add("|---|---|---|---|")
            for fp in hold["false_positive_examples"]:
                add(f"| `{fp['probe']}` | {fp['expected_row']} | {fp['got_row']} | "
                    f"{fp['similarity']} |")
            add("")

        add("#### 4.1 Similarity threshold sweep")
        add("")
        add(f"Fitted on {cache['n_fit_rows']} rows' held-out paraphrases and reported on "
            f"the other {cache['n_holdout_rows']} rows' (seed {report['margin_sweep']['seed']}). "
            f"Selection rule, declared before the split: {cache['selection_rule']}.")
        add("")
        add("| Threshold | Hit rate | Correct | False positives | Guard rejections |")
        add("|---|---|---|---|---|")
        for row in cache["fit_sweep"]:
            marker = " <-selected" if row["threshold"] == cache["selected_threshold"] else ""
            add(f"| {row['threshold']}{marker} | {row['hit_rate_pct']}% | "
                f"{row['correct_hit_rate_pct']}% | {row['false_positive_rate_pct']}% "
                f"({row['false_positives']}) | {row['guard_rejections']} |")
        add("")
        add("Held-out rows:")
        add("")
        add("| Threshold | Hit rate | Correct | False positives |")
        add("|---|---|---|---|")
        for row in cache["holdout"].values():
            add(f"| {row['threshold']} | {row['hit_rate_pct']}% | "
                f"{row['correct_hit_rate_pct']}% | {row['false_positive_rate_pct']}% "
                f"({row['false_positives']}) |")
        add("")
        add(f"Shipped threshold = {cache['shipped_threshold']}; the fit half selected "
            f"{cache['selected_threshold']}. {cache['verdict']}")
        add("")

    # ---- 5
    add("## 5. Gate ablation")
    add("")
    add("Same gold set, gates switched on cumulatively. This is what each gate buys.")
    add("")
    add("| Stage | Gates | accuracy@1 | Wrong | Abstained | Precision |")
    add("|---|---|---|---|---|---|")
    for a in report["ablation"]:
        add(f"| {a['stage']} | {a['gates']} | {a['accuracy_at_1_pct']}% | {a['wrong_pct']}% | "
            f"{a['abstained_pct']}% | {a['precision_pct']}% |")
    add("")
    first, last = report["ablation"][0], report["ablation"][-1]
    add(f"Wrong deeplinks fall from {first['wrong_pct']}% to {last['wrong_pct']}% and "
        f"precision rises from {first['precision_pct']}% to {last['precision_pct']}% across "
        "the four gates.")
    add("")
    add("Gate [3] reads its scope qualifier from the catalog fields "
        f"`{report['config']['scope_fields']}`. Reading `description` as well rejects "
        "candidates whose own message a step legitimately names:")
    add("")
    add("| Gate [3] reads | accuracy@1 | Wrong | Abstained | Precision |")
    add("|---|---|---|---|---|")
    for label, row in scope.items():
        add(f"| {' + '.join(row['scope_fields'])} | {row['accuracy_at_1_pct']}% | "
            f"{row['wrong_pct']}% | {row['abstained_pct']}% | {row['precision_pct']}% |")
    add("")

    # ---- 5.1
    add("### 5.1 Margin threshold sweep")
    add("")
    add(f"`MARGIN_DELTA` fitted on a seeded random half ({sweep['n_fit']} cases, seed "
        f"{sweep['seed']}) and reported on the held-out half ({sweep['n_holdout']} cases). "
        f"Selection rule, declared before the split: {sweep['selection_rule']}.")
    add("")
    add("Fit half:")
    add("")
    add("| δ | accuracy@1 | Wrong | Abstained | Precision |")
    add("|---|---|---|---|---|")
    for row in sweep["fit_sweep"]:
        marker = " ←selected" if row["margin_delta"] == sweep["selected_margin_delta"] else ""
        add(f"| {row['margin_delta']}{marker} | {row['accuracy_at_1_pct']}% | "
            f"{row['wrong_pct']}% | {row['abstained_pct']}% | {row['precision_pct']}% |")
    add("")
    add("Held-out half, for the fit-selected δ and the shipped δ only:")
    add("")
    add("| δ | accuracy@1 | Wrong | Abstained | Precision |")
    add("|---|---|---|---|---|")
    for row in sweep["holdout"].values():
        add(f"| {row['margin_delta']} | {row['accuracy_at_1_pct']}% | {row['wrong_pct']}% | "
            f"{row['abstained_pct']}% | {row['precision_pct']}% |")
    add("")
    add(f"Shipped δ = {sweep['shipped_margin_delta']}; fit half selected "
        f"δ = {sweep['selected_margin_delta']}. {sweep['verdict']}")
    add("")

    # ---- 6
    add("## 6. Per-row results")
    add("")
    add("| Row | Actions | Exact | Dummy | Null | Schema | Rules | URL leak | Fallback | Latency |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in c["per_row"]:
        add(f"| {row['id']} | {row['actions']} | {row['resolved']} | {row['dummy_positive']} | "
            f"{row['null']} | {'PASS' if row['schema_valid'] else 'FAIL'} | "
            f"{'PASS' if row['rule_compliant'] else 'FAIL'} | "
            f"{'yes' if row['url_leak'] else 'no'} | {row['fallback'] or '-'} | "
            f"{row['latency_ms']} ms |")
    add("")

    # ---- 7
    add("## 7. Limitations")
    add("")
    add("- **The gold set is derived, not human-labelled.** It is generated from the "
        "catalog's own `message` strings, so it measures whether the resolver can recover "
        "the entry a step was phrased from. It does not measure whether a real article's "
        "wording would reach the right entry. One consequence is specific enough to name: "
        "gate [2] reads polarity after subtracting the candidate's own subject from the "
        "step, and in this gold set that subject is always present verbatim, so the "
        "subtraction always succeeds. On a real article that paraphrases a setting rather "
        "than naming it, the subtraction falls back to reading the whole step, and the "
        "gate is correspondingly weaker than these numbers suggest.")
    add(f"- **{dup['classes_with_duplicates']} message classes are not separable at all.** "
        f"`{dup['largest_classes'][0]['message']}` covers "
        f"{dup['largest_classes'][0]['n']} different screens, distinguishable only by "
        "`description`, which gate [3] does not read. Those cases are scored against the "
        "whole class because no resolver could do better from the step alone.")
    scope_stage = next((a for a in report["ablation"] if a["stage"] == "V2"), None)
    polarity_stage = next((a for a in report["ablation"] if a["stage"] == "V1"), None)
    if scope_stage and polarity_stage and \
            scope_stage["accuracy_at_1_pct"] == polarity_stage["accuracy_at_1_pct"]:
        add("- **The ablation cannot measure gate [3].** V1 and V2 are identical "
            f"({scope_stage['accuracy_at_1_pct']}% accuracy@1 for both), and this is a "
            "property of the gold set, not of the gate. Every generated step is phrased "
            "from its own entry's `message`, so it always carries that entry's qualifier "
            "and the gate never rejects the answer; the candidates it does reject were "
            "outranked anyway. The gate's value shows on steps NOT derived from catalog "
            "wording — the factory-reset trap in "
            "`backend/tests/test_resolver.py` is the production case it was built for. "
            "Read the V2 row as 'costs nothing here', not as 'does nothing'.")
    cache = report["cache"]
    if cache.get("status") == "measured":
        add("- **The cache is measured on 20 rows' paraphrases, not on production "
            "traffic.** The held-out paraphrases come from the same extractor that wrote "
            "the seeds, so they are more consistent in register than real users would be. "
            "The false-positive number is the one to watch as the corpus grows: more "
            "stored plans means more chances to land on the wrong one.")
    add("- **Latency excludes network time to the LLM** when run against the offline stub. "
        "A live provider run reports real extraction latency in the same table.")
    add("- **Cost is an estimate** from published per-token rates, not a billed amount.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="stub",
                        help="extraction provider for the compliance run (stub|replay|gemini)")
    parser.add_argument("--rate-limit-rpm", type=int, default=0,
                        help="pace provider calls, e.g. 12 to stay under Gemini's "
                             "15/min free tier. 0 disables pacing.")
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args.provider, rate_limit_rpm=args.rate_limit_rpm)
    report["wall_clock_s"] = round(time.perf_counter() - started, 1)

    report_path, metrics_path = Path(args.report), Path(args.metrics)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics_path.write_text(render_metrics(report), encoding="utf-8")

    c, r = report["compliance"], report["resolution"]
    print(f"provider           {c['provider']}")
    print(f"gold cases         {r['n_cases']}")
    print(f"accuracy@1         {r['accuracy_at_1_pct']}%")
    print(f"wrong              {r['wrong_pct']}%")
    print(f"abstained          {r['abstained_pct']}%")
    print(f"precision          {r['precision_pct']}%")
    print(f"schema-valid       {c['schema_valid_pct']}%   rule-compliant "
          f"{c['rule_compliant_pct']}%")
    print(f"url leaks          {c['url_leaks']}   catalog-invalid "
          f"{c['catalog_invalid_deeplinks']}")
    print(f"latency p50/p95    {c['latency_p50_ms']} / {c['latency_p95_ms']} ms")
    def _show(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:  # --report/--metrics pointed outside the repo
            return str(path)

    print(f"wrote {_show(report_path)} and {_show(metrics_path)} "
          f"in {report['wall_clock_s']}s")


if __name__ == "__main__":
    main()
