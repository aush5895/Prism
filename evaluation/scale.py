"""Scale proof: what happens as the scenario corpus grows toward Samsung's 10k+ target.

    python -m evaluation.scale --max 10000

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
It is not possible to synthesise 10,000 genuine Samsung knowledge articles. Nothing here
pretends to. What this module measures is what growing a corpus actually stresses:

    the semantic cache      lookup latency at both tiers, and the store-then-hit path
    collision behaviour     does a bigger store start answering the wrong scenario
    the retrieval index     resolver latency over the fixed 578-entry catalog
    memory                  what the cache costs to hold
    cold-path spend         projected from the measured per-query cost

It does NOT measure extraction quality across unseen domains. Every scenario here is
recombined from the 11 distinct articles the supplied kit contains, so the language model
never sees a genuinely new subject area. A corpus of 10,000 real articles would test
whether the prompt generalises; this tests whether the machinery around it holds up. Those
are different questions and only the second one is answered here.

EVERYTHING GENERATED IS LABELLED SYNTHETIC. Scenario ids are `SYN-*`, article ids are
`SYN-ART-*`, and the plans are placeholder objects of the right shape. No synthetic
scenario is ever presented as Samsung data, in the report or in the chart.

Generation is seeded, so two runs produce the same corpus and the same numbers.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config  # noqa: E402
from app.pipeline.cache import SemanticCache, evidence_key  # noqa: E402
from app.pipeline.deeplinks import get_catalog  # noqa: E402
from app.pipeline.embeddings import get_embedder  # noqa: E402
from app.pipeline.enrich import enrich  # noqa: E402
from app.pipeline.ground import normalize_siis  # noqa: E402

REPORT_PATH = ROOT / "evaluation" / "report.json"
METRICS_PATH = ROOT / "docs" / "metrics.md"
CHART_PATH = ROOT / "docs" / "scale_p95.png"

SIZES = (100, 1_000, 5_000, 10_000)
SEED = 20260924
LATENCY_BUDGET_MS = 300.0          # Theme 2 guide §6.2

# Seeds stored per scenario: the canonical query plus two paraphrases. The remaining
# paraphrases are held out and are what the probes use, so a hit is generalisation rather
# than a dictionary lookup.
SEED_VARIATIONS = 2
PROBE_SAMPLE = 200                 # probes per size; enough for a stable p95, cheap enough to run
# Roughly how many scenarios share one article. Real corpora cluster: many complaints map
# to the same knowledge article, which is exactly where within-article collisions live.
SCENARIOS_PER_ARTICLE = 8


# ----------------------------------------------------------------- synthesis
@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    article_id: str
    query: str
    variations: Tuple[str, ...]
    evidence: str
    plan: Dict[str, Any]
    # Which of the supplied complaints this scenario re-voices. Two scenarios with the
    # same article AND the same complaint differ only in generator-added context, handset
    # and register, so a hit between them is the same-article case run_eval reports
    # separately -- not a false positive. -1 means "unknown", which never matches.
    complaint_id: int = -1


_DEVICES = ("Galaxy S22", "Galaxy S23 Ultra", "Galaxy Z Flip 5", "Galaxy A54",
            "Galaxy Tab S9", "Galaxy Note 20", "Galaxy S21 FE", "Galaxy Z Fold 4")
# Real complaints carry circumstance, and it is what keeps two reports of the same
# symptom distinguishable. Without it the canonical forms collide far more than a real
# corpus would, and the measurement becomes a study of this generator instead.
_CONTEXTS = (
    "", " after the latest software update", " ever since I dropped it",
    " only when the battery is low", " when I am charging it", " in bright sunlight",
    " since yesterday morning", " after I installed a new case",
    " whenever I open the camera", " during video calls", " when the phone gets warm",
    " right after a restart",
)

_REGISTERS = (
    "{d} {c}",
    "my {d} {c} and it is really annoying",
    "why does my {d} {c}?",
    "{c} on my {d}, please help",
    "hi, i have a {d} and {c}",
    "{d}: {c}",
    "ive had this {d} for months and now {c}",
    "{c} — {d}, happens every day",
)


def _sections(article: str) -> List[str]:
    """Split a real article into its numbered/headed sections."""
    parts = re.split(r"\n(?=#{1,4}\s|\d+\.\s)", article)
    return [p.strip() for p in parts if len(p.strip()) > 60] or [article]


def _core_complaint(query: str) -> str:
    """The symptom, with a leading device mention stripped so a new one can be swapped in."""
    text = query.strip().rstrip(".")
    text = re.sub(r"^(my|the)\s+", "", text, flags=re.I)
    text = re.sub(r"^(samsung\s+)?(galaxy\s+)?[a-z]?[\*\d]{1,6}[a-z0-9]*"
                  r"(\s+(ultra|plus|pro|fe|5g))?\s*", "", text, flags=re.I)
    return text[0].lower() + text[1:] if text else "the screen is not responding"


def build_corpus(n: int, seed: int = SEED) -> List[Scenario]:
    """Recombine the supplied kit into `n` labelled-synthetic scenarios.

    Articles are variants: a real article's own sections, sub-sampled and reordered, so
    the text is genuine Samsung prose in an arrangement the kit does not contain. Queries
    are the real complaints' symptoms re-voiced across registers and devices.
    """
    rng = random.Random(seed)
    rows = json.loads((ROOT / "data" / "siis_responses.json").read_text(encoding="utf-8"))["responses"]

    real_articles: Dict[str, str] = {}
    for row in rows:
        text = normalize_siis(row["siis_response"]).text
        real_articles.setdefault(evidence_key(row["siis_response"]), text)
    article_pool = list(real_articles.values())
    complaints = [_core_complaint(r["original_query"]) for r in rows]

    n_articles = max(1, n // SCENARIOS_PER_ARTICLE)
    variants: List[Tuple[str, str]] = []
    for i in range(n_articles):
        base = article_pool[i % len(article_pool)]
        chunks = _sections(base)
        keep = max(2, int(len(chunks) * rng.uniform(0.5, 1.0)))
        picked = rng.sample(chunks, min(keep, len(chunks)))
        rng.shuffle(picked)
        header = f"[SYNTHETIC ARTICLE SYN-ART-{i:05d}] recombined from supplied sections"
        variants.append((f"SYN-ART-{i:05d}", header + "\n" + "\n".join(picked)))

    scenarios: List[Scenario] = []
    for i in range(n):
        article_id, article_text = variants[i % len(variants)]
        complaint_id = i % len(complaints)
        complaint = complaints[complaint_id] + _CONTEXTS[rng.randrange(len(_CONTEXTS))]
        device = _DEVICES[rng.randrange(len(_DEVICES))]
        forms = list(_REGISTERS)
        rng.shuffle(forms)
        phrasings = tuple(dict.fromkeys(
            f.format(d=device, c=complaint) for f in forms
        ))
        scenarios.append(Scenario(
            scenario_id=f"SYN-{i:06d}",
            article_id=article_id,
            query=phrasings[0],
            variations=phrasings[1:],
            evidence=article_text,
            plan=_synthetic_plan(i),
            complaint_id=complaint_id,
        ))
    return scenarios


def _synthetic_plan(i: int) -> Dict[str, Any]:
    """A placeholder of the right SHAPE. The cache never reads below `actions`, and this
    is never scored as output -- it exists so the store holds something plan-sized."""
    return {"contexts": [{
        "goal": "Follow these steps to perform this Synthetic Troubleshooting",
        "title": "Synthetic scenario",
        "score": 0.5,
        "actions": [{
            "actionName": f"Synthetic Action {i}",
            "description": "It will stand in for a real plan",
            "category": "auto",
            "stepGroups": [{"steps": ["Open Settings.", "Tap Display."],
                            "actionableDeeplink": None, "validationDeeplink": None}],
        }],
    }]}


# ----------------------------------------------------------------- helpers
def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, int(round(q * (len(ordered) - 1))))], 3)


def _populate(cache: SemanticCache, scenarios: Sequence[Scenario]) -> None:
    for s in scenarios:
        cache.store(enrich(s.query), s.plan, s.variations[:SEED_VARIATIONS],
                    evidence=evidence_key(s.evidence))


# ----------------------------------------------------------------- one corpus size
def measure(scenarios: Sequence[Scenario], rng: random.Random) -> Dict[str, Any]:
    n = len(scenarios)
    # Attribution is by PLAN IDENTITY. Two scenarios can share a canonical query -- two
    # users describing the same symptom on the same handset -- so matching on query text
    # would misattribute exactly the hits this section is trying to count.
    by_plan = {id(s.plan): s for s in scenarios}

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    cache = SemanticCache(similarity_min=config.CACHE_SIMILARITY_MIN)
    _populate(cache, scenarios)
    warm = scenarios[0]
    cache.lookup(enrich(warm.query), evidence=evidence_key(warm.evidence))  # build the matrix
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    matrix = cache._ensure_matrix()  # noqa: SLF001 - measuring internals is the point here
    sample = rng.sample(list(scenarios), min(PROBE_SAMPLE, n))

    # ---- L0: the exact canonical query.
    # Not asserted to hit. Measured on this corpus: at 1,000 scenarios, 259 canonical
    # forms are shared by more than one scenario and 257 of those span more than one
    # article. Two users typing the same words about different underlying articles is a
    # real thing, and the evidence guard turns it into a miss rather than the wrong plan.
    # That is the correct outcome and it is what the l0 hit rate below reports.
    l0: List[float] = []
    l0_hits = l0_guard_rejected = 0
    for s in sample:
        started = time.perf_counter()
        result = cache.lookup(enrich(s.query), evidence=evidence_key(s.evidence))
        l0.append((time.perf_counter() - started) * 1000.0)
        if result.tier == "L0":
            l0_hits += 1
        elif result.guard_rejected:
            l0_guard_rejected += 1

    # ---- L1: a held-out paraphrase the cache has never stored
    l1: List[float] = []
    correct = equivalent = wrong_same_article = cross_article = misses = 0
    for s in sample:
        held_out = s.variations[SEED_VARIATIONS:]
        if not held_out:
            continue
        probe = held_out[rng.randrange(len(held_out))]
        started = time.perf_counter()
        result = cache.lookup(enrich(probe), evidence=evidence_key(s.evidence))
        elapsed = (time.perf_counter() - started) * 1000.0
        if not result.hit:
            misses += 1
            continue
        l1.append(elapsed)
        landed = by_plan.get(id(result.plan.response)) if result.plan else None
        if landed is None or landed.scenario_id == s.scenario_id:
            correct += 1
        elif landed.article_id == s.article_id and landed.complaint_id == s.complaint_id \
                and landed.complaint_id >= 0:
            equivalent += 1
        elif landed.article_id == s.article_id:
            wrong_same_article += 1
        else:
            cross_article += 1

    # ---- store-then-hit: the path where the O(n) rebuild used to live
    interleaved: List[float] = []
    # The probe must actually reach L1. An exact canonical query short-circuits at L0
    # without touching the matrix, which is precisely the cost being measured -- and on a
    # corpus this size a held-out paraphrase can collide into L0 by chance, so scan for
    # one that genuinely lands on the semantic tier.
    probe_scenario, probe, probe_evidence = None, None, None
    for candidate in sample:
        candidate_evidence = evidence_key(candidate.evidence)
        for text in candidate.variations[SEED_VARIATIONS:]:
            enriched = enrich(text)
            if cache.lookup(enriched, evidence=candidate_evidence).tier == "L1":
                probe_scenario, probe, probe_evidence = candidate, enriched, candidate_evidence
                break
        if probe is not None:
            break
    if probe is None:                      # nothing reaches L1; nothing to measure here
        probe_scenario = sample[0]
        probe = enrich(probe_scenario.query)
        probe_evidence = evidence_key(probe_scenario.evidence)
    for i in range(25):
        extra = Scenario(f"SYN-EXTRA-{i:04d}", probe_scenario.article_id,
                         f"interleaved synthetic complaint {i} {probe_scenario.query}",
                         (f"interleaved paraphrase {i}",), probe_scenario.evidence,
                         probe_scenario.plan)
        _populate(cache, [extra])
        started = time.perf_counter()
        cache.lookup(probe, evidence=probe_evidence)
        interleaved.append((time.perf_counter() - started) * 1000.0)

    # ---- resolver: fixed 578-entry catalog, so this should not move with corpus size
    catalog = get_catalog()
    steps = ["Tap the switch next to Touch sensitivity to enable it.",
             "Tap Navigation bar.", "Navigate to Settings, tap Connections, and tap Wi-Fi.",
             "Tap Factory data reset.", "Turn off Auto-Sync."]
    resolver: List[float] = []
    for i in range(100):
        step = steps[i % len(steps)]
        started = time.perf_counter()
        catalog.resolve_step(step)
        resolver.append((time.perf_counter() - started) * 1000.0)

    decided = correct + equivalent + wrong_same_article + cross_article
    keys_bytes = sum(len(k.encode("utf-8")) for k in cache._keys)  # noqa: SLF001
    return {
        "scenarios": n,
        "articles": len({s.article_id for s in scenarios}),
        "cache_keys": len(cache),
        "probes": decided + misses,
        "l0_p50_ms": _pct(l0, 0.50), "l0_p95_ms": _pct(l0, 0.95),
        "l0_hit_rate_pct": round(100.0 * l0_hits / max(1, len(l0)), 1),
        "l0_guard_rejected_pct": round(100.0 * l0_guard_rejected / max(1, len(l0)), 1),
        "l1_p50_ms": _pct(l1, 0.50), "l1_p95_ms": _pct(l1, 0.95),
        "store_then_hit_p50_ms": _pct(interleaved, 0.50),
        "store_then_hit_p95_ms": _pct(interleaved, 0.95),
        "resolver_p50_ms": _pct(resolver, 0.50), "resolver_p95_ms": _pct(resolver, 0.95),
        "hit_rate_pct": round(100.0 * decided / max(1, decided + misses), 1),
        "correct_hit_pct": round(100.0 * correct / max(1, decided + misses), 1),
        # Same article, same supplied complaint, different generated wording/handset.
        "equivalent_hit_pct": round(100.0 * equivalent / max(1, decided + misses), 1),
        # The strictest reading: any hit on a different synthetic scenario at all.
        "other_scenario_hit_pct": round(
            100.0 * (equivalent + wrong_same_article + cross_article)
            / max(1, decided + misses), 1),
        "wrong_scenario_same_article_pct": round(
            100.0 * wrong_same_article / max(1, decided + misses), 1),
        "cross_article_pct": round(100.0 * cross_article / max(1, decided + misses), 1),
        "false_positive_pct": round(
            100.0 * (wrong_same_article + cross_article) / max(1, decided + misses), 1),
        # The ALLOCATION, not the view: the matrix lives in a buffer that grows by 1.5x.
        "matrix_mb": round((cache._buf.nbytes if cache._buf is not None  # noqa: SLF001
                            else (matrix.nbytes if matrix is not None else 0)) / 1e6, 2),
        "keys_mb": round(keys_bytes / 1e6, 2),
        "peak_traced_mb": round((peak - baseline) / 1e6, 1),
    }


# ----------------------------------------------------------------- chart
def write_chart(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["scenarios"] for r in rows]
    series = [
        ("Cache L0 hit", [r["l0_p95_ms"] for r in rows], "#3ddc84", "o"),
        ("Cache L1 hit", [r["l1_p95_ms"] for r in rows], "#5eb3ff", "s"),
        ("Store, then hit", [r["store_then_hit_p95_ms"] for r in rows], "#ffb454", "^"),
        ("Resolver (gates 1-5)", [r["resolver_p95_ms"] for r in rows], "#c58cff", "d"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    fig.patch.set_facecolor("#12151a")
    ax.set_facecolor("#12151a")

    ax.axhline(LATENCY_BUDGET_MS, color="#ff6b6b", linewidth=2, linestyle="--", zorder=1)
    ax.text(sizes[0], LATENCY_BUDGET_MS * 1.12,
            f"Samsung fast-path budget — {LATENCY_BUDGET_MS:.0f} ms (guide §6.2)",
            color="#ff6b6b", fontsize=11, fontweight="bold")

    for label, values, colour, marker in series:
        ax.plot(sizes, values, marker=marker, color=colour, linewidth=2.2,
                markersize=7, label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("Synthetic scenarios in the corpus", color="#e8edf4", fontsize=12)
    ax.set_ylabel("p95 latency (ms, log scale)", color="#e8edf4", fontsize=12)
    # The title states what the data shows. It used to claim "every path stays inside the
    # 300 ms budget" unconditionally, and was drawn over a run where one path took 24 s.
    worst = max(max(r["l0_p95_ms"], r["l1_p95_ms"], r["store_then_hit_p95_ms"],
                    r["resolver_p95_ms"]) for r in rows)
    title = ("Latency against corpus size — every path stays inside the "
             f"{LATENCY_BUDGET_MS:.0f} ms budget" if worst <= LATENCY_BUDGET_MS else
             f"Latency against corpus size — worst p95 {worst:,.0f} ms exceeds the "
             f"{LATENCY_BUDGET_MS:.0f} ms budget")
    ax.set_title(title, color="#e8edf4", fontsize=14, fontweight="bold", pad=16)

    ax.tick_params(colors="#97a3b6")
    for spine in ax.spines.values():
        spine.set_color("#333c4a")
    ax.grid(True, which="both", color="#222834", linewidth=0.8, zorder=0)
    legend = ax.legend(facecolor="#1a1f27", edgecolor="#333c4a", fontsize=11)
    for text in legend.get_texts():
        text.set_color("#e8edf4")

    fig.text(0.5, 0.012,
             "Synthetic corpus recombined from the 11 supplied articles. Measures cache, "
             "index and memory scaling — not extraction quality on unseen domains.",
             ha="center", color="#97a3b6", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------- entry point
def first_degradation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The first metric that moves in the wrong direction, and where.

    Reported rather than asserted: a scale claim with nothing degrading is usually a
    claim that was not measured hard enough.
    """
    baseline, findings = rows[0], []
    for row in rows[1:]:
        if row["false_positive_pct"] > baseline["false_positive_pct"]:
            findings.append({
                "metric": "cross-scenario false positives",
                "at_scenarios": row["scenarios"],
                "from_pct": baseline["false_positive_pct"],
                "to_pct": row["false_positive_pct"],
            })
            break
    for row in rows[1:]:
        worst = max(row["l0_p95_ms"], row["l1_p95_ms"], row["store_then_hit_p95_ms"])
        if worst > LATENCY_BUDGET_MS:
            findings.append({"metric": "fast-path p95 over budget",
                             "at_scenarios": row["scenarios"], "to_ms": worst})
            break
    growth = rows[-1]["l1_p95_ms"] / max(baseline["l1_p95_ms"], 1e-6)
    return {
        "findings": findings,
        "l1_p95_growth_factor": round(growth, 2),
        "memory_mb_at_max": rows[-1]["matrix_mb"] + rows[-1]["keys_mb"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=10_000,
                        help="largest corpus size to measure")
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    sizes = [s for s in SIZES if s <= args.max] or [args.max]
    corpus = build_corpus(max(sizes))
    embedder = get_embedder()
    print(f"embedder: {embedder.name}")
    print(f"corpus:   {len(corpus)} synthetic scenarios over "
          f"{len({s.article_id for s in corpus})} synthetic article variants (seed {SEED})")

    rows = []
    for size in sizes:
        started = time.perf_counter()
        row = measure(corpus[:size], random.Random(SEED + size))
        row["measured_in_s"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(f"  {size:>6,} scenarios | {row['cache_keys']:>6,} keys | "
              f"L0 p95 {row['l0_p95_ms']:>6.2f} | L1 p95 {row['l1_p95_ms']:>6.2f} | "
              f"store+hit p95 {row['store_then_hit_p95_ms']:>7.2f} | "
              f"resolver p95 {row['resolver_p95_ms']:>5.2f} | "
              f"FP {row['false_positive_pct']:>4.1f}% | "
              f"{row['matrix_mb'] + row['keys_mb']:>6.1f} MB | {row['measured_in_s']}s",
              flush=True)

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    per_query_cost = 0.0
    compliance = report.get("compliance") or {}
    if compliance.get("n_rows"):
        per_query_cost = compliance["total_cost_usd"] / compliance["n_rows"]

    report["scale"] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "embedder": embedder.name,
        "corpus": "synthetic, recombined from the 11 distinct articles in the supplied kit",
        "measures": "cache, retrieval index and memory scaling",
        "does_not_measure": "extraction quality on unseen domains",
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "seed_variations_per_scenario": SEED_VARIATIONS,
        "probes_per_size": PROBE_SAMPLE,
        "per_query_cost_usd": round(per_query_cost, 6),
        "projected_cold_path_cost_usd_at_10k": round(per_query_cost * 10_000, 2),
        "by_size": rows,
        "verdict": first_degradation(rows),
        "chart": str(CHART_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_chart(rows, CHART_PATH)
    print(f"\nwrote {report_path.relative_to(ROOT)} (key 'scale') and "
          f"{CHART_PATH.relative_to(ROOT)}")

    from evaluation.run_eval import render_metrics
    METRICS_PATH.write_text(render_metrics(report), encoding="utf-8")
    print(f"re-rendered {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
