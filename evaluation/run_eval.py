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
from app.llm.replay import ReplayProvider  # noqa: E402
from app.main import run_pipeline  # noqa: E402
from app.pipeline import validate  # noqa: E402
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
        env = run_pipeline(req, provider=provider)
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
        rationale = (
            f"δ = {selected_delta} is worth only {gained_correct} more correct answers on "
            f"the held-out half and produces {extra_wrong:+d} wrong ones, inside the "
            f"{MATERIAL_DELTA_CASES}-case materiality bar. The shipped δ = {shipped_delta} "
            f"also yields the lower wrong rate ({shipped_row['wrong_pct']}% against "
            f"{selected_row['wrong_pct']}%), which is the error this project weights "
            f"hardest. Keep δ = {shipped_delta}.")

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


# ----------------------------------------------------------------- report
def build_report(provider_name: str, rate_limit_rpm: int = 0) -> Dict[str, Any]:
    cases = synthetic.build_cases()
    catalog = get_catalog()

    resolution = score_cases(cases, gates=GATES_ALL)
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
        "compliance": run_compliance(provider_name, rate_limit_rpm=rate_limit_rpm),
        "resolution": resolution,
        "ablation": run_ablation(cases),
        "margin_sweep": run_margin_sweep(cases),
        "scope_fields_comparison": run_scope_field_comparison(cases),
        "cache": {
            "status": "not implemented (D3b)",
            "exact_hit_rate_pct": None,
            "paraphrase_hit_rate_pct": None,
            "fast_path_p95_ms": None,
        },
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
    add(f"| Exact-hash cache hit rate | {report['cache']['status']} |")
    add(f"| Paraphrase cache hit rate | {report['cache']['status']} |")
    add(f"| Cache fast-path p95 | {report['cache']['status']} |")
    add("")
    add("Cost is an estimate from published per-token rates, not a billed figure. The "
        "offline stub provider costs nothing, so a stub run reports $0.")
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
        "wording would reach the right entry.")
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
    add("- **The semantic cache is not implemented** (D3b). Every cache figure in section 4 "
        "is marked accordingly rather than estimated.")
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
