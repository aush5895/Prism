"""Run the vertical slice on a supplied row and print the resolver's reasoning.

    python -m tools.demo_row21 [--row row_21] [--provider stub|replay|gemini]
    python -m tools.demo_row21 --write-contract-example
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.contracts import TroubleshootRequest  # noqa: E402
from app.llm import get_provider  # noqa: E402
from app.llm.replay import ReplayProvider  # noqa: E402
from app.main import run_pipeline  # noqa: E402
from app.pipeline.deeplinks import get_catalog  # noqa: E402
from app.schema_samsung import ContextDeeplinkResponse  # noqa: E402

FIXTURE = ROOT / "backend" / "tests" / "fixtures" / "extraction_row21.json"
CONTRACT_EXAMPLE = ROOT / "docs" / "worked_example_row21.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="row_21")
    ap.add_argument("--provider", default="replay")
    ap.add_argument("--write-contract-example", action="store_true")
    args = ap.parse_args()

    rows = json.loads((ROOT / "data" / "siis_responses.json").read_text())["responses"]
    row = next(r for r in rows if r["id"] == args.row)
    query = row["original_query"].split(". ", 1)[-1]

    provider = ReplayProvider(FIXTURE) if args.provider == "replay" else get_provider(args.provider)
    env = run_pipeline(TroubleshootRequest(query=query, siis_response=row["siis_response"]),
                       provider=provider)

    print("=" * 78)
    print(f"QUERY      {query[:70]}")
    print(f"EVIDENCE   {row['siis_response']['title']}  (supplied with request)")
    print(f"PROVIDER   {env.meta.model}")
    print("=" * 78)

    if not env.response["contexts"]:
        print(f"\nno plan produced — fallback = {env.meta.fallback}\n")
        return

    goal = env.response["contexts"][0]
    print(f"\ngoal   {goal['goal']}\ntitle  {goal['title']}\nscore  {goal['score']}\n")

    catalog = get_catalog()
    for i, action in enumerate(goal["actions"], 1):
        print(f"{i}. [{action['category']:8s}] {action['actionName']}")
        print(f"   {action['description']}")
        for group in action["stepGroups"]:
            dl = group["actionableDeeplink"]
            for step in group["steps"]:
                print(f"     - {step}")
            if dl:
                catalog.verify_identity(dl)
                print(f"     -> {dl['deeplink']}  [{dl['originalType']}]  {dl['message']}")
                if group["validationDeeplink"]:
                    v = group["validationDeeplink"]
                    extra = f" {v.get('condition')} {v.get('value')}" if v.get("condition") else ""
                    print(f"        validate: {v['deeplink']}  key={v['key']!r}{extra}")
            else:
                print("     -> null")
        print()

    ContextDeeplinkResponse(**env.response)
    print("-" * 78)
    print(f"schema validation: PASS   |  total {env.meta.latency_ms} ms")
    print(f"stages: {json.dumps(env.meta.stage_latency_ms)}")
    print(f"variations: {len(env.query_variations)}   fallback: {env.meta.fallback}")

    if args.write_contract_example:
        payload = {
            "query": env.query,
            "query_variations": env.query_variations,
            "response": env.response,
            "meta": {"latency_ms": None, "cache_hit": False, "model": None, "cost_usd": None,
                     "_note": "telemetry layer fills these at runtime; never authored"},
        }
        CONTRACT_EXAMPLE.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {CONTRACT_EXAMPLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
