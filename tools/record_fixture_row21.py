"""Generate backend/tests/fixtures/extraction_row21.json from the LIVE provider.

This is a RECORDED extraction: whatever the configured LLMProvider (Gemini by default)
returns for the `Touchscreen issues` reference article, pinned to disk so
backend/tests/test_row21_integration.py can replay it deterministically without an API
key or model drift turning the suite red for the wrong reason. It lives in tools/ +
tests/fixtures/, never in backend/app (directive 17) — no fixture content is hardcoded
into application code.

    python -m tools.record_fixture_row21                  # uses LLM_PROVIDER from env/config
    LLM_PROVIDER=stub python -m tools.record_fixture_row21 # offline dry run, for smoke-testing this script

Re-record whenever the prompt (backend/app/llm/base.py) or the model changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

ROW_ID = "row_21"
OUT = ROOT / "backend" / "tests" / "fixtures" / "extraction_row21.json"


def main() -> None:
    from app.llm import get_provider
    from app.pipeline.ground import normalize_siis

    rows = json.loads((ROOT / "data" / "siis_responses.json").read_text())["responses"]
    row = next(r for r in rows if r["id"] == ROW_ID)
    evidence = normalize_siis(row["siis_response"]).text
    query = row["original_query"]

    provider = get_provider()
    extraction = provider.extract(query, evidence)

    payload = {
        "_recorded_from": f"{provider.name}:{provider.model}",
        "_row_id": ROW_ID,
        **extraction.model_dump(mode="json"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    n_actions = len(extraction.actions)
    n_steps = sum(len(g.steps) for a in extraction.actions for g in a.step_groups)
    usage = provider.last_usage()
    print(f"wrote {OUT.relative_to(ROOT)}: recorded from {payload['_recorded_from']}, "
          f"{n_actions} actions, {n_steps} steps")
    print(f"usage: {usage}")


if __name__ == "__main__":
    main()
