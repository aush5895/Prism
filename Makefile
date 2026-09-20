.PHONY: install run test test-row21 demo contract-example record-fixtures eval eval-live clean

PY ?= python3

install:
	$(PY) -m pip install -r requirements.txt

## Start the API on :8000. Runs on the offline stub unless GEMINI_API_KEY is exported.
run:
	$(PY) -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

## Full suite. Needs no API key.
test:
	$(PY) -m pytest

## The D1 milestone on its own.
test-row21:
	$(PY) -m pytest backend/tests/test_row21_integration.py -v

## Print the row_21 plan end to end, with the resolver's accept/reject trace.
demo:
	$(PY) -m tools.demo_row21

## Regenerate docs/worked_example_row21.json from the live pipeline.
contract-example:
	$(PY) -m tools.demo_row21 --write-contract-example

## Re-record the extraction fixture (requires a configured provider).
record-fixtures:
	$(PY) -m tools.record_fixture_row21

## Regenerate evaluation/report.json + docs/metrics.md. No API key needed.
eval:
	$(PY) -m evaluation.run_eval

## The same measurements with extraction served by the live provider.
eval-live:
	$(PY) -m evaluation.run_eval --provider gemini

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache
