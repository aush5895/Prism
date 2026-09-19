# Smart Guided Troubleshooting Engine
**Samsung PRISM Generative AI Hackathon, 3rd Edition (Y2026–27) — Theme 02**

Turns a vague Galaxy device complaint into a grounded, ordered, schema-valid troubleshooting plan with verified in-app Settings deeplinks.

> **Status: D1 — backend vertical slice complete.** One LLM call, everything downstream deterministic. 68 tests green, no API key required to run any of them.

---

## Quick start

```bash
pip install -r requirements.txt

make test           # full suite, offline, no API key           -> 68 passed
make test-row21     # the D1 milestone on its own               -> 19 passed
make demo           # print the row_21 plan with resolver trace
make run            # serve on http://localhost:8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/troubleshoot \
  -H 'Content-Type: application/json' \
  -d '{"query":"my galaxy phone touch screen is really laggy and slow to respond"}'
```

With a Gemini key: `export GEMINI_API_KEY=...` — the service picks it up automatically and logs a warning + falls back to the offline provider if it is missing. Tests never need it.

---

## Where the design comes from

Every load-bearing decision is cited to a numbered section of the supplied Samsung materials in **[`docs/API_CONTRACT.md` §1 (C1–C12)](docs/API_CONTRACT.md)**. Two are worth repeating here because they shape the whole architecture:

- **`siis_response` is supplied with the request, not retrieved by us.** Theme 2 guide §5 types it in the request body and says that when it is omitted *"the engine performs semantic lookup against pre-warmed cache entries"*; §2's pipeline table calls Structure Extraction a *parser* of that reference text; and `siis_responses.json`'s own `_readme` calls it *"the payload your API must accept"*. (C1–C4)
- **`deeplinks.json` is the corpus that gets searched.** Guide §2 — *"Matches step groups to specific device settings screens using semantic and keyword search across masked catalog URIs"* — with §7.4 requiring the match be made on `description`/`message`/`qna_description` and **never** on the URI. Guide §6.2's three graded IR criteria are screen-resolution accuracy, paraphrase hit rate and plan hierarchy; article retrieval is not among them. (C5–C8)

---

## Architecture

```
POST /v1/troubleshoot
  ├─ contracts.py        request validation (unknown fields rejected, not ignored)
  ├─ enrich.py           device / domain / symptoms / canonical query / cache key
  ├─ ground.py           siis_response used directly (C1); else fallback index (C2)
  ├─ llm/                ONE constrained call -> ir.Extraction   ← the only LLM on the path
  │                      (the IR has no deeplink field: the model cannot name a URI)
  ├─ deeplinks.py        gates [0]-[6] over the 578-entry catalog
  ├─ ordering.py         rules-over-model categorisation + 5 disruption tiers
  ├─ assemble.py         Samsung-schema dicts + score formula
  └─ validate.py         rule gates + URL scrub + ContextDeeplinkResponse
```

### Resolver gates

```
[0] short-circuit   manual category (guide §4.1) or hardware-button step -> null, no retrieval
[1] retrieval       BM25 over message + description + qna_description
[2] polarity        ON⇒onURL · OFF⇒offURL · VIEW⇒onClickURL · UPDATE⇒updateURL|onClickURL
[3] scope           reject a qualifier the step does not carry ("auto factory reset")
[4] target-concept  |content(candidate.message) ∩ tokens(step)| / |content(candidate.message)| ≥ 0.60
[5] margin          top1 − top2 ≥ δ; a single survivor needs no comparison
[6] identity        URI ∈ catalog and every field byte-equal to the catalog record
```

Three legal outcomes and no fourth: **exact catalog entry**, **`bixby://dummy_positive`**, **`null`**.

Gate [4] exists because gates [1]–[3]+[5] alone produced two measured failures on the real catalog — a false accept (*View Check Samsung Care+ subscription* returned for a software-update step) and a false abstain (*View Navigation bar*, an exact match, discarded when a distractor sharing the token "bar" collapsed the margin). Both are locked down by regression tests. The direction of the coverage ratio is the fix: it asks how much of the **candidate's own subject** the step covers, not the reverse. See `backend/app/text.py` for the tokenization policy and `docs/API_CONTRACT.md` §5.1 for the measurements.

### What the LLM is and is not

It is a parser. It receives the reference article and returns structure with character spans back into that article. It has no access to the catalog, and `ir.py` gives it no field in which a URI could be written. Word-count and casing rules are enforced in `validate.py`, not requested in the prompt — guide §7.5 is explicit that asking a model to respect them is unreliable.

---

## Layout

```
data/                       Samsung-supplied, unmodified, checksummed
  schema.py                 immutable contract (sha256 649440e0…6ebf6b)
  deeplinks.json            578 entries · siis_responses.json 20 rows · input.txt 20 complaints
  lexicons/                 category, tier, scope and domain lexicons — DATA, not code
backend/app/
  config.py  text.py  contracts.py  ir.py  telemetry.py  main.py
  schema_samsung.py         byte-identical copy of data/schema.py (asserted in CI)
  pipeline/                 enrich · ground · deeplinks · ordering · assemble · validate
  llm/                      base (interface + prompt) · stub · replay · gemini
backend/tests/              68 tests; fixtures/ holds recorded extractions
tools/                      demo_row21 · record_fixture_row21
docs/                       PHASE0_REQUIREMENTS_ANALYSIS.md · API_CONTRACT.md · worked_example_row21.json
```

## API

| Endpoint | Source | Notes |
|---|---|---|
| `POST /v1/troubleshoot` | Theme 2 guide §5 | `{query, siis_response?}` → `{query, query_variations, response, meta}`. `response` alone validates against `schema.py`. |
| `GET /health` | Theme 2 guide §5 | `503` until catalog and indexes are warm, then `200 {"status":"ok"}`. |
| `GET /metrics` | **ours, not in Samsung's contract** | Per-stage latency percentiles, cache counters, cost. |

## Guarantees enforced by tests

| Guarantee | Test |
|---|---|
| `schema.py` unmodified; vendored copy identical | `test_contract.py` |
| Every emitted URI is in the catalog, field for field | `test_resolver.py`, `test_generality.py` |
| The masked URI is never matched on | `test_uri_is_never_part_of_the_match` |
| `manual` actions never carry a deeplink | `test_row21_integration.py`, `test_generality.py` |
| `critical` actions ordered last | `test_row21_integration.py`, `test_generality.py` |
| Zero URL leakage | `test_zero_url_leakage` |
| No fixture text hardcoded in `backend/app` | `test_no_fixture_text_is_hardcoded_in_backend_app` |
| Pipeline is scenario-agnostic (all 20 rows via the offline parser) | `test_offline_stub_drives_every_supplied_row_to_valid_output` |

## Not built yet (by design)

Semantic cache (D3), repair/retry loop (D3), dense retrieval leg (D2, to be justified by ablation), frontend (D4), evaluation harness and `metrics.md` (D4). No Redis, no FAISS, no cross-encoder, no multi-agent orchestration — see `docs/PHASE0_REQUIREMENTS_ANALYSIS.md` §7 for why each was rejected.
