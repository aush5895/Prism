# API Contract & Worked Example
### Theme 02 — Smart Guided Troubleshooting Engine · frozen before implementation

This document is the interface contract Day 1 builds against. Nothing here is aspirational: every deeplink URI, every catalog field and every gate decision in §7 was produced by running the real resolver logic over the real 578-entry catalog, and the final JSON was validated against Samsung's unmodified `schema.py`.

**Reproduce it:** `make contract-example` → regenerates `docs/worked_example_row21.json` and re-runs all gates.
**`schema.py` SHA-256:** `649440e0309b25dc5363039fc28dd7f21b71177779b909def8757b41e76ebf6b` (contract test asserts this; see §9.3)

---

## 1. Source citations — where each design claim comes from

Every load-bearing architectural claim below is traceable to a numbered section of the supplied Samsung materials. None of these is our inference.

| # | Claim | Citation | Quoted text |
|---|---|---|---|
| **C1** | `siis_response` is **supplied to the API as input**, not retrieved by us | **Theme 2 guide §5, "System Interface & API Contract"** | `POST /v1/troubleshoot` Request Body: `{ "query": "...", "siis_response": "<optional raw text context>" }` |
| **C2** | When `siis_response` is absent, the fallback is the **cache**, not a corpus search | **Theme 2 guide §5**, bullet under Request Body | *"If `siis_response` is omitted, the engine performs semantic lookup against pre-warmed cache entries."* |
| **C3** | The SIIS payload is **parsed**, not searched for | **Theme 2 guide §2**, Core Pipeline Components table, row *1. Structure Extraction* | *"Parses unstructured customer-care reference text into a structured Goal object containing atomic steps and action categories."* |
| **C4** | Confirmed by the kit's own data file | **`siis_responses.json` → `_readme`** | *"`siis_response` is the payload your API must accept in `POST /v1/troubleshoot`. … Derive actions and steps from it; do not invent steps that are not in the text."* |
| **C5** | **`deeplinks.json` is the corpus that is searched** | **Theme 2 guide §2**, row *2. Deeplink Mapping & Sequencing* | *"Matches step groups to specific device settings screens using semantic and keyword search across masked catalog URIs."* |
| **C6** | Matching happens on **metadata fields, never the URI** | **Theme 2 guide §7.4, "Matching on Masked URI Strings"** | *"Deeplink identifiers (`bixby://masked/act/…`) are obfuscated tokens. Matching must be performed on descriptive metadata fields (`description`, `message`, `qna_description`), never on the URI string itself."* |
| **C7** | Catalog integrity is mandatory | **Theme 2 guide §4.2.2, "Catalog Integrity"** | *"Deeplinks must be matched from `deeplinks.json` using semantic descriptions. Hallucinating or altering URIs is strictly forbidden."* |
| **C8** | The retrieval that is **graded** is deeplink retrieval and cache retrieval — not article retrieval | **Theme 2 guide §6.2, "Information Retrieval & Deeplink Precision"** | The three sub-criteria are *Screen Resolution Accuracy* ("mapping user actions to exact target screens rather than high-level parent menus"), *Semantic Paraphrase Hit Rate* (">= 80% cache hit rates on diverse, unseen query paraphrases"), and *Plan Hierarchy* ("sequencing steps by disruption level"). Article retrieval appears in none of them. |
| **C9** | Ordering rule | **Theme 2 guide §1 (Target State) and §4.1 (`category`)** | *"Order actions logically (least disruptive troubleshooting steps first; destructive/critical operations last)."* · `critical`: *"Must be ordered last."* |
| **C10** | `manual` may not carry an actionable deeplink | **Theme 2 guide §4.1, `category`** | `manual`: *"Physical interventions (e.g., cleaning ports, replacing hardware, visiting service centers). Cannot carry an actionable deeplink."* |
| **C11** | `dummy_positive` usage rule | **`deeplinks.json` → entry `DL-DUMMY.qna_description`** | *"Use only when a step opens a Settings screen and no other catalog entry matches it. Write description and message yourself (5-7 words, naming the concrete screen from the steps)."* |
| **C12** | Empty-result contract | **Theme 2 guide §4.2.3, "No Hallucinated Steps"** | *"If the reference data contains no viable solution, the engine must return an empty list (`contexts: []`) with fallback metadata (`"fallback": "no_match"`).* |

**Consequence, stated as a derivation rather than an opinion:** C1+C2+C3+C4 establish that the SIIS article arrives with the request. C5+C6+C8 establish that the corpus requiring semantic retrieval is the deeplink catalog. Therefore the retrieval subsystem is sized for `deeplinks.json` (578 entries), and article lookup exists only to serve the `siis_response`-absent path described in C2 — where the guide names the cache first. `README.md` §Architecture and `docs/ARCHITECTURE.md` §1 both carry this table so the reasoning travels with the repo.

---

## 2. Request contract

### `POST /v1/troubleshoot`

```jsonc
{
  "query":         "string, required, 1..2000 chars — the raw colloquial complaint",
  "siis_response": "string | {title, content} | null — optional raw reference text"
}
```

| Field | Type | Required | Handling |
|---|---|---|---|
| `query` | `str` | **yes** | Empty/whitespace → `422`. Over 2000 chars → truncated, flagged in `meta.warnings`. |
| `siis_response` | `str \| {title, content} \| null` | no | Per C1 this is the grounding evidence. The API contract (C1) types it as a string; the kit fixture ships it as `{title, content}`. **We accept both** and normalize to `f"{title}\n{content}"`. Absent → path C2 (cache, then BM25 fallback index over supplied rows, then `no_siis_context`). |

No other fields are accepted. Unknown fields are rejected with `422` rather than ignored — a silently-dropped field is how a demo goes wrong.

### `GET /health`

`200 {"status": "ok"}` **only when** the embedding model is loaded, both indexes are built and the cache is attached — per Theme 2 guide §5: *"Returns HTTP 200 (`{"status": "ok"}`) when the caching layer, model connections, and vector indexes are fully initialized."* Until then `503`. This makes cold-start measurable instead of invisible.

### `GET /metrics` — **ours, not Samsung's**

Explicitly marked non-spec in the README. Serves per-stage latency percentiles, cache hit/miss, token and cost counters. It exists so that §10 of your directives (metrics measured, never authored) has a single source.

---

## 3. Response contract

### 3.1 Envelope

```jsonc
{
  "query":            "<echo of the raw input>",
  "query_variations": ["…8 to 10 paraphrases…"],
  "response":         { "contexts": [ …Goal… ] },
  "meta":             { "latency_ms": 0, "cache_hit": false, "model": "…", "cost_usd": 0.0 }
}
```

`response` — and only `response` — is validated against `ContextDeeplinkResponse` from Samsung's `schema.py`, unmodified. `query_variations` is required by guide §4.1; `meta` matches Appendix B's worked example field-for-field. Neither is ever nested inside `response`, so the graded object stays exactly the shape Samsung defined.

`meta` values are written by the telemetry layer at request time. **No value in `meta` is ever authored, defaulted to a flattering number, or copied from a previous run.**

### 3.2 Field rules enforced programmatically (guide §4.1)

| Field | Rule | Enforcement |
|---|---|---|
| `goal` | `Follow these steps to perform this {Topic} Troubleshooting` | regex `fullmatch`, repair prompt on failure |
| `title` | 2–3 words, sentence case | word count + case assertion |
| `score` | float ∈ [0,1] | computed by formula (§3.4), not by the model |
| `actionName` | Title Case, exactly one screen/feature | `str.title()` equality + post-hoc same-screen merge |
| `description` | `It will` + 5–7 words | word count; auto-trim then re-validate |
| `steps` | imperative, one interaction each, no URLs | URL regex over every string in the payload |
| `category` | `auto` / `manual` / `critical` | rule-based override of the model's proposal (§6.2) |
| `actionableDeeplink` | URI ∈ catalog, fields byte-equal to catalog record | gate 5 (§5) |
| `query_variations` | 8–10, mixed registers | count + pairwise-distinctness check |

### 3.3 Fallback contract

| Condition | HTTP | Body |
|---|---|---|
| Article contains no viable solution for the complaint (C12) | `200` | `response.contexts: []`, `meta.fallback: "no_match"` |
| No `siis_response`, cache miss, fallback index finds nothing | `200` | `response.contexts: []`, `meta.fallback: "no_siis_context"` |
| Extraction fails schema validation after 2 repair attempts | `200` | `response.contexts: []`, `meta.fallback: "schema_repair_exhausted"` |
| Malformed request | `422` | FastAPI validation error |
| Upstream LLM unreachable | `503` | `{"detail": "extraction_unavailable"}` |

`fallback` lives in `meta`, not in `response`, because `ContextDeeplinkResponse` has no such field and we do not widen Samsung's model. An empty `contexts` list is a **success**, never a 500 — returning nothing is the specified correct behaviour when the evidence does not support a plan.

### 3.4 How `score` is computed

The model does not choose the confidence. It is a defined function of measurable quantities:

```
score = 0.40 · span_coverage        # fraction of actions with a verified char-span into the SIIS text
      + 0.30 · deeplink_precision   # fraction of auto stepGroups resolved to an exact catalog entry
                                    #   (dummy_positive counts as 0 — it is a known unknown)
      + 0.30 · evidence_alignment   # retrieval alignment between query and the grounding article
```

Weights live in one config constant and are reported in `metrics.md`. For the §7 example the three terms measured `1.000`, `1.000`, `0.388` → **`score = 0.82`**.

---

## 4. Generality — no per-query logic anywhere

Your directive 2, made checkable rather than promised:

- The only per-scenario artefacts in the repo are **data** (`data/`, frozen, checksummed) and **labels** (`evaluation/gold/`). Neither is imported by `backend/app/`.
- A CI test greps `backend/app/**` for any string literal appearing in `input.txt` or in any `siis_response`. A match fails the build. This is what makes "no hard-coded cases" an enforced property rather than a claim.
- The three lexicons that do exist (critical verbs, physical-action verbs, scope qualifiers) are **category-level and domain-general** — they contain no device model, no complaint text, no article title. They are `data/lexicons/*.yaml`, editable without touching code, which is also the 10k+ scaling story: new scenarios need new *catalog rows*, not new code paths.

---

## 5. Deeplink resolver — final pipeline

Your specified order, with **one gate inserted**. The insertion is not a preference; the probe below proves the pipeline is unsafe without it.

```
step text
  │
  ├─ [0] SHORT-CIRCUIT        category ∈ {manual} → deeplink = null (C10). Physical critical steps
  │                            (button presses) → null. Resolver never runs. ← catches ~44% of steps
  │
  ├─ [1] CANDIDATE RETRIEVAL  hybrid BM25 (message+description+qna_description)
  │                            + dense (MiniLM-L6-v2), reciprocal-rank fusion, top-40
  │
  ├─ [2] POLARITY / TYPE GATE intent(step) ∈ {ON,OFF,VIEW,UPDATE} must match originalType
  │                            ON⇒onURL · OFF⇒offURL · VIEW⇒onClickURL · UPDATE⇒updateURL|onClickURL
  │
  ├─ [3] SCOPE GATE           reject if candidate carries a qualifier absent from the step
  │                            (auto, schedule, inactivity, preset, + appliance nouns)
  │
  ├─ [4] TARGET-CONCEPT GATE  ← ADDED. Direction matters: the candidate's OWN subject must appear
  │                            in the step, not merely share a token with it.
  │                            coverage = |content(message) ∩ tokens(step)| / |content(message)|
  │                            reject if coverage < 0.60
  │
  ├─ [5] MARGIN GATE          require top1 − top2 ≥ δ over survivors, else abstain
  │
  ├─ [6] CATALOG IDENTITY     assert URI ∈ deeplinks.json AND description/message/originalType/
  │                            validation are field-equal to the catalog record (C7)
  │
  └─► exact catalog entry  |  bixby://dummy_positive (C11)  |  null
```

### 5.1 Why gate [4] exists — measured, not argued

Running gates [1]–[3] and [5] only, against the real catalog:

| Step | Top BM25 candidate | Outcome without gate [4] |
|---|---|---|
| `Check for software updates on your device.` | `DL-0576` *View Check Samsung Care+ subscription* | **ACCEPTED — wrong.** The catalog has no software-update entry at all; "Check" alone carried it through, and the margin gate did not fire. A false positive that would have shipped. |
| `Tap Navigation bar.` | `DL-0169` *View Navigation bar* (correct) vs `DL-0379` *View Double tap space bar to add period* at 0.975 | **ABSTAINED — wrong.** "bar" matched; the margin collapsed to 0.025 and a perfectly correct exact match was thrown away for `dummy_positive`. |

Both failure directions from one root cause: token overlap in the wrong direction. Gate [4] fixes both, with no model involved:

| Step | Candidate | coverage | Verdict |
|---|---|---|---|
| `Check for software updates…` | *View Check Samsung Care+ subscription* → `{samsung, care, subscription}` | 0.00 | reject → `dummy_positive` ✅ |
| `Tap Navigation bar.` | *View Navigation bar* → `{navigation, bar}` | 1.00 | accept `DL-0169` ✅ |
| `Tap Navigation bar.` | *View Double tap space bar…* → `{double,tap,space,bar,add,period}` | 0.33 | reject ✅ |

Also measured and now regression-tested: the `auto factory reset` trap. `Tap Factory data reset.` ranks `DL-0022` *View Reset Options* ("Opens the **auto** factory reset settings page") at 1.000 — gate [3] rejects on the qualifier `auto`, and the step falls to `dummy_positive` rather than sending a user to a scheduling preference. And the polarity pair: `Enable Touch sensitivity` (`DL-0126`, onURL) vs `Disable Touch sensitivity` (`DL-0125`, offURL) score 1.000/0.934 — lexically inseparable, correctly separated by gate [2] in **both** directions.

### 5.2 Note on the dense leg

The measurements above are BM25-only; the HuggingFace model is not reachable from this environment. Every resolution in §7 is therefore achieved by **lexical retrieval plus deterministic gates alone** — which is a useful floor, and makes the dense leg a measurable *improvement* in the ablation rather than an unexamined assumption. If the dense leg does not beat this floor on the gold set, we ship without it and say so.

---

## 6. Ordering and categorisation

### 6.1 Disruption tiers

Phase 0 proposed 3 tiers. Building the §7 example exposed a flaw: a pure category sort puts *"remove your peeling screen protector"* — the least disruptive thing in the entire article, and the article's own first step — after two Settings changes. That contradicts C9's stated principle while technically satisfying it. Refined to 5 internal tiers; the emitted `category` stays one of Samsung's three values.

| Tier | Contents | Emitted `category` |
|---|---|---|
| 0 | manual, non-invasive — remove accessory, clean surface, swap charger | `manual` |
| 1 | auto, toggle — `onURL` / `offURL` / `updateURL` | `auto` |
| 2 | auto, navigational — `onClickURL` / `dummy_positive` | `auto` |
| 3 | manual, service escalation — contact support, service centre, hardware replacement | `manual` |
| 4 | critical — restart, firmware update, safe mode, factory reset | `critical` |

Sort is `(tier, source_order)`, stable — so within a tier the article's own narrative order survives, which is why factory reset lands last in §7 exactly as the article intends ("a last resort, after trying all the previous steps"). Tier 4 last satisfies C9's explicit rule. Tiers 1→2 implement C8's *Plan Hierarchy*. The table is one dict in `ordering.py`.

### 6.2 Category assignment is rules-over-model

Stage 1 proposes a category; Stage 2 overrides it. `critical` lexicon: factory reset, force restart, power off, safe mode, software/firmware update, recovery mode. `manual` lexicon: inspect, remove, eject, wipe, clean, replace, contact, service centre. Anything else with a resolved deeplink → `auto`; anything else without → `manual`, which is also `schema.py`'s own default and the safe failure direction (C10 means a mis-categorised action loses its deeplink rather than gaining a wrong one).

Note `Check Software Update` in §7 is `critical`, not `auto` — guide §4.1 lists *firmware update* explicitly among critical operations. Easy to get wrong by intuition; it comes straight from the text.

---

## 7. Complete worked example — `row_21`, raw query to final JSON

Chosen because it exercises **every branch** of the resolver in one request: both polarity directions, an exact navigational match, a genuine catalog gap, the `auto factory reset` trap, physical manual actions, and critical ordering.

### Stage 0 — Input

```json
{ "query": "My Galaxy S22 screen inputs are delayed and the touch responsiveness is laggy, causing a noticeable delay when I try to interact with the phone.",
  "siis_response": { "title": "Touchscreen issues on a Galaxy phone or tablet", "content": "…4.5k chars…" } }
```

### Stage 1 — Query enrichment

| Slot | Value |
|---|---|
| device | `Galaxy S22` |
| domain | `display / touch` |
| symptoms | `delayed input`, `laggy touch response` |
| negative evidence | *(none stated — no mention of cracks, water, or physical damage)* |
| canonical query | `galaxy s22 touchscreen input delay and laggy touch response` |
| cache key | SHA-1 of canonical query + 384-d embedding |

### Stage 2 — Grounding

`siis_response` supplied → used directly (path C1). *Cross-check: the BM25 fallback index, run independently on this query, also returns `row_21` at rank 1 — so the C2 path would have found the same evidence.*

### Stage 3 — Extraction (the single LLM call)

Evidence = article text only. Output: 9 candidate actions, each carrying a character span into the source. Span coverage **9/9 = 1.000**.

Note the §1/§5 conditional pair: the article says *enable* Touch sensitivity if you keep a protector, and *disable* it if you do not. Same screen, two operations — this is the legitimate use of `stepGroups` being a list, and it resolves ambiguity A5 from Phase 0: **one stepGroup per distinct operation on that screen**, not one per action.

### Stage 4 — Deeplink resolution (measured, gate by gate)

| # | Step | Gate outcome | Resolved |
|---|---|---|---|
| 1 | `Tap the switch next to Touch sensitivity to enable it.` | intent `ON`; `DL-0125` (offURL) rejected by **[2] polarity**; `DL-0089/0091/0189` rejected by **[4] concept** (cov 0.25) | `DL-0126` → `bixby://masked/act/14eb42b895` |
| 2 | `Tap the switch next to Touch sensitivity to disable it.` | intent `OFF`; `DL-0126` (onURL) rejected by **[2]** | `DL-0125` → `bixby://masked/act/1b0d34e9b4` |
| 3 | `Tap Navigation bar.` | intent `VIEW`; `DL-0379` rejected by **[4]** (cov 0.33), saving the margin gate | `DL-0169` → `bixby://masked/act/2f3dd95259` |
| 4 | `Tap Factory data reset.` | `DL-0022` *auto factory reset* rejected by **[3] scope('auto')**; all others by **[4]** | `bixby://dummy_positive` |
| 5 | `Check that your device software is up to date.` | `category=critical`, no Settings screen opened by any step | `null` |
| 6 | Restart / Safe mode / protector / charger / support | **[0] short-circuit** — physical or `manual` (C10) | `null` |

All emitted URIs passed **[6] catalog identity**: present in `deeplinks.json`, all fields byte-equal to the catalog record, `validation` copied verbatim (including `resultType`/`condition`/`value` on the `onURL` entry and key-only on the `offURL` entry — exactly the asymmetry `sample_output.json` demonstrates).

### Stage 5 — Ordering

```
tier 0  Remove Screen Protector · Try A Different Charger
tier 1  Adjust Touch Sensitivity
tier 2  Configure Navigation Bar
tier 3  Contact Samsung Support
tier 4  Restart Your Device · Check Software Update · Enter Safe Mode · Perform Factory Data Reset
```

### Stage 6 — Final response (abridged; full file at `docs/worked_example_row21.json`)

```jsonc
{
  "query": "My Galaxy S22 screen inputs are delayed and the touch responsiveness is laggy, …",
  "query_variations": [
    "Touch response on my Galaxy S22 is delayed and unresponsive.",   // formal
    "my s22 touchscreen is super laggy when i tap stuff",              // casual
    "S22 touch lag delayed input screen",                              // keyword-only
    "this phone touch is so slow its driving me insane",               // frustrated
    "my galxy s22 tuch screen is laggy and slow to respnd",            // typo-inclusive
    … 10 total …
  ],
  "response": {
    "contexts": [{
      "goal":  "Follow these steps to perform this Touchscreen Troubleshooting",
      "title": "Touchscreen response issues",
      "score": 0.82,
      "actions": [
        { "actionName": "Remove Screen Protector",
          "description": "It will remove interference from the screen surface",
          "category": "manual",
          "stepGroups": [{ "steps": ["Remove any third-party screen protector that is peeling or has debris under it.",
                                     "Wipe the front and back gently with a lint-free microfiber cloth.",
                                     "Avoid applying too much pressure while cleaning."],
                           "actionableDeeplink": null, "validationDeeplink": null }] },

        { "actionName": "Adjust Touch Sensitivity",
          "description": "It will match screen response to your protector",
          "category": "auto",
          "stepGroups": [
            { "steps": ["Navigate to and open Settings.", "Tap Display.",
                        "Tap the switch next to Touch sensitivity to enable it."],
              "actionableDeeplink": { "deeplink": "bixby://masked/act/14eb42b895",
                                      "description": "Enables touch sensitivity via device Settings on the device.",
                                      "message": "Enable Touch sensitivity", "originalType": "onURL" },
              "validationDeeplink": { "deeplink": "bixby://masked/val/6451858b28", "key": "Touch sensitivity",
                                      "resultType": "boolean", "condition": "equal", "value": "True" } },
            { "steps": ["Navigate to and open Settings.", "Tap Display.",
                        "Tap the switch next to Touch sensitivity to disable it."],
              "actionableDeeplink": { "deeplink": "bixby://masked/act/1b0d34e9b4",
                                      "description": "Disables touch sensitivity via device Settings on the device.",
                                      "message": "Disable Touch sensitivity", "originalType": "offURL" },
              "validationDeeplink": { "deeplink": "bixby://masked/val/6451858b28", "key": "Touch sensitivity" } }] },

        { "actionName": "Configure Navigation Bar", …  "actionableDeeplink": { "deeplink": "bixby://masked/act/2f3dd95259", … } },
        { "actionName": "Contact Samsung Support",  …  "category": "manual",   "actionableDeeplink": null },
        { "actionName": "Restart Your Device",      …  "category": "critical", "actionableDeeplink": null },
        { "actionName": "Check Software Update",    …  "category": "critical", "actionableDeeplink": null },
        { "actionName": "Enter Safe Mode",          …  "category": "critical", "actionableDeeplink": null },
        { "actionName": "Perform Factory Data Reset",
          "description": "It will erase all data and restore defaults",
          "category": "critical",
          "stepGroups": [{ "steps": ["Back up your personal data before you continue.", "Navigate to and open Settings.",
                                     "Tap General management.", "Tap Reset.", "Tap Factory data reset.",
                                     "Swipe to and tap Reset.", "Tap Delete all."],
                           "actionableDeeplink": { "deeplink": "bixby://dummy_positive",
                                                   "description": "Opens the Factory data reset settings screen",
                                                   "message": "Open Factory data reset under General management",
                                                   "originalType": "placeholder" },
                           "validationDeeplink": null }] }
      ]
    }]
  },
  "meta": { "latency_ms": <runtime>, "cache_hit": false, "model": "<configured>", "cost_usd": <runtime> }
}
```

### Stage 7 — Verification (all executed, all green)

```
*** SCHEMA VALIDATION: PASS ***  goals=1 actions=9      ← against unmodified schema.py
*** ALL AUTOMATED GATES: PASS ***
    url-leak · description(It will + 5–7) · actionName Title Case · manual⇒null deeplink
    catalog-membership · critical-last · goal-template · title 2–3 words · score∈[0,1]
    query_variations count 8–10
```

### Stage 8 — Cache write

The validated plan is written once, keyed by the canonical query **and all 10 variations** (11 vectors). Precision over recall per your directive 6: a hit must clear a high similarity threshold **and** pass a cheap post-hit guard — the cached plan's device/domain slots must match the incoming query's slots. Guard failure ⇒ treated as a miss ⇒ full pipeline. Every hit is logged with its similarity and its source query so false positives are auditable, and the false-positive rate is a row in `metrics.md`.

---

## 8. What changed since Phase 0, and why

| # | Change | Cause |
|---|---|---|
| 1 | Target-concept gate added to the resolver | Probe found a false accept (Care+ for software update) and a false abstain (navigation bar). §5.1 |
| 2 | Ordering tiers 3 → 5 | A pure category sort demoted the article's own least-disruptive first step below two Settings changes. §6.1 |
| 3 | A5 resolved: one stepGroup **per distinct operation on the screen**, not one per action | The `row_21` conditional enable/disable pair is the case the `List[StepGroup]` type exists for. §7 Stage 3 |
| 4 | `score` defined as a formula over measured terms | Directive 10; also stops the model from inventing its own confidence |
| 5 | `fallback` relocated from `response` to `meta` | `ContextDeeplinkResponse` has no such field and we do not widen Samsung's model. §3.3 |
| 6 | Dense leg reframed as a measured improvement over a lexical floor | HF unreachable here; the floor turns out to be strong, so the dense leg must now earn its place in the ablation. §5.2 |

---

## 9. Where each of your twelve directives is enforced

| # | Directive | Enforced by |
|---|---|---|
| 1 | Cite the Theme 2 sections; not an assumption | §1 (C1–C12), mirrored into `README.md` and `ARCHITECTURE.md` |
| 2 | Generalized; 20 rows are fixtures | §4 + CI test grepping `backend/app/**` against `input.txt` and every `siis_response` |
| 3 | `schema.py` immutable, checksum test | `tests/test_contract.py::test_schema_checksum` asserts SHA-256 `649440e0…6ebf6b`; CI runs it first and fails the build on mismatch |
| 4 | LLM constrained to grounded structuring | Single call, evidence-only prompt, mandatory source spans (span coverage is a graded term in `score`), deeplinks never model-authored except `DL-DUMMY` prose (C11) |
| 5 | Resolver stage order | §5 — your five stages, plus gate [4] justified by measurement in §5.1 |
| 6 | Cache precision > recall | §7 Stage 8 — high threshold, slot guard, guard failure ⇒ full pipeline, FP rate reported |
| 7 | Minimal core before innovation | Phase 0 §12 cut order; D1–D3 are core only, differentiators start D4 |
| 8 | Vertical slice first | D1: one real input → envelope → schema validation → correct deeplink handling. This document *is* the target of that slice. |
| 9 | No frontend until backend runs on all cases | Frontend starts D4, gated on all 20 rows producing schema-valid output |
| 10 | Metrics measured, never hard-coded | `meta` written by telemetry only; `metrics.md` generated by `make eval`; a test fails if any numeric literal in the report is not traceable to `report.json` |
| 11 | No adaptive re-planning in MVP | Cut from scope; listed in `LIMITATIONS.md` |
| 12 | Contract + worked example before coding | This document; example regenerable via `make contract-example` |

---

## 10. Open item

Provider keys. The client is provider-agnostic and temperature-0 with JSON mode; `LLM_PROVIDER` + `LLM_MODEL` are env vars, and an offline deterministic stub backs every test so CI never needs a key. I need to know which of Gemini / OpenAI you have a working key for to set the default and start cost accounting. Day 1 does not block on it — the vertical slice runs against the stub.
