"""D1 vertical slice, end to end, on a real supplied Theme 2 case.

Request -> enrichment -> grounding -> extraction -> gates [0]-[6] -> 5-tier ordering
-> Samsung schema validation -> envelope. The extraction is replayed from a recording so
this test pins the DETERMINISTIC half; every deeplink below is computed live by the
resolver against the real 578-entry catalog, not replayed.

D3a note: extraction_row21.json was re-recorded from the live gemini-3.1-flash-lite
provider (see tools/record_fixture_row21.py); it was previously a hand-authored D1
placeholder. The assertions below were updated to match what the live model actually
produces. Two things changed for reasons worth knowing, not pipeline bugs:
  - The "enable Touch sensitivity" step lost its explicit "to enable it" wording once it
    was split into its own action (a prompt fix for a different bug -- see git history on
    backend/app/llm/base.py). Gate [2] polarity correctly can't tell which catalog entry
    an unqualified "tap the switch" means, so it declines to resolve rather than guess --
    this is the resolver doing its job, not a regression. The polarity-gate MECHANISM
    itself (does "enable" vs "disable" wording route to the right catalog entry) is
    covered independently in test_resolver.py::test_polarity_gate_separates_enable_from_disable
    with controlled text, so this file no longer re-asserts it against row_21's specific,
    now-ambiguous phrasing.
  - The article's "Software Updates" section (no explicit UI path, just "keep software
    current") was judged not independently actionable and folded out; there is no
    "Check Software Update" action to test a null deeplink against any more.
"""
import pytest

from app.contracts import TroubleshootRequest
from app.main import run_pipeline
from app.llm.replay import ReplayProvider
from app.schema_samsung import ContextDeeplinkResponse

from conftest import FIXTURES

TOUCH_SENSITIVITY_ON = "bixby://masked/act/14eb42b895"   # DL-0126, onURL
TOUCH_SENSITIVITY_OFF = "bixby://masked/act/1b0d34e9b4"  # DL-0125, offURL
NAVIGATION_BAR = "bixby://masked/act/2f3dd95259"         # DL-0169, onClickURL
DUMMY = "bixby://dummy_positive"


@pytest.fixture(scope="module")
def envelope(request):
    rows = __import__("json").loads(
        (request.config.rootpath / "data" / "siis_responses.json").read_text())["responses"]
    row = next(r for r in rows if r["id"] == "row_21")
    req = TroubleshootRequest(query=row["original_query"].split(". ", 1)[-1],
                              siis_response=row["siis_response"])
    return run_pipeline(req, provider=ReplayProvider(FIXTURES / "extraction_row21.json"))


@pytest.fixture(scope="module")
def goal(envelope):
    return envelope.response["contexts"][0]


@pytest.fixture(scope="module")
def actions(goal):
    return {a["actionName"]: a for a in goal["actions"]}


# ----------------------------------------------------------------- schema
def test_response_validates_against_unmodified_samsung_schema(envelope):
    parsed = ContextDeeplinkResponse(**envelope.response)
    assert len(parsed.contexts) == 1
    assert len(parsed.contexts[0].actions) == 10


def test_envelope_shape(envelope):
    assert envelope.query
    assert 8 <= len(envelope.query_variations) <= 10
    assert envelope.meta.fallback is None
    assert envelope.meta.latency_ms is not None and envelope.meta.latency_ms > 0
    assert set(envelope.meta.stage_latency_ms) >= {
        "enrich", "ground", "extract", "alignment", "resolve_and_order", "validate"}


def test_goal_and_title_follow_the_required_syntax(goal):
    import re
    assert re.fullmatch(r"Follow these steps to perform this .+ Troubleshooting", goal["goal"])
    assert "Touchscreen" in goal["goal"], "the goal must name the article's topic"
    # The exact wording is the model's and shifts between recordings; the RULE is ours.
    assert 2 <= len(goal["title"].split()) <= 3
    assert goal["title"][0].isupper() and goal["title"] == goal["title"][0] + goal["title"][1:].lower()
    assert 0.0 <= goal["score"] <= 1.0


def test_score_comes_from_the_formula_not_the_model(goal):
    """Contract §3.4: span_coverage 0.40 + deeplink_precision 0.30 + alignment 0.30.

    0.82 until span verification landed, because span_coverage fell from a fabricated
    1.00 to a measured value: the extractor's character offsets were guesses and the old
    check accepted any offset that was merely in BOUNDS. See test_spans.py.
    """
    assert goal["score"] == pytest.approx(0.75, abs=0.01)


# ----------------------------------------------------------------- grounding
def test_extraction_is_grounded_with_verified_spans(envelope, request):
    import json
    rows = json.loads((request.config.rootpath / "data" / "siis_responses.json").read_text())
    row = next(r for r in rows["responses"] if r["id"] == "row_21")
    evidence_len = len(f"{row['siis_response']['title']}\n{row['siis_response']['content']}")
    recording = json.loads((FIXTURES / "extraction_row21.json").read_text())
    spans = [s["source_span"] for a in recording["actions"]
             for g in a["step_groups"] for s in g["steps"]]
    assert spans and all(0 <= a < b <= evidence_len for a, b in spans)


# ----------------------------------------------------------------- polarity
def test_both_toggle_polarities_resolve_to_their_own_catalog_entry(actions):
    """The pair is lexically inseparable and must be separated by polarity alone.

    This regressed twice and is worth pinning. It first broke when the extractor wrote
    both directions as the bare "Tap the switch next to Touch sensitivity.", which gate
    [2] rightly refuses to guess at -- the prompt now requires every toggle step to say
    which way it is being set. It broke again when "Adjust touch sensitivity" was
    classified `manual` because its DESCRIPTION mentioned a screen protector, and a manual
    action may never carry a deeplink.
    """
    on = actions["Adjust Touch Sensitivity"]
    off = actions["Disable Touch Sensitivity"]
    assert on["category"] == "auto" and off["category"] == "auto"

    on_group, off_group = on["stepGroups"][0], off["stepGroups"][0]
    assert on_group["actionableDeeplink"]["deeplink"] == TOUCH_SENSITIVITY_ON
    assert on_group["actionableDeeplink"]["originalType"] == "onURL"
    assert off_group["actionableDeeplink"]["deeplink"] == TOUCH_SENSITIVITY_OFF
    assert off_group["actionableDeeplink"]["originalType"] == "offURL"
    assert on_group["actionableDeeplink"]["deeplink"] != off_group["actionableDeeplink"]["deeplink"]

    # catalog asymmetry, copied not authored: onURL carries the full validation shape
    assert on_group["validationDeeplink"]["resultType"] == "boolean"
    assert on_group["validationDeeplink"]["value"] == "True"
    assert set(off_group["validationDeeplink"]) == {"deeplink", "key"}


def test_qualified_toggle_wording_resolves(actions):
    """"Disable Touch Sensitivity"'s step explicitly says "to disable it", so it
    resolves to the offURL catalog entry."""
    act = actions["Disable Touch Sensitivity"]
    assert act["category"] == "auto"
    group = act["stepGroups"][0]
    assert group["actionableDeeplink"]["deeplink"] == TOUCH_SENSITIVITY_OFF
    assert group["actionableDeeplink"]["originalType"] == "offURL"
    assert set(group["validationDeeplink"]) == {"deeplink", "key"}


# ----------------------------------------------------------------- exact match
def test_navigation_bar_matches_exactly(actions):
    group = actions["Disable Full Screen Gestures"]["stepGroups"][0]
    assert group["actionableDeeplink"]["deeplink"] == NAVIGATION_BAR
    assert group["actionableDeeplink"]["message"] == "View Navigation bar"
    assert group["validationDeeplink"]["key"] == "Navigation bar"


# ----------------------------------------------------------------- dummy / null
def test_factory_reset_falls_back_to_dummy_positive(actions):
    act = actions["Perform Factory Reset"]
    assert act["category"] == "critical"
    group = act["stepGroups"][0]
    assert group["actionableDeeplink"]["deeplink"] == DUMMY
    assert "Factory data reset" in group["actionableDeeplink"]["message"]
    assert group["validationDeeplink"] is None


def test_no_manual_action_anywhere_carries_a_deeplink(goal):
    """Guide §4.1. Asserted over every manual action in the plan rather than a list of
    names, which drifts each time the extraction is re-recorded."""
    manual = [a for a in goal["actions"] if a["category"] == "manual"]
    assert len(manual) >= 3, "row_21 should yield several hands-on actions"
    for action in manual:
        for group in action["stepGroups"]:
            assert group["actionableDeeplink"] is None, action["actionName"]
            assert group["validationDeeplink"] is None, action["actionName"]


@pytest.mark.parametrize("name", ["Restart The Device", "Enter Safe Mode"])
def test_physical_critical_actions_carry_no_deeplink(actions, name):
    act = actions[name]
    assert act["category"] == "critical"
    assert act["stepGroups"][0]["actionableDeeplink"] is None


# ----------------------------------------------------------------- ordering
def test_five_tier_ordering(goal):
    """Contract §6.1: tiers ascend, and the emitted names are not the contract.

    Pinning the exact name sequence made this fail on every re-recording for reasons
    like "Contact Support" becoming "Contact Samsung Support". The ordering property is
    what matters: hands-on checks first, then the Settings actions, then service
    escalation, then the disruptive operations last.
    """
    names = [a["actionName"] for a in goal["actions"]]
    categories = [a["category"] for a in goal["actions"]]

    first_auto = categories.index("auto")
    last_auto = len(categories) - 1 - categories[::-1].index("auto")
    first_critical = categories.index("critical")

    assert first_auto > 0, "non-invasive manual checks come before the Settings actions"
    assert last_auto - first_auto == categories.count("auto") - 1, "autos are contiguous"
    assert first_critical > last_auto, "disruptive operations come after the Settings ones"

    # the one manual permitted after the autos is the service escalation (tier 3)
    for i, (name, category) in enumerate(zip(names, categories)):
        if category == "manual" and i > last_auto:
            assert "support" in name.lower(), f"{name!r} should not sort after the autos"


def test_critical_actions_are_last(goal):
    categories = [a["category"] for a in goal["actions"]]
    first_critical = categories.index("critical")
    assert all(c == "critical" for c in categories[first_critical:])


# ----------------------------------------------------------------- hygiene
def test_zero_url_leakage(envelope):
    import json
    import re
    blob = json.dumps(envelope.model_dump())
    assert not re.search(r"https?://|www\.|\]\(", blob)


def test_every_emitted_uri_is_catalog_backed(goal, catalog):
    seen = 0
    for action in goal["actions"]:
        for group in action["stepGroups"]:
            dl = group["actionableDeeplink"]
            if dl:
                catalog.verify_identity(dl)
                seen += 1
    assert seen == 4  # 2 toggles + 1 navigational + 1 dummy_positive


def test_matches_the_frozen_contract_example(envelope, request):
    """docs/worked_example_row21.json is the frozen §7 example. Drift is a contract break."""
    import json
    frozen = json.loads((request.config.rootpath / "docs" / "worked_example_row21.json").read_text())
    assert envelope.response == frozen["response"]
    assert envelope.query_variations == frozen["query_variations"]
