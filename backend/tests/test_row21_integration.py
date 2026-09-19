"""D1 vertical slice, end to end, on a real supplied Theme 2 case.

Request -> enrichment -> grounding -> extraction -> gates [0]-[6] -> 5-tier ordering
-> Samsung schema validation -> envelope. The extraction is replayed from a recording so
this test pins the DETERMINISTIC half; every deeplink below is computed live by the
resolver against the real 578-entry catalog, not replayed.
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
    assert len(parsed.contexts[0].actions) == 9


def test_envelope_shape(envelope):
    assert envelope.query
    assert 8 <= len(envelope.query_variations) <= 10
    assert envelope.meta.fallback is None
    assert envelope.meta.latency_ms is not None and envelope.meta.latency_ms > 0
    assert set(envelope.meta.stage_latency_ms) >= {
        "enrich", "ground", "extract", "alignment", "resolve_and_order", "validate"}


def test_goal_and_title_follow_the_required_syntax(goal):
    assert goal["goal"] == "Follow these steps to perform this Touchscreen Troubleshooting"
    assert goal["title"] == "Touchscreen response issues"
    assert 0.0 <= goal["score"] <= 1.0


def test_score_comes_from_the_formula_not_the_model(goal):
    # span_coverage 1.0, deeplink_precision 1.0, evidence_alignment measured by BM25
    assert goal["score"] == pytest.approx(0.82, abs=0.01)


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
def test_both_touch_sensitivity_polarities_resolve_on_one_screen(actions):
    """One action = one screen; two distinct operations on it = two stepGroups."""
    act = actions["Adjust Touch Sensitivity"]
    assert act["category"] == "auto"
    assert len(act["stepGroups"]) == 2

    on, off = act["stepGroups"]
    assert on["actionableDeeplink"]["deeplink"] == TOUCH_SENSITIVITY_ON
    assert on["actionableDeeplink"]["originalType"] == "onURL"
    assert on["validationDeeplink"]["resultType"] == "boolean"
    assert on["validationDeeplink"]["value"] == "True"

    assert off["actionableDeeplink"]["deeplink"] == TOUCH_SENSITIVITY_OFF
    assert off["actionableDeeplink"]["originalType"] == "offURL"
    assert set(off["validationDeeplink"]) == {"deeplink", "key"}


# ----------------------------------------------------------------- exact match
def test_navigation_bar_matches_exactly(actions):
    group = actions["Configure Navigation Bar"]["stepGroups"][0]
    assert group["actionableDeeplink"]["deeplink"] == NAVIGATION_BAR
    assert group["actionableDeeplink"]["message"] == "View Navigation bar"
    assert group["validationDeeplink"]["key"] == "Navigation bar"


# ----------------------------------------------------------------- dummy / null
def test_factory_reset_falls_back_to_dummy_positive(actions):
    act = actions["Perform Factory Data Reset"]
    assert act["category"] == "critical"
    group = act["stepGroups"][0]
    assert group["actionableDeeplink"]["deeplink"] == DUMMY
    assert "Factory data reset" in group["actionableDeeplink"]["message"]
    assert group["validationDeeplink"] is None


def test_software_update_gets_null_not_dummy(actions):
    """No catalog entry AND no Settings screen opened by any step -> null."""
    group = actions["Check Software Update"]["stepGroups"][0]
    assert group["actionableDeeplink"] is None


@pytest.mark.parametrize("name", ["Remove Screen Protector", "Try A Different Charger",
                                  "Contact Samsung Support"])
def test_manual_actions_carry_no_deeplink(actions, name):
    act = actions[name]
    assert act["category"] == "manual"
    for group in act["stepGroups"]:
        assert group["actionableDeeplink"] is None
        assert group["validationDeeplink"] is None


@pytest.mark.parametrize("name", ["Restart Your Device", "Enter Safe Mode"])
def test_physical_critical_actions_carry_no_deeplink(actions, name):
    act = actions[name]
    assert act["category"] == "critical"
    assert act["stepGroups"][0]["actionableDeeplink"] is None


# ----------------------------------------------------------------- ordering
def test_five_tier_ordering(goal):
    assert [a["actionName"] for a in goal["actions"]] == [
        "Remove Screen Protector",      # tier 0 manual, non-invasive
        "Try A Different Charger",      # tier 0
        "Adjust Touch Sensitivity",     # tier 1 auto toggle
        "Configure Navigation Bar",     # tier 2 auto navigational
        "Contact Samsung Support",      # tier 3 service escalation
        "Restart Your Device",          # tier 4 critical, article order preserved
        "Check Software Update",
        "Enter Safe Mode",
        "Perform Factory Data Reset",   # "last resort" stays last
    ]


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
