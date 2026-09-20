"""Resolver gate regressions (directives 5-11).

Cases A/B/C and the factory-reset trap are the exact failures the probe found before the
target-concept gate existed (contract §5.1). They exist so those failures cannot return.
"""
import pytest

from app import config
from app.ir import ExtractedAction, ExtractedStep, ExtractedStepGroup, Extraction
from app.pipeline import assemble, ordering
from app.pipeline.deeplinks import parse_intent
from app.text import coverage


def _action(name: str, steps: list[str], hint: str | None = None) -> ExtractedAction:
    """Build a one-group action from step text. Synthetic, domain-general: no fixture
    string and no catalog message (directive 17)."""
    return ExtractedAction(
        action_name=name,
        description=f"It will address {name.lower()} on your device",
        category_hint=hint,
        step_groups=[ExtractedStepGroup(steps=[ExtractedStep(text=s) for s in steps])],
    )


# ------------------------------------------------------------ gate [2] polarity
@pytest.mark.parametrize("step,intent", [
    ("Tap the switch next to Touch sensitivity to enable it.", "ON"),
    ("Tap the switch next to Touch sensitivity to disable it.", "OFF"),
    ("Tap Navigation bar.", "VIEW"),
    ("Navigate to and open Settings.", "VIEW"),
    ("Set the brightness to a lower level.", "UPDATE"),
])
def test_intent_parsing(step, intent):
    assert parse_intent(step) == intent


def test_polarity_gate_separates_enable_from_disable(catalog):
    """The pair is lexically inseparable (BM25 1.000 vs 0.934) and must be separated by
    originalType alone, in BOTH directions."""
    on = catalog.resolve_step("Tap the switch next to Touch sensitivity to enable it.")
    off = catalog.resolve_step("Tap the switch next to Touch sensitivity to disable it.")

    assert on.is_exact and on.catalog_id == "DL-0126"
    assert on.deeplink["originalType"] == "onURL"
    assert off.is_exact and off.catalog_id == "DL-0125"
    assert off.deeplink["originalType"] == "offURL"
    assert on.deeplink["deeplink"] != off.deeplink["deeplink"]

    # each rejects the other on polarity, not on score
    assert any(t.catalog_id == "DL-0125" and t.verdict == "reject:polarity" for t in on.trace)
    assert any(t.catalog_id == "DL-0126" and t.verdict == "reject:polarity" for t in off.trace)


def test_onurl_carries_full_validation_and_offurl_key_only(catalog):
    """Catalog asymmetry that sample_output.json demonstrates: onURL entries carry
    resultType/condition/value, offURL entries carry key only. Copied, never authored."""
    on = catalog.resolve_step("Tap the switch next to Touch sensitivity to enable it.")
    off = catalog.resolve_step("Tap the switch next to Touch sensitivity to disable it.")
    assert on.validation["resultType"] == "boolean"
    assert on.validation["condition"] == "equal" and on.validation["value"] == "True"
    assert set(off.validation) == {"deeplink", "key"}
    assert on.validation["key"] == off.validation["key"] == "Touch sensitivity"


# ------------------------------------------------------------ gate [3] scope
def test_factory_reset_trap_is_never_emitted(catalog):
    """REGRESSION: DL-0022 'auto factory reset' ranks #1 on BM25 for this step. It is a
    scheduling preference, not the reset action, and must never be emitted.

    Asserts the CONTRACT (it is rejected), not the MECHANISM (which gate rejects it).
    It has moved gates once already: it was a scope rejection while gate [3] read the
    candidate's `description`, and became a concept rejection at coverage 0.50 when
    config.SCOPE_FIELDS narrowed gate [3] to `message`. Both are correct outcomes; pinning
    the gate made a real improvement look like a regression.
    """
    res = catalog.resolve_step("Tap Factory data reset.")
    assert not res.is_exact
    assert res.catalog_id != "DL-0022"
    verdicts = {t.catalog_id: t.verdict for t in res.trace}
    assert verdicts["DL-0022"].startswith("reject:")


def test_scope_gate_rejects_a_qualifier_the_step_does_not_ask_for(catalog):
    """'Sync' must not reach 'Enable Auto-Sync': automatic syncing is a different feature
    from syncing, separable only by the qualifier the step never uses."""
    res = catalog.resolve_step("Tap the switch next to Sync to enable it.")
    assert not res.is_exact
    rejected = [t for t in res.trace if t.verdict == "reject:scope(auto)"]
    assert any(t.message == "Enable Auto-Sync" for t in rejected)


def test_scope_qualifier_admitted_when_the_step_shares_it(catalog):
    """The gate rejects unmatched qualifiers, not the qualifier itself — a step that does
    ask for the automatic variant must still reach it."""
    res = catalog.resolve_step("Tap the switch next to Auto Restart to enable it.")
    assert res.is_exact and res.catalog_id == "DL-0480"
    assert res.deeplink["message"] == "Enable Auto Restart"


def test_scope_gate_reads_only_the_configured_fields(catalog):
    """42 of the 578 entries carry a qualifier in `description` that never appears in
    `message` (e.g. DL-0330 'View Speak usage hints' is described as a TalkBack screen).
    Gate [3] must not reject those on a word the candidate's own message never claims."""
    assert config.SCOPE_FIELDS == ("message",)
    entry = catalog.by_id["DL-0330"]
    assert "talkback" in entry["description"].lower()
    assert "talkback" not in entry["message"].lower()
    assert "talkback" not in catalog._scope_blob(entry).lower()


# ------------------------------------------------------------ gate [4] target concept
def test_regression_A_software_update_must_not_resolve_to_care_plus(catalog):
    """REGRESSION A: before the concept gate this ACCEPTED DL-0576
    'View Check Samsung Care+ subscription' at rank 1. The catalog has no software-update
    entry; the shared token was the generic verb 'Check'."""
    step = "Check for software updates on your device."
    res = catalog.resolve_step(step)
    assert not res.is_exact
    assert res.catalog_id != "DL-0576"
    assert coverage("View Check Samsung Care+ subscription", step) == 0.0
    assert any(t.catalog_id == "DL-0576" and t.verdict == "reject:concept" for t in res.trace)


def test_regression_B_navigation_bar_resolves_exactly(catalog):
    """REGRESSION B: before the concept gate this ABSTAINED to dummy_positive because a
    distractor sharing the token 'bar' collapsed the margin."""
    res = catalog.resolve_step("Tap Navigation bar.")
    assert res.is_exact and res.catalog_id == "DL-0169"
    assert res.deeplink["message"] == "View Navigation bar"
    assert res.deeplink["originalType"] == "onClickURL"


def test_regression_C_space_bar_distractor_is_rejected(catalog):
    """REGRESSION C: DL-0379 scored 0.975 on 'Tap Navigation bar.' purely via 'bar'."""
    res = catalog.resolve_step("Tap Navigation bar.")
    assert any(t.catalog_id == "DL-0379" and t.verdict == "reject:concept" for t in res.trace)
    assert coverage("View Double tap space bar to add period", "Tap Navigation bar.") < \
        config.CONCEPT_COVERAGE_MIN


def test_coverage_direction_is_candidate_to_step():
    """The gate measures how much of the CANDIDATE the step covers, not the reverse.
    A long step must not be able to drag in a narrow unrelated candidate."""
    assert coverage("View Navigation bar", "Tap Navigation bar.") == 1.0
    assert coverage("View Check Samsung Care+ subscription",
                    "Check for software updates on your device.") == 0.0


# ------------------------------------------------------------ gate [5] margin
def test_single_survivor_skips_the_margin_comparison(catalog):
    """Directive 7: one survivor needs no top1-top2 gap."""
    res = catalog.resolve_step("Tap Navigation bar.")
    eligible = [t for t in res.trace if t.verdict == "eligible"]
    assert len(eligible) == 1
    assert res.is_exact


# ------------------------------------------------------------ gate [6] identity
def test_emitted_deeplinks_are_verbatim_catalog_copies(catalog):
    for step in ("Tap Navigation bar.",
                 "Tap the switch next to Touch sensitivity to enable it."):
        res = catalog.resolve_step(step)
        entry = catalog.by_uri[res.deeplink["deeplink"]]
        for field in ("deeplink", "description", "message", "originalType"):
            assert res.deeplink[field] == entry[field]
        assert res.validation == entry["validation"]


def test_identity_check_rejects_a_fabricated_uri(catalog):
    with pytest.raises(ValueError):
        catalog.verify_identity({"deeplink": "bixby://masked/act/deadbeef99"})


def test_identity_check_rejects_field_drift(catalog):
    good = catalog._emit(catalog.by_id["DL-0169"])
    catalog.verify_identity(good)
    with pytest.raises(ValueError):
        catalog.verify_identity({**good, "message": "View Navigation settings"})


def test_uri_is_never_part_of_the_match(catalog):
    """Guide §7.4 — matching on the masked token is forbidden. Searching for a URI must
    not retrieve its own entry."""
    res = catalog.resolve_step("Open bixby://masked/act/2f3dd95259")
    assert res.catalog_id != "DL-0169"


# ------------------------------------------------------------ gate [0] short-circuit
def test_manual_category_never_receives_a_deeplink(catalog):
    res = catalog.resolve_step_group(["Tap Navigation bar."], category="manual", physical=False)
    assert res.decision == "none" and res.deeplink is None
    assert res.reason == "manual_category_forbids_deeplink"
    assert res.trace == []  # retrieval never ran


def test_physical_interaction_never_receives_a_deeplink(catalog):
    res = catalog.resolve_step_group(["Tap Navigation bar."], category="critical", physical=True)
    assert res.decision == "none" and res.deeplink is None


def test_device_operation_gets_null_however_the_step_is_worded(catalog):
    """REGRESSION: gate [0]'s physical check keys on step WORDING. An extraction that
    writes a reboot as 'Tap Restart.' carries no hardware-button phrase, so the resolver
    used to reach the catalog and hand a device REBOOT a Settings deeplink. Observed
    output before the fix: dummy_positive 'Open Restart again under Restart'.

    A restart is a state the device enters, not a screen Settings can open.
    """
    action = _action("Restart the device", ["Tap Restart.", "Tap Restart again to confirm."],
                     hint="critical")
    assert ordering.is_physical(action) is False      # the wording alone catches nothing
    assert ordering.is_device_operation(action) is True

    resolved = assemble.resolve_actions(Extraction(goal_topic="T", title="t", actions=[action]),
                                        catalog)
    for res in resolved[0].resolutions:
        assert res.decision == "none"
        assert res.deeplink is None


def test_factory_reset_is_not_a_device_operation_and_keeps_dummy_positive(catalog):
    """The counterpart: factory reset genuinely lives at Settings > General management >
    Reset, so it must stay OUT of the device-operation lexicon and keep its
    dummy_positive. This is the line the lexicon has to draw."""
    action = _action("Perform factory reset",
                     ["Navigate to and open Settings.", "Tap General management.",
                      "Tap Reset.", "Tap Factory data reset."], hint="critical")
    assert ordering.is_device_operation(action) is False

    resolved = assemble.resolve_actions(Extraction(goal_topic="T", title="t", actions=[action]),
                                        catalog)
    res = resolved[0].resolutions[0]
    assert res.decision == "dummy_positive"
    assert res.deeplink["deeplink"] == config.DUMMY_DEEPLINK
    assert "Factory data reset" in res.deeplink["message"]


# ------------------------------------------------------------ dummy_positive
def test_dummy_positive_names_the_concrete_screen(catalog):
    steps = ["Navigate to and open Settings.", "Tap General management.", "Tap Reset.",
             "Tap Factory data reset.", "Swipe to and tap Reset.", "Tap Delete all."]
    res = catalog.resolve_step_group(steps, category="critical", physical=False)
    assert res.decision == "dummy_positive"
    assert res.deeplink["deeplink"] == config.DUMMY_DEEPLINK
    assert "Factory data reset" in res.deeplink["description"]
    assert "Factory data reset" in res.deeplink["message"]
    for prose in (res.deeplink["description"], res.deeplink["message"]):
        assert 5 <= len(prose.split()) <= 7  # DL-DUMMY's own rule
    assert res.validation is None


def test_no_settings_screen_yields_null_not_dummy(catalog):
    """dummy_positive is for a Settings screen with no catalog entry — not for anything
    that failed to resolve."""
    res = catalog.resolve_step_group(
        ["Check that your device software is up to date."], category="critical", physical=False)
    assert res.decision == "none" and res.deeplink is None
