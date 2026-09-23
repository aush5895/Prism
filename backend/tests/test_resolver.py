"""Resolver gate regressions (directives 5-11).

Cases A/B/C and the factory-reset trap are the exact failures the probe found before the
target-concept gate existed (contract §5.1). They exist so those failures cannot return.
"""
import pytest

from app import config
from app.ir import ExtractedAction, ExtractedStep, ExtractedStepGroup, Extraction
from app.pipeline import assemble, ordering
from app.pipeline.deeplinks import intent_for_candidate, load_lexicons, parse_intent
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


# ------------------------------------------------- gate [2] per-candidate intent
@pytest.mark.parametrize("step,candidate_message,intent", [
    # REGRESSION: the setting's own NAME contains a polarity phrase and used to outvote
    # the instruction. Both of these read OFF from the whole step before the fix, so
    # "enable" resolved to the Disable entry — the user asked to turn something on and
    # was sent to turn it off.
    ("Tap the switch next to Double tap to turn off screen to enable it.",
     "Enable Double tap to turn off screen", "ON"),
    ("Turn on Double tap to turn off screen.",
     "Enable Double tap to turn off screen", "ON"),
    # The mirror case. Taking the LAST polarity marker instead of removing the candidate's
    # name would fix the two above and break this one, which is why it is here.
    ("Turn off Double tap to turn on screen.",
     "Disable Double tap to turn on screen", "OFF"),
    # Ordinary wording must be unaffected.
    ("Tap the switch next to Touch sensitivity to disable it.",
     "Disable Touch sensitivity", "OFF"),
    ("Tap Navigation bar.", "View Navigation bar", "VIEW"),
    # The subject is ENTIRELY a polarity phrase; subtracting it must still leave the
    # instruction readable.
    ("Turn on Turn on now.", "Enable Turn on now", "ON"),
])
def test_intent_is_read_per_candidate_with_its_own_name_removed(step, candidate_message, intent):
    assert intent_for_candidate(step, candidate_message) == intent


def test_polarity_hijack_resolved_to_the_opposite_polarity_before_the_fix(catalog):
    """The end-to-end consequence, not just the parse: these must reach the ENABLE entry."""
    for step in ("Turn on Double tap to turn off screen.",
                 "Tap the switch next to Double tap to turn off screen to enable it."):
        res = catalog.resolve_step(step)
        assert res.is_exact, step
        assert res.deeplink["originalType"] == "onURL", step
        assert res.deeplink["message"].startswith("Enable"), step


def test_whole_step_parse_intent_survives_for_the_fallback_path(catalog):
    """parse_intent stays the module-level fallback, used when a candidate's subject is
    not present in the step to subtract."""
    assert parse_intent("Tap the switch next to Touch sensitivity to enable it.") == "ON"
    # a candidate whose name shares nothing with the step falls back to the whole-step read
    assert intent_for_candidate("Tap the switch next to Touch sensitivity to enable it.",
                                "View Navigation bar") == "ON"


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


# ------------------------------------------- compound names (tokenisation)
def test_a_hyphenated_name_reaches_its_closed_up_catalog_entry(catalog):
    """REGRESSION: tokens("Wi-Fi") was ["wi", "fi"] and tokens("WiFi") was ["wifi"], two
    sets that never intersect. This step ranked DL-0308 "View WiFi Settings" at BM25 0.77
    and then killed it at gate [4] on coverage 0.00.

    Gold is the equivalence CLASS, not one id. Six catalog entries share the message
    "View WiFi Settings" -- Wi-Fi scanning, Wi-Fi settings, automatic Wi-Fi turn on,
    Intelligent Wi-Fi, Hotspot 2.0, switch to mobile data on poor Wi-Fi -- separable only
    by `description`, which gate [3] deliberately does not read. Asserting DL-0308
    specifically would be asserting a coin flip between six.
    """
    step = "Navigate to Settings, tap Connections, and then tap Wi-Fi."
    res = catalog.resolve_step(step)
    assert res.is_exact
    assert res.deeplink["message"] == "View WiFi Settings"

    wifi_class = {e["id"] for e in catalog.entries if e["message"] == "View WiFi Settings"}
    assert "DL-0308" in wifi_class
    assert res.catalog_id in wifi_class


@pytest.mark.parametrize("written,expected", [
    ("Wi-Fi", ["wifi"]), ("WiFi", ["wifi"]), ("wi_fi", ["wifi"]),
    ("Auto-Sync", ["autosync"]), ("e-SIM", ["esim"]),
])
def test_compound_names_tokenise_identically_however_they_are_written(written, expected):
    from app.text import tokens
    assert tokens(written) == expected


def test_the_joiner_strip_applies_to_the_catalog_side_too(catalog):
    """Normalising only the step would move the mismatch rather than remove it."""
    from app.text import coverage
    assert coverage("View WiFi Settings", "tap Wi-Fi") == 1.0
    assert coverage("View Wi-Fi Settings", "tap WiFi") == 1.0


# ------------------------------------------- gate [0] hardware vs touchscreen
def test_a_long_press_on_an_onscreen_element_is_not_physical(catalog):
    """REGRESSION: "touch and hold" was itself a physical trigger, so a touchscreen
    gesture on a named on-screen element read as a hardware interaction. Gate [0] tests
    the WHOLE action, so this one step nulled every step group in its action and the
    action lost a valid Wi-Fi deeplink.

    Measured over the 20 supplied articles: "touch and hold" occurs 5 times, 2 on hardware
    and 3 on ordinary screen elements.
    """
    action = _action("Verify internet connection",
                     ["Touch and hold the Wi-Fi icon to check your connection status.",
                      "Navigate to Settings, tap Connections, and then tap Wi-Fi."],
                     hint="auto")
    assert ordering.is_physical(action) is False

    resolved = assemble.resolve_actions(Extraction(goal_topic="T", title="t",
                                                   actions=[action]), catalog)
    res = resolved[0].resolutions[0]
    assert res.is_exact, "a touchscreen gesture must not cost the action its deeplink"
    assert res.deeplink["message"] == "View WiFi Settings"


@pytest.mark.parametrize("steps", [
    ["Press and hold the Power button, then tap Restart."],
    ["Touch and hold Power off, then tap Safe mode."],
    ["Press and hold the Power button and the Volume down button at the same time."],
])
def test_a_long_press_on_a_hardware_control_is_still_physical(catalog, steps):
    """The other direction. The control is named in the same action, so it fires on the
    control rather than on the gesture, and the deeplink stays null."""
    action = _action("Restart the device", steps, hint="critical")
    assert ordering.is_physical(action) is True

    resolved = assemble.resolve_actions(Extraction(goal_topic="T", title="t",
                                                   actions=[action]), catalog)
    assert all(r.deeplink is None for r in resolved[0].resolutions)


def test_bare_hold_gestures_are_data_but_never_a_trigger():
    """Kept in the lexicon so the distinction is visible, and deliberately not consulted
    by is_physical."""
    lex = load_lexicons()
    assert "hold_gestures" in lex
    assert "touch and hold" in lex["hold_gestures"]
    assert "touch and hold" not in lex["physical_interaction"]
    assert "press and hold" not in lex["physical_interaction"]


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
