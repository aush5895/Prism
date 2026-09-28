"""The catalog audit (pipeline/catalog_audit.py) and its one use in gate [4].

The gold set is built FROM catalog labels, so it cannot see a resolver that confidently
follows a wrong label. These tests are where that failure is visible.
"""
from __future__ import annotations

import json

import pytest

from app.pipeline.catalog_audit import audit, label_supported, unsupported_ids
from app.pipeline.deeplinks import get_catalog


@pytest.fixture(scope="module")
def entries(repo_root):
    return json.loads((repo_root / "data" / "deeplinks.json").read_text(encoding="utf-8"))["deeplinks"]


@pytest.fixture(scope="module")
def by_id(entries):
    return {e["id"]: e for e in entries}


@pytest.mark.parametrize("entry_id, what_it_really_is", [
    ("DL-0163", "Enable Grayscale -> enables mono audio"),
    ("DL-0312", "Enable WiFi -> enables Mobile Hotspot"),
    ("DL-0117", "View Timeout Settings -> guest mode"),
    ("DL-0295", "label is literally 'Onurl'"),
])
def test_entries_whose_label_names_a_different_screen_are_flagged(by_id, entry_id,
                                                                 what_it_really_is):
    assert not label_supported(by_id[entry_id]), what_it_really_is


@pytest.mark.parametrize("entry_id, why_it_is_fine", [
    ("DL-0070", "View Sound Settings -> mute all sounds: generic but honest (plural)"),
    ("DL-0319", "View Alarm Settings -> alarms in Do Not Disturb (plural)"),
    ("DL-0310", "Enable WiFi -> Wi-Fi scanning (joiner: Wi-Fi == WiFi)"),
    ("DL-0026", "Enable Auto-Sync -> auto sync (two words == one label word)"),
])
def test_honest_labels_are_not_flagged(by_id, entry_id, why_it_is_fine):
    """The check is deliberately lenient: an entry is only distrusted when its label and
    its own description share nothing at all."""
    assert label_supported(by_id[entry_id]), why_it_is_fine


def test_the_placeholder_is_never_audited(entries):
    assert "DL-DUMMY" not in unsupported_ids(entries)


def test_grayscale_no_longer_resolves_to_mono_audio():
    """REGRESSION. 'Tap the switch next to Grayscale to enable it.' resolved exactly to
    DL-0163, whose own description is 'Enables mono audio'. The gold set scored it
    correct, because it matched the label."""
    catalog = get_catalog()
    result = catalog.resolve_step("Tap the switch next to Grayscale to enable it.")
    assert result.catalog_id != "DL-0163"
    assert any(t.catalog_id == "DL-0163" and t.verdict == "reject:label" for t in result.trace)


def test_no_resolution_in_the_gold_set_lands_on_an_unsupported_label():
    from evaluation import synthetic
    catalog = get_catalog()
    for case in synthetic.build_cases():
        result = catalog.resolve_step(case.step)
        if result.is_exact:
            assert result.catalog_id not in catalog.unsupported_labels, case.step


def test_the_audit_reports_the_structural_defects_it_measures(entries):
    report = audit(entries)
    assert report["n_entries"] == 577
    assert ["DL-0035", "DL-0222"] in report["identical_text_groups"]     # Dwell action
    assert report["off_without_on"] == ["charging feedback"]
    assert report["shared_label_entries"] == 193
    assert {x["id"] for x in report["label_unsupported"]} == unsupported_ids(entries)


def test_the_health_report_renders_only_measured_figures(entries):
    """docs/catalog_health.md is generated. Every count in it must come from the audit."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from tools.catalog_health import render

    report = audit(entries)
    unresolved = {"provider": "test:fixture", "decisions": [
        {"decision": "dummy_positive", "reason": "settings_screen_absent_from_catalog", "n": 2}],
        "gaps": [{"row": "row_x", "article": "A", "action": "Factory Data Reset",
                  "decision": "dummy_positive",
                  "reason": "settings_screen_absent_from_catalog", "nearest": []}]}
    md = render(report, unresolved, "2026-01-01T00:00:00Z")
    assert f"| {len(report['label_unsupported'])} |" in md
    assert f"{report['shared_label_groups']} labels, {report['shared_label_entries']} entries" in md
    assert "`test:fixture`" in md
    for x in report["label_unsupported"]:
        assert f"`{x['id']}`" in md
