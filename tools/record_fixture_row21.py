"""Generate backend/tests/fixtures/extraction_row21.json.

This is a RECORDED extraction: the structure a compliant provider returns for the
`Touchscreen issues` reference article. It lives in tools/ + tests/fixtures/, never in
backend/app (directive 17). Source spans are located in the real article text at build
time rather than typed by hand, so every span is genuinely verifiable.

Re-record from the live provider once a Gemini key is configured:
    LLM_PROVIDER=gemini python -m tools.record_fixture_row21 --from-provider
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

ROW_ID = "row_21"
OUT = ROOT / "backend" / "tests" / "fixtures" / "extraction_row21.json"

# (step text, anchor phrase used to locate the supporting span in the article)
PLAN = [
    ("Remove Screen Protector", "It will remove interference from the screen surface", "manual", [[
        ("Remove any third-party screen protector that is peeling or has debris under it.",
         "If your screen protector is peeling or has debris under it, please remove it."),
        ("Wipe the front and back gently with a lint-free microfiber cloth.",
         "gently wipe the front and back with a lint-free, soft microfiber cloth"),
        ("Avoid applying too much pressure while cleaning.",
         "Please avoid applying too much pressure when cleaning."),
    ]]),
    ("Adjust Touch Sensitivity", "It will match screen response to your protector", "auto", [
        [("Navigate to and open Settings.", "go to Settings"),
         ("Tap Display.", "tap Display"),
         ("Tap the switch next to Touch sensitivity to enable it.",
          'you can try enabling the "Touch sensitivity" option')],
        [("Navigate to and open Settings.", "navigate to Settings"),
         ("Tap Display.", "tap Display, and then tap the switch next to Touch sensitivity"),
         ("Tap the switch next to Touch sensitivity to disable it.",
          "tap the switch next to Touch sensitivity to disable it")],
    ]),
    ("Restart Your Device", "It will clear temporary glitches affecting touch input", "critical", [[
        ("Press and hold the Power button.", "Press and hold the Power button, then tap Restart."),
        ("Tap Restart.", "then tap Restart"),
        ("Tap Restart again to confirm.", "Tap Restart again to confirm."),
    ]]),
    ("Try A Different Charger", "It will rule out insufficient charging power", "manual", [[
        ("Connect the device to a different, undamaged charger.",
         "please try using a different, undamaged charger"),
        ("Avoid using the touchscreen while the device is charging.",
         "avoid using the touchscreen while your device is charging"),
    ]]),
    ("Configure Navigation Bar", "It will stop gestures from misreading your taps", "auto", [[
        ("Navigate to and open Settings.", "Go to Settings, tap Display, and then tap Navigation bar."),
        ("Tap Display.", "tap Display, and then tap Navigation bar"),
        ("Tap Navigation bar.", "tap Navigation bar"),
        ("Select Buttons to turn off full screen gestures.",
         "Select Buttons to turn off full screen gestures."),
    ]]),
    ("Check Software Update", "It will keep device software fully current", "critical", [[
        ("Check that your device software is up to date.",
         "Keeping your device's software up to date is important for smooth performance."),
        ("Check for updates to any third-party apps you use.",
         "You should also check for updates for any third-party apps you use"),
    ]]),
    ("Enter Safe Mode", "It will reveal whether an app causes this", "critical", [[
        ("Press and hold the Power button and the Volume down button at the same time.",
         "Press and hold the Power button (or Side button) and the Volume down button at the same time."),
        ("Touch and hold Power off.", "Touch and hold Power off"),
        ("Tap Safe mode.", "then tap Safe mode"),
        ("Remove any third-party app installed around the time the issue began.",
         "you can try removing any third-party apps that were recently updated or installed"),
        ("Restart the device to exit Safe mode.",
         "To exit Safe mode, simply restart your phone or tablet"),
    ]]),
    ("Perform Factory Data Reset", "It will erase all data and restore defaults", "critical", [[
        ("Back up your personal data before you continue.",
         "it is crucial to back up your personal data to avoid losing it"),
        ("Navigate to and open Settings.", "Navigate to and open Settings."),
        ("Tap General management.", "Tap General management."),
        ("Tap Reset.", "Tap Reset."),
        ("Tap Factory data reset.", "Tap Factory data reset."),
        ("Swipe to and tap Reset.", "Swipe to and tap Reset."),
        ("Tap Delete all.", "Tap Delete all."),
    ]]),
    ("Contact Samsung Support", "It will connect you with Samsung support staff", "manual", [[
        ("Contact Samsung Support if the issue persists after these steps.",
         "please reach out to Samsung Support for further assistance"),
    ]]),
]

VARIATIONS = [
    "Touch response on my Galaxy S22 is delayed and unresponsive.",
    "my s22 touchscreen is super laggy when i tap stuff",
    "S22 touch lag delayed input screen",
    "Why does my Galaxy S22 take so long to register a tap?",
    "Screen input lag on Galaxy S22 after tapping.",
    "this phone touch is so slow its driving me insane",
    "Galaxy S22 touchscreen responds late to every touch.",
    "my galxy s22 tuch screen is laggy and slow to respnd",
    "Delayed touch registration on Samsung Galaxy S22 display.",
    "S22 screen takes a second to react when I press anything.",
]


def main() -> None:
    from app.pipeline.ground import normalize_siis  # noqa: E402

    rows = json.loads((ROOT / "data" / "siis_responses.json").read_text())["responses"]
    row = next(r for r in rows if r["id"] == ROW_ID)
    evidence = normalize_siis(row["siis_response"]).text

    def span(anchor: str):
        i = evidence.find(anchor)
        if i < 0:
            raise SystemExit(f"anchor not found in article: {anchor!r}")
        return [i, i + len(anchor)]

    actions = []
    for name, desc, hint, groups in PLAN:
        built_groups, lo, hi = [], len(evidence), 0
        for group in groups:
            steps = []
            for text, anchor in group:
                s = span(anchor)
                lo, hi = min(lo, s[0]), max(hi, s[1])
                steps.append({"text": text, "source_span": s})
            built_groups.append({"steps": steps})
        actions.append({"action_name": name, "description": desc, "category_hint": hint,
                        "source_span": [lo, hi], "step_groups": built_groups})

    payload = {
        "_recorded_from": "hand-recorded reference extraction (D1); re-record with Gemini",
        "_row_id": ROW_ID,
        "goal_topic": "Touchscreen",
        "title": "Touchscreen response issues",
        "actions": actions,
        "query_variations": VARIATIONS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    n_steps = sum(len(g["steps"]) for a in actions for g in a["step_groups"])
    print(f"wrote {OUT.relative_to(ROOT)}: {len(actions)} actions, {n_steps} steps, all spans located")


if __name__ == "__main__":
    main()
