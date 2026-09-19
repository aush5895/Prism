"""The pipeline must be scenario-agnostic (directive 2/17).

Two independent guarantees:
  1. A static check that no fixture text was ever copied into backend/app.
  2. A behavioural check that the offline provider — which has never seen these articles —
     drives the same pipeline to schema-valid output on EVERY supplied row.
"""
import ast
import json
from pathlib import Path

import pytest

from app.contracts import TroubleshootRequest
from app.llm.stub import StubProvider
from app.main import run_pipeline
from app.schema_samsung import ContextDeeplinkResponse

MIN_LITERAL = 15  # shorter literals are generic English, not copied scenario content
MIN_CONTENT_WORDS = 3  # a literal with fewer subject words cannot encode a scenario answer

# The one catalog URI the specification itself names, so code must reference it by value
# (Theme 2 guide §3; DL-DUMMY._readme). Every other masked URI is banned outright by
# test_no_catalog_uri_appears_in_backend_app below.
SPEC_NAMED_CONSTANTS = {"bixby://dummy_positive"}


def _app_string_literals(app_dir: Path):
    for path in sorted(app_dir.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.strip()
                if len(value) >= MIN_LITERAL:
                    yield path, value


def test_no_fixture_text_is_hardcoded_in_backend_app(repo_root):
    """CI gate. If this fails, scenario content leaked out of data/ and into code."""
    rows = json.loads((repo_root / "data" / "siis_responses.json").read_text())["responses"]
    catalog = json.loads((repo_root / "data" / "deeplinks.json").read_text())["deeplinks"]

    corpus = "\n".join(
        [(repo_root / "data" / "input.txt").read_text()]
        + [r["original_query"] for r in rows]
        + [r["siis_response"]["title"] for r in rows]
        + [r["siis_response"]["content"] for r in rows]
        + [e["message"] for e in catalog]
        + [e["description"] for e in catalog]
        + [e["deeplink"] for e in catalog]
    ).lower()

    from app.text import content as content_tokens

    offenders = []
    for path, value in _app_string_literals(repo_root / "backend" / "app"):
        if value in SPEC_NAMED_CONSTANTS:
            continue
        # A literal can only encode a scenario answer if it carries subject words.
        # "settings screen" and "in device Settings" are generic UI English; the same
        # content-token policy the resolver uses decides that here too.
        if len(content_tokens(value)) < MIN_CONTENT_WORDS:
            continue
        if value.lower() in corpus:
            offenders.append(f"{path.name}: {value[:70]!r}")
    assert not offenders, "fixture content found in backend/app:\n  " + "\n  ".join(offenders)


def test_no_catalog_uri_appears_in_backend_app(repo_root):
    """A hardcoded URI would be the most direct form of benchmark gaming."""
    for path, value in _app_string_literals(repo_root / "backend" / "app"):
        assert "bixby://masked/" not in value, f"{path.name} hardcodes a masked URI"


@pytest.mark.parametrize("row_index", range(20))
def test_offline_stub_drives_every_supplied_row_to_valid_output(siis_rows, row_index):
    """The stub is a generic structural parser with no knowledge of these articles. If
    the deterministic half of the pipeline were scenario-coupled, most of these would
    fail or fall back."""
    row = siis_rows[row_index]
    req = TroubleshootRequest(
        query=row["original_query"].split(". ", 1)[-1],
        siis_response=row["siis_response"],
    )
    env = run_pipeline(req, provider=StubProvider())

    # Always schema-valid, whether or not a plan was produced.
    parsed = ContextDeeplinkResponse(**env.response)
    assert env.meta.fallback in (None, "no_match")
    if env.meta.fallback is None:
        assert parsed.contexts and parsed.contexts[0].actions
        categories = [a.category.value for a in parsed.contexts[0].actions]
        if "critical" in categories:
            first = categories.index("critical")
            assert all(c == "critical" for c in categories[first:])
    assert 8 <= len(env.query_variations) <= 10


def test_every_emitted_deeplink_across_all_rows_is_catalog_backed(siis_rows, catalog):
    """Catalog integrity (guide §4.2.2) asserted over the whole supplied set, not one row."""
    emitted = 0
    for row in siis_rows:
        req = TroubleshootRequest(query=row["original_query"].split(". ", 1)[-1],
                                  siis_response=row["siis_response"])
        env = run_pipeline(req, provider=StubProvider())
        for goal in env.response.get("contexts", []):
            for action in goal["actions"]:
                for group in action["stepGroups"]:
                    dl = group["actionableDeeplink"]
                    if dl:
                        catalog.verify_identity(dl)
                        emitted += 1
                    if action["category"] == "manual":
                        assert dl is None
    assert emitted > 0, "no deeplinks resolved across 20 rows — resolver is inert"
