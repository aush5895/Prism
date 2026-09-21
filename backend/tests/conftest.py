import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolate_semantic_cache():
    """The cache is process-global by design — sharing plans across requests is the whole
    point in production. In a test session that makes it shared mutable state: one test's
    stored plan can answer another test's query and silently change what is under test.
    Every test therefore starts cold.
    """
    from app.pipeline.cache import reset_cache
    reset_cache()
    yield
    reset_cache()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def catalog():
    from app.pipeline.deeplinks import get_catalog
    return get_catalog()


@pytest.fixture(scope="session")
def siis_rows():
    return json.loads((ROOT / "data" / "siis_responses.json").read_text())["responses"]


@pytest.fixture(scope="session")
def row21(siis_rows):
    return next(r for r in siis_rows if r["id"] == "row_21")
