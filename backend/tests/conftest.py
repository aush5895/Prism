import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

FIXTURES = Path(__file__).parent / "fixtures"


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
