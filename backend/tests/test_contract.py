"""Samsung's schema.py is immutable (directive 1/3). These tests run first in CI."""
import hashlib

from app import config


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_samsung_schema_checksum_unchanged(repo_root):
    """If this fails, someone edited Samsung's contract to make our output fit.
    Fix the output, not the schema."""
    assert _sha256(repo_root / "data" / "schema.py") == config.SCHEMA_SHA256


def test_vendored_schema_is_byte_identical(repo_root):
    """backend/app/schema_samsung.py must stay a verbatim copy of the supplied file."""
    assert (_sha256(repo_root / "backend" / "app" / "schema_samsung.py")
            == _sha256(repo_root / "data" / "schema.py"))


def test_schema_exposes_the_expected_contract():
    from app.schema_samsung import (Action, ContextDeeplinkResponse, Deeplink, Goal,
                                    StepGroup, ValidationDeepLink, actionCategory)

    assert set(Goal.model_fields) == {"goal", "title", "actions", "score"}
    assert set(Action.model_fields) == {"actionName", "description", "stepGroups", "category"}
    assert set(StepGroup.model_fields) == {"steps", "validationDeeplink", "actionableDeeplink"}
    assert set(Deeplink.model_fields) == {"deeplink", "description", "message", "classes", "originalType"}
    assert set(ValidationDeepLink.model_fields) == {"deeplink", "key", "resultType", "condition", "value"}
    assert {c.value for c in actionCategory} == {"auto", "manual", "critical"}
    # category defaults to the safe direction: a mis-classified action loses its deeplink
    assert Action.model_fields["category"].default == actionCategory.manual
    assert ContextDeeplinkResponse().contexts == []


def test_catalog_loads_and_is_uri_unique(catalog):
    assert len(catalog) == 578
    assert len(catalog.by_uri) == 578
    assert config.DUMMY_DEEPLINK in catalog.by_uri
