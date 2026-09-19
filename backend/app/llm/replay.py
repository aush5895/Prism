"""Replay provider — loads a previously RECORDED extraction from disk.

Purpose: pin the deterministic half of the pipeline (resolver, gates, ordering, scoring,
schema assembly) under test without an API key and without letting model drift turn an
integration test red for the wrong reason.

The recorded file lives under backend/tests/fixtures/ and is passed in by path. No
fixture content ever enters backend/app (directive 17). Recordings are regenerated from
the live provider with `make record-fixtures` and are expected to be re-recorded when the
prompt changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..ir import Extraction
from .base import LLMProvider


class ReplayProvider(LLMProvider):
    name = "replay"

    def __init__(self, recording_path: str | Path):
        self.path = Path(recording_path)
        if not self.path.exists():
            raise FileNotFoundError(f"no recording at {self.path}")
        self._payload = json.loads(self.path.read_text())
        self.model = self._payload.get("_recorded_from", "recorded")

    def extract(self, query: str, evidence: str) -> Extraction:  # noqa: ARG002
        return Extraction(**{k: v for k, v in self._payload.items() if not k.startswith("_")})
