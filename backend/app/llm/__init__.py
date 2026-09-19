"""Provider factory. Adding Grok later means adding one module and one branch here —
no pipeline change (directive: keep the provider abstract)."""
from __future__ import annotations

from .. import config
from .base import LLMProvider
from .replay import ReplayProvider
from .stub import StubProvider

__all__ = ["LLMProvider", "StubProvider", "ReplayProvider", "get_provider"]


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    name = (name or config.LLM_PROVIDER).lower()
    if name == "stub":
        return StubProvider()
    if name == "replay":
        return ReplayProvider(**kwargs)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(**kwargs)
    raise ValueError(f"unknown LLM provider: {name!r}")
