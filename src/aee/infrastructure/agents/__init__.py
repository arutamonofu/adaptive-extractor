# src/aee/infrastructure/agents/__init__.py

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aee.infrastructure.agents.extractor import UniversalExtractor


def __getattr__(name: str):
    """Lazy loading — extractor imports dspy."""
    if name == "UniversalExtractor":
        from aee.infrastructure.agents.extractor import UniversalExtractor

        return UniversalExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["UniversalExtractor"]
