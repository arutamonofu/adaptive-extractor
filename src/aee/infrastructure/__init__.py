"""Infrastructure layer for AutoEvoExtractor.

The infrastructure layer handles external integrations including:
- LLM providers (Ollama, OpenAI, etc.)
- Document parsers (Marker)
- Storage repositories (agents, predictions, ground truth)
- Tracking systems (MLflow)
- Configuration management (settings, logging)
- Agent implementations (DSPy modules)
- Cache management (DSPy persistent cache)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import agents, cache, config, llm, parsers, storage


def __getattr__(name: str):
    """Lazy loading of subpackages to avoid importing heavy dependencies."""
    if name in ("agents", "cache", "config", "llm", "parsers", "storage"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    "agents",
    "cache",
    "config",
    "llm",
    "parsers",
    "storage",
]
