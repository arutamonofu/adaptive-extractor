"""Infrastructure layer for AutoEvoExtractor.

The infrastructure layer handles external integrations including:
- LLM providers (Ollama, OpenAI, etc.)
- Document parsers (Docling, Marker)
- Storage repositories (agents, predictions, ground truth)
- Tracking systems (MLflow)
- Configuration management (settings, logging)
- Agent implementations (DSPy modules)
- Cache management (DSPy persistent cache)
"""

from . import agents, cache, config, llm, parsers, storage

__all__ = [
    "agents",
    "cache",
    "config",
    "llm",
    "parsers",
    "storage",
]
