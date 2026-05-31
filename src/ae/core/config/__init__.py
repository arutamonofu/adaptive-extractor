# src/ae/core/config/__init__.py
"""Configuration module for Adaptive Extractor."""

from __future__ import annotations

from ae.core.config.logging import setup_logging
from ae.core.config.settings import (
    AEVisualParserConfig,
    ApiConfig,
    CircuitBreakerConfig,
    GeminiParserConfig,
    IngestionConfig,
    LLMInstanceConfig,
    OllamaConfig,
    Settings,
)

__all__ = [
    "Settings",
    "setup_logging",
    "LLMInstanceConfig",
    "OllamaConfig",
    "ApiConfig",
    "CircuitBreakerConfig",
    "GeminiParserConfig",
    "AEVisualParserConfig",
    "IngestionConfig",
]


def __getattr__(name: str):
    """Lazy loading for heavy config classes."""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
