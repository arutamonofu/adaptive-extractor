# src/aee/infrastructure/config/__init__.py
"""Configuration module for AutoEvoExtractor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aee.infrastructure.config.environments import (
    load_dev_settings,
    load_prod_settings,
    load_settings_for_environment,
    load_test_settings,
)
from aee.infrastructure.config.logging import setup_logging
from aee.infrastructure.config.settings import (
    ApiConfig,
    CircuitBreakerConfig,
    GeminiParserConfig,
    IngestionConfig,
    LLMInstanceConfig,
    MarkerConfig,
    OllamaConfig,
    Settings,
)

if TYPE_CHECKING:
    from aee.infrastructure.config.settings import TransformersConfig

__all__ = [
    "Settings",
    "setup_logging",
    "load_settings_for_environment",
    "load_dev_settings",
    "load_test_settings",
    "load_prod_settings",
    "LLMInstanceConfig",
    "OllamaConfig",
    "ApiConfig",
    "CircuitBreakerConfig",
    "MarkerConfig",
    "GeminiParserConfig",
    "IngestionConfig",
    "TransformersConfig",
]


def __getattr__(name: str):
    """Lazy loading for heavy config classes."""
    if name == "TransformersConfig":
        from aee.infrastructure.config.settings import TransformersConfig

        return TransformersConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
