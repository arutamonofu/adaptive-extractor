"""LLM provider infrastructure for AutoEvoExtractor.

This module provides LLM provider abstractions for interfacing with
various language models (Ollama, OpenAI, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState

if TYPE_CHECKING:
    from .provider import (
        BaseHTTPProvider,
        BaseLMProvider,
        OllamaLM,
        OpenRouterLM,
        TransformersLM,
        create_lm,
        setup_student,
        setup_teacher,
    )


def __getattr__(name: str):
    """Lazy loading — provider.py imports dspy, which is heavy."""
    if name == "BaseLMProvider":
        from .provider import BaseLMProvider

        return BaseLMProvider
    if name == "BaseHTTPProvider":
        from .provider import BaseHTTPProvider

        return BaseHTTPProvider
    if name == "TransformersLM":
        from .provider import TransformersLM

        return TransformersLM
    if name == "OllamaLM":
        from .provider import OllamaLM

        return OllamaLM
    if name == "OpenRouterLM":
        from .provider import OpenRouterLM

        return OpenRouterLM
    if name == "create_lm":
        from .provider import create_lm

        return create_lm
    if name == "setup_student":
        from .provider import setup_student

        return setup_student
    if name == "setup_teacher":
        from .provider import setup_teacher

        return setup_teacher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    "BaseLMProvider",
    "BaseHTTPProvider",
    "TransformersLM",
    "OllamaLM",
    "OpenRouterLM",
    "create_lm",
    "setup_student",
    "setup_teacher",
]
