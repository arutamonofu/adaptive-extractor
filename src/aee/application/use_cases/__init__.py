"""Use cases for AutoEvoExtractor.

This module contains the application use cases that orchestrate
services and domain logic for specific workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimize_agent import (
        OptimizeAgentRequest,
        OptimizeAgentResponse,
        OptimizeAgentUseCase,
    )
    from .parse_documents import (
        ParseDocumentsRequest,
        ParseDocumentsResponse,
        ParseDocumentsUseCase,
    )
    from .predict_batch import (
        BatchPredictionRequest,
        BatchPredictionResponse,
        BatchPredictionUseCase,
    )


def __getattr__(name: str):
    """Lazy loading — use cases pull in heavy dependencies (dspy, marker, mlflow)."""
    if name == "OptimizeAgentRequest":
        from .optimize_agent import OptimizeAgentRequest

        return OptimizeAgentRequest
    if name == "OptimizeAgentResponse":
        from .optimize_agent import OptimizeAgentResponse

        return OptimizeAgentResponse
    if name == "OptimizeAgentUseCase":
        from .optimize_agent import OptimizeAgentUseCase

        return OptimizeAgentUseCase
    if name == "ParseDocumentsRequest":
        from .parse_documents import ParseDocumentsRequest

        return ParseDocumentsRequest
    if name == "ParseDocumentsResponse":
        from .parse_documents import ParseDocumentsResponse

        return ParseDocumentsResponse
    if name == "ParseDocumentsUseCase":
        from .parse_documents import ParseDocumentsUseCase

        return ParseDocumentsUseCase
    if name == "BatchPredictionRequest":
        from .predict_batch import BatchPredictionRequest

        return BatchPredictionRequest
    if name == "BatchPredictionResponse":
        from .predict_batch import BatchPredictionResponse

        return BatchPredictionResponse
    if name == "BatchPredictionUseCase":
        from .predict_batch import BatchPredictionUseCase

        return BatchPredictionUseCase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    "OptimizeAgentRequest",
    "OptimizeAgentResponse",
    "OptimizeAgentUseCase",
    "ParseDocumentsRequest",
    "ParseDocumentsResponse",
    "ParseDocumentsUseCase",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "BatchPredictionUseCase",
]
