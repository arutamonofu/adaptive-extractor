"""Use cases for AutoEvoExtractor.

This module contains the application use cases that orchestrate
services and domain logic for specific workflows.
"""

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
