"""Extraction module for Adaptive Extractor.

This module handles running extraction on documents using trained agents,
managing agent lifecycle, persistence, and batch predictions.
"""

from .agent import BaseAgent, UniversalExtractor
from .manager import AgentManager
from .pipeline import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionUseCase,
)

__all__ = [
    "BaseAgent",
    "UniversalExtractor",
    "AgentManager",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "BatchPredictionUseCase",
]
