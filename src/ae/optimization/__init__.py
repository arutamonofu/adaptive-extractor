"""Optimization module for Adaptive Extractor.

This module handles agent instruction and prompt optimization using MIPROv2,
along with dataset creation, validation, and experiment tracking.
"""

from .dataset import DatasetBuilder, DataValidator, ValidationResult
from .orchestrator import (
    OptimizeAgentRequest,
    OptimizeAgentResponse,
    OptimizeAgentUseCase,
)
from .tracking import ExperimentTracker

__all__ = [
    "DatasetBuilder",
    "DataValidator",
    "ValidationResult",
    "OptimizeAgentRequest",
    "OptimizeAgentResponse",
    "OptimizeAgentUseCase",
    "ExperimentTracker",
]
