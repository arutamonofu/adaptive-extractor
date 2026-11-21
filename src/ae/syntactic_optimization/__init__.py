"""Syntactic Prompt Optimization module for Adaptive Extractor.

This module handles syntactic prompt optimization (instructions and demonstrations)
using DSPy MIPROv2, along with dataset creation, validation, and experiment tracking.
"""

from .dataset_builder import DatasetBuilder
from .data_validator import DataValidator, ValidationResult
from .use_case import (
    OptimizeAgentRequest,
    OptimizeAgentResponse,
    SyntacticOptimizationUseCase,
)
from .tracking import ExperimentTracker

__all__ = [
    "DatasetBuilder",
    "DataValidator",
    "ValidationResult",
    "OptimizeAgentRequest",
    "OptimizeAgentResponse",
    "SyntacticOptimizationUseCase",
    "ExperimentTracker",
]
