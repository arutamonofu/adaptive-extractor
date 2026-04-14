"""Application services for AutoEvoExtractor.

This module provides high-level services that orchestrate domain
and infrastructure components for common operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .agent_manager import AgentManager, SerializableAgent
from .data_validator import DataValidator, ValidationResult

if TYPE_CHECKING:
    from .dataset_builder import DatasetBuilder
    from .experiment_tracker import ExperimentTracker


def __getattr__(name: str):
    """Lazy loading — DatasetBuilder imports dspy, ExperimentTracker imports mlflow."""
    if name == "DatasetBuilder":
        from .dataset_builder import DatasetBuilder

        return DatasetBuilder
    if name == "ExperimentTracker":
        from .experiment_tracker import ExperimentTracker

        return ExperimentTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    "AgentManager",
    "DataValidator",
    "DatasetBuilder",
    "ExperimentTracker",
    "ValidationResult",
    "SerializableAgent",
]
