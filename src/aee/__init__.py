# src/aee/__init__.py

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.4.0"

if TYPE_CHECKING:
    # Core modules
    from aee.application.services import DatasetBuilder
    from aee.domain.evaluation import ExperimentMatcher, TaskMetric
    from aee.infrastructure.agents import UniversalExtractor
    from aee.infrastructure.config import Settings, setup_logging
    from aee.infrastructure.llm import create_lm, setup_student, setup_teacher
    from aee.infrastructure.parsers import BaseParser, MarkerParser
    from aee.infrastructure.storage import (
        DataSplitRepository,
        ExtractionRepository,
        GroundTruthRepository,
    )


def __getattr__(name: str):
    """Lazy module loading to avoid importing heavy dependencies at startup."""
    if name == "Settings":
        from aee.infrastructure.config import Settings

        return Settings
    if name == "setup_logging":
        from aee.infrastructure.config import setup_logging

        return setup_logging
    if name == "BaseParser":
        from aee.infrastructure.parsers import BaseParser

        return BaseParser
    if name == "MarkerParser":
        from aee.infrastructure.parsers import MarkerParser

        return MarkerParser
    if name == "UniversalExtractor":
        from aee.infrastructure.agents import UniversalExtractor

        return UniversalExtractor
    if name == "TaskMetric":
        from aee.domain.evaluation import TaskMetric

        return TaskMetric
    if name == "ExperimentMatcher":
        from aee.domain.evaluation import ExperimentMatcher

        return ExperimentMatcher
    if name == "setup_student":
        from aee.infrastructure.llm import setup_student

        return setup_student
    if name == "setup_teacher":
        from aee.infrastructure.llm import setup_teacher

        return setup_teacher
    if name == "create_lm":
        from aee.infrastructure.llm import create_lm

        return create_lm
    if name == "GroundTruthRepository":
        from aee.infrastructure.storage import GroundTruthRepository

        return GroundTruthRepository
    if name == "ExtractionRepository":
        from aee.infrastructure.storage import ExtractionRepository

        return ExtractionRepository
    if name == "DataSplitRepository":
        from aee.infrastructure.storage import DataSplitRepository

        return DataSplitRepository
    if name == "DatasetBuilder":
        from aee.application.services import DatasetBuilder

        return DatasetBuilder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    # Version
    "__version__",
    # Config
    "Settings",
    "setup_logging",
    # Infrastructure - Parsers
    "BaseParser",
    "MarkerParser",
    # Agents
    "UniversalExtractor",
    # Domain - Evaluation
    "TaskMetric",
    "ExperimentMatcher",
    # Infrastructure - LLM
    "setup_student",
    "setup_teacher",
    "create_lm",
    # Infrastructure - Storage (Repositories)
    "GroundTruthRepository",
    "ExtractionRepository",
    "DataSplitRepository",
    # Application - Services
    "DatasetBuilder",
]
