"""Domain evaluation package.

This package contains evaluation logic for matching and scoring records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ae.core.evaluation.entities import ExtractionOutput, ExtractionResult, Record

if TYPE_CHECKING:
    from ae.core.evaluation.matcher import ExperimentEntity, ExperimentMatcher, RecordEntity, RecordMatcher
    from ae.core.evaluation.metrics import TaskMetric


def __getattr__(name: str):
    """Lazy loading — evaluation imports numpy, scipy, tabulate, dspy."""
    if name == "TaskMetric":
        from ae.core.evaluation.metrics import TaskMetric

        return TaskMetric
    if name in ("RecordMatcher", "ExperimentMatcher"):
        from ae.core.evaluation.matcher import RecordMatcher

        return RecordMatcher
    if name in ("RecordEntity", "ExperimentEntity"):
        from ae.core.evaluation.matcher import RecordEntity

        return RecordEntity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    "Record",
    "ExtractionOutput",
    "ExtractionResult",
    "RecordEntity",
    "ExperimentEntity",
    "RecordMatcher",
    "ExperimentMatcher",
    "TaskMetric",
]
