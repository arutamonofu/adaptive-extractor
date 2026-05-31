"""Domain evaluation package.

This package contains evaluation logic for matching and scoring experiments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ae.core.evaluation.matcher import ExperimentMatcher
    from ae.core.evaluation.metrics import TaskMetric


def __getattr__(name: str):
    """Lazy loading — evaluation imports numpy, scipy, tabulate, dspy."""
    if name == "TaskMetric":
        from ae.core.evaluation.metrics import TaskMetric

        return TaskMetric
    if name == "ExperimentMatcher":
        from ae.core.evaluation.matcher import ExperimentMatcher

        return ExperimentMatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    "ExperimentMatcher",
    "TaskMetric",
]
