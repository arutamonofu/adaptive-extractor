"""Storage infrastructure for AutoEvoExtractor.

This module provides repository pattern implementations for managing
various types of data: agents, ground truth, extractions, documents, and splits.

Functional API:
    New code should prefer the functional API for simpler use cases:
    - save_agent, load_agent, list_agents (from .agents_fn)
    - load_ground_truth (from .ground_truth_fn)
    - load_split, load_all_splits (from .splits_fn)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .agents import AgentMetadata, AgentRepository
from .agents_fn import (
    delete_agent,
    get_agent_info,
    get_latest_agent,
    list_agents,
    load_agent,
    save_agent,
)
from .documents import DocumentRepository
from .extractions import ExtractionRepository
from .splits import DataSplitRepository
from .splits_fn import (
    create_random_split,
    load_all_splits,
    load_split,
    save_splits,
    validate_splits,
)

# Lazy imports — ground_truth.py and migrations.py import pandas
if TYPE_CHECKING:
    from .ground_truth import GroundTruthRepository
    from .ground_truth_fn import load_ground_truth
    from .ground_truth_fn import validate_coverage as validate_gt_coverage
    from .migrations import (
        AgentMigrator,
        GroundTruthMigrator,
        migrate_all_agents,
        migrate_all_ground_truth,
    )


def __getattr__(name: str):
    """Lazy loading — ground_truth imports pandas, migrations imports pandas."""
    if name == "GroundTruthRepository":
        from .ground_truth import GroundTruthRepository

        return GroundTruthRepository
    if name == "load_ground_truth":
        from .ground_truth_fn import load_ground_truth

        return load_ground_truth
    if name == "validate_gt_coverage":
        from .ground_truth_fn import validate_coverage

        return validate_coverage
    if name == "AgentMigrator":
        from .migrations import AgentMigrator

        return AgentMigrator
    if name == "GroundTruthMigrator":
        from .migrations import GroundTruthMigrator

        return GroundTruthMigrator
    if name == "migrate_all_agents":
        from .migrations import migrate_all_agents

        return migrate_all_agents
    if name == "migrate_all_ground_truth":
        from .migrations import migrate_all_ground_truth

        return migrate_all_ground_truth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    # Classic repository classes (backward compatibility)
    "AgentMetadata",
    "AgentRepository",
    "ExtractionRepository",
    "GroundTruthRepository",
    "DocumentRepository",
    "DataSplitRepository",
    # Functional API - Agents
    "save_agent",
    "load_agent",
    "list_agents",
    "get_latest_agent",
    "delete_agent",
    "get_agent_info",
    # Functional API - Ground Truth
    "load_ground_truth",
    "validate_gt_coverage",
    # Functional API - Splits
    "load_split",
    "load_all_splits",
    "save_splits",
    "create_random_split",
    "validate_splits",
    # Migrations
    "AgentMigrator",
    "GroundTruthMigrator",
    "migrate_all_agents",
    "migrate_all_ground_truth",
]
