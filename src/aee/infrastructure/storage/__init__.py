"""Storage infrastructure for AutoEvoExtractor.

This module provides repository pattern implementations for managing
various types of data: agents, ground truth, predictions, documents, and splits.
"""

from .agents import AgentMetadata, AgentRepository
from .documents import DocumentRepository
from .ground_truth import GroundTruthRepository
from .predictions import PredictionRepository
from .splits import DataSplitRepository
from .migrations import (
    AgentMigrator,
    GroundTruthMigrator,
    migrate_all_agents,
    migrate_all_ground_truth,
)

__all__ = [
    "AgentMetadata",
    "AgentRepository",
    "GroundTruthRepository",
    "PredictionRepository",
    "DocumentRepository",
    "DataSplitRepository",
    # Migrations
    "AgentMigrator",
    "GroundTruthMigrator",
    "migrate_all_agents",
    "migrate_all_ground_truth",
]
