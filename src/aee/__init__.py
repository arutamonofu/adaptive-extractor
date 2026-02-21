# src/aee/__init__.py

__version__ = "0.4.0"

# Core modules
from aee.infrastructure.config import Settings, setup_logging
from aee.domain.entities import DocumentMetadata, ProcessedDocument
from aee.infrastructure.parsers import BaseParser, DoclingParser, MarkerParser, TextCleaner
from aee.infrastructure.agents import UniversalExtractor
from aee.domain.evaluation import TaskMetric, ExperimentMatcher
from aee.infrastructure.llm import setup_student, setup_teacher, create_lm
from aee.infrastructure.storage import GroundTruthRepository, ExtractionRepository, DataSplitRepository
from aee.application.services import DatasetBuilder

__all__ = [
    # Version
    "__version__",

    # Config
    "Settings",
    "setup_logging",

    # Domain Entities
    "DocumentMetadata",
    "ProcessedDocument",

    # Infrastructure - Parsers
    "BaseParser",
    "DoclingParser",
    "MarkerParser",
    "TextCleaner",

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
