# src/ae/__init__.py

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at package initialization.
# This ensures that litellm/dspy imported downstream at module load time
# respect env settings before initialization.
_project_root = Path(__file__).resolve().parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

# Set LiteLLM to use local model cost map to avoid handshake timeout warnings
# on startup in firewalled/offline environments.
if os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") is None:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from typing import TYPE_CHECKING

__version__ = "0.4.0"

if TYPE_CHECKING:
    from ae.core.config import Settings, setup_logging
    from ae.core.evaluation import ExperimentMatcher, TaskMetric
    from ae.core.llm import create_lm, setup_student, setup_teacher
    from ae.core.storage import (
        DataSplitRepository,
        ExtractionRepository,
        GroundTruthRepository,
    )
    from ae.extraction.agent import UniversalExtractor
    from ae.ingestion.base_parser import BaseParser
    from ae.optimization.dataset import DatasetBuilder


def __getattr__(name: str):
    """Lazy module loading to avoid importing heavy dependencies at startup."""
    if name == "Settings":
        from ae.core.config import Settings
        return Settings
    if name == "setup_logging":
        from ae.core.config import setup_logging
        return setup_logging
    if name == "BaseParser":
        from ae.ingestion.base_parser import BaseParser
        return BaseParser
    if name == "UniversalExtractor":
        from ae.extraction.agent import UniversalExtractor
        return UniversalExtractor
    if name == "TaskMetric":
        from ae.core.evaluation import TaskMetric
        return TaskMetric
    if name == "ExperimentMatcher":
        from ae.core.evaluation import ExperimentMatcher
        return ExperimentMatcher
    if name == "setup_student":
        from ae.core.llm import setup_student
        return setup_student
    if name == "setup_teacher":
        from ae.core.llm import setup_teacher
        return setup_teacher
    if name == "create_lm":
        from ae.core.llm import create_lm
        return create_lm
    if name == "GroundTruthRepository":
        from ae.core.storage import GroundTruthRepository
        return GroundTruthRepository
    if name == "ExtractionRepository":
        from ae.core.storage import ExtractionRepository
        return ExtractionRepository
    if name == "DataSplitRepository":
        from ae.core.storage import DataSplitRepository
        return DataSplitRepository
    if name == "DatasetBuilder":
        from ae.optimization.dataset import DatasetBuilder
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
    # Parsers
    "BaseParser",
    # Agents
    "UniversalExtractor",
    # Evaluation
    "TaskMetric",
    "ExperimentMatcher",
    # LLM
    "setup_student",
    "setup_teacher",
    "create_lm",
    # Storage (Repositories)
    "GroundTruthRepository",
    "ExtractionRepository",
    "DataSplitRepository",
    # Optimization Services
    "DatasetBuilder",
]
