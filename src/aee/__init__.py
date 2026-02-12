# src/aee/__init__.py

__version__ = "0.2.0"

# Core modules
from aee.config import settings, setup_logging
from aee.data import DocumentMetadata, ProcessedDocument
from aee.ingestion import BaseParser, DoclingParser, MarkerParser, TextCleaner
from aee.models import NanozymeExperiment, NanozymeExtractionOutput, NanozymeSignature
from aee.agents import UniversalExtractor
from aee.evaluation import TaskMetric, ExperimentMatcher
from aee.llm import setup_student, setup_teacher, create_lm
from aee.utils import load_ground_truth, load_predictions, get_split_files, create_dataset_from_ids

__all__ = [
    # Version
    "__version__",
    
    # Config
    "settings", 
    "setup_logging",
    
    # Data
    "DocumentMetadata", 
    "ProcessedDocument",
    
    # Ingestion
    "BaseParser", 
    "DoclingParser", 
    "MarkerParser", 
    "TextCleaner",
    
    # Models
    "NanozymeExperiment", 
    "NanozymeExtractionOutput", 
    "NanozymeSignature",
    
    # Agents
    "UniversalExtractor",
    
    # Evaluation
    "TaskMetric", 
    "ExperimentMatcher",
    
    # LLM
    "setup_student", 
    "setup_teacher", 
    "create_lm",
    
    # Utils
    "load_ground_truth",
    "load_predictions",
    "get_split_files",
    "create_dataset_from_ids",
]