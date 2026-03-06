"""Domain entities package.

This package contains core domain entities like experiments and extractions.

Note: ProcessedDocument and DocumentMetadata have been removed. Parsed documents
are now stored as plain Markdown files.
"""

from aee.domain.entities.experiment import Experiment
from aee.domain.entities.extraction import ExtractionResult, ExtractionOutput

__all__ = [
    "Experiment",
    "ExtractionResult",
    "ExtractionOutput",
]
