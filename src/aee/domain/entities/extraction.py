"""Extraction result entities.

This module defines the types and structures for extraction results,
including outputs from LLM-based extraction and post-processing.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .experiment import Experiment

# Type variable for experiment types
E = TypeVar("E", bound=Experiment)


class ExtractionResult(BaseModel, Generic[E]):
    """Result of an extraction operation.

    This generic class wraps the extracted experiments along with metadata
    about the extraction process.

    Type Parameters:
        E: The experiment type (e.g., NanozymeExperiment).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiments: List[E] = Field(default_factory=list)
    source_document: Optional[str] = None
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionOutput(BaseModel):
    """Base class for task-specific extraction outputs.

    Task plugins should define their own extraction output classes
    that inherit from this base class. This ensures consistent structure
    across different extraction tasks.
    """

    experiments: List[Any] = Field(default_factory=list)

    def get_experiment_count(self) -> int:
        """Get the number of extracted experiments.

        Returns:
            Number of experiments in this output.
        """
        return len(self.experiments)

    def is_empty(self) -> bool:
        """Check if extraction produced any results.

        Returns:
            True if no experiments were extracted, False otherwise.
        """
        return len(self.experiments) == 0
