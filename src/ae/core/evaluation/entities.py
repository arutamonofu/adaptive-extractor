"""Extraction result and experiment domain entities."""

from abc import ABC
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class Experiment(BaseModel, ABC):
    """Base class for all experiment types.

    Each task plugin should define its own experiment model that inherits from
    this base class. The base class ensures all experiments are Pydantic models
    and can be serialized/validated consistently.

    Task-specific fields should be defined in the concrete experiment classes.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary.

        Returns:
            Dictionary representation of the experiment.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create experiment from dictionary.

        Args:
            data: Dictionary containing experiment data.

        Returns:
            Experiment instance.
        """
        return cls(**data)


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
