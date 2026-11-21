"""Extraction result and record domain entities."""

from abc import ABC
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Record(BaseModel, ABC):
    """Base class for all record types.

    Each task plugin should define its own record model that inherits from
    this base class. The base class ensures all records are Pydantic models
    and can be serialized/validated consistently.

    Task-specific fields should be defined in the concrete record classes.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary.

        Returns:
            Dictionary representation of the record.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Record":
        """Create record from dictionary.

        Args:
            data: Dictionary containing record data.

        Returns:
            Record instance.
        """
        return cls(**data)


# Type variable for record types
R = TypeVar("R", bound=Record)


class ExtractionResult(BaseModel, Generic[R]):
    """Result of an extraction operation.

    This generic class wraps the extracted records along with metadata
    about the extraction process.

    Type Parameters:
        R: The record type (e.g., NanozymeRecord).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    records: List[R] = Field(
        default_factory=list,
        validation_alias=AliasChoices("records", "experiments"),
    )
    source_document: Optional[str] = None
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def experiments(self) -> List[R]:
        """Backward-compatible alias for records."""
        return self.records


class ExtractionOutput(BaseModel):
    """Base class for task-specific extraction outputs.

    Task plugins should define their own extraction output classes
    that inherit from this base class. This ensures consistent structure
    across different extraction tasks.
    """

    records: List[Any] = Field(
        default_factory=list,
        validation_alias=AliasChoices("records", "experiments"),
    )

    @property
    def experiments(self) -> List[Any]:
        """Backward-compatible alias for records."""
        return self.records

    def get_record_count(self) -> int:
        """Get the number of extracted records.

        Returns:
            Number of records in this output.
        """
        return len(self.records)

    def get_experiment_count(self) -> int:
        """Backward-compatible alias."""
        return self.get_record_count()

    def is_empty(self) -> bool:
        """Check if extraction produced any results.

        Returns:
            True if no records were extracted, False otherwise.
        """
        return len(self.records) == 0
