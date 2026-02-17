"""Experiment-related domain entities.

This module defines the base interfaces and types for experiments across
all extraction tasks. Task-specific experiment types (e.g., NanozymeExperiment)
are defined in their respective task plugin modules.
"""

from abc import ABC
from typing import Any, Dict

from pydantic import BaseModel


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
