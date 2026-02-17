"""Base task definition for the plugin system.

This module defines the abstract base class that all task plugins must implement.
Tasks define how to extract, validate, and evaluate specific types of experiments.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

import dspy
from pydantic import BaseModel

from aee.shared.exceptions import TaskValidationError


class TaskDefinition(ABC):
    """Abstract base class for task definitions.

    Each task plugin must inherit from this class and implement all abstract
    properties and methods. This ensures consistent task structure and behavior
    across different extraction tasks.

    Example:
        ```python
        class NanozymeTask(TaskDefinition):
            @property
            def name(self) -> str:
                return "nanozymes"

            @property
            def signature(self) -> Type[dspy.Signature]:
                return NanozymeSignature

            # ... implement other abstract methods
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the task.

        Returns:
            Task name (e.g., "nanozymes", "catalysts").
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the task.

        Returns:
            Description of what this task extracts.
        """
        pass

    @property
    @abstractmethod
    def signature(self) -> Type[dspy.Signature]:
        """DSPy signature class for this task.

        The signature defines the input/output interface for the LLM extraction.

        Returns:
            DSPy Signature class.
        """
        pass

    @property
    @abstractmethod
    def output_model(self) -> Type[BaseModel]:
        """Pydantic model for extraction output.

        This model should contain a list of experiments and must be compatible
        with the signature's output field.

        Returns:
            Pydantic BaseModel class for extraction output.
        """
        pass

    @property
    @abstractmethod
    def experiment_model(self) -> Type[BaseModel]:
        """Pydantic model for individual experiments.

        This model defines the structure of a single experiment extracted by
        this task.

        Returns:
            Pydantic BaseModel class for experiments.
        """
        pass

    @property
    @abstractmethod
    def row_converter(self) -> Callable[[Any], Optional[BaseModel]]:
        """Function to convert CSV row to experiment model.

        This function is used to load ground truth data from CSV files.

        Returns:
            Function that takes a pandas Series and returns an experiment instance.
        """
        pass

    @property
    @abstractmethod
    def compare_fields(self) -> List[str]:
        """List of field names to compare during evaluation.

        These fields are used by the ExperimentMatcher to calculate metrics.

        Returns:
            List of field names as strings.
        """
        pass

    @property
    def float_tolerance(self) -> float:
        """Tolerance for floating-point comparisons.

        Can be overridden by subclasses to use task-specific tolerance.

        Returns:
            Float tolerance (default 0.05 = 5%).
        """
        return 0.05

    @property
    def config(self) -> Dict[str, Any]:
        """Additional task-specific configuration.

        Can be overridden by subclasses to provide additional configuration.

        Returns:
            Configuration dictionary.
        """
        return {}

    def validate(self) -> None:
        """Validate task definition completeness and consistency.

        Raises:
            TaskValidationError: If validation fails.
        """
        errors: List[str] = []

        # Check required properties
        try:
            if not self.name or not isinstance(self.name, str):
                errors.append("Task name must be a non-empty string")
        except Exception as e:
            errors.append(f"Error accessing 'name' property: {e}")

        try:
            if not self.description or not isinstance(self.description, str):
                errors.append("Task description must be a non-empty string")
        except Exception as e:
            errors.append(f"Error accessing 'description' property: {e}")

        try:
            if not issubclass(self.signature, dspy.Signature):
                errors.append("Signature must be a DSPy Signature class")
        except Exception as e:
            errors.append(f"Error accessing 'signature' property: {e}")

        try:
            if not issubclass(self.output_model, BaseModel):
                errors.append("Output model must be a Pydantic BaseModel class")
        except Exception as e:
            errors.append(f"Error accessing 'output_model' property: {e}")

        try:
            if not issubclass(self.experiment_model, BaseModel):
                errors.append("Experiment model must be a Pydantic BaseModel class")
        except Exception as e:
            errors.append(f"Error accessing 'experiment_model' property: {e}")

        try:
            if not callable(self.row_converter):
                errors.append("Row converter must be callable")
        except Exception as e:
            errors.append(f"Error accessing 'row_converter' property: {e}")

        try:
            if not isinstance(self.compare_fields, list) or not self.compare_fields:
                errors.append("Compare fields must be a non-empty list")
            elif not all(isinstance(f, str) for f in self.compare_fields):
                errors.append("All compare fields must be strings")
        except Exception as e:
            errors.append(f"Error accessing 'compare_fields' property: {e}")

        try:
            if not isinstance(self.float_tolerance, (int, float)) or not 0 <= self.float_tolerance <= 1:
                errors.append("Float tolerance must be a number between 0 and 1")
        except Exception as e:
            errors.append(f"Error accessing 'float_tolerance' property: {e}")

        # Validate output_model has 'experiments' field
        try:
            output_fields = self.output_model.model_fields
            if "experiments" not in output_fields:
                errors.append("Output model must have an 'experiments' field")
        except Exception as e:
            errors.append(f"Error inspecting output_model fields: {e}")

        # Validate compare_fields exist in experiment_model
        try:
            experiment_fields = set(self.experiment_model.model_fields.keys())
            invalid_fields = [f for f in self.compare_fields if f not in experiment_fields]
            if invalid_fields:
                errors.append(
                    f"Compare fields {invalid_fields} not found in experiment model. "
                    f"Available fields: {sorted(experiment_fields)}"
                )
        except Exception as e:
            errors.append(f"Error validating compare_fields: {e}")

        if errors:
            raise TaskValidationError(self.name, errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task definition to dictionary representation.

        Returns:
            Dictionary with task configuration.
        """
        return {
            "name": self.name,
            "description": self.description,
            "signature": self.signature.__name__,
            "output_model": self.output_model.__name__,
            "experiment_model": self.experiment_model.__name__,
            "compare_fields": self.compare_fields,
            "float_tolerance": self.float_tolerance,
            "config": self.config,
        }

    def __repr__(self) -> str:
        """String representation of task."""
        return f"<TaskDefinition: {self.name}>"
