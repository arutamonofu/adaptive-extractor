"""Dynamic wrapper for TaskConfig to implement TaskDefinition interface.

This module provides a compatibility layer that allows TaskConfig to be used
interchangeably with TaskDefinition, enabling gradual migration to the new
configuration-based approach.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type

import dspy
from pydantic import BaseModel

from aee.domain.tasks.base import TaskDefinition
from aee.domain.tasks.config import TaskConfig
from aee.domain.tasks.dynamic_models import create_all_models, create_row_converter
from aee.domain.tasks.signature import create_signature
from aee.shared.exceptions import TaskValidationError

logger = logging.getLogger(__name__)


class ConfigBackedTask(TaskDefinition):
    """TaskDefinition wrapper backed by TaskConfig.

    This class implements the TaskDefinition interface using a TaskConfig
    as the source of truth. It dynamically generates models, signatures,
    and converters from the configuration.

    This enables:
    - Using TaskConfig with existing code that expects TaskDefinition
    - Gradual migration from class-based to config-based tasks
    - YAML-based task definitions with full TaskDefinition compatibility

    Example:
        ```python
        config = load_task_from_yaml("tasks/nanozymes/task.yaml")
        task = ConfigBackedTask(config)

        # Use like any TaskDefinition
        task.validate()
        signature = task.signature
        model = task.experiment_model
        ```
    """

    def __init__(self, config: TaskConfig):
        """Initialize ConfigBackedTask from TaskConfig.

        Args:
            config: Task configuration to use.
        """
        self._config = config

        # Generate models and signature lazily
        self._experiment_model: Optional[Type[BaseModel]] = None
        self._output_model: Optional[Type[BaseModel]] = None
        self._signature: Optional[Type[dspy.Signature]] = None
        self._row_converter: Optional[Callable] = None

    @property
    def config(self) -> TaskConfig:
        """Get the underlying TaskConfig.

        Returns:
            TaskConfig instance.
        """
        return self._config

    @property
    def name(self) -> str:
        """Task identifier.

        Returns:
            Task name from config.
        """
        return self._config.name

    @property
    def description(self) -> str:
        """Task description.

        Returns:
            Task description from config.
        """
        return self._config.description

    @property
    def signature(self) -> Type[dspy.Signature]:
        """DSPy signature for this task.

        Lazily generates the signature from config on first access.

        Returns:
            DSPy Signature class.
        """
        if self._signature is None:
            self._ensure_models_loaded()

        return self._signature

    @property
    def output_model(self) -> Type[BaseModel]:
        """Pydantic model for extraction output.

        Lazily generates the model from config on first access.

        Returns:
            Pydantic BaseModel class.
        """
        if self._output_model is None:
            self._ensure_models_loaded()

        return self._output_model

    @property
    def experiment_model(self) -> Type[BaseModel]:
        """Pydantic model for individual experiments.

        Lazily generates the model from config on first access.

        Returns:
            Pydantic BaseModel class.
        """
        if self._experiment_model is None:
            self._ensure_models_loaded()

        return self._experiment_model

    @property
    def row_converter(self) -> Callable:
        """Function to convert CSV row to experiment model.

        Lazily generates the converter from config on first access.

        Returns:
            Converter function.
        """
        if self._row_converter is None:
            self._ensure_models_loaded()

        return self._row_converter

    @property
    def compare_fields(self) -> List[str]:
        """Fields to compare during evaluation.

        Returns:
            List of field names from config.
        """
        return self._config.compare_fields

    @property
    def float_tolerance(self) -> float:
        """Tolerance for float comparisons.

        Returns:
            Float tolerance from config.
        """
        return self._config.float_tolerance

    def _ensure_models_loaded(self) -> None:
        """Ensure all models and components are loaded.

        Lazily generates models, signature, and converter from config.
        """
        if self._experiment_model is None:
            logger.info(f"Generating models for task '{self.name}' from config")

            self._experiment_model, self._output_model = create_all_models(
                self._config,
                base_class=self._config.base_class,
            )

            self._signature = create_signature(
                self._config,
                self._experiment_model,
                self._output_model,
            )

            self._row_converter = create_row_converter(
                self._config,
                self._experiment_model,
            )

    def validate(self) -> None:
        """Validate task definition.

        Delegates validation to TaskConfig and checks model compatibility.

        Raises:
            TaskValidationError: If validation fails.
        """
        errors = self._config.validate()

        # Additional validation: ensure models can be generated
        if not errors:
            try:
                self._ensure_models_loaded()
            except Exception as e:
                errors.append(f"Failed to generate models: {e}")

        if errors:
            raise TaskValidationError(self.name, errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation.

        Returns:
            Dictionary with task configuration.
        """
        result = self._config.to_dict()

        # Add generated model names
        if self._experiment_model is not None:
            result["experiment_model"] = self._experiment_model.__name__

        if self._output_model is not None:
            result["output_model"] = self._output_model.__name__

        if self._signature is not None:
            result["signature"] = self._signature.__name__

        return result

    def __repr__(self) -> str:
        """String representation of task."""
        return f"<ConfigBackedTask: {self.name}>"
