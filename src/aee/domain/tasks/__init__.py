"""Task plugin system for AutoEvoExtractor.

This module provides the task plugin infrastructure including base classes,
registry, and utilities for managing extraction tasks.

Supports both classic TaskDefinition approach and new dynamic TaskConfig approach.
"""

from .base import TaskDefinition
from .config import FieldSpec, RowConverterConfig, TaskConfig
from .dynamic_models import (
    create_all_models,
    create_experiment_model,
    create_output_model,
    create_row_converter,
)
from .dynamic_wrapper import ConfigBackedTask
from .loader import (
    load_task_complete,
    load_task_from_yaml,
    load_task_with_models,
    save_task_to_yaml,
)
from .registry import (
    TaskRegistry,
    get_global_registry,
    get_task,
    load_and_register_task,
    register_task,
)
from .signature import create_signature, generate_default_instruction

__all__ = [
    # Base classes
    "TaskDefinition",
    "ConfigBackedTask",
    # Configuration
    "TaskConfig",
    "FieldSpec",
    "RowConverterConfig",
    # Dynamic model generation
    "create_experiment_model",
    "create_output_model",
    "create_all_models",
    "create_row_converter",
    # Signature generation
    "create_signature",
    "generate_default_instruction",
    # YAML loading/saving
    "load_task_from_yaml",
    "load_task_with_models",
    "load_task_complete",
    "save_task_to_yaml",
    # Registry
    "TaskRegistry",
    "get_global_registry",
    "get_task",
    "register_task",
    "load_and_register_task",
]
