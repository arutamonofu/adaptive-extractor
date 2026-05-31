"""Task system for Adaptive Extractor.

This module provides the task infrastructure including configuration,
dynamic model generation, and registry for managing extraction tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import FieldSpec, RowConverterConfig, TaskConfig
from .dynamic_models import (
    create_all_models,
    create_experiment_model,
    create_output_model,
    create_row_converter,
)
from .loader import (
    load_task_complete,
    load_task_from_yaml,
    load_task_with_models,
    save_task_to_yaml,
)
from .registry import (
    TaskRegistry,
    get_config,
    get_global_registry,
    get_task,
    load_and_register_task,
    load_task_with_instruction,
    register_config,
)

if TYPE_CHECKING:
    from .signature import create_signature


def __getattr__(name: str):
    """Lazy loading — create_signature imports dspy."""
    if name == "create_signature":
        from .signature import create_signature

        return create_signature
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
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
    # YAML loading/saving
    "load_task_from_yaml",
    "load_task_with_models",
    "load_task_complete",
    "load_task_with_instruction",
    "save_task_to_yaml",
    # Registry
    "TaskRegistry",
    "get_global_registry",
    "get_config",
    "get_task",
    "register_config",
    "load_and_register_task",
]
