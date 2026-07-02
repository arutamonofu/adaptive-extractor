"""Schema system for Adaptive Extractor.

This module provides the schema infrastructure including configuration,
dynamic model generation, and loading/saving schemas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import FieldSpec, RowConverterConfig, SchemaConfig, ExtractionBundle
from .dynamic_models import (
    create_all_models,
    create_experiment_model,
    create_output_model,
    create_row_converter,
)
from .loader import (
    load_schema_complete,
    load_schema_from_yaml,
    load_schema_with_models,
    save_schema_to_yaml,
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
    "SchemaConfig",
    "FieldSpec",
    "RowConverterConfig",
    "ExtractionBundle",
    # Dynamic model generation
    "create_experiment_model",
    "create_output_model",
    "create_all_models",
    "create_row_converter",
    # Signature generation
    "create_signature",
    # YAML loading/saving
    "load_schema_from_yaml",
    "load_schema_with_models",
    "load_schema_complete",
    "save_schema_to_yaml",
]
