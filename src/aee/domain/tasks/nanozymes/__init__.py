"""Nanozyme task plugin.

This module provides the complete nanozyme extraction task implementation,
including models, signature, converters, and task definition.
"""

from typing import Callable, List, Optional, Type

import dspy
from pydantic import BaseModel

from aee.domain.tasks.base import TaskDefinition
from aee.domain.tasks import register_task

from .converters import row_to_nanozyme
from .models import NanozymeExperiment, NanozymeExtractionOutput
from .signature import create_nanozyme_signature


class NanozymeTask(TaskDefinition):
    """Task definition for nanozyme experiment extraction.

    This task extracts detailed information about nanozyme experiments
    from scientific articles, including material properties, catalytic
    activities, and kinetic parameters.

    The initial instruction for the DSPy signature must be provided explicitly
    via the initial_instruction parameter. There is no fallback default instruction.
    """

    def __init__(self, initial_instruction: str):
        """Initialize the NanozymeTask.

        Args:
            initial_instruction: The initial instruction text to use for the DSPy signature.
                          Must be loaded from a configuration file.

        Raises:
            ValueError: If the initial_instruction is empty or contains only whitespace.
        """
        if not initial_instruction or not initial_instruction.strip():
            raise ValueError(
                "NanozymeTask requires a non-empty initial_instruction. "
                "The instruction must be loaded from a configuration file."
            )

        self._initial_instruction = initial_instruction
        self._signature_class = create_nanozyme_signature(initial_instruction)

    @property
    def name(self) -> str:
        """Task identifier."""
        return "nanozymes"

    @property
    def description(self) -> str:
        """Task description."""
        return (
            "Extract nanozyme experiment data including chemical formulas, "
            "catalytic activities, kinetic parameters (Km, Vmax), reaction "
            "conditions (pH, temperature), and material properties."
        )

    @property
    def signature(self) -> Type[dspy.Signature]:
        """DSPy signature for nanozyme extraction."""
        return self._signature_class

    @property
    def output_model(self) -> Type[BaseModel]:
        """Output model containing extracted experiments."""
        return NanozymeExtractionOutput

    @property
    def experiment_model(self) -> Type[BaseModel]:
        """Model for individual nanozyme experiments."""
        return NanozymeExperiment

    @property
    def row_converter(self) -> Callable:
        """Converter function for CSV rows to experiments."""
        return row_to_nanozyme

    @property
    def compare_fields(self) -> List[str]:
        """Fields to compare during evaluation."""
        return [
            "formula",
            "activity",
            "syngony",
            "surface",
            "length",
            "width",
            "depth",
            "reaction_type",
            "km_value",
            "km_unit",
            "vmax_value",
            "vmax_unit",
            "ph",
            "temperature",
            "c_min",
            "c_max",
            "c_const",
            "c_const_unit",
            "ccat_value",
            "ccat_unit",
        ]

    @property
    def float_tolerance(self) -> float:
        """Tolerance for float comparisons (10%)."""
        return 0.10

    @property
    def initial_instruction(self) -> str:
        """The initial instruction used for this task."""
        return self._initial_instruction

    @property
    def instruction_metadata(self) -> dict:
        """Metadata about the initial instruction."""
        import hashlib
        instruction_hash = hashlib.sha256(self._initial_instruction.encode()).hexdigest()[:12]
        return {
            "instruction_hash": instruction_hash,
            "instruction_length": len(self._initial_instruction),
        }


# Note: Task registration now happens after loading the instruction from config.
# See scripts/optimize.py for the registration pattern with config-loaded instructions.
__all__ = [
    "NanozymeTask",
    "NanozymeExperiment",
    "NanozymeExtractionOutput",
    "create_nanozyme_signature",
    "row_to_nanozyme",
]
