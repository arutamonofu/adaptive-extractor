"""DSPy signature for nanozyme extraction.

This module defines the DSPy signature that specifies the input/output
interface for LLM-based nanozyme extraction.
"""

from typing import Type

import dspy

from .models import NanozymeExtractionOutput


def create_nanozyme_signature(instruction: str) -> Type[dspy.Signature]:
    """Create a NanozymeSignature class with the given instruction.

    Args:
        instruction: The initial instruction text to use as the signature docstring.
                Should be loaded from a configuration file.

    Returns:
        A DSPy Signature class configured with the instruction.

    Raises:
        ValueError: If the instruction is empty or contains only whitespace.
    """
    if not instruction or not instruction.strip():
        raise ValueError("Initial instruction for NanozymeSignature cannot be empty")

    class NanozymeSignature(dspy.Signature):
        """DSPy signature for extracting nanozyme experiments from scientific articles.

        This signature provides detailed instructions to the LLM about what
        information to extract and how to structure it. The instructions are
        loaded from an external instruction file specified in the configuration.
        """

        __doc__ = instruction

        document_text: str = dspy.InputField(
            desc="Full text content of the scientific article."
        )
        extracted_data: NanozymeExtractionOutput = dspy.OutputField(
            desc="A list of structured experiments matching the schema."
        )

    return NanozymeSignature
