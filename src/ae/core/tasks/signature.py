"""Dynamic DSPy signature generation for tasks.

This module provides functions to dynamically generate DSPy signatures
from TaskConfig, enabling flexible task definitions without hardcoded signatures.
"""

import logging
from typing import TYPE_CHECKING, Optional, Type

from pydantic import BaseModel

from .config import TaskConfig

if TYPE_CHECKING:
    import dspy

logger = logging.getLogger(__name__)


def create_signature(
    task_config: TaskConfig,
    experiment_model: Type[BaseModel],
    output_model: Optional[Type[BaseModel]] = None,
    instruction: Optional[str] = None,
    schema_in_prompt: bool = False,
) -> "Type[dspy.Signature]":
    """Dynamically create a DSPy signature from TaskConfig.

    This function generates a DSPy Signature class with input/output fields
    based on the task configuration and generated models.

    Args:
        task_config: Task configuration.
        experiment_model: Generated experiment model class.
        output_model: Generated output model class (optional, will be created if not provided).
        instruction: Instruction text (optional, will use task_config if not provided).
        schema_in_prompt: Whether the schema is embedded in the prompt.

    Returns:
        Dynamically generated DSPy Signature class.

    Raises:
        ValueError: If no instruction is available.
        FileNotFoundError: If instruction file not found.
    """
    import dspy
    import hashlib

    # Get instruction from config (no fallback - strict mode)
    if instruction is None:
        instruction = task_config.get_instruction()

    if not instruction or not instruction.strip():
        raise ValueError("Instruction cannot be empty")

    # Create output model if not provided
    if output_model is None:
        from .dynamic_models import create_output_model
        # Use minimal descriptions if schema_in_prompt is True
        if schema_in_prompt:
            minimal_config = task_config.get_minimal_config()
            # We also need a corresponding minimal experiment model
            from .dynamic_models import create_experiment_model
            minimal_exp_model = create_experiment_model(minimal_config)
            output_model = create_output_model(minimal_config, minimal_exp_model)
        else:
            output_model = create_output_model(task_config, experiment_model)

    compiled_docstring = instruction.strip()
    prompt_hash = hashlib.md5(compiled_docstring.encode("utf-8")).hexdigest()[:8]
    signature_name = f"DynamicExtractionSignature_{prompt_hash}"

    class_attributes = {
        "__doc__": compiled_docstring,
        "document_text": dspy.InputField(
            desc="Full text content of the scientific article or document."
        ),
        "extracted_data": dspy.OutputField(
            desc=f"Extracted {task_config.name} experiments as structured data."
        )
    }

    # Dynamically create the class inheriting from dspy.Signature
    DynamicSignature = type(signature_name, (dspy.Signature,), class_attributes)
    # Set the output model as attribute/type hint for the output field so TypedPredictor knows it
    DynamicSignature.__annotations__ = {
        "document_text": str,
        "extracted_data": output_model
    }

    logger.info(f"Created DSPy signature '{signature_name}' with hash '{prompt_hash}'")

    return DynamicSignature
