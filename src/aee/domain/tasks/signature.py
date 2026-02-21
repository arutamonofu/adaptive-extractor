"""Dynamic DSPy signature generation for tasks.

This module provides functions to dynamically generate DSPy signatures
from TaskConfig, enabling flexible task definitions without hardcoded signatures.
"""

import logging
from typing import Any, Dict, List, Optional, Type

import dspy
from pydantic import BaseModel

from .config import TaskConfig

logger = logging.getLogger(__name__)


def create_signature(
    task_config: TaskConfig,
    experiment_model: Type[BaseModel],
    output_model: Optional[Type[BaseModel]] = None,
    instruction: Optional[str] = None,
) -> Type[dspy.Signature]:
    """Dynamically create a DSPy signature from TaskConfig.

    This function generates a DSPy Signature class with input/output fields
    based on the task configuration and generated models.

    Args:
        task_config: Task configuration.
        experiment_model: Generated experiment model class.
        output_model: Generated output model class (optional, will be created if not provided).
        instruction: Instruction text (optional, will use task_config if not provided).

    Returns:
        Dynamically generated DSPy Signature class.

    Raises:
        ValueError: If no instruction is available.
        FileNotFoundError: If instruction file not found.

    Example:
        ```python
        config = TaskConfig(
            name="nanozymes",
            experiment_fields={...},
            compare_fields=["formula", "activity"],
            initial_instruction_file="config/initial_instructions/nanozymes.txt"
        )

        ExperimentModel = create_experiment_model(config)
        OutputModel = create_output_model(config, ExperimentModel)
        Signature = create_signature(config, ExperimentModel, OutputModel)

        # Use signature with DSPy
        module = dspy.Predict(Signature)
        result = module(document_text="...")
        ```
    """
    # Get instruction from config (no fallback - strict mode)
    if instruction is None:
        instruction = task_config.get_instruction()

    if not instruction or not instruction.strip():
        raise ValueError("Instruction cannot be empty")

    # Create output model if not provided
    if output_model is None:
        from .dynamic_models import create_output_model
        output_model = create_output_model(task_config, experiment_model)

    # Create dynamic signature class
    signature_name = f"{task_config.name.title()}Signature"

    class DynamicSignature(dspy.Signature):
        __doc__ = instruction

        document_text: str = dspy.InputField(
            desc="Full text content of the scientific article or document."
        )
        extracted_data: output_model = dspy.OutputField(
            desc=f"Extracted {task_config.name} experiments as structured data."
        )

    # Set the class name dynamically
    DynamicSignature.__name__ = signature_name

    logger.info(f"Created DSPy signature '{signature_name}'")

    return DynamicSignature
