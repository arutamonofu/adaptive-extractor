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


def generate_default_instruction(task_config: TaskConfig) -> str:
    """Generate a default instruction from TaskConfig.

    Creates a reasonable default instruction based on task name and fields.

    Args:
        task_config: Task configuration.

    Returns:
        Generated instruction text.
    """
    field_list = ", ".join(task_config.experiment_fields.keys())

    instruction = (
        f"Extract {task_config.name} experiments from scientific articles.\n\n"
        f"For each experiment, identify the following fields: {field_list}.\n\n"
        f"Return all experiments found in the text as a structured list."
    )

    return instruction


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

    Example:
        ```python
        config = TaskConfig(
            name="nanozymes",
            experiment_fields={...},
            compare_fields=["formula", "activity"],
            initial_instruction="Extract nanozyme experiments..."
        )

        ExperimentModel = create_experiment_model(config)
        OutputModel = create_output_model(config, ExperimentModel)
        Signature = create_signature(config, ExperimentModel, OutputModel)

        # Use signature with DSPy
        module = dspy.Predict(Signature)
        result = module(document_text="...")
        ```
    """
    # Get instruction from parameter, config, or generate default
    if instruction is None:
        try:
            instruction = task_config.get_instruction()
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Could not load instruction: {e}. Generating default.")
            instruction = generate_default_instruction(task_config)

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


def create_signature_with_instructions(
    task_config: TaskConfig,
    experiment_model: Type[BaseModel],
    output_model: Type[BaseModel],
    instruction_template: Optional[str] = None,
    **template_vars: Any,
) -> Type[dspy.Signature]:
    """Create signature with templated instruction.

    Allows using a template string for the instruction with variable substitution.

    Args:
        task_config: Task configuration.
        experiment_model: Generated experiment model.
        output_model: Generated output model.
        instruction_template: Template string with {variables}.
        **template_vars: Variables to substitute in template.

    Returns:
        DSPy Signature class.

    Example:
        ```python
        template = "Extract {task_name} data with fields: {fields}"
        Signature = create_signature_with_instructions(
            config, ExperimentModel, OutputModel,
            template,
            task_name="nanozymes",
            fields="formula, activity, km_value"
        )
        ```
    """
    if instruction_template:
        instruction = instruction_template.format(
            task_name=task_config.name,
            description=task_config.description,
            fields=", ".join(task_config.experiment_fields.keys()),
            **template_vars,
        )
    else:
        instruction = task_config.get_instruction()

    return create_signature(
        task_config,
        experiment_model,
        output_model,
        instruction=instruction,
    )


def update_signature_instruction(
    signature_class: Type[dspy.Signature],
    new_instruction: str,
) -> Type[dspy.Signature]:
    """Update the instruction of an existing signature class.

    Creates a new signature class with the same fields but new instruction.

    Args:
        signature_class: Original signature class.
        new_instruction: New instruction text.

    Returns:
        New signature class with updated instruction.
    """
    if not new_instruction or not new_instruction.strip():
        raise ValueError("New instruction cannot be empty")

    # Get existing fields
    fields = {}
    for name, field_info in signature_class.model_fields.items():
        fields[name] = (field_info.annotation, field_info)

    # Create new signature with same fields but new instruction
    class UpdatedSignature(dspy.Signature):
        __doc__ = new_instruction

    # Copy fields to new class
    for name, (field_type, field_info) in fields.items():
        setattr(UpdatedSignature, name, field_info)

    logger.info("Updated signature instruction")

    return UpdatedSignature
