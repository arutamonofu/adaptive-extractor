"""YAML loader for task configurations.

This module provides functions to load TaskConfig from YAML manifest files,
enabling declarative task definitions without Python code changes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml

from aee.domain.entities import Experiment

from .config import FieldSpec, RowConverterConfig, TaskConfig
from .dynamic_models import create_all_models, create_row_converter
from .signature import create_signature

logger = logging.getLogger(__name__)


def _find_project_root(start_path: Path) -> Path:
    """Find project root by looking for pyproject.toml.

    Args:
        start_path: Starting path to search from.

    Returns:
        Path to project root directory.
    """
    current = start_path.resolve()

    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback to start path if not found
    return start_path.parent.parent.parent  # Fallback to reasonable default


def _parse_field_spec(field_name: str, field_data: Dict[str, Any]) -> FieldSpec:
    """Parse FieldSpec from YAML data.

    Args:
        field_name: Name of the field.
        field_data: Dictionary with field specification.

    Returns:
        FieldSpec instance.

    Raises:
        ValueError: If field specification is invalid.
    """
    # Get type - can be string or Python type
    type_spec = field_data.get("type", "str")

    # Handle string type specifications
    if isinstance(type_spec, str):
        # Check for Literal type with choices
        if "choices" in field_data:
            choices = field_data["choices"]
            # Create Literal type string representation
            choices_str = ", ".join(f"'{c}'" for c in choices)
            type_spec = f"Literal[{choices_str}]"

    # Map string type names to Python types
    type_mapping: Dict[str, Type] = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
    }

    if isinstance(type_spec, str):
        type_spec = type_mapping.get(type_spec.lower(), str)

    return FieldSpec(
        type=type_spec,
        description=field_data.get("description", ""),
        required=field_data.get("required", True),
        default=field_data.get("default", None),
        choices=field_data.get("choices", None),
        min_value=field_data.get("min_value", None),
        max_value=field_data.get("max_value", None),
        pattern=field_data.get("pattern", None),
        alt_names=field_data.get("alt_names", []),
    )


def _parse_row_converter(
    converter_data: Dict[str, Any],
) -> RowConverterConfig:
    """Parse RowConverterConfig from YAML data.

    Args:
        converter_data: Dictionary with row converter configuration.

    Returns:
        RowConverterConfig instance.
    """
    mapping = {}

    if isinstance(converter_data, dict):
        for field_name, columns in converter_data.items():
            if isinstance(columns, list):
                mapping[field_name] = columns
            else:
                mapping[field_name] = [columns]

    return RowConverterConfig(mapping=mapping)


def load_task_from_yaml(
    yaml_path: str | Path,
    base_class: Optional[Type[Experiment]] = None,
) -> TaskConfig:
    """Load TaskConfig from a YAML file.

    Reads and parses a YAML manifest file into a TaskConfig object.
    Validates the configuration and resolves relative paths.

    Args:
        yaml_path: Path to the YAML manifest file.
        base_class: Optional base class for experiment model (default: Experiment).

    Returns:
        TaskConfig instance.

    Raises:
        FileNotFoundError: If YAML file not found.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If configuration is invalid.

    Example:
        ```python
        config = load_task_from_yaml("src/aee/domain/tasks/nanozymes/task.yaml")
        print(f"Task: {config.name}, Fields: {len(config.experiment_fields)}")
        ```
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Task YAML file not found: {yaml_path}")

    logger.info(f"Loading task configuration from {yaml_path}")

    # Read and parse YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if not isinstance(yaml_data, dict):
        raise ValueError(f"Invalid YAML format: expected dictionary, got {type(yaml_data)}")

    # Parse experiment fields
    experiment_fields: Dict[str, FieldSpec] = {}
    fields_data = yaml_data.get("fields", {})

    for field_name, field_spec in fields_data.items():
        experiment_fields[field_name] = _parse_field_spec(field_name, field_spec)

    # Parse row converter
    row_converter = RowConverterConfig()
    if "row_converter" in yaml_data:
        row_converter = _parse_row_converter(yaml_data["row_converter"])

    # Handle instruction - either inline or from file
    initial_instruction = yaml_data.get("initial_instruction", None)
    instruction_file_path = yaml_data.get("instruction_file", None)

    # Resolve relative instruction file path
    if instruction_file_path:
        instruction_path = Path(instruction_file_path)
        if not instruction_path.is_absolute():
            # Try to resolve relative to project root (find pyproject.toml)
            project_root = _find_project_root(yaml_path)
            instruction_path = (project_root / instruction_path).resolve()
        instruction_file_path = str(instruction_path)

    # Create TaskConfig
    config = TaskConfig(
        name=yaml_data.get("name", ""),
        description=yaml_data.get("description", ""),
        experiment_fields=experiment_fields,
        compare_fields=yaml_data.get("compare_fields", []),
        float_tolerance=yaml_data.get("float_tolerance", 0.05),
        initial_instruction=initial_instruction,
        instruction_file_path=instruction_file_path,
        row_converter=row_converter,
        output_model_name=yaml_data.get("output_model_name", "ExtractionOutput"),
        experiment_model_name=yaml_data.get("experiment_model_name", "Experiment"),
        base_class=base_class,
        tags=yaml_data.get("tags", []),
        version=yaml_data.get("version", "1.0.0"),
        metadata=yaml_data.get("metadata", {}),
    )

    logger.info(f"Loaded task configuration: {config.name}")

    return config


def load_task_with_models(
    yaml_path: str | Path,
    base_class: Optional[Type[Experiment]] = None,
) -> tuple[TaskConfig, Type, Type]:
    """Load TaskConfig and generate models from YAML.

    Convenience function that loads the configuration and creates
    the experiment and output models.

    Args:
        yaml_path: Path to YAML manifest file.
        base_class: Optional base class for experiment model.

    Returns:
        Tuple of (TaskConfig, experiment_model, output_model).

    Example:
        ```python
        config, ExperimentModel, OutputModel = load_task_with_models(
            "src/aee/domain/tasks/nanozymes/task.yaml"
        )
        ```
    """
    config = load_task_from_yaml(yaml_path, base_class=base_class)

    experiment_model, output_model = create_all_models(config)

    return config, experiment_model, output_model


def load_task_complete(
    yaml_path: str | Path,
    base_class: Optional[Type[Experiment]] = None,
) -> dict[str, Any]:
    """Load complete task setup from YAML.

    Loads configuration, generates models, signature, and converter.
    Returns a dictionary with all components needed for a task.

    Args:
        yaml_path: Path to YAML manifest file.
        base_class: Optional base class for experiment model.

    Returns:
        Dictionary with keys:
        - config: TaskConfig instance
        - experiment_model: Generated Pydantic model
        - output_model: Generated output model
        - signature: Generated DSPy signature
        - row_converter: Generated converter function

    Example:
        ```python
        task = load_task_complete("tasks/nanozymes/task.yaml")
        config = task["config"]
        ExperimentModel = task["experiment_model"]
        Signature = task["signature"]
        converter = task["row_converter"]
        ```
    """
    config, experiment_model, output_model = load_task_with_models(
        yaml_path, base_class=base_class
    )

    signature = create_signature(config, experiment_model, output_model)
    row_converter = create_row_converter(config, experiment_model)

    return {
        "config": config,
        "experiment_model": experiment_model,
        "output_model": output_model,
        "signature": signature,
        "row_converter": row_converter,
    }


def save_task_to_yaml(
    config: TaskConfig,
    output_path: str | Path,
    include_metadata: bool = True,
) -> Path:
    """Save TaskConfig to a YAML file.

    Serializes a TaskConfig object to a YAML manifest file.

    Args:
        config: TaskConfig to save.
        output_path: Path for output YAML file.
        include_metadata: Whether to include metadata fields.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)

    # Convert config to dictionary
    yaml_data = {
        "name": config.name,
        "description": config.description,
        "compare_fields": config.compare_fields,
        "float_tolerance": config.float_tolerance,
    }

    # Add fields
    fields_data = {}
    for field_name, spec in config.experiment_fields.items():
        field_dict = {
            "type": spec.type.__name__ if hasattr(spec.type, "__name__") else str(spec.type),
            "description": spec.description,
            "required": spec.required,
        }

        if spec.default is not None:
            field_dict["default"] = spec.default

        if spec.choices:
            field_dict["choices"] = spec.choices

        if spec.alt_names:
            field_dict["alt_names"] = spec.alt_names

        fields_data[field_name] = field_dict

    yaml_data["fields"] = fields_data

    # Add row converter mapping
    if config.row_converter.mapping:
        yaml_data["row_converter"] = config.row_converter.mapping

    # Add instruction
    if config.initial_instruction:
        yaml_data["initial_instruction"] = config.initial_instruction
    elif config.instruction_file_path:
        yaml_data["instruction_file"] = config.instruction_file_path

    # Add optional metadata
    if include_metadata:
        yaml_data["version"] = config.version
        yaml_data["tags"] = config.tags
        if config.metadata:
            yaml_data["metadata"] = config.metadata

    # Write YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved task configuration to {output_path}")

    return output_path
