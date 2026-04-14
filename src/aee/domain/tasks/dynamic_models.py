"""Dynamic model generation for tasks.

This module provides functions to dynamically generate Pydantic models
from TaskConfig, enabling flexible task definitions without hardcoded models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Type, Union

import pandas
from pydantic import BaseModel, Field, create_model
from pydantic.functional_validators import BeforeValidator

from aee.domain.entities import Experiment

from .config import FieldSpec, TaskConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _get_python_type(type_spec: Union[Type, str]) -> Type:
    """Convert type specification to Python type.

    Args:
        type_spec: Type specification (string or type object).

    Returns:
        Corresponding Python type.
    """
    if isinstance(type_spec, type):
        return type_spec

    if isinstance(type_spec, str):
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        if type_spec in type_mapping:
            return type_mapping[type_spec]

        # Handle Literal types from string representation
        if type_spec.startswith("Literal[") and type_spec.endswith("]"):
            # Extract choices from Literal[...] string
            choices_str = type_spec[8:-1]
            choices = [c.strip().strip("'\"") for c in choices_str.split(",")]
            return Literal[tuple(choices)]  # type: ignore

    raise ValueError(f"Cannot convert type specification: {type_spec}")


def _create_field_type(spec: FieldSpec) -> Type[Any]:
    """Create the type annotation for a field based on FieldSpec.

    Args:
        spec: Field specification.

    Returns:
        Type annotation for the field.
    """
    base_type: Type[Any] = _get_python_type(spec.type)

    # Handle Literal types with choices
    if spec.choices:
        base_type = Literal[tuple(spec.choices)]  # type: ignore[assignment]

    if spec.required:
        return base_type
    else:
        return Optional[base_type]  # type: ignore[return-value]


def _string_coerce_validator(v: Any) -> Any:
    """Coerce numeric values to strings.

    This handles the case where LLM returns JSON with numeric values
    (e.g., 12.0, 4.0) but the field is defined as str type in YAML.

    Args:
        v: Value to coerce.

    Returns:
        String representation of the value.
    """
    if v is None:
        return v
    if isinstance(v, bool):
        # Booleans should be converted to "true"/"false" strings
        return str(v).lower()
    if isinstance(v, float):
        # Clean up float representation: 12.0 -> "12", 1.5e-07 -> "1.5e-07"
        if v.is_integer():
            return str(int(v))
        return str(v)
    if isinstance(v, int):
        return str(v)
    # Already a string or other type - convert to string
    return str(v)


# Reusable validator for str-typed fields
StringCoerce = BeforeValidator(_string_coerce_validator)


def create_experiment_model(
    task_config: TaskConfig,
    base_class: Optional[Type[BaseModel]] = None,
) -> Type[BaseModel]:
    """Dynamically create a Pydantic model for experiments.

    This function generates a Pydantic model class based on the field
    specifications in TaskConfig. The generated model includes all fields
    with appropriate types, validation, and descriptions.

    For fields defined as str type, automatic conversion from numeric values
    (int, float) to strings is applied to handle LLM JSON responses.

    Args:
        task_config: Task configuration with field specifications.
        base_class: Optional base class (default: Experiment).

    Returns:
        Dynamically generated Pydantic model class.

    Example:
        ```python
        config = TaskConfig(
            name="nanozymes",
            experiment_fields={
                "formula": FieldSpec(type=str, description="Chemical formula"),
                "activity": FieldSpec(
                    type=str,
                    description="Catalytic activity",
                    choices=["peroxidase", "oxidase", "catalase"]
                ),
                "km_value": FieldSpec(type=float, description="Michaelis constant", required=False),
            },
            compare_fields=["formula", "activity"],
        )

        ExperimentModel = create_experiment_model(config)
        experiment = ExperimentModel(formula="Fe3O4", activity="peroxidase")
        ```
    """
    if base_class is None:
        base_class = Experiment

    fields: Dict[str, Any] = {}

    for field_name, spec in task_config.experiment_fields.items():
        field_type = _create_field_type(spec)
        pydantic_field = spec.to_pydantic_field()

        # For str-typed fields, add coercion validator to handle numeric JSON values from LLM
        if spec.type is str or spec.type == "str" or spec.type == "string":
            # Use Annotated type with BeforeValidator for automatic coercion
            from typing import Annotated
            field_type = Annotated[field_type, StringCoerce]  # type: ignore[assignment]

        fields[field_name] = (field_type, pydantic_field)  # type: ignore[assignment]

    # Create the dynamic model with fields
    model = create_model("Experiment", __base__=base_class, **fields)  # type: ignore[arg-type]

    logger.info(f"Created experiment model with {len(fields)} fields")

    return model


def create_output_model(
    task_config: TaskConfig,
    experiment_model: Type[BaseModel],
) -> Type[BaseModel]:
    """Dynamically create a Pydantic model for extraction output.

    Creates a wrapper model containing a list of experiments.

    Args:
        task_config: Task configuration.
        experiment_model: Generated experiment model.

    Returns:
        Dynamically generated output model class.

    Example:
        ```python
        OutputModel = create_output_model(config, ExperimentModel)
        output = OutputModel(experiments=[exp1, exp2])
        ```
    """
    from typing import List

    fields: Dict[str, Any] = {
        "experiments": (List[experiment_model], Field(default_factory=list))  # type: ignore[arg-type,valid-type]
    }

    model = create_model("ExtractionOutput", __base__=BaseModel, **fields)

    logger.info("Created output model 'ExtractionOutput'")

    return model


def _convert_value_to_type(
    value: Any,
    spec: FieldSpec,
) -> Any:
    """Convert a value to the specified type.

    Args:
        value: Value to convert.
        spec: Field specification with target type.

    Returns:
        Converted value or default on failure.
    """
    try:
        if spec.type is float:
            return float(value)
        elif spec.type is int:
            return int(value)
        elif spec.type is bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        else:
            return str(value)
    except (ValueError, TypeError):
        logger.debug(f"Failed to convert '{value}' to {spec.type}")
        return spec.default if not spec.required else None


def _extract_field_value(
    row: pandas.Series,
    field_name: str,
    spec: FieldSpec,
    row_converter: Any,
) -> Any:
    """Extract and convert a single field value from a row.

    Args:
        row: Pandas Series with data.
        field_name: Name of the field.
        spec: Field specification.
        row_converter: RowConverterConfig for column name mapping.

    Returns:
        Extracted and converted value.
    """
    import pandas

    # Get possible column names for this field
    column_names = row_converter.get_column_names(field_name)

    # Try each column name in order
    value = None
    for col in column_names:
        if col in row.index:
            val = row.get(col)
            if not (pandas.isna(val) or val == ""):
                value = val
                break

    # Handle None/missing values
    if value is None or (isinstance(value, float) and pandas.isna(value)):
        if spec.required:
            return None  # Will cause validation to fail
        return spec.default

    # Convert to appropriate type
    return _convert_value_to_type(value, spec)


def create_row_converter(
    task_config: TaskConfig,
    experiment_model: Type[BaseModel],
):
    """Dynamically create a row-to-experiment converter function.

    Creates a function that converts a pandas Series to an experiment model
    instance based on the row_converter configuration.

    Args:
        task_config: Task configuration with row_converter mapping.
        experiment_model: Generated experiment model class.

    Returns:
        Function that converts pandas Series to experiment model.

    Example:
        ```python
        converter = create_row_converter(config, ExperimentModel)
        experiment = converter(row)  # row is pandas Series
        ```
    """

    def converter(row: pandas.Series):
        """Convert pandas Series to experiment model.

        Args:
            row: Pandas Series containing experiment data.

        Returns:
            Experiment model instance or None if required fields missing.
        """
        # Extract all field values
        field_values = {}

        for field_name, spec in task_config.experiment_fields.items():
            value = _extract_field_value(
                row=row,
                field_name=field_name,
                spec=spec,
                row_converter=task_config.row_converter,
            )

            # Check if required field is missing
            if value is None and spec.required:
                logger.debug(f"Missing required field '{field_name}'")
                return None

            field_values[field_name] = value

        # Create experiment instance
        try:
            return experiment_model(**field_values)
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return None

    return converter  # type: ignore[return-value]


def create_all_models(
    task_config: TaskConfig,
    base_class: Optional[Type[BaseModel]] = None,
) -> tuple[Type[BaseModel], Type[BaseModel]]:
    """Create both experiment and output models from TaskConfig.

    Convenience function that creates both models needed for a task.

    Args:
        task_config: Task configuration.
        base_class: Optional base class for experiment model.

    Returns:
        Tuple of (experiment_model, output_model).
    """
    experiment_model = create_experiment_model(
        task_config,
        base_class=base_class or task_config.base_class,
    )

    output_model = create_output_model(task_config, experiment_model)

    return experiment_model, output_model
