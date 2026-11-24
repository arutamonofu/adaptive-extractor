"""Dynamic model generation for tasks.

This module provides functions to dynamically generate Pydantic models
from TaskConfig, enabling flexible task definitions without hardcoded models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Type, Union

import pandas
from pydantic import AliasChoices, BaseModel, Field, create_model
from pydantic.functional_validators import BeforeValidator

from ae.core.evaluation.entities import Record

from .config import FieldPatternSpec, FieldSpec, SchemaConfig
from .tools import ExtractionToolSpec, normalize_tool_spec
from .validators import FieldValidatorSpec

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

    if spec.required or spec.default is not None:
        return base_type
    else:
        return Optional[base_type]  # type: ignore[return-value]


def _string_coerce_validator(v: Any) -> Any:
    """Coerce numeric values to strings and trim whitespace.

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
    if isinstance(v, str):
        return v.strip()
    # Already other type - convert to string
    return str(v)


# Reusable validator for str-typed fields
StringCoerce = BeforeValidator(_string_coerce_validator)


def create_record_model(
    schema_config: SchemaConfig,
    base_class: Optional[Type[BaseModel]] = None,
) -> Type[BaseModel]:
    """Dynamically create a Pydantic model for records.

    This function generates a Pydantic model class based on the field
    specifications in SchemaConfig. The generated model includes all fields
    with appropriate types, validation, and descriptions.

    For fields defined as str type, automatic conversion from numeric values
    (int, float) to strings is applied to handle LLM JSON responses.

    Args:
        schema_config: Schema configuration with field specifications.
        base_class: Optional base class (default: Record).

    Returns:
        Dynamically generated Pydantic model class.
    """
    if base_class is None:
        base_class = Record

    fields: Dict[str, Any] = {}

    for field_name, spec in schema_config.fields.items():
        field_type = _create_field_type(spec)
        pydantic_field = spec.to_pydantic_field()

        # For str-typed fields, add coercion validator to handle numeric JSON values from LLM
        metadata_items: List[Any] = []
        if spec.type is str or spec.type == "str" or spec.type == "string":
            if spec.default is not None:
                def _make_default_coerce(def_val: Any):
                    def _coerce(v: Any) -> Any:
                        if v is None:
                            return def_val
                        coerced = _string_coerce_validator(v)
                        if isinstance(coerced, str) and not coerced:
                            return def_val
                        return coerced
                    return _coerce
                metadata_items.append(BeforeValidator(_make_default_coerce(spec.default)))
            else:
                metadata_items.append(StringCoerce)

        # Attach declarative field validators and pattern metadata (excluded from JSON Schema)
        effective_validators = spec.get_effective_validators()
        if effective_validators:
            metadata_items.extend(effective_validators)
        if spec.pattern:
            metadata_items.append(FieldPatternSpec(pattern=spec.pattern))

        if metadata_items:
            field_type = Annotated[(field_type, *metadata_items)]  # type: ignore[assignment]

        fields[field_name] = (field_type, pydantic_field)  # type: ignore[assignment]

    # Create the dynamic model with fields
    model = create_model("Record", __base__=base_class, **fields)  # type: ignore[arg-type]

    logger.info(f"Created record model with {len(fields)} fields")

    return model


# Backward-compatible alias
create_experiment_model = create_record_model


def create_output_model(
    schema_config: SchemaConfig,
    record_model: Type[BaseModel],
) -> Type[BaseModel]:
    """Dynamically create a Pydantic model for extraction output.

    Creates a wrapper model containing a list of records.

    Args:
        schema_config: Schema configuration.
        record_model: Generated record model.

    Returns:
        Dynamically generated output model class.
    """
    from typing import List
    from pydantic import model_validator

    fields: Dict[str, Any] = {
        "records": (
            List[record_model],  # type: ignore[valid-type]
            Field(default_factory=list, validation_alias=AliasChoices("records", "experiments")),
        )
    }

    class BaseOutput(BaseModel):
        @property
        def experiments(self) -> List[Any]:
            return getattr(self, "records", [])

        @model_validator(mode="before")
        @classmethod
        def normalize_records_input(cls, data: Any) -> Any:
            """Unwrap list representations returned by structured output LLM."""
            if isinstance(data, list):
                if len(data) == 1 and isinstance(data[0], dict) and "records" in data[0]:
                    return data[0]
                if all(isinstance(x, dict) and "records" not in x for x in data):
                    return {"records": data}
            return data

    model = create_model("ExtractionOutput", __base__=BaseOutput, **fields)
    model.__ae_extraction_tools__ = tuple(
        normalize_tool_spec(t) for t in (getattr(schema_config, "tools", None) or ())
    )

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
    row: Any,
    field_name: str,
    spec: FieldSpec,
    row_converter: Any,
) -> Any:
    """Extract and convert a single field value from a row.

    Args:
        row: Pandas Series or mapping with data.
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
        has_col = (
            (col in row.index)
            if hasattr(row, "index") and not isinstance(row, (dict, list, tuple))
            else (col in row)
        )
        if has_col:
            val = row.get(col) if hasattr(row, "get") else row[col]
            if not pandas.isna(val) and not (isinstance(val, str) and val.strip() == ""):
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
    schema_config: SchemaConfig,
    record_model: Type[BaseModel],
):
    """Dynamically create a row-to-record converter function.

    Creates a function that converts a pandas Series to a record model
    instance based on the row_converter configuration.

    Args:
        schema_config: Schema configuration with row_converter mapping.
        record_model: Generated record model class.

    Returns:
        Function that converts pandas Series to record model.
    """

    def converter(row: pandas.Series):
        """Convert pandas Series to record model.

        Args:
            row: Pandas Series containing record data.

        Returns:
            Record model instance or None if required fields missing.
        """
        # Extract all field values
        field_values = {}

        for field_name, spec in schema_config.fields.items():
            value = _extract_field_value(
                row=row,
                field_name=field_name,
                spec=spec,
                row_converter=schema_config.row_converter,
            )

            # Check if required field is missing
            if value is None and spec.required:
                logger.debug(f"Missing required field '{field_name}'")
                return None

            field_values[field_name] = value

        # Create record instance
        try:
            return record_model(**field_values)
        except Exception as e:
            logger.error(f"Failed to create record: {e}")
            return None

    return converter  # type: ignore[return-value]


def create_all_models(
    schema_config: SchemaConfig,
    base_class: Optional[Type[BaseModel]] = None,
) -> tuple[Type[BaseModel], Type[BaseModel]]:
    """Create both record and output models from SchemaConfig.

    Convenience function that creates both models needed for a task.

    Args:
        schema_config: Schema configuration.
        base_class: Optional base class for record model.

    Returns:
        Tuple of (record_model, output_model).
    """
    record_model = create_record_model(
        schema_config,
        base_class=base_class or schema_config.base_class,
    )

    output_model = create_output_model(schema_config, record_model)

    return record_model, output_model


def _resolve_record_model(output_model: Type[BaseModel]) -> Type[BaseModel]:
    """Return the underlying record BaseModel from a collection output model, or output_model itself."""
    if hasattr(output_model, "model_fields") and "records" in output_model.model_fields:
        annotation = output_model.model_fields["records"].annotation
        args = getattr(annotation, "__args__", None)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    return output_model


def extract_field_validators(output_model: Type[BaseModel]) -> Dict[str, List[FieldValidatorSpec]]:
    """Extract declarative FieldValidatorSpec metadata attached to each record field."""
    if not output_model or not hasattr(output_model, "model_fields"):
        return {}

    target_model = _resolve_record_model(output_model)
    validators_by_field: Dict[str, List[FieldValidatorSpec]] = {}
    for name, field_info in target_model.model_fields.items():
        meta = getattr(field_info, "metadata", None) or ()
        specs = [item for item in meta if isinstance(item, FieldValidatorSpec)]
        if specs:
            validators_by_field[name] = specs
    return validators_by_field


def extract_schema_tools(output_model: Optional[Type[BaseModel]]) -> List[ExtractionToolSpec]:
    """Extract declarative ExtractionToolSpec entries attached to an ExtractionOutput model."""
    if not output_model:
        return []
    raw_tools = getattr(output_model, "__ae_extraction_tools__", None) or ()
    return [normalize_tool_spec(t) for t in raw_tools]


def extract_schema_field_constraints(output_model: Type[BaseModel]) -> Dict[str, Dict[str, Any]]:
    """Extract normalized JSON-Schema constraints for each field of an output or record model.

    Returns:
        Mapping from field name to dict containing optional keys:
        'required', 'default', 'description', 'enum', 'pattern', 'minimum', 'maximum', 'type'.
    """
    if not output_model or not hasattr(output_model, "model_fields"):
        return {}

    target_model = _resolve_record_model(output_model)
    try:
        json_schema = target_model.model_json_schema()
    except Exception:
        json_schema = {}
    properties = json_schema.get("properties", {}) if isinstance(json_schema, dict) else {}

    from pydantic_core import PydanticUndefined

    constraints: Dict[str, Dict[str, Any]] = {}
    for name, field in target_model.model_fields.items():
        prop = properties.get(name, {}) if isinstance(properties, dict) else {}
        # Flatten anyOf (e.g. Optional[Literal[...]] or Optional[str]) to inspect inner constraints
        candidates = [prop]
        if isinstance(prop.get("anyOf"), list):
            candidates.extend(sub for sub in prop["anyOf"] if isinstance(sub, dict) and sub.get("type") != "null")

        info: Dict[str, Any] = {
            "required": field.is_required(),
        }
        if not field.is_required() and field.default is not PydanticUndefined:
            info["default"] = field.default
        if field.description and field.description.strip():
            info["description"] = field.description.strip()

        for cand in candidates:
            if "enum" in cand and "enum" not in info:
                info["enum"] = list(cand["enum"])
            if "pattern" in cand and "pattern" not in info:
                info["pattern"] = cand["pattern"]
            if "minimum" in cand and "minimum" not in info:
                info["minimum"] = cand["minimum"]
            if "maximum" in cand and "maximum" not in info:
                info["maximum"] = cand["maximum"]
            if "type" in cand and cand["type"] != "null" and "type" not in info:
                info["type"] = cand["type"]

        field_meta = getattr(field, "metadata", None) or ()
        val_names = [m.name for m in field_meta if isinstance(m, FieldValidatorSpec)]
        if val_names:
            info["validators"] = val_names
        for m in field_meta:
            if isinstance(m, FieldPatternSpec) and "pattern" not in info:
                info["pattern"] = m.pattern

        constraints[name] = info

    return constraints
