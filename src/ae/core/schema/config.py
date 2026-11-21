"""Schema configuration dataclasses for dynamic schema generation.

This module provides the core data structures for defining schemas declaratively
using SchemaConfig and FieldSpec, enabling dynamic model and signature generation.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel
from pydantic import Field as PydanticField

from .tools import (
    ExtractionToolSpec,
    normalize_tool_spec,
    validate_tool_config,
)
from .validators import (
    FieldValidatorSpec,
    has_field_validator,
    normalize_validator_spec,
    validate_validator_config,
)


@dataclass
class FieldSpec:
    """Specification for a single experiment field.

    Defines the type, description, and validation rules for a field
    that will be extracted by the task.

    Attributes:
        type: Python type for the field (str, float, int, Literal, etc.)
        description: Human-readable description of the field
        required: Whether the field is required (default True)
        default: Default value for optional fields (default None)
        choices: List of valid choices for Literal fields
        min_value: Minimum value for numeric fields
        max_value: Maximum value for numeric fields
        pattern: Regex pattern for string fields
        format: Optional semantic/syntactic format validator name (e.g., "smiles")
        validators: Optional list of declarative field validators
    """

    type: Union[Type[str], Type[int], Type[float], Type]
    description: str = ""
    required: bool = True
    default: Any = None
    choices: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    field_name: Optional[str] = None
    format: Optional[str] = None
    validators: List[Union[str, Dict[str, Any], FieldValidatorSpec]] = field(default_factory=list)

    def __post_init__(self):
        """Validate field specification after initialization."""
        if self.choices and self.type is not str:
            raise ValueError("choices can only be used with str type")

        if self.pattern and self.type is not str:
            raise ValueError("pattern can only be used with str type")

        if self.min_value is not None and self.type not in (int, float):
            raise ValueError("min_value can only be used with numeric types")

        if self.max_value is not None and self.type not in (int, float):
            raise ValueError("max_value can only be used with numeric types")

        if self.format is not None:
            if not isinstance(self.format, str) or not self.format.strip():
                raise ValueError("format must be a non-empty string")
            self.format = self.format.strip()

        if self.validators is None:
            self.validators = []
        elif not isinstance(self.validators, list):
            raise ValueError("validators must be a list")
        else:
            self.validators = [normalize_validator_spec(v) for v in self.validators]

        if not self.required and self.default is None and self.choices is None:
            # Optional fields without default should have None as default
            self.default = None

    def get_effective_validators(self) -> List[FieldValidatorSpec]:
        """Return normalized list of validators implied by ``format`` and ``validators``."""
        specs: List[FieldValidatorSpec] = [normalize_validator_spec(v) for v in self.validators]
        if self.format:
            existing_names = {s.name for s in specs}
            if self.format not in existing_names:
                specs.insert(0, FieldValidatorSpec(name=self.format))
        return specs

    def to_pydantic_field(self) -> "PydanticField":  # type: ignore[valid-type]
        """Convert FieldSpec to Pydantic Field.

        Returns:
            Pydantic Field with appropriate constraints.
        """
        desc = (self.description or "").strip()
        field_kwargs: Dict[str, Any] = {}
        if desc:
            field_kwargs["description"] = desc

        if not self.required:
            field_kwargs["default"] = self.default

        if self.min_value is not None:
            field_kwargs["ge"] = self.min_value

        if self.max_value is not None:
            field_kwargs["le"] = self.max_value

        if self.pattern:
            field_kwargs["pattern"] = self.pattern

        return PydanticField(**field_kwargs)


@dataclass
class RowConverterConfig:
    """Configuration for row-to-experiment conversion.

    Maps CSV column names to field names with fallback alternatives.

    Attributes:
        mapping: Dictionary mapping field names to list of possible CSV column names
    """

    mapping: Dict[str, List[str]] = field(default_factory=dict)

    def get_column_names(self, field_name: str) -> List[str]:
        """Get possible column names for a field.

        Args:
            field_name: Name of the field.

        Returns:
            List of possible CSV column names in priority order.
        """
        return self.mapping.get(field_name, [field_name])


@dataclass
class SchemaConfig:
    """Complete configuration for an extraction schema.

    This is the single source of truth for schema definition. All models,
    signatures, and converters are generated from this configuration.

    Attributes:
        name: Unique schema identifier (e.g., "nanozymes", "catalysts")
        fields: Dictionary of field specifications
        compare_fields: List of field names to compare during evaluation
        float_tolerance: Tolerance for float comparisons (0.0 to 1.0)
        instruction_file: Path to instruction file for DSPy signature (relative to project root)
        row_converter: Configuration for CSV row conversion
        base_class: Base class for record model (e.g., Record)
    """

    name: str
    fields: Dict[str, FieldSpec] = field(default_factory=dict)
    compare_fields: List[str] = field(default_factory=list)
    float_tolerance: float = 0.05
    instruction_file: Optional[str] = None
    row_converter: RowConverterConfig = field(default_factory=RowConverterConfig)
    base_class: Optional[Type[BaseModel]] = None
    judge_field_descriptions: Optional[Dict[str, str]] = None
    tools: List[Union[str, Dict[str, Any], ExtractionToolSpec]] = field(default_factory=list)

    def __init__(
        self,
        name: str,
        fields: Optional[Dict[str, FieldSpec]] = None,
        compare_fields: Optional[List[str]] = None,
        float_tolerance: float = 0.05,
        instruction_file: Optional[str] = None,
        row_converter: Optional[RowConverterConfig] = None,
        base_class: Optional[Type[BaseModel]] = None,
        judge_field_descriptions: Optional[Dict[str, str]] = None,
        experiment_fields: Optional[Dict[str, FieldSpec]] = None,
        tools: Optional[List[Union[str, Dict[str, Any], ExtractionToolSpec]]] = None,
    ):
        self.name = name
        self.fields = fields if fields is not None else (experiment_fields or {})
        self.compare_fields = compare_fields or []
        self.float_tolerance = float_tolerance
        self.instruction_file = instruction_file
        self.row_converter = row_converter or RowConverterConfig()
        self.base_class = base_class
        self.judge_field_descriptions = judge_field_descriptions
        self.tools = list(tools) if tools is not None else []
        self.__post_init__()

    @property
    def experiment_fields(self) -> Dict[str, FieldSpec]:
        """Backward-compatible alias for fields."""
        return self.fields

    def __post_init__(self):
        """Validate schema configuration after initialization."""
        for name, spec in self.fields.items():
            if hasattr(spec, "field_name") and spec.field_name is None:
                spec.field_name = name

        if not self.name or not isinstance(self.name, str):
            raise ValueError("Schema name must be a non-empty string")

        if not self.fields:
            raise ValueError("Schema must have at least one field")

        if not self.compare_fields:
            raise ValueError("Schema must have at least one compare field")

        if not isinstance(self.compare_fields, list):
            raise ValueError("compare_fields must be a list")

        if not 0 <= self.float_tolerance <= 1:
            raise ValueError("float_tolerance must be between 0 and 1")

        # Validate compare_fields exist in fields
        field_names = set(self.fields.keys())
        invalid_fields = [f for f in self.compare_fields if f not in field_names]
        if invalid_fields:
            raise ValueError(
                f"compare_fields {invalid_fields} not found in fields. "
                f"Available fields: {sorted(field_names)}"
            )

        if self.tools is None:
            self.tools = []
        elif not isinstance(self.tools, list):
            raise ValueError("tools must be a list")
        else:
            self.tools = [normalize_tool_spec(t) for t in self.tools]

    def get_instruction(self) -> str:
        """Get the instruction text from instruction file.

        Returns:
            Instruction text.

        Raises:
            ValueError: If no instruction file is specified.
            FileNotFoundError: If instruction file not found.
        """
        if not self.instruction_file:
            raise ValueError(
                f"No instruction file specified for schema '{self.name}'."
            )

        from pathlib import Path

        instruction_path = Path(self.instruction_file)
        if not instruction_path.exists():
            raise FileNotFoundError(
                f"Instruction file not found: {self.instruction_file}. "
                "Ensure path is relative to project root"
            )
        return instruction_path.read_text(encoding="utf-8")

    def get_instruction_hash(self) -> str:
        """Get SHA256 hash of the instruction.

        Returns:
            First 12 characters of SHA256 hash.
        """
        instruction = self.get_instruction()
        return hashlib.sha256(instruction.encode()).hexdigest()[:12]

    def get_schema_hash(self) -> str:
        """Compute a structural compatibility hash of the schema fields and types.

        Only structural properties (field names, types, and required flags) affect
        agent compatibility; runtime constraints and validators (``format``,
        ``validators``, ``choices``, ``default``, ``min_value``, ``max_value``,
        ``pattern``) are excluded so refined schemas and validator updates do not
        invalidate previously saved agents.

        Returns:
            SHA256 hash string (first 12 characters).
        """
        import json

        canonical_fields = {}
        for name, spec in sorted(self.fields.items()):
            canonical_fields[name] = {
                "type": getattr(spec.type, "__name__", str(spec.type)),
                "required": spec.required,
                "default": None,
                "choices": None,
                "min_value": None,
                "max_value": None,
                "pattern": None,
            }

        canonical_json = json.dumps(canonical_fields, sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]

    def get_validation_hash(self) -> str:
        """Compute a hash of runtime validation rules and field constraints.

        Captures ``default``, ``choices``, ``min_value``, ``max_value``,
        ``pattern``, ``format``, and ``validators`` separately from structural
        agent compatibility.

        Returns:
            SHA256 hash string (first 12 characters).
        """
        import json

        canonical_constraints = {}
        for name, spec in sorted(self.fields.items()):
            entry: Dict[str, Any] = {
                "default": spec.default,
                "choices": spec.choices,
                "min_value": spec.min_value,
                "max_value": spec.max_value,
                "pattern": spec.pattern,
            }
            if spec.format is not None:
                entry["format"] = spec.format
            if spec.validators:
                entry["validators"] = [
                    normalize_validator_spec(v).to_canonical_dict()
                    for v in spec.validators
                ]
            canonical_constraints[name] = entry

        if self.tools:
            canonical_constraints["__tools__"] = [
                normalize_tool_spec(t).to_canonical_dict() for t in self.tools
            ]

        canonical_json = json.dumps(canonical_constraints, sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]

    def get_required_fields(self) -> List[str]:
        """Get list of required field names.

        Returns:
            List of field names that are required.
        """
        return [
            name for name, spec in self.fields.items()
            if spec.required
        ]

    def get_optional_fields(self) -> List[str]:
        """Get list of optional field names.

        Returns:
            List of field names that are optional.
        """
        return [
            name for name, spec in self.fields.items()
            if not spec.required
        ]

    @property
    def field_descriptions(self) -> Dict[str, str]:
        """Get field descriptions for semantic judge.

        Returns:
            Dictionary mapping field names to their descriptions.
            Only includes fields that have a description.
        """
        if self.judge_field_descriptions is not None:
            return self.judge_field_descriptions
        return {
            name: spec.description
            for name, spec in self.fields.items()
            if spec.description
        }

    @property
    def field_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get explicit field constraints (choices, pattern, numeric bounds) for evaluation."""
        constraints: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.fields.items():
            c: Dict[str, Any] = {}
            if spec.choices:
                c["choices"] = list(spec.choices)
            if spec.pattern:
                c["pattern"] = spec.pattern
            if spec.min_value is not None:
                c["min_value"] = spec.min_value
            if spec.max_value is not None:
                c["max_value"] = spec.max_value
            if c:
                constraints[name] = c
        return constraints

    @property
    def field_validators(self) -> Dict[str, List[FieldValidatorSpec]]:
        """Get normalized declarative validators configured for each field."""
        validators: Dict[str, List[FieldValidatorSpec]] = {}
        for name, spec in self.fields.items():
            effective = spec.get_effective_validators()
            if effective:
                validators[name] = effective
        return validators

    def get_field_choices(self, field_name: str) -> Optional[List[str]]:
        """Get choices for a field if it's a Literal type.

        Args:
            field_name: Name of the field.

        Returns:
            List of choices or None if not a Literal field.
        """
        if field_name not in self.fields:
            return None
        return self.fields[field_name].choices

    def to_dict(self) -> Dict[str, Any]:
        """Convert SchemaConfig to dictionary.

        Returns:
            Dictionary representation of the config.
        """
        from dataclasses import asdict
        return asdict(self)

    def validate(self) -> list[str]:
        """Validate SchemaConfig completeness and consistency.

        Performs comprehensive validation of the schema configuration,
        checking all fields, compare_fields, and instruction.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Validate name
        if not self.name or not isinstance(self.name, str):
            errors.append("Schema name must be a non-empty string")

        # Validate fields
        if not self.fields:
            errors.append("Schema must have at least one field")
        else:
            for field_name, spec in self.fields.items():
                field_errors = self._validate_field_spec(field_name, spec)
                errors.extend(field_errors)

        # Validate compare_fields
        if not self.compare_fields:
            errors.append("Schema must have at least one compare field")
        elif not isinstance(self.compare_fields, list):
            errors.append("compare_fields must be a list")
        else:
            field_names = set(self.fields.keys())
            invalid_fields = [f for f in self.compare_fields if f not in field_names]
            if invalid_fields:
                errors.append(
                    f"compare_fields {invalid_fields} not found in fields. "
                    f"Available fields: {sorted(field_names)}"
                )

        # Validate float_tolerance
        if not isinstance(self.float_tolerance, (int, float)):
            errors.append("float_tolerance must be a number")
        elif not 0 <= self.float_tolerance <= 1:
            errors.append("float_tolerance must be between 0 and 1")

        # Validate row_converter mapping
        for field_name in self.row_converter.mapping:
            if field_name not in self.fields:
                errors.append(
                    f"row_converter references unknown field '{field_name}'"
                )

        # Validate declarative extraction tools
        for raw_tool in self.tools:
            try:
                t_spec = normalize_tool_spec(raw_tool)
            except Exception as exc:
                errors.append(f"Invalid extraction tool spec: {exc}")
                continue
            for t_err in validate_tool_config(t_spec):
                errors.append(f"Extraction tool: {t_err}")

        return errors

    def _validate_field_spec(
        self, field_name: str, spec: FieldSpec
    ) -> list[str]:
        """Validate a single FieldSpec.

        Args:
            field_name: Name of the field.
            spec: Field specification to validate.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []

        # Validate description
        if not spec.description:
            errors.append(f"Field '{field_name}' must have a description")

        # Validate choices
        if spec.choices:
            if spec.type is not str:
                errors.append(
                    f"Field '{field_name}': choices can only be used with str type"
                )
            elif not isinstance(spec.choices, list) or not spec.choices:
                errors.append(
                    f"Field '{field_name}': choices must be a non-empty list"
                )
            elif spec.default is not None and spec.default not in spec.choices:
                errors.append(
                    f"Field '{field_name}': default {spec.default!r} must be one of choices {spec.choices}"
                )

        # Validate numeric constraints
        if spec.min_value is not None and spec.type not in (int, float):
            errors.append(
                f"Field '{field_name}': min_value can only be used with numeric types"
            )

        if spec.max_value is not None and spec.type not in (int, float):
            errors.append(
                f"Field '{field_name}': max_value can only be used with numeric types"
            )

        if (
            spec.min_value is not None
            and spec.max_value is not None
            and spec.min_value > spec.max_value
        ):
            errors.append(
                f"Field '{field_name}': min_value cannot be greater than max_value"
            )

        # Validate pattern
        if spec.pattern and spec.type is not str:
            errors.append(
                f"Field '{field_name}': pattern can only be used with str type"
            )

        # Validate format
        if spec.format is not None and not has_field_validator(spec.format):
            errors.append(
                f"Field '{field_name}': unknown format validator {spec.format!r}"
            )

        # Validate declarative validators
        for raw_val in spec.validators:
            try:
                v_spec = normalize_validator_spec(raw_val)
            except Exception as exc:
                errors.append(f"Field '{field_name}': invalid validator spec: {exc}")
                continue
            for v_err in validate_validator_config(v_spec):
                errors.append(f"Field '{field_name}': {v_err}")
            if v_spec.on_error == "nullify" and spec.required and spec.default is None:
                errors.append(
                    f"Field '{field_name}': validator {v_spec.name!r} uses on_error='nullify' "
                    "but field is required with no default"
                )

        return errors

    def validate_or_raise(self) -> None:
        """Validate SchemaConfig and raise exception if invalid.

        Raises:
            ValueError: If validation fails.
        """
        errors = self.validate()
        if errors:
            raise ValueError(
                f"SchemaConfig validation failed for '{self.name}':\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def __repr__(self) -> str:
        """String representation of SchemaConfig."""
        field_count = len(self.fields)
        return f"<SchemaConfig: {self.name} ({field_count} fields)>"


@dataclass
class ExtractionBundle:
    """Class-container representing a fully loaded schema setup with its models, signature, and converter.

    Attributes:
        config: The core SchemaConfig instance.
        signature: Generated DSPy signature class.
        record_model: Generated Pydantic model for individual records.
        output_model: Generated Pydantic output model.
        row_converter: Generated converter function (Callable).
    """

    config: SchemaConfig
    signature: Type
    record_model: Type[BaseModel]
    output_model: Type[BaseModel]
    row_converter: Any

    @property
    def experiment_model(self) -> Type[BaseModel]:
        """Backward-compatible alias for record_model."""
        return self.record_model

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return ["config", "signature", "record_model", "output_model", "row_converter"]
