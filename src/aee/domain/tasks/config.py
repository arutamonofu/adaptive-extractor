"""Task configuration dataclasses for dynamic task generation.

This module provides the core data structures for defining tasks declaratively
using TaskConfig and FieldSpec, enabling dynamic model and signature generation.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field as PydanticField


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
        alt_names: Alternative names for the field (for CSV mapping)
    """

    type: Union[Type[str], Type[int], Type[float], Type]
    description: str
    required: bool = True
    default: Any = None
    choices: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    alt_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate field specification after initialization."""
        if self.choices and self.type != str:
            raise ValueError("choices can only be used with str type")

        if self.pattern and self.type != str:
            raise ValueError("pattern can only be used with str type")

        if self.min_value is not None and self.type not in (int, float):
            raise ValueError("min_value can only be used with numeric types")

        if self.max_value is not None and self.type not in (int, float):
            raise ValueError("max_value can only be used with numeric types")

        if not self.required and self.default is None and self.choices is None:
            # Optional fields without default should have None as default
            self.default = None

    def to_pydantic_field(self) -> PydanticField:
        """Convert FieldSpec to Pydantic Field.

        Returns:
            Pydantic Field with appropriate constraints.
        """
        field_kwargs: Dict[str, Any] = {
            "description": self.description,
        }

        if not self.required:
            field_kwargs["default"] = self.default

        if self.choices:
            field_kwargs["choices"] = self.choices

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
class TaskConfig:
    """Complete configuration for an extraction task.

    This is the single source of truth for task definition. All models,
    signatures, and converters are generated from this configuration.

    Attributes:
        name: Unique task identifier (e.g., "nanozymes", "catalysts")
        experiment_fields: Dictionary of field specifications
        compare_fields: List of field names to compare during evaluation
        float_tolerance: Tolerance for float comparisons (0.0 to 1.0)
        initial_instruction_file: Path to instruction file for DSPy signature (from config/default.yaml)
        row_converter: Configuration for CSV row conversion
        output_model_name: Name for the generated output model class
        experiment_model_name: Name for the generated experiment model class
        base_class: Base class for experiment model (e.g., Experiment)
        tags: Optional tags for categorizing tasks
        version: Task configuration version
        metadata: Additional metadata
    """

    name: str
    experiment_fields: Dict[str, FieldSpec]
    compare_fields: List[str]
    float_tolerance: float
    initial_instruction_file: Optional[str] = None
    row_converter: RowConverterConfig = field(default_factory=RowConverterConfig)
    output_model_name: str = "ExtractionOutput"
    experiment_model_name: str = "Experiment"
    base_class: Optional[Type[BaseModel]] = None
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task configuration after initialization."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Task name must be a non-empty string")

        if not self.experiment_fields:
            raise ValueError("Task must have at least one experiment field")

        if not self.compare_fields:
            raise ValueError("Task must have at least one compare field")

        if not isinstance(self.compare_fields, list):
            raise ValueError("compare_fields must be a list")

        if not 0 <= self.float_tolerance <= 1:
            raise ValueError("float_tolerance must be between 0 and 1")

        # Validate compare_fields exist in experiment_fields
        field_names = set(self.experiment_fields.keys())
        invalid_fields = [f for f in self.compare_fields if f not in field_names]
        if invalid_fields:
            raise ValueError(
                f"compare_fields {invalid_fields} not found in experiment_fields. "
                f"Available fields: {sorted(field_names)}"
            )

    def get_instruction(self) -> str:
        """Get the instruction text from instruction file.

        Returns:
            Instruction text.

        Raises:
            ValueError: If no instruction file is specified.
            FileNotFoundError: If instruction file not found.
        """
        if not self.initial_instruction_file:
            raise ValueError(
                f"No instruction file specified for task '{self.name}'. "
                "Set 'task.initial_instruction_file' in config/default.yaml"
            )

        from pathlib import Path

        instruction_path = Path(self.initial_instruction_file)
        if not instruction_path.exists():
            raise FileNotFoundError(
                f"Instruction file not found: {self.initial_instruction_file}. "
                "Check 'task.initial_instruction_file' in config/default.yaml"
            )
        return instruction_path.read_text(encoding="utf-8")

    def get_instruction_hash(self) -> str:
        """Get SHA256 hash of the instruction.

        Returns:
            First 12 characters of SHA256 hash.
        """
        instruction = self.get_instruction()
        return hashlib.sha256(instruction.encode()).hexdigest()[:12]

    def get_required_fields(self) -> List[str]:
        """Get list of required field names.

        Returns:
            List of field names that are required.
        """
        return [
            name for name, spec in self.experiment_fields.items()
            if spec.required
        ]

    def get_optional_fields(self) -> List[str]:
        """Get list of optional field names.

        Returns:
            List of field names that are optional.
        """
        return [
            name for name, spec in self.experiment_fields.items()
            if not spec.required
        ]

    def get_field_choices(self, field_name: str) -> Optional[List[str]]:
        """Get choices for a field if it's a Literal type.

        Args:
            field_name: Name of the field.

        Returns:
            List of choices or None if not a Literal field.
        """
        if field_name not in self.experiment_fields:
            return None
        return self.experiment_fields[field_name].choices

    def to_dict(self) -> Dict[str, Any]:
        """Convert TaskConfig to dictionary.

        Returns:
            Dictionary representation of the config.
        """
        from dataclasses import asdict
        return asdict(self)

    def validate(self) -> list[str]:
        """Validate TaskConfig completeness and consistency.

        Performs comprehensive validation of the task configuration,
        checking all fields, compare_fields, and instruction.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Validate name
        if not self.name or not isinstance(self.name, str):
            errors.append("Task name must be a non-empty string")

        # Validate experiment_fields
        if not self.experiment_fields:
            errors.append("Task must have at least one experiment field")
        else:
            for field_name, spec in self.experiment_fields.items():
                field_errors = self._validate_field_spec(field_name, spec)
                errors.extend(field_errors)

        # Validate compare_fields
        if not self.compare_fields:
            errors.append("Task must have at least one compare field")
        elif not isinstance(self.compare_fields, list):
            errors.append("compare_fields must be a list")
        else:
            field_names = set(self.experiment_fields.keys())
            invalid_fields = [f for f in self.compare_fields if f not in field_names]
            if invalid_fields:
                errors.append(
                    f"compare_fields {invalid_fields} not found in experiment_fields. "
                    f"Available fields: {sorted(field_names)}"
                )

        # Validate float_tolerance
        if not isinstance(self.float_tolerance, (int, float)):
            errors.append("float_tolerance must be a number")
        elif not 0 <= self.float_tolerance <= 1:
            errors.append("float_tolerance must be between 0 and 1")

        # Validate instruction
        try:
            instruction = self.get_instruction()
            if not instruction or not instruction.strip():
                errors.append("Instruction cannot be empty")
        except FileNotFoundError as e:
            errors.append(str(e))
        except ValueError as e:
            errors.append(str(e))

        # Validate row_converter mapping
        for field_name in self.row_converter.mapping:
            if field_name not in self.experiment_fields:
                errors.append(
                    f"row_converter references unknown field '{field_name}'"
                )

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
            if spec.type != str:
                errors.append(
                    f"Field '{field_name}': choices can only be used with str type"
                )
            elif not isinstance(spec.choices, list) or not spec.choices:
                errors.append(
                    f"Field '{field_name}': choices must be a non-empty list"
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
        if spec.pattern and spec.type != str:
            errors.append(
                f"Field '{field_name}': pattern can only be used with str type"
            )

        return errors

    def validate_or_raise(self) -> None:
        """Validate TaskConfig and raise exception if invalid.

        Raises:
            ValueError: If validation fails.
        """
        errors = self.validate()
        if errors:
            raise ValueError(
                f"TaskConfig validation failed for '{self.name}':\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def __repr__(self) -> str:
        """String representation of TaskConfig."""
        field_count = len(self.experiment_fields)
        return f"<TaskConfig: {self.name} ({field_count} fields)>"
