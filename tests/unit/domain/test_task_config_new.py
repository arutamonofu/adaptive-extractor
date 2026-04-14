"""Unit tests for TaskConfig and FieldSpec.

Tests cover:
- FieldSpec creation and validation
- TaskConfig creation and validation
- FieldSpec to Pydantic conversion
- TaskConfig utility methods
"""

import pytest

from aee.domain.tasks import FieldSpec, RowConverterConfig, TaskConfig


@pytest.mark.unit
class TestFieldSpec:
    """Tests for FieldSpec dataclass."""

    def test_create_minimal_field_spec(self):
        """Test creating minimal FieldSpec."""
        spec = FieldSpec(
            type=str,
            description="Test field",
        )

        assert spec.type is str
        assert spec.description == "Test field"
        assert spec.required is True
        assert spec.default is None
        assert spec.choices is None

    def test_create_optional_field_spec(self):
        """Test creating optional FieldSpec with default."""
        spec = FieldSpec(
            type=float,
            description="Optional value",
            required=False,
            default=0.0,
        )

        assert spec.type is float
        assert spec.required is False
        assert spec.default == 0.0

    def test_create_field_with_choices(self):
        """Test creating FieldSpec with choices."""
        spec = FieldSpec(
            type=str,
            description="Activity type",
            choices=["peroxidase", "oxidase", "catalase"],
        )

        assert spec.choices == ["peroxidase", "oxidase", "catalase"]

    def test_create_field_with_numeric_constraints(self):
        """Test creating FieldSpec with numeric constraints."""
        spec = FieldSpec(
            type=float,
            description="pH value",
            min_value=0.0,
            max_value=14.0,
        )

        assert spec.min_value == 0.0
        assert spec.max_value == 14.0

    def test_choices_with_non_string_type_raises(self):
        """Test that choices with non-str type raises error."""
        with pytest.raises(ValueError, match="choices can only be used with str type"):
            FieldSpec(
                type=float,
                description="Value",
                choices=[1.0, 2.0, 3.0],
            )

    def test_pattern_with_non_string_type_raises(self):
        """Test that pattern with non-str type raises error."""
        with pytest.raises(ValueError, match="pattern can only be used with str type"):
            FieldSpec(
                type=int,
                description="Count",
                pattern=r"\d+",
            )

    def test_min_value_with_non_numeric_type_raises(self):
        """Test that min_value with non-numeric type raises error."""
        with pytest.raises(ValueError, match="min_value can only be used with numeric types"):
            FieldSpec(
                type=str,
                description="Name",
                min_value=0,
            )

    def test_max_value_with_non_numeric_type_raises(self):
        """Test that max_value with non-numeric type raises error."""
        with pytest.raises(ValueError, match="max_value can only be used with numeric types"):
            FieldSpec(
                type=str,
                description="Name",
                max_value=100,
            )

    def test_to_pydantic_field_required(self):
        """Test converting required FieldSpec to Pydantic Field."""
        spec = FieldSpec(
            type=str,
            description="Required field",
            required=True,
        )

        field = spec.to_pydantic_field()
        assert field.description == "Required field"
        # Required fields should not have default

    def test_to_pydantic_field_optional(self):
        """Test converting optional FieldSpec to Pydantic Field."""
        spec = FieldSpec(
            type=float,
            description="Optional field",
            required=False,
            default=0.0,
        )

        field = spec.to_pydantic_field()
        assert field.description == "Optional field"
        assert field.default == 0.0

    def test_to_pydantic_field_with_choices(self):
        """Test converting FieldSpec with choices."""
        spec = FieldSpec(
            type=str,
            description="Activity",
            choices=["peroxidase", "oxidase"],
        )

        field = spec.to_pydantic_field()
        assert field.description == "Activity"
        # Check json_schema_extra for choices
        assert field.json_schema_extra is not None

    def test_to_pydantic_field_with_constraints(self):
        """Test converting FieldSpec with numeric constraints."""
        spec = FieldSpec(
            type=float,
            description="pH",
            min_value=0.0,
            max_value=14.0,
        )

        field = spec.to_pydantic_field()
        assert field.description == "pH"
        # Pydantic v2 uses different attribute names
        # Check that constraints are present in field metadata
        assert hasattr(field, 'ge') or field.metadata


@pytest.mark.unit
class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    @pytest.fixture
    def sample_fields(self):
        """Sample field specifications."""
        return {
            "formula": FieldSpec(type=str, description="Chemical formula"),
            "activity": FieldSpec(
                type=str,
                description="Catalytic activity",
                choices=["peroxidase", "oxidase", "catalase"],
            ),
            "km_value": FieldSpec(
                type=float,
                description="Michaelis constant",
                required=False,
            ),
        }

    def test_create_minimal_task_config(self, sample_fields, tmp_path):
        """Test creating minimal TaskConfig."""
        # Create temporary instruction file
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test instruction")

        config = TaskConfig(
            name="test_task",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        assert config.name == "test_task"
        assert len(config.experiment_fields) == 3
        assert config.float_tolerance == 0.05

    def test_task_config_with_all_options(self, sample_fields, tmp_path):
        """Test creating TaskConfig with all options."""
        # Create temporary instruction file
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test instruction")

        config = TaskConfig(
            name="test_task",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.10,
            initial_instruction_file=str(instruction_file),
        )

        assert config.float_tolerance == 0.10
        assert config.initial_instruction_file == str(instruction_file)

    def test_empty_name_raises(self, sample_fields, tmp_path):
        """Test that empty name raises error."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        with pytest.raises(ValueError, match="name must be a non-empty string"):
            TaskConfig(
                name="",
                experiment_fields=sample_fields,
                compare_fields=["formula"],
                float_tolerance=0.05,
                initial_instruction_file=str(instruction_file),
            )

    def test_empty_fields_raises(self, tmp_path):
        """Test that empty experiment_fields raises error."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        with pytest.raises(ValueError, match="must have at least one experiment field"):
            TaskConfig(
                name="test",
                experiment_fields={},
                compare_fields=["formula"],
                float_tolerance=0.05,
                initial_instruction_file=str(instruction_file),
            )

    def test_empty_compare_fields_raises(self, sample_fields, tmp_path):
        """Test that empty compare_fields raises error."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        with pytest.raises(ValueError, match="must have at least one compare field"):
            TaskConfig(
                name="test",
                experiment_fields=sample_fields,
                compare_fields=[],
                float_tolerance=0.05,
                initial_instruction_file=str(instruction_file),
            )

    def test_invalid_float_tolerance_raises(self, sample_fields, tmp_path):
        """Test that invalid float_tolerance raises error."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        with pytest.raises(ValueError, match="float_tolerance must be between 0 and 1"):
            TaskConfig(
                name="test",
                experiment_fields=sample_fields,
                compare_fields=["formula"],
                float_tolerance=1.5,
                initial_instruction_file=str(instruction_file),
            )

        with pytest.raises(ValueError, match="float_tolerance must be between 0 and 1"):
            TaskConfig(
                name="test",
                experiment_fields=sample_fields,
                compare_fields=["formula"],
                float_tolerance=-0.1,
                initial_instruction_file=str(instruction_file),
            )

    def test_compare_fields_not_in_experiment_fields_raises(self, sample_fields, tmp_path):
        """Test that compare_fields not in experiment_fields raises error."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        with pytest.raises(ValueError, match="not found in experiment_fields"):
            TaskConfig(
                name="test",
                experiment_fields=sample_fields,
                compare_fields=["formula", "nonexistent_field"],
                float_tolerance=0.05,
                initial_instruction_file=str(instruction_file),
            )

    def test_get_required_fields(self, sample_fields, tmp_path):
        """Test getting required field names."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        required = config.get_required_fields()
        assert "formula" in required
        assert "activity" in required
        assert "km_value" not in required  # optional

    def test_get_optional_fields(self, sample_fields, tmp_path):
        """Test getting optional field names."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        optional = config.get_optional_fields()
        assert "km_value" in optional
        assert "formula" not in optional

    def test_get_field_choices(self, sample_fields, tmp_path):
        """Test getting field choices."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        choices = config.get_field_choices("activity")
        assert choices == ["peroxidase", "oxidase", "catalase"]

        # No choices for field without choices
        assert config.get_field_choices("formula") is None

    def test_to_dict(self, sample_fields, tmp_path):
        """Test converting TaskConfig to dictionary."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        config_dict = config.to_dict()
        assert config_dict["name"] == "test"
        assert "description" not in config_dict
        assert len(config_dict["experiment_fields"]) == 3

    def test_validate_success(self, sample_fields, tmp_path):
        """Test successful validation."""
        # Create temporary instruction file
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test instruction")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        errors = config.validate()
        assert errors == []

    # Note: Instruction file validation is now deferred to get_instruction() call time
    # (e.g., during DSPy signature creation), not during TaskConfig.validate().
    # This allows TaskConfig to be used in tests without requiring a physical file.

    def test_validate_or_raise_success(self, sample_fields, tmp_path):
        """Test validate_or_raise with valid config."""
        # Create temporary instruction file
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test instruction")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        # Should not raise
        config.validate_or_raise()

    def test_validate_or_raise_invalid_compare_fields(self, sample_fields, tmp_path):
        """Test validate_or_raise raises when compare_fields are invalid."""
        # Create temporary instruction file
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test instruction")

        # Create config with invalid compare_fields - will raise during __post_init__
        with pytest.raises(ValueError, match="not found in experiment_fields"):
            TaskConfig(
                name="test",
                experiment_fields=sample_fields,
                compare_fields=["nonexistent_field"],
                float_tolerance=0.05,
                initial_instruction_file=str(instruction_file),
            )

    def test_get_instruction_file_not_found(self, sample_fields):
        """Test get_instruction raises FileNotFoundError when file not found."""
        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file="/nonexistent/path.txt",
        )

        with pytest.raises(FileNotFoundError, match="not found"):
            config.get_instruction()

    def test_get_instruction_no_file_specified(self, sample_fields):
        """Test get_instruction raises ValueError when no file specified."""
        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
        )

        with pytest.raises(ValueError, match="No instruction file"):
            config.get_instruction()

    def test_get_instruction_success(self, sample_fields, tmp_path):
        """Test get_instruction reads file content."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test instruction content")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        instruction = config.get_instruction()
        assert instruction == "Test instruction content"

    def test_get_instruction_hash(self, sample_fields, tmp_path):
        """Test get_instruction_hash returns correct hash."""
        instruction_file = tmp_path / "test_instruction.txt"
        instruction_file.write_text("Test instruction")

        config = TaskConfig(
            name="test",
            experiment_fields=sample_fields,
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )

        hash_value = config.get_instruction_hash()
        assert len(hash_value) == 12  # First 12 characters
        assert isinstance(hash_value, str)


@pytest.mark.unit
class TestRowConverterConfig:
    """Tests for RowConverterConfig."""

    def test_create_empty_config(self):
        """Test creating empty RowConverterConfig."""
        config = RowConverterConfig()
        assert config.mapping == {}

    def test_create_config_with_mapping(self):
        """Test creating RowConverterConfig with mapping."""
        config = RowConverterConfig(
            mapping={
                "formula": ["formula", "name"],
                "activity": ["activity", "type"],
            }
        )

        assert config.mapping["formula"] == ["formula", "name"]
        assert config.mapping["activity"] == ["activity", "type"]

    def test_get_column_names_exists(self):
        """Test getting column names for existing field."""
        config = RowConverterConfig(
            mapping={
                "formula": ["formula", "name"],
            }
        )

        columns = config.get_column_names("formula")
        assert columns == ["formula", "name"]

    def test_get_column_names_not_exists(self):
        """Test getting column names for non-existing field."""
        config = RowConverterConfig()

        columns = config.get_column_names("nonexistent")
        assert columns == ["nonexistent"]  # Returns field name as default


@pytest.mark.unit
class TestDynamicModelStringValidation:
    """Tests for string field validation in dynamically created models."""

    def test_float_to_string_conversion(self):
        """Float values should convert to strings for str-typed fields."""
        from aee.domain.tasks import create_experiment_model

        config = TaskConfig(
            name="test_string_conv",
            experiment_fields={
                "length": FieldSpec(type=str, description="Length", required=False),
                "formula": FieldSpec(type=str, description="Formula", required=True),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )

        ExperimentModel = create_experiment_model(config)
        exp = ExperimentModel(formula="Fe3O4", length=12.0)

        assert exp.length == "12"

    def test_int_to_string_conversion(self):
        """Int values should convert to strings for str-typed fields."""
        from aee.domain.tasks import create_experiment_model

        config = TaskConfig(
            name="test_int_conv",
            experiment_fields={
                "count": FieldSpec(type=str, description="Count", required=False),
                "formula": FieldSpec(type=str, description="Formula", required=True),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )

        ExperimentModel = create_experiment_model(config)
        exp = ExperimentModel(formula="Fe3O4", count=5)

        assert exp.count == "5"

    def test_string_passthrough(self):
        """String values should pass through unchanged."""
        from aee.domain.tasks import create_experiment_model

        config = TaskConfig(
            name="test_str_pass",
            experiment_fields={
                "length": FieldSpec(type=str, description="Length", required=False),
                "formula": FieldSpec(type=str, description="Formula", required=True),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )

        ExperimentModel = create_experiment_model(config)
        exp = ExperimentModel(formula="Fe3O4", length="12")

        assert exp.length == "12"

    def test_range_string_accepted(self):
        """Range strings like '5-10' should still work."""
        from aee.domain.tasks import create_experiment_model

        config = TaskConfig(
            name="test_range",
            experiment_fields={
                "length": FieldSpec(type=str, description="Length", required=False),
                "formula": FieldSpec(type=str, description="Formula", required=True),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )

        ExperimentModel = create_experiment_model(config)
        exp = ExperimentModel(formula="Fe3O4", length="5-10")

        assert exp.length == "5-10"

    def test_scientific_notation(self):
        """Scientific notation floats should convert properly."""
        from aee.domain.tasks import create_experiment_model

        config = TaskConfig(
            name="test_sci",
            experiment_fields={
                "value": FieldSpec(type=str, description="Value", required=False),
                "formula": FieldSpec(type=str, description="Formula", required=True),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )

        ExperimentModel = create_experiment_model(config)
        exp = ExperimentModel(formula="Fe3O4", value=1.502e-07)

        assert exp.value == "1.502e-07"
