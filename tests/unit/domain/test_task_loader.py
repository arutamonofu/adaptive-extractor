"""Unit tests for YAML task loader.

Tests cover:
- Loading TaskConfig from YAML
- Parsing field specifications
- Row converter configuration
- Instruction file resolution
- Saving TaskConfig to YAML
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from aee.domain.tasks.config import FieldSpec, TaskConfig
from aee.domain.tasks.loader import (
    _parse_field_spec,
    _parse_row_converter,
    load_task_from_yaml,
    load_task_with_models,
    save_task_to_yaml,
)


class TestParseFieldSpec:
    """Tests for _parse_field_spec function."""

    def test_parse_minimal_field_spec(self):
        """Test parsing minimal field specification."""
        field_data = {
            "type": "str",
            "description": "Test field",
        }

        spec = _parse_field_spec("test_field", field_data)

        assert spec.type == str
        assert spec.description == "Test field"
        assert spec.required is True

    def test_parse_field_spec_with_type_variations(self):
        """Test parsing field spec with different type specifications."""
        # String type
        spec = _parse_field_spec("f", {"type": "string", "description": "d"})
        assert spec.type == str

        # Integer type
        spec = _parse_field_spec("f", {"type": "int", "description": "d"})
        assert spec.type == int

        # Float type
        spec = _parse_field_spec("f", {"type": "float", "description": "d"})
        assert spec.type == float

    def test_parse_field_spec_with_choices(self):
        """Test parsing field spec with choices."""
        field_data = {
            "type": "str",
            "description": "Activity",
            "choices": ["peroxidase", "oxidase"],
        }

        spec = _parse_field_spec("activity", field_data)

        assert spec.choices == ["peroxidase", "oxidase"]

    def test_parse_field_spec_with_constraints(self):
        """Test parsing field spec with numeric constraints."""
        field_data = {
            "type": "float",
            "description": "pH",
            "min_value": 0.0,
            "max_value": 14.0,
        }

        spec = _parse_field_spec("ph", field_data)

        assert spec.min_value == 0.0
        assert spec.max_value == 14.0

    def test_parse_field_spec_optional(self):
        """Test parsing optional field spec."""
        field_data = {
            "type": "float",
            "description": "Optional value",
            "required": False,
            "default": 0.0,
        }

        spec = _parse_field_spec("value", field_data)

        assert spec.required is False
        assert spec.default == 0.0

    def test_parse_field_spec_with_alt_names(self):
        """Test parsing field spec with alternative names."""
        field_data = {
            "type": "float",
            "description": "Km value",
            "alt_names": ["km", "km_val"],
        }

        spec = _parse_field_spec("km_value", field_data)

        assert spec.alt_names == ["km", "km_val"]


class TestParseRowConverter:
    """Tests for _parse_row_converter function."""

    def test_parse_empty_converter(self):
        """Test parsing empty row converter."""
        config = _parse_row_converter({})
        assert config.mapping == {}

    def test_parse_converter_with_mapping(self):
        """Test parsing row converter with mapping."""
        converter_data = {
            "formula": ["formula", "name"],
            "activity": ["activity"],
        }

        config = _parse_row_converter(converter_data)

        assert config.mapping["formula"] == ["formula", "name"]
        assert config.mapping["activity"] == ["activity"]

    def test_parse_converter_single_column(self):
        """Test parsing converter with single column names."""
        converter_data = {
            "formula": "formula",
            "activity": "activity",
        }

        config = _parse_row_converter(converter_data)

        assert config.mapping["formula"] == ["formula"]
        assert config.mapping["activity"] == ["activity"]


class TestLoadTaskFromYaml:
    """Tests for load_task_from_yaml function."""

    def test_load_valid_yaml(self, tmp_path: Path):
        """Test loading valid YAML configuration."""
        yaml_content = """
name: test_task
description: Test task description
compare_fields:
  - field1
  - field2
float_tolerance: 0.10
fields:
  field1:
    type: str
    description: First field
  field2:
    type: float
    description: Second field
    required: false
initial_instruction: Test instruction
"""
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml_content)

        config = load_task_from_yaml(yaml_path)

        assert config.name == "test_task"
        assert config.description == "Test task description"
        assert config.compare_fields == ["field1", "field2"]
        assert config.float_tolerance == 0.10
        assert len(config.experiment_fields) == 2
        assert config.initial_instruction == "Test instruction"

    def test_load_yaml_with_row_converter(self, tmp_path: Path):
        """Test loading YAML with row converter configuration."""
        yaml_content = """
name: test_task
description: Test
compare_fields:
  - formula
fields:
  formula:
    type: str
    description: Formula
row_converter:
  formula:
    - formula
    - name
initial_instruction: Test
"""
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml_content)

        config = load_task_from_yaml(yaml_path)

        assert "formula" in config.row_converter.mapping
        assert config.row_converter.mapping["formula"] == ["formula", "name"]

    def test_load_yaml_with_instruction_file(self, tmp_path: Path):
        """Test loading YAML with instruction file reference."""
        # Create instruction file in same directory as YAML
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("Test instruction from file")

        # Use absolute path for instruction file
        yaml_content = f"""
name: test_task
description: Test
compare_fields:
  - field1
fields:
  field1:
    type: str
    description: Field
instruction_file: {instruction_file.absolute()}
"""
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml_content)

        config = load_task_from_yaml(yaml_path)

        # Instruction should be loaded from file
        instruction = config.get_instruction()
        assert instruction == "Test instruction from file"

    def test_load_nonexistent_yaml_raises(self, tmp_path: Path):
        """Test that loading nonexistent YAML raises error."""
        with pytest.raises(FileNotFoundError):
            load_task_from_yaml(tmp_path / "nonexistent.yaml")

    def test_load_yaml_validation_errors(self, tmp_path: Path):
        """Test loading YAML with validation errors."""
        # This test verifies that validation catches invalid compare_fields
        # Note: Validation happens during TaskConfig creation, not during YAML parsing
        yaml_content = """
name: test_task
description: Test
compare_fields:
  - field1
fields:
  field1:
    type: str
    description: Field
initial_instruction: Test instruction
"""
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml_content)

        # Config should load successfully since compare_fields is valid
        config = load_task_from_yaml(yaml_path)

        # Validate - should pass since field1 exists
        errors = config.validate()
        assert errors == []

        # Now test with truly invalid compare_fields
        # This will fail during TaskConfig creation due to __post_init__
        yaml_content_invalid = """
name: test_task
description: Test
compare_fields:
  - nonexistent_field
fields:
  field1:
    type: str
    description: Field
"""
        yaml_path_invalid = tmp_path / "task_invalid.yaml"
        yaml_path_invalid.write_text(yaml_content_invalid)

        # This should raise ValueError during __post_init__
        with pytest.raises(ValueError, match="not found in experiment_fields"):
            load_task_from_yaml(yaml_path_invalid)


class TestLoadTaskWithModels:
    """Tests for load_task_with_models function."""

    def test_load_with_models_success(self, tmp_path: Path):
        """Test loading task with generated models."""
        yaml_content = """
name: test_task
description: Test
compare_fields:
  - formula
fields:
  formula:
    type: str
    description: Chemical formula
  activity:
    type: str
    description: Activity
    choices:
      - peroxidase
      - oxidase
initial_instruction: Test instruction
"""
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml_content)

        config, experiment_model, output_model = load_task_with_models(yaml_path)

        assert config.name == "test_task"
        assert experiment_model.__name__ == "Experiment"
        assert output_model.__name__ == "ExtractionOutput"

        # Test creating experiment instance
        exp = experiment_model(formula="Fe3O4", activity="peroxidase")
        assert exp.formula == "Fe3O4"
        assert exp.activity == "peroxidase"


class TestSaveTaskToYaml:
    """Tests for save_task_to_yaml function."""

    def test_save_and_reload_roundtrip(self, tmp_path: Path):
        """Test saving and reloading TaskConfig."""
        # Create original config
        original_config = TaskConfig(
            name="test_task",
            description="Test description",
            experiment_fields={
                "formula": FieldSpec(type=str, description="Formula"),
                "activity": FieldSpec(
                    type=str,
                    description="Activity",
                    choices=["peroxidase", "oxidase"],
                ),
                "km_value": FieldSpec(
                    type=float,
                    description="Km value",
                    required=False,
                    default=0.0,
                ),
            },
            compare_fields=["formula", "activity"],
            float_tolerance=0.10,
            initial_instruction="Test instruction",
            tags=["test"],
            version="1.0.0",
        )

        # Save to YAML
        yaml_path = tmp_path / "saved_task.yaml"
        saved_path = save_task_to_yaml(original_config, yaml_path)

        assert saved_path.exists()

        # Reload from YAML
        reloaded_config = load_task_from_yaml(saved_path)

        # Compare key fields
        assert reloaded_config.name == original_config.name
        assert reloaded_config.description == original_config.description
        assert reloaded_config.compare_fields == original_config.compare_fields
        assert reloaded_config.float_tolerance == original_config.float_tolerance
        assert len(reloaded_config.experiment_fields) == len(original_config.experiment_fields)

    def test_save_yaml_format(self, tmp_path: Path):
        """Test that saved YAML is properly formatted."""
        config = TaskConfig(
            name="test",
            description="Test",
            experiment_fields={
                "field1": FieldSpec(type=str, description="Field 1"),
            },
            compare_fields=["field1"],
            initial_instruction="Test",
        )

        yaml_path = tmp_path / "task.yaml"
        save_task_to_yaml(config, yaml_path)

        # Read and parse YAML
        content = yaml_path.read_text()
        yaml_data = yaml.safe_load(content)

        assert yaml_data["name"] == "test"
        assert "fields" in yaml_data
        assert "field1" in yaml_data["fields"]


class TestProjectRootResolution:
    """Tests for project root resolution in loader."""

    def test_find_project_root(self, tmp_path: Path):
        """Test finding project root with pyproject.toml."""
        # Create fake project structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text("[tool.poetry]")

        tasks_dir = project_root / "src" / "tasks"
        tasks_dir.mkdir(parents=True)

        from aee.domain.tasks.loader import _find_project_root

        root = _find_project_root(tasks_dir / "task.yaml")
        assert root == project_root
