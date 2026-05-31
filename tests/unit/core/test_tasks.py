
import pytest

from ae.core.tasks import (
    FieldSpec,
    TaskConfig,
    create_experiment_model,
    load_task_from_yaml,
    load_task_with_models,
    save_task_to_yaml,
)
from ae.core.tasks.loader import (
    _find_project_root,
    _parse_row_converter,
)


@pytest.mark.unit
class TestFieldSpec:
    """Tests for FieldSpec dataclass creation, constraints, and Pydantic conversion."""

    def test_field_spec_creation_and_constraints(self):
        # Minimal string spec
        spec = FieldSpec(type=str, description="Test field")
        assert spec.type is str
        assert spec.required is True

        # Optional with default
        spec_opt = FieldSpec(type=float, description="Opt", required=False, default=0.0)
        assert spec_opt.required is False
        assert spec_opt.default == 0.0

        # Choices and validation errors for invalid constraints
        with pytest.raises(ValueError, match="choices can only be used with str type"):
            FieldSpec(type=float, description="Val", choices=[1.0, 2.0])

        with pytest.raises(ValueError, match="pattern can only be used with str type"):
            FieldSpec(type=int, description="Val", pattern=r"\d+")

        with pytest.raises(ValueError, match="min_value can only be used with numeric types"):
            FieldSpec(type=str, description="Val", min_value=0)

    def test_to_pydantic_field(self):
        spec = FieldSpec(type=str, description="Test", choices=["A", "B"])
        field = spec.to_pydantic_field()
        assert field.description == "Test"
        assert field.json_schema_extra is not None


@pytest.mark.unit
class TestRowConverterConfig:
    """Tests for RowConverterConfig parsing and validation."""

    def test_parse_row_converter(self):
        config = _parse_row_converter({"formula": ["formula", "name"], "activity": "activity"})
        assert config.mapping["formula"] == ["formula", "name"]
        assert config.mapping["activity"] == ["activity"]


@pytest.mark.unit
class TestTaskConfig:
    """Tests for TaskConfig properties and validation."""

    def test_task_config_validation(self, tmp_path):
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("Test instruction")

        fields = {
            "formula": FieldSpec(type=str, description="Formula"),
            "ph": FieldSpec(type=float, description="pH", required=False),
        }

        # Valid config
        config = TaskConfig(
            name="test_task",
            experiment_fields=fields,
            compare_fields=["formula"],
            float_tolerance=0.05,
            initial_instruction_file=str(instruction_file),
        )
        assert config.name == "test_task"
        assert len(config.validate()) == 0

        # Compare field not in fields raises ValueError on post-init
        with pytest.raises(ValueError, match="not found in experiment_fields"):
            TaskConfig(
                name="invalid",
                experiment_fields=fields,
                compare_fields=["nonexistent"],
                float_tolerance=0.05,
            )


@pytest.mark.unit
class TestTaskLoaderAndSaver:
    """Tests for YAML loader and saver roundtrip and root path resolution."""

    def test_load_and_save_task_yaml(self, tmp_path):
        yaml_content = """
name: test_yaml_task
compare_fields:
  - formula
float_tolerance: 0.10
fields:
  formula:
    type: str
    description: Chemical formula
  activity:
    type: str
    description: Catalytic activity
    choices:
      - peroxidase
      - oxidase
row_converter:
  formula:
    - formula
    - name
"""
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml_content)

        # Load with models
        config, experiment_model, output_model = load_task_with_models(yaml_path)
        assert config.name == "test_yaml_task"
        assert experiment_model.__name__ == "Experiment"
        assert output_model.__name__ == "ExtractionOutput"
        assert config.row_converter.mapping["formula"] == ["formula", "name"]

        # Roundtrip save and reload
        save_path = tmp_path / "saved_task.yaml"
        save_task_to_yaml(config, save_path)
        assert save_path.exists()

        reloaded = load_task_from_yaml(save_path)
        assert reloaded.name == "test_yaml_task"
        assert reloaded.float_tolerance == 0.10

    def test_find_project_root(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text("[tool.poetry]")

        tasks_dir = project_root / "src" / "tasks"
        tasks_dir.mkdir(parents=True)

        root = _find_project_root(tasks_dir / "task.yaml")
        assert root == project_root


@pytest.mark.unit
class TestDynamicModelValidation:
    """Tests for Pydantic dynamic model generation, string coercion, and type casting."""

    def test_string_field_coercion(self):
        config = TaskConfig(
            name="test_coercion",
            experiment_fields={
                "formula": FieldSpec(type=str, description="Formula", required=True),
                "length": FieldSpec(type=str, description="Length in nm", required=False),
                "active": FieldSpec(type=str, description="Active status", required=False),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )
        ExperimentModel = create_experiment_model(config)

        # Coerce numeric types to string
        exp = ExperimentModel(formula="Fe3O4", length=12.0, active=True)
        assert exp.length == "12"
        assert exp.active == "true"

        # Pass ranges and strings directly
        exp2 = ExperimentModel(formula="CuO", length="10-15", active="false")
        assert exp2.length == "10-15"
        assert exp2.active == "false"

    def test_float_field_no_string_coercion(self):
        config = TaskConfig(
            name="test_float",
            experiment_fields={
                "formula": FieldSpec(type=str, description="Formula", required=True),
                "km": FieldSpec(type=float, description="Km value", required=False),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )
        ExperimentModel = create_experiment_model(config)

        exp = ExperimentModel(formula="Fe3O4", km=1.502)
        assert isinstance(exp.km, float)
        assert exp.km == 1.502

        # Convert string representation of float to float
        exp2 = ExperimentModel(formula="Fe3O4", km="0.065")
        assert isinstance(exp2.km, float)
        assert exp2.km == 0.065
