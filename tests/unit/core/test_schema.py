
import pytest

from ae.core.schema import (
    FieldSpec,
    SchemaConfig,
    create_experiment_model,
    create_record_model,
    load_schema_from_yaml,
    load_schema_with_models,
    save_schema_to_yaml,
)
from ae.core.schema.loader import (
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
        assert field.json_schema_extra is None

        # Generated JSON schema must emit standard 'enum' without non-standard 'choices'
        cfg = SchemaConfig(
            name="test_enum_schema",
            fields={"cat": spec},
            compare_fields=["cat"],
        )
        from ae.core.schema.dynamic_models import create_output_model
        rec_model = create_record_model(cfg)
        out_model = create_output_model(cfg, rec_model)
        schema_json = out_model.model_json_schema()
        cat_prop = schema_json["$defs"]["Record"]["properties"]["cat"]
        assert cat_prop["enum"] == ["A", "B"]
        assert "choices" not in cat_prop

        from google.genai import _transformers, types
        processed = dict(schema_json)
        _transformers.process_schema(processed, None)
        validated = types.Schema.model_validate(processed)
        assert validated is not None

        # FieldSpec without description does not populate description in Pydantic Field
        spec_no_desc = FieldSpec(type=str)
        assert spec_no_desc.description == ""
        field_no_desc = spec_no_desc.to_pydantic_field()
        assert field_no_desc.description is None


@pytest.mark.unit
class TestRowConverterConfig:
    """Tests for RowConverterConfig parsing and validation."""

    def test_parse_row_converter(self):
        config = _parse_row_converter({"formula": ["formula", "name"], "activity": "activity"})
        assert config.mapping["formula"] == ["formula", "name"]
        assert config.mapping["activity"] == ["activity"]


@pytest.mark.unit
class TestSchemaConfig:
    """Tests for SchemaConfig properties and validation."""

    def test_schema_config_validation(self, tmp_path):
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("Test instruction")

        fields = {
            "formula": FieldSpec(type=str, description="Formula"),
            "ph": FieldSpec(type=float, description="pH", required=False),
        }

        # Valid config
        config = SchemaConfig(
            name="test_task",
            experiment_fields=fields,
            compare_fields=["formula"],
            float_tolerance=0.05,
            instruction_file=str(instruction_file),
        )
        assert config.name == "test_task"
        assert len(config.validate()) == 0

        # Compare field not in fields raises ValueError on post-init
        with pytest.raises(ValueError, match="not found in (fields|experiment_fields)"):
            SchemaConfig(
                name="invalid",
                experiment_fields=fields,
                compare_fields=["nonexistent"],
                float_tolerance=0.05,
            )


@pytest.mark.unit
class TestSchemaLoaderAndSaver:
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
        config, experiment_model, output_model = load_schema_with_models(yaml_path)
        assert config.name == "test_yaml_task"
        assert experiment_model.__name__ == "Record"
        assert output_model.__name__ == "ExtractionOutput"
        assert config.row_converter.mapping["formula"] == ["formula", "name"]

        # Roundtrip save and reload
        save_path = tmp_path / "saved_task.yaml"
        save_schema_to_yaml(config, save_path)
        assert save_path.exists()

        reloaded = load_schema_from_yaml(save_path)
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
        config = SchemaConfig(
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
        config = SchemaConfig(
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

    def test_string_field_default_and_whitespace_coercion(self) -> None:
        """Test that string fields with default values handle None, empty strings, and whitespace."""
        config = SchemaConfig(
            name="test_defaults",
            experiment_fields={
                "target_relation": FieldSpec(
                    type=str,
                    default="=",
                    required=False,
                ),
                "target_value": FieldSpec(
                    type=str,
                    required=False,
                ),
            },
            compare_fields=["target_relation"],
        )
        model = create_record_model(config)

        # Omitted field should take default
        rec1 = model()
        assert rec1.target_relation == "="

        # None should be coerced to default
        rec2 = model(target_relation=None)
        assert rec2.target_relation == "="

        # Empty string should be coerced to default
        rec3 = model(target_relation="")
        assert rec3.target_relation == "="

        # Whitespace should be trimmed
        rec4 = model(target_relation=" = ")
        assert rec4.target_relation == "="

        # Explicit non-default value should be preserved (with whitespace trimmed)
        rec5 = model(target_relation=" > ")
        assert rec5.target_relation == ">"

        # Field without default should still accept None
        rec6 = model(target_value=None)
        assert rec6.target_value is None

        # Field without default trims whitespace when string provided
        rec7 = model(target_value="  12.5  ")
        assert rec7.target_value == "12.5"

    def test_save_and_load_preserves_all_constraints(self, tmp_path):
        from ae.core.schema.dynamic_models import create_all_models, extract_schema_field_constraints

        cfg = SchemaConfig(
            name="full_constraints",
            fields={
                "rel": FieldSpec(
                    type=str,
                    description="Relation operator",
                    required=False,
                    default="=",
                    choices=["=", ">", "<"],
                ),
                "code": FieldSpec(
                    type=str,
                    description="Compound code",
                    pattern=r"^[A-Za-z0-9_-]+$",
                ),
                "score": FieldSpec(
                    type=float,
                    description="Score",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
            compare_fields=["rel", "code"],
        )
        assert cfg.validate() == []

        yaml_path = tmp_path / "constraints.yaml"
        save_schema_to_yaml(cfg, yaml_path)
        reloaded = load_schema_from_yaml(yaml_path)

        assert reloaded.fields["rel"].choices == ["=", ">", "<"]
        assert reloaded.fields["rel"].default == "="
        assert reloaded.fields["code"].pattern == r"^[A-Za-z0-9_-]+$"
        assert reloaded.fields["score"].min_value == 0.0
        assert reloaded.fields["score"].max_value == 100.0

        _, out_model = create_all_models(reloaded)
        extracted = extract_schema_field_constraints(out_model)
        assert extracted["rel"]["enum"] == ["=", ">", "<"]
        assert extracted["rel"]["default"] == "="
        assert extracted["code"]["pattern"] == r"^[A-Za-z0-9_-]+$"
        assert extracted["score"]["minimum"] == 0.0
        assert extracted["score"]["maximum"] == 100.0

    def test_validate_rejects_default_outside_choices(self):
        cfg = SchemaConfig(
            name="bad_default",
            fields={
                "rel": FieldSpec(
                    type=str,
                    description="Relation",
                    required=False,
                    default="!=",
                    choices=["=", ">", "<"],
                )
            },
            compare_fields=["rel"],
        )
        errors = cfg.validate()
        assert any("must be one of choices" in e for e in errors)

    def test_row_converter_accepts_dict_and_series_with_missing_values(self):
        import pandas as pd
        from ae.core.schema.dynamic_models import create_all_models, create_row_converter

        cfg = SchemaConfig(
            name="dict_series_test",
            fields={
                "compound_id": FieldSpec(type=str, required=False),
                "target_value": FieldSpec(type=str, required=False),
                "score": FieldSpec(type=float, required=False),
            },
            compare_fields=["compound_id", "target_value"],
        )
        record_model, _ = create_all_models(cfg)
        converter = create_row_converter(cfg, record_model)

        rec_from_dict = converter({"compound_id": "1a", "target_value": "", "score": float("nan")})
        assert rec_from_dict is not None
        assert rec_from_dict.compound_id == "1a"
        assert rec_from_dict.target_value is None
        assert rec_from_dict.score is None

        rec_literal = converter({"compound_id": "N/A", "target_value": None, "score": None})
        assert rec_literal is not None
        assert rec_literal.compound_id == "N/A"
        assert rec_literal.target_value is None

        rec_from_series = converter(pd.Series({"compound_id": "2b", "target_value": "12.5", "score": 3.5}))
        assert rec_from_series is not None
        assert rec_from_series.compound_id == "2b"
        assert rec_from_series.target_value == "12.5"
        assert rec_from_series.score == 3.5
