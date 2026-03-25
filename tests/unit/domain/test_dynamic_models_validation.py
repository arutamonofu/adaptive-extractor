"""Unit tests for dynamic model validation with string field coercion.

Tests verify that numeric values from LLM JSON responses are properly
converted to strings for str-typed fields in dynamically generated models.
"""

import pytest

from aee.domain.tasks.config import FieldSpec, TaskConfig
from aee.domain.tasks.dynamic_models import create_experiment_model


@pytest.mark.unit
class TestStringFieldValidation:
    """Test automatic float→str conversion for string fields."""

    @pytest.fixture
    def string_field_config(self) -> TaskConfig:
        """Create a TaskConfig with str-typed fields for testing."""
        return TaskConfig(
            name="test_validation",
            experiment_fields={
                "length": FieldSpec(type=str, description="Length in nm", required=False),
                "width": FieldSpec(type=str, description="Width in nm", required=False),
                "ph": FieldSpec(type=str, description="pH value", required=False),
                "temperature": FieldSpec(type=str, description="Temperature in °C", required=False),
                "formula": FieldSpec(type=str, description="Chemical formula", required=True),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )

    def test_float_to_string_conversion(self, string_field_config: TaskConfig):
        """Float values should be converted to strings for str-typed fields."""
        ExperimentModel = create_experiment_model(string_field_config)

        # Should not raise - float converted to str
        exp = ExperimentModel(formula="Fe3O4", length=12.0, ph=4.0, temperature=40.0)

        assert exp.length == "12"
        assert exp.ph == "4"
        assert exp.temperature == "40"

    def test_int_to_string_conversion(self, string_field_config: TaskConfig):
        """Int values should be converted to strings."""
        ExperimentModel = create_experiment_model(string_field_config)

        exp = ExperimentModel(formula="Fe3O4", length=12, width=15)

        assert exp.length == "12"
        assert exp.width == "15"

    def test_string_passthrough(self, string_field_config: TaskConfig):
        """String values should pass through unchanged."""
        ExperimentModel = create_experiment_model(string_field_config)

        exp = ExperimentModel(formula="Fe3O4", length="12", ph="4.0")

        assert exp.length == "12"
        assert exp.ph == "4.0"

    def test_range_strings_accepted(self, string_field_config: TaskConfig):
        """Range strings like '5-10' should still work."""
        ExperimentModel = create_experiment_model(string_field_config)

        exp = ExperimentModel(formula="Fe3O4", length="5-10", temperature="25-30")

        assert exp.length == "5-10"
        assert exp.temperature == "25-30"

    def test_scientific_notation_float(self, string_field_config: TaskConfig):
        """Scientific notation floats should convert properly."""
        ExperimentModel = create_experiment_model(string_field_config)

        exp = ExperimentModel(formula="Fe3O4", length=1.502e-07)

        # Scientific notation should be preserved as string
        assert exp.length == "1.502e-07"

    def test_small_float_values(self, string_field_config: TaskConfig):
        """Small float values like 0.065 should convert properly."""
        ExperimentModel = create_experiment_model(string_field_config)

        exp = ExperimentModel(formula="Fe3O4", ph=0.065)

        assert exp.ph == "0.065"

    def test_none_values_handled(self, string_field_config: TaskConfig):
        """None values should be handled correctly for optional fields."""
        ExperimentModel = create_experiment_model(string_field_config)

        # Optional fields can be None
        exp = ExperimentModel(formula="Fe3O4", length=None, ph=None)

        assert exp.length is None
        assert exp.ph is None

    def test_required_field_still_required(self, string_field_config: TaskConfig):
        """Required fields should still be validated."""
        ExperimentModel = create_experiment_model(string_field_config)

        # Missing required field should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            ExperimentModel(length=12.0, ph=4.0)  # Missing 'formula'

    def test_mixed_numeric_and_string_values(self, string_field_config: TaskConfig):
        """Model should handle mix of numeric and string values."""
        ExperimentModel = create_experiment_model(string_field_config)

        exp = ExperimentModel(
            formula="Fe3O4",
            length=12.0,  # float
            width="15",   # string
            ph=4,         # int
            temperature="40.0"  # string
        )

        assert exp.length == "12"
        assert exp.width == "15"
        assert exp.ph == "4"
        assert exp.temperature == "40.0"

    def test_boolean_to_string_conversion(self, string_field_config: TaskConfig):
        """Boolean values should convert to 'true'/'false' strings."""
        # Add a boolean-like field test
        config_with_bool = TaskConfig(
            name="test_bool",
            experiment_fields={
                "formula": FieldSpec(type=str, description="Formula", required=True),
                "active": FieldSpec(type=str, description="Active flag", required=False),
            },
            compare_fields=["formula"],
            float_tolerance=0.05,
        )
        BoolModel = create_experiment_model(config_with_bool)

        exp = BoolModel(formula="Fe3O4", active=True)
        assert exp.active == "true"

        exp2 = BoolModel(formula="Fe3O4", active=False)
        assert exp2.active == "false"


@pytest.mark.unit
class TestFloatFieldNoConversion:
    """Test that float-typed fields don't get string conversion."""

    @pytest.fixture
    def float_field_config(self) -> TaskConfig:
        """Create a TaskConfig with float-typed fields."""
        return TaskConfig(
            name="test_float",
            experiment_fields={
                "km_value": FieldSpec(type=float, description="Km value", required=False),
                "vmax_value": FieldSpec(type=float, description="Vmax value", required=False),
                "formula": FieldSpec(type=str, description="Formula", required=True),
            },
            compare_fields=["formula", "km_value"],
            float_tolerance=0.05,
        )

    def test_float_fields_stay_float(self, float_field_config: TaskConfig):
        """Float-typed fields should remain as floats."""
        ExperimentModel = create_experiment_model(float_field_config)

        exp = ExperimentModel(formula="Fe3O4", km_value=1.502, vmax_value=1.472e-07)

        assert isinstance(exp.km_value, float)
        assert exp.km_value == 1.502
        assert isinstance(exp.vmax_value, float)
        assert exp.vmax_value == 1.472e-07

    def test_string_to_float_conversion(self, float_field_config: TaskConfig):
        """String values should be convertible to float for float fields."""
        ExperimentModel = create_experiment_model(float_field_config)

        exp = ExperimentModel(formula="Fe3O4", km_value="1.502")

        assert isinstance(exp.km_value, float)
        assert exp.km_value == 1.502


@pytest.mark.unit
class TestNanozymesConfig:
    """Test with actual nanozymes-like configuration."""

    @pytest.fixture
    def nanozymes_like_config(self) -> TaskConfig:
        """Create a config similar to nanozymes.yaml."""
        return TaskConfig(
            name="nanozymes_test",
            experiment_fields={
                "formula": FieldSpec(type=str, description="Chemical formula", required=True),
                "activity": FieldSpec(
                    type=str,
                    description="Catalytic activity",
                    choices=["peroxidase", "oxidase", "catalase", "laccase"],
                    required=True,
                ),
                "length": FieldSpec(type=str, description="Length in nm", required=False),
                "width": FieldSpec(type=str, description="Width in nm", required=False),
                "depth": FieldSpec(type=str, description="Depth in nm", required=False),
                "km_value": FieldSpec(type=str, description="Km value", required=False),
                "vmax_value": FieldSpec(type=str, description="Vmax value", required=False),
                "ph": FieldSpec(type=str, description="pH value", required=False),
                "temperature": FieldSpec(type=str, description="Temperature in °C", required=False),
                "c_const": FieldSpec(type=str, description="Constant concentration", required=False),
            },
            compare_fields=["formula", "activity", "length", "ph"],
            float_tolerance=0.05,
        )

    def test_nanozymes_like_float_values(self, nanozymes_like_config: TaskConfig):
        """Test nanozymes-like config with float values from LLM."""
        ExperimentModel = create_experiment_model(nanozymes_like_config)

        # Simulate LLM response with numeric values
        exp = ExperimentModel(
            formula="Fe3O4",
            activity="peroxidase",
            length=12.0,
            width=12.0,
            depth=12.0,
            km_value=1.502,
            vmax_value=1.472e-07,
            ph=4.0,
            temperature=40.0,
            c_const=20.0,
        )

        # All str fields should have string values
        assert exp.length == "12"
        assert exp.width == "12"
        assert exp.depth == "12"
        assert exp.km_value == "1.502"
        assert exp.vmax_value == "1.472e-07"
        assert exp.ph == "4"
        assert exp.temperature == "40"
        assert exp.c_const == "20"

    def test_nanozymes_like_mixed_values(self, nanozymes_like_config: TaskConfig):
        """Test nanozymes-like config with mixed value types."""
        ExperimentModel = create_experiment_model(nanozymes_like_config)

        exp = ExperimentModel(
            formula="CeO2",
            activity="catalase",
            length="10-15",  # Range string
            width=8.5,       # Float
            depth="5",       # String
            km_value=0.065,  # Small float
            vmax_value="5.65e-08",  # String scientific notation
            ph=4.0,
            temperature=40,  # Int
            c_const=0.32,    # Float
        )

        assert exp.length == "10-15"
        assert exp.width == "8.5"
        assert exp.depth == "5"
        assert exp.km_value == "0.065"
        assert exp.vmax_value == "5.65e-08"
        assert exp.ph == "4"
        assert exp.temperature == "40"
        assert exp.c_const == "0.32"
