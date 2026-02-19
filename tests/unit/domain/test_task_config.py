"""Unit tests for task configuration and dynamic model generation.

Tests cover:
- TaskDefinition implementation (NanozymeTask)
- DSPy signature creation
- Dynamic experiment model generation
- Task validation
"""

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from aee.domain.tasks.nanozymes import (
    NanozymeExperiment,
    NanozymeExtractionOutput,
    row_to_nanozyme,
)
from aee.shared.exceptions import TaskValidationError


class TestNanozymeTask:
    """Tests for NanozymeTask implementation."""

    def test_task_name(self, nanozyme_task):
        """Test task name property."""
        assert nanozyme_task.name == "nanozymes"

    def test_task_description(self, nanozyme_task):
        """Test task description property."""
        assert "nanozyme" in nanozyme_task.description.lower()
        assert len(nanozyme_task.description) > 20

    def test_task_validate_success(self, nanozyme_task):
        """Test task validation passes for valid task."""
        # Should not raise
        nanozyme_task.validate()

    def test_task_compare_fields(self, nanozyme_task):
        """Test compare fields are defined."""
        compare_fields = nanozyme_task.compare_fields
        assert isinstance(compare_fields, list)
        assert len(compare_fields) > 0
        assert "formula" in compare_fields
        assert "activity" in compare_fields

    def test_task_float_tolerance(self, nanozyme_task):
        """Test float tolerance property."""
        assert nanozyme_task.float_tolerance == 0.10

    def test_task_to_dict(self, nanozyme_task):
        """Test task serialization to dictionary."""
        task_dict = nanozyme_task.to_dict()
        assert task_dict["name"] == "nanozymes"
        assert "signature" in task_dict
        assert "output_model" in task_dict
        assert "experiment_model" in task_dict
        assert "compare_fields" in task_dict


class TestNanozymeSignature:
    """Tests for DSPy signature creation."""

    def test_signature_creation(self, nanozyme_signature_class):
        """Test signature class is created successfully."""
        assert nanozyme_signature_class is not None
        
    def test_signature_has_input_field(self, nanozyme_signature_class):
        """Test signature has document_text input field."""
        # Check signature has the expected fields via model_fields
        assert hasattr(nanozyme_signature_class, "model_fields")
        assert "document_text" in nanozyme_signature_class.model_fields
        
    def test_signature_has_output_field(self, nanozyme_signature_class):
        """Test signature has extracted_data output field."""
        assert hasattr(nanozyme_signature_class, "model_fields")
        assert "extracted_data" in nanozyme_signature_class.model_fields

    def test_signature_with_empty_instruction_raises(self):
        """Test that empty instruction raises ValueError."""
        from aee.domain.tasks.nanozymes.signature import create_nanozyme_signature
        
        with pytest.raises(ValueError, match="cannot be empty"):
            create_nanozyme_signature(instruction="")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            create_nanozyme_signature(instruction="   ")


class TestNanozymeExperimentModel:
    """Tests for NanozymeExperiment Pydantic model."""

    def test_create_minimal_experiment(self):
        """Test creating experiment with required fields only."""
        experiment = NanozymeExperiment(
            formula="Fe3O4",
            activity="peroxidase",
        )
        assert experiment.formula == "Fe3O4"
        assert experiment.activity == "peroxidase"

    def test_create_full_experiment(self):
        """Test creating experiment with all fields."""
        experiment = NanozymeExperiment(
            formula="Fe3O4",
            activity="peroxidase",
            surface="naked",
            syngony="cubic",
            length=10.0,
            width=5.0,
            depth=3.0,
            reaction_type="TMB+H2O2",
            km_value=0.05,
            km_unit="mM",
            vmax_value=100.0,
            vmax_unit="M/s",
            ph=7.0,
            temperature=25.0,
            c_min=0.1,
            c_max=1.0,
            c_const=0.5,
            c_const_unit="mM",
            ccat_value=10.0,
            ccat_unit="mcg/ml",
        )
        assert experiment.formula == "Fe3O4"
        assert experiment.length == 10.0
        assert experiment.km_value == 0.05
        assert experiment.ph == 7.0

    def test_experiment_with_none_values(self):
        """Test creating experiment with optional None values."""
        experiment = NanozymeExperiment(
            formula="Fe3O4",
            activity="peroxidase",
            surface=None,
            length=None,
            km_value=None,
        )
        assert experiment.surface is None
        assert experiment.length is None
        assert experiment.km_value is None

    def test_experiment_validation_invalid_activity(self):
        """Test that invalid activity raises validation error."""
        with pytest.raises(ValidationError):
            NanozymeExperiment(
                formula="Fe3O4",
                activity="invalid_activity",  # type: ignore
            )

    def test_experiment_validation_valid_activities(self):
        """Test all valid activity values."""
        valid_activities = [
            "peroxidase",
            "oxidase",
            "catalase",
            "laccase",
            "superoxide_dismutase",
            "glucose oxidase",
            "other",
        ]
        for activity in valid_activities:
            experiment = NanozymeExperiment(
                formula="Fe3O4",
                activity=activity,  # type: ignore
            )
            assert experiment.activity == activity

    def test_experiment_to_dict(self):
        """Test experiment serialization to dictionary."""
        experiment = NanozymeExperiment(
            formula="Fe3O4",
            activity="peroxidase",
            length=10.0,
        )
        exp_dict = experiment.to_dict()
        assert exp_dict["formula"] == "Fe3O4"
        assert exp_dict["activity"] == "peroxidase"
        assert exp_dict["length"] == 10.0

    def test_experiment_from_dict(self):
        """Test experiment deserialization from dictionary."""
        data = {
            "formula": "CuO",
            "activity": "oxidase",
            "ph": 7.5,
        }
        experiment = NanozymeExperiment.from_dict(data)
        assert experiment.formula == "CuO"
        assert experiment.activity == "oxidase"
        assert experiment.ph == 7.5


class TestNanozymeExtractionOutput:
    """Tests for NanozymeExtractionOutput model."""

    def test_create_empty_output(self):
        """Test creating empty extraction output."""
        output = NanozymeExtractionOutput()
        assert output.experiments == []

    def test_create_output_with_experiments(self, nanozyme_experiments):
        """Test creating output with experiments list."""
        output = NanozymeExtractionOutput(experiments=nanozyme_experiments)
        assert len(output.experiments) == 2
        assert output.experiments[0].formula == "Fe3O4"

    def test_output_model_has_experiments_field(self, nanozyme_task):
        """Test that output model has 'experiments' field (required by TaskDefinition)."""
        output_model = nanozyme_task.output_model
        assert "experiments" in output_model.model_fields


class TestRowConverter:
    """Tests for row_to_nanozyme converter function."""

    def test_convert_valid_row(self, sample_gt_dataframe):
        """Test converting a valid CSV row."""
        row = sample_gt_dataframe.iloc[0]
        experiment = row_to_nanozyme(row)
        
        assert experiment is not None
        assert experiment.formula == "Fe3O4"
        assert experiment.activity == "peroxidase"
        assert experiment.length == 10.0
        assert experiment.km_value == 0.05

    def test_convert_row_with_alt_keys(self):
        """Test converter handles alternative column names."""
        row = pd.Series({
            "formula": "Au",
            "activity": "peroxidase",
            "km_val": 0.03,  # Alternative key
            "vmax_val": 80.0,  # Alternative key
        })
        experiment = row_to_nanozyme(row)
        
        assert experiment is not None
        assert experiment.km_value == 0.03
        assert experiment.vmax_value == 80.0

    def test_convert_row_missing_required_field(self):
        """Test converter returns None when required field missing."""
        row = pd.Series({
            "activity": "peroxidase",
            # Missing 'formula'
        })
        experiment = row_to_nanozyme(row)
        assert experiment is None

    def test_convert_row_with_nan_values(self):
        """Test converter handles NaN values correctly."""
        import numpy as np
        
        row = pd.Series({
            "formula": "Fe3O4",
            "activity": "peroxidase",
            "length": np.nan,
            "km_value": np.nan,
            "surface": np.nan,
        })
        experiment = row_to_nanozyme(row)
        
        assert experiment is not None
        assert experiment.formula == "Fe3O4"
        assert experiment.length is None
        assert experiment.km_value is None
