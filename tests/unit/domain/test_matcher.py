"""Unit tests for ExperimentMatcher evaluation engine.

Tests cover:
- String normalization and exact match
- Float comparison with tolerance
- Hungarian algorithm alignment
- F1/Precision/Recall computation
"""

import pytest

from aee.domain.evaluation.matcher import ExperimentMatcher


# Make experiment_model available module-level
@pytest.fixture(autouse=True)
def _setup_experiment_model(nanozyme_task, request):
    """Setup experiment_model at module level."""
    # Store in module globals for access by tests
    request.module.experiment_model = nanozyme_task["experiment_model"]


class TestStringNormalization:
    """Tests for string normalization in matcher."""

    def test_normalize_exact_match(self):
        """Test exact string match after normalization."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        assert matcher._normalize_text("Fe3O4") == "fe3o4"
        assert matcher._normalize_text("FE3O4") == "fe3o4"
        assert matcher._normalize_text("  Fe3O4  ") == "fe3o4"

    def test_normalize_dash_variants(self):
        """Test normalization of different dash characters."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        # Different dash types should normalize to same string
        assert matcher._normalize_text("Fe−3O4") == matcher._normalize_text("Fe-3O4")
        assert matcher._normalize_text("Fe–3O4") == matcher._normalize_text("Fe-3O4")
        assert matcher._normalize_text("Fe—3O4") == matcher._normalize_text("Fe-3O4")

    def test_normalize_whitespace(self):
        """Test whitespace removal in normalization."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        assert matcher._normalize_text("Fe 3 O4") == "fe3o4"
        assert matcher._normalize_text("Fe   3   O4") == "fe3o4"

    def test_normalize_none_value(self):
        """Test None value normalization."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        assert matcher._normalize_text(None) == ""

    def test_normalize_numeric_value(self):
        """Test numeric value normalization."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        assert matcher._normalize_text(123) == "123"
        assert matcher._normalize_text(12.5) == "12.5"


class TestFloatComparison:
    """Tests for float comparison with tolerance."""

    def test_exact_float_match(self):
        """Test exact float match."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)
        
        assert matcher._compare_floats(0.05, 0.05) is True

    def test_float_within_tolerance(self):
        """Test floats within tolerance are considered equal."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.10)
        
        # 0.055 is ~9.09% higher than 0.05 - should match
        assert matcher._compare_floats(0.055, 0.05) is True
        # 0.046 is 8% lower than 0.05 - should match (within 10%)
        assert matcher._compare_floats(0.046, 0.05) is True
        
        # 0.054 is 8% higher - should match with 10% tolerance
        assert matcher._compare_floats(0.054, 0.05) is True
        
        # Edge case: 0.045 is exactly 10% lower, but due to floating point precision
        # it's slightly over 10%, so we test with a slightly higher tolerance
        matcher_11 = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.11)
        assert matcher_11._compare_floats(0.045, 0.05) is True

    def test_float_outside_tolerance(self):
        """Test floats outside tolerance are different."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)
        
        # 0.06 is 20% higher than 0.05
        assert matcher._compare_floats(0.06, 0.05) is False
        # 0.04 is 20% lower than 0.05
        assert matcher._compare_floats(0.04, 0.05) is False
        
        # 0.053 is 6% higher - outside 5% tolerance
        assert matcher._compare_floats(0.053, 0.05) is False

    def test_zero_comparison(self):
        """Test comparison with zero values."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)
        
        assert matcher._compare_floats(0.0, 0.0) is True
        # Very small number vs zero - uses absolute tolerance for zero
        # The implementation uses 1e-9 threshold
        assert matcher._compare_floats(1e-10, 0.0) is True  # Within 1e-9
        assert matcher._compare_floats(1e-8, 0.0) is False  # Outside 1e-9

    def test_custom_tolerance(self):
        """Test custom tolerance values."""
        strict_matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.01)
        lenient_matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.20)
        
        # 0.052 is 4% higher than 0.05
        assert strict_matcher._compare_floats(0.052, 0.05) is False  # 4% > 1%
        assert lenient_matcher._compare_floats(0.052, 0.05) is True  # 4% < 20%


class TestIsMatch:
    """Tests for general value matching."""

    def test_string_match(self):
        """Test string value matching."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        assert matcher._is_match("Fe3O4", "Fe3O4") is True
        assert matcher._is_match("FE3O4", "fe3o4") is True
        assert matcher._is_match("Fe3O4", "CuO") is False

    def test_float_match(self):
        """Test float value matching."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)
        
        assert matcher._is_match(0.05, 0.05) is True
        assert matcher._is_match(0.051, 0.05) is True
        assert matcher._is_match(0.06, 0.05) is False

    def test_none_comparison(self):
        """Test None value comparison."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        # Both None = match
        assert matcher._is_match(None, None) is True
        # One None = no match
        assert matcher._is_match("Fe3O4", None) is False
        assert matcher._is_match(None, "Fe3O4") is False

    def test_mixed_types(self):
        """Test comparison of mixed types."""
        matcher = ExperimentMatcher(fields_to_compare=["length"], float_tolerance=0.05)

        # String number vs float
        assert matcher._is_match("10.0", 10.0) is True
        assert matcher._is_match(10.0, "10.0") is True


class TestAlignPairs:
    """Tests for Hungarian algorithm alignment."""

    @pytest.mark.usefixtures("experiment_model")
    def test_align_empty_lists(self):
        """Test alignment of empty lists."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        pairs = matcher.align_pairs([], [])
        assert pairs == []

    def test_align_preds_empty(self):
        """Test alignment when predictions are empty."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]

        pairs = matcher.align_pairs([], gts)

        # All GTs should be paired with None (False Negatives)
        assert len(pairs) == 2
        assert all(pred is None for pred, _ in pairs)

    def test_align_gts_empty(self):
        """Test alignment when ground truths are empty."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)
        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]

        pairs = matcher.align_pairs(preds, [])

        # All preds should be paired with None (False Positives)
        assert len(pairs) == 2
        assert all(gt is None for _, gt in pairs)

    def test_align_perfect_match(self):
        """Test alignment with perfect matches."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]

        pairs = matcher.align_pairs(preds, gts)

        assert len(pairs) == 2
        # All should be matched (no None pairs)
        assert all(pred is not None and gt is not None for pred, gt in pairs)

    def test_align_partial_match(self):
        """Test alignment with partial matches."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="Au", activity="catalase"),  # Extra (FP)
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="ZnO", activity="catalase"),  # Missing (FN)
        ]

        pairs = matcher.align_pairs(preds, gts)

        # Should have 2 pairs (Hungarian algorithm balances)
        # 1 matched pair (Fe3O4) + 1 mismatched pair (Au vs ZnO)
        assert len(pairs) == 2

        # Fe3O4 should be matched correctly
        fe_matches = [
            (p, g) for p, g in pairs
            if p and p.formula == "Fe3O4" and g and g.formula == "Fe3O4"
        ]
        assert len(fe_matches) == 1

    def test_align_multiple_candidates(self):
        """Test alignment with multiple similar candidates."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity", "length"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase", length=10.0),
            experiment_model(formula="Fe3O4", activity="peroxidase", length=12.0),  # Closer to GT
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase", length=11.0),
        ]

        pairs = matcher.align_pairs(preds, gts)

        # Should have 2 pairs (1 matched + 1 unmatched pred)
        matched_pairs = [(p, g) for p, g in pairs if p is not None and g is not None]
        assert len(matched_pairs) == 1

        # The closer one (length=12.0, diff=1.0) should be matched
        # vs (length=10.0, diff=1.0) - both have same diff, so first might be chosen
        # Just verify one of them is matched
        assert matched_pairs[0][0].formula == "Fe3O4"
        assert matched_pairs[0][0].length in [10.0, 12.0]


class TestF1Computation:
    """Tests for F1/Precision/Recall computation."""

    def test_perfect_prediction(self):
        """Test F1 for perfect prediction."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]
        
        report = matcher.get_detailed_report(preds, gts)
        
        assert report["f1"] == 1.0
        assert report["precision"] == 1.0
        assert report["recall"] == 1.0

    def test_false_positives(self):
        """Test F1 with false positives."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="Au", activity="catalase"),  # FP
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
        ]
        
        report = matcher.get_detailed_report(preds, gts)
        
        assert report["f1"] < 1.0
        assert report["precision"] < 1.0
        assert report["recall"] == 1.0

    def test_false_negatives(self):
        """Test F1 with false negatives."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),  # FN
        ]
        
        report = matcher.get_detailed_report(preds, gts)
        
        assert report["f1"] < 1.0
        assert report["precision"] == 1.0
        assert report["recall"] < 1.0

    def test_complete_miss(self):
        """Test F1 for complete miss (no correct predictions)."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Au", activity="catalase"),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
        ]
        
        report = matcher.get_detailed_report(preds, gts)
        
        assert report["f1"] == 0.0
        assert report["precision"] == 0.0
        assert report["recall"] == 0.0

    def test_optimization_score(self):
        """Test get_optimization_score returns F1."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]
        
        f1 = matcher.get_optimization_score(preds, gts)
        
        assert 0.0 <= f1 <= 1.0
        assert isinstance(f1, float)

    def test_field_level_scores(self):
        """Test per-field score computation."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity", "length"], float_tolerance=0.05)

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase", length=10.0),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="oxidase", length=10.0),  # Wrong activity
        ]
        
        report = matcher.get_detailed_report(preds, gts)
        
        assert "fields" in report
        assert "formula" in report["fields"]
        assert "activity" in report["fields"]
        assert "length" in report["fields"]
        
        # Formula and length should match, activity should not
        assert report["fields"]["formula"] == 1.0
        assert report["fields"]["length"] == 1.0
        assert report["fields"]["activity"] < 1.0


class TestMatcherInitialization:
    """Tests for ExperimentMatcher initialization and validation."""

    def test_empty_fields_raises(self):
        """Test that empty fields_to_compare raises ValueError."""
        with pytest.raises(ValueError, match="fields_to_compare"):
            ExperimentMatcher(fields_to_compare=[], float_tolerance=0.05)

    def test_invalid_tolerance_raises(self):
        """Test that invalid tolerance raises ValueError."""
        with pytest.raises(ValueError, match="float_tolerance"):
            ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=-0.1)
        
        with pytest.raises(ValueError, match="float_tolerance"):
            ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=1.5)

    def test_valid_initialization(self):
        """Test valid matcher initialization."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.10,
        )
        
        assert matcher.fields == ["formula", "activity"]
        assert matcher.tolerance == 0.10
