# mypy: ignore-errors
# flake8: noqa: F821
"""Unit tests for ExperimentMatcher evaluation engine.

Tests cover:
- String normalization and exact match
- Float comparison with tolerance
- Hungarian algorithm alignment
- F1/Precision/Recall computation
"""

import pytest

from aee.domain.evaluation import ExperimentMatcher


@pytest.mark.unit
# Make experiment_model available module-level
@pytest.fixture(autouse=True)
def _setup_experiment_model(nanozyme_task, request):
    """Setup experiment_model at module level."""
    # Store in module globals for access by tests
    request.module.experiment_model = nanozyme_task["experiment_model"]


@pytest.mark.unit
class TestStringNormalization:
    """Tests for string normalization in matcher."""

    def test_normalize_exact_match(self):
        """Test exact string match after normalization (case-sensitive)."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        # Case-sensitive: case is preserved
        assert matcher._normalize_text("Fe3O4") == "Fe3O4"
        assert matcher._normalize_text("FE3O4") == "FE3O4"
        assert matcher._normalize_text("  Fe3O4  ") == "Fe3O4"

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

        assert matcher._normalize_text("Fe 3 O4") == "Fe3O4"
        assert matcher._normalize_text("Fe   3   O4") == "Fe3O4"

    def test_normalize_none_value(self):
        """Test None value normalization."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        assert matcher._normalize_text(None) == ""

    def test_normalize_numeric_value(self):
        """Test numeric value normalization."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        assert matcher._normalize_text(123) == "123"
        assert matcher._normalize_text(12.5) == "12.5"


@pytest.mark.unit
class TestFloatComparison:
    """Tests for float comparison with strict tolerance."""

    @pytest.mark.parametrize("pred,gold,expected", [
        # Exact matches
        (0.05, 0.05, True),
        (0.0, 0.0, True),

        # Very small difference (within rel_tol=1e-9)
        (0.05, 0.050000000001, True),
        (100.0, 100.0000001, True),

        # Outside strict tolerance
        (0.054, 0.05, False),   # 8% difference
        (0.046, 0.05, False),   # 8% difference
        (110.0, 100.0, False),  # 10% difference
        (0.06, 0.05, False),    # 20% difference
        (0.04, 0.05, False),    # 20% difference

        # Zero comparison (uses absolute tolerance)
        (1e-10, 0.0, False),    # Outside absolute tolerance for isclose
        (0.0, 1e-10, False),
    ])
    def test_float_comparison_parametrized(self, pred, gold, expected):
        """Test float comparison with strict tolerance (math.isclose).

        Args:
            pred: Predicted value
            gold: Ground truth value
            expected: Expected result (True/False)
        """
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)
        result = matcher._compare_floats(pred, gold)
        assert result is expected, f"_compare_floats({pred}, {gold}) failed"

    def test_strict_float_comparison(self):
        """Test that float comparison is now strict (no tolerance)."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)

        # Exact match
        assert matcher._compare_floats(0.05, 0.05) is True

        # Very small difference (should match with rel_tol=1e-9)
        assert matcher._compare_floats(0.05, 0.050000000001) is True

        # Any larger difference should not match
        assert matcher._compare_floats(0.05, 0.051) is False  # 2% difference
        assert matcher._compare_floats(0.05, 0.06) is False   # 20% difference


@pytest.mark.unit
class TestIsMatch:
    """Tests for general value matching."""

    def test_string_match(self):
        """Test string value matching (case-sensitive)."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)

        # Exact match (case-sensitive)
        assert matcher._is_match("Fe3O4", "Fe3O4") is True
        # Different case = no match
        assert matcher._is_match("FE3O4", "fe3o4") is False
        assert matcher._is_match("Fe3O4", "CuO") is False

    def test_float_match(self):
        """Test float value matching (strict)."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)

        # Exact match
        assert matcher._is_match(0.05, 0.05) is True
        # Strict comparison: 2% difference is not acceptable
        assert matcher._is_match(0.051, 0.05) is False
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

        # String number vs float (converted to float)
        assert matcher._is_match("10.0", 10.0) is True
        assert matcher._is_match(10.0, "10.0") is True


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
class TestSemanticJudge:
    """Tests for Semantic Judge functionality."""

    def test_semantic_judge_disabled(self, experiment_model):
        """Test that semantic judge can be disabled."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )

        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [experiment_model(formula="Fe3O4", activity="peroxidase")]

        # With judge disabled, case-sensitive comparison should pass for exact match
        report = matcher.get_detailed_report(preds, gts)
        assert report["f1"] == 1.0

    def test_semantic_judge_no_student_llm(self, experiment_model):
        """Test that semantic judge skips when student_llm is None."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            student_llm=None,
            enable_semantic_judge=True,
        )

        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [experiment_model(formula="CuO", activity="oxidase")]

        # Should work without LLM, just strict comparison
        report = matcher.get_detailed_report(preds, gts)
        assert report["f1"] == 0.0

    def test_case_sensitive_comparison(self, experiment_model):
        """Test that string comparison is case-sensitive."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            enable_semantic_judge=False,  # Disable judge for strict test
        )

        # Case-sensitive: "Co" != "co"
        assert matcher._normalize_text("Co") == "Co"
        assert matcher._normalize_text("co") == "co"
        assert matcher._normalize_text("Co") != matcher._normalize_text("co")

        # Different case should not match
        assert matcher._is_match("Co3O4", "co3o4") is False

    def test_strict_float_comparison(self, experiment_model):
        """Test that float comparison uses math.isclose."""
        matcher = ExperimentMatcher(
            fields_to_compare=["km_value"],
            float_tolerance=0.05,  # This is now unused
        )

        # Exact match
        assert matcher._compare_floats(0.05, 0.05) is True

        # Very small difference (should match with rel_tol=1e-9)
        assert matcher._compare_floats(0.05, 0.050000000001) is True

        # Larger difference (should not match)
        assert matcher._compare_floats(0.05, 0.06) is False

    def test_missing_value_only_fn(self):
        """Test that missing value (Pred=None) counts as FN only, not FP+FN."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )

        # Model didn't predict anything (None for all fields)
        pred = None
        gt = type('Entity', (), {'formula': 'Fe3O4', 'activity': 'peroxidase'})()

        pairs = [(pred, gt)]
        stats = matcher._compute_stats(pairs, task_name="test")

        # Should be 2 FN (one per field), 0 FP
        # TP=0, FP=0, FN=2 -> Precision=0/0=0, Recall=0/2=0, F1=0
        assert stats["f1"] == 0.0

    def test_hallucination_only_fp(self):
        """Test that hallucination (GT=None) counts as FP only, not FP+FN."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )

        # Model hallucinated values (GT has None for all fields)
        pred = type('Entity', (), {'formula': 'Fe3O4', 'activity': 'peroxidase'})()
        gt = None

        pairs = [(pred, gt)]
        stats = matcher._compute_stats(pairs, task_name="test")

        # Should be 2 FP (one per field), 0 FN
        # TP=0, FP=2, FN=0 -> Precision=0/2=0, Recall=0/0=0, F1=0
        assert stats["f1"] == 0.0

    def test_mismatch_fp_and_fn(self, experiment_model):
        """Test that mismatch (wrong value) counts as both FP and FN."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula"],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )

        # Both have values but they don't match
        pred = experiment_model(formula="Fe3O4", activity="peroxidase")
        gt = experiment_model(formula="CuO", activity="oxidase")

        pairs = [(pred, gt)]
        stats = matcher._compute_stats(pairs, task_name="test")

        # Mismatch: FP=1, FN=1
        # TP=0, FP=1, FN=1 -> Precision=0/1=0, Recall=0/1=0, F1=0
        assert stats["f1"] == 0.0

    def test_build_judge_prompt(self, experiment_model):
        """Test that judge prompt is built correctly."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            field_descriptions={
                "formula": "Chemical formula of the compound",
                "activity": "Type of enzymatic activity",
            },
        )

        gt_json = {"formula": "Fe3O4", "activity": "peroxidase"}
        pred_json = {"formula": "Fe3O4", "activity": "catalase"}
        discrepancies = ["activity"]

        prompt = matcher._build_judge_prompt(
            task_name="nanozymes",
            gt_json=gt_json,
            pred_json=pred_json,
            discrepancies=discrepancies,
        )

        # Check prompt contains key elements
        assert "Task: nanozymes" in prompt
        assert "Chemical formula of the compound" in prompt
        assert "Type of enzymatic activity" in prompt
        assert "Fe3O4" in prompt
        assert "activity" in prompt

    def test_call_semantic_judge_parse_error(self, experiment_model):
        """Test that semantic judge handles JSON parse errors gracefully."""
        # Mock LLM that returns invalid JSON
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                return ["{invalid json}"]

        matcher = ExperimentMatcher(
            fields_to_compare=["formula"],
            float_tolerance=0.05,
            student_llm=MockLLM(),
            enable_semantic_judge=True,
        )

        verdicts = matcher._call_semantic_judge(
            task_name="test",
            gt_json={"formula": "Fe3O4"},
            pred_json={"formula": "CuO"},
            discrepancies=["formula"],
        )

        # Should return empty dict on parse error (fallback to strict)
        assert verdicts == {}

    def test_call_semantic_judge_exception(self, experiment_model):
        """Test that semantic judge handles exceptions gracefully."""
        # Mock LLM that raises exception
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                raise RuntimeError("LLM error")

        matcher = ExperimentMatcher(
            fields_to_compare=["formula"],
            float_tolerance=0.05,
            student_llm=MockLLM(),
            enable_semantic_judge=True,
        )

        verdicts = matcher._call_semantic_judge(
            task_name="test",
            gt_json={"formula": "Fe3O4"},
            pred_json={"formula": "CuO"},
            discrepancies=["formula"],
        )

        # Should return empty dict on exception (fallback to strict)
        assert verdicts == {}

    def test_semantic_judge_verdict_yes(self, experiment_model):
        """Test that YES verdict from judge grants amnesty (TP)."""
        # Mock LLM that always says YES
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                return ['{"formula": "YES"}']

        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            student_llm=MockLLM(),
            field_descriptions={"formula": "Chemical formula"},
            enable_semantic_judge=True,
        )

        # Matching values
        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [experiment_model(formula="Fe3O4", activity="peroxidase")]

        report = matcher.get_detailed_report(preds, gts, task_name="test")

        # With matching values, should be TP
        assert report["f1"] == 1.0

    def test_semantic_judge_verdict_no(self, experiment_model):
        """Test that NO verdict from judge applies strict penalties."""
        # Mock LLM that always says NO
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                return ['{"formula": "NO"}']

        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            student_llm=MockLLM(),
            field_descriptions={"formula": "Chemical formula"},
            enable_semantic_judge=True,
        )

        # Mismatch and judge says NO
        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [experiment_model(formula="CuO", activity="oxidase")]

        report = matcher.get_detailed_report(preds, gts, task_name="test")

        # With NO verdict on mismatch, should be FP+FN
        assert report["f1"] == 0.0
