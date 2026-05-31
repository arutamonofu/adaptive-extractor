# mypy: ignore-errors
# flake8: noqa: F821

import pytest

from ae.core.evaluation import ExperimentMatcher
from ae.core.evaluation.matcher import _extract_first_json


@pytest.fixture(autouse=True)
def _setup_experiment_model(nanozyme_task, request):
    """Setup experiment_model at module level for matcher tests."""
    request.module.experiment_model = nanozyme_task["experiment_model"]


@pytest.mark.unit
class TestEvaluationEngine:
    """Consolidated unit tests for the evaluation matcher and judges."""

    def test_string_normalization(self):
        """Test normalization variants (whitespace, dash characters, case-sensitive)."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)
        assert matcher._normalize_text("Fe3O4") == "Fe3O4"
        assert matcher._normalize_text("  Fe3O4  ") == "Fe3O4"
        assert matcher._normalize_text("Fe−3O4") == matcher._normalize_text("Fe-3O4")
        assert matcher._normalize_text("Fe 3 O4") == "Fe3O4"
        assert matcher._normalize_text(None) == ""
        assert matcher._normalize_text(123) == "123"

    @pytest.mark.parametrize("pred,gold,expected", [
        (0.05, 0.05, True),
        (0.05, 0.050000000001, True),
        (0.054, 0.05, False),
        (1e-10, 0.0, False),
    ])
    def test_float_comparison(self, pred, gold, expected):
        """Test float comparison using strict tolerance (rel_tol=1e-9)."""
        matcher = ExperimentMatcher(fields_to_compare=["km_value"], float_tolerance=0.05)
        assert matcher._compare_floats(pred, gold) is expected

    def test_general_value_matching(self):
        """Test general is_match logic for mixed types, None values and strings."""
        matcher = ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=0.05)
        assert matcher._is_match("Fe3O4", "Fe3O4") is True
        assert matcher._is_match("FE3O4", "fe3o4") is False
        assert matcher._is_match(None, None) is True
        assert matcher._is_match("Fe3O4", None) is False
        assert matcher._is_match("10.0", 10.0) is True

    def test_hungarian_pairs_alignment(self):
        """Test alignment of predictions and ground truths via Hungarian algorithm."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        # Empty
        assert matcher.align_pairs([], []) == []

        preds = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="Au", activity="catalase"),
        ]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="ZnO", activity="catalase"),
        ]

        pairs = matcher.align_pairs(preds, gts)
        assert len(pairs) == 2
        # Fe3O4 should be paired with Fe3O4
        matched = [p for p in pairs if p[0] and p[1] and p[0].formula == "Fe3O4"]
        assert len(matched) == 1

    def test_f1_computation(self):
        """Test detailed precision, recall, and F1 score reporting."""
        matcher = ExperimentMatcher(fields_to_compare=["formula", "activity"], float_tolerance=0.05)

        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [
            experiment_model(formula="Fe3O4", activity="peroxidase"),
            experiment_model(formula="CuO", activity="oxidase"),
        ]

        report = matcher.get_detailed_report(preds, gts)
        assert report["precision"] == 1.0
        assert report["recall"] == 0.5
        assert report["f1"] == 2/3

        # Fields scores
        assert report["fields"]["formula"] == 0.5
        assert report["fields"]["activity"] == 0.5

    def test_matcher_validation_rules(self):
        """Test initialization constraints for comparison fields and tolerance."""
        with pytest.raises(ValueError, match="fields_to_compare"):
            ExperimentMatcher(fields_to_compare=[], float_tolerance=0.05)

        with pytest.raises(ValueError, match="float_tolerance"):
            ExperimentMatcher(fields_to_compare=["formula"], float_tolerance=-0.1)

    def test_semantic_judge(self):
        """Test Semantic Judge workflow, prompts, mock verdicts, and exceptions."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity"],
            float_tolerance=0.05,
            enable_semantic_judge=False,
        )
        # Verify it works correctly with judge disabled
        preds = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        gts = [experiment_model(formula="Fe3O4", activity="peroxidase")]
        report = matcher.get_detailed_report(preds, gts)
        assert report["f1"] == 1.0

        # Mock LLM that returns YES verdict
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                return ['{"formula": "YES"}']

        matcher_with_judge = ExperimentMatcher(
            fields_to_compare=["formula"],
            float_tolerance=0.05,
            student_llm=MockLLM(),
            enable_semantic_judge=True,
        )

        verdicts = matcher_with_judge._call_semantic_judge(
            task_name="test",
            gt_json={"formula": "Fe3O4"},
            pred_json={"formula": "iron oxide"},
            discrepancies=["formula"],
        )
        assert verdicts == {"formula": "YES"}

        # Exception fallback
        class FailLLM:
            def __call__(self, prompt, **kwargs):
                raise RuntimeError("LLM Failure")

        matcher_fail = ExperimentMatcher(
            fields_to_compare=["formula"],
            float_tolerance=0.05,
            student_llm=FailLLM(),
            enable_semantic_judge=True,
        )
        assert matcher_fail._call_semantic_judge("test", {"f": "A"}, {"f": "B"}, ["f"]) == {}


@pytest.mark.unit
class TestExtractFirstJson:
    """Tests for the brace-balancing JSON extractor utility."""

    def test_balanced_json_parsing(self):
        """Test extraction of clean, nested, and markdown-wrapped JSON."""
        assert _extract_first_json('{"a": 1}') == '{"a": 1}'
        assert _extract_first_json('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert _extract_first_json('{"outer": {"inner": 2}}') == '{"outer": {"inner": 2}}'
        assert _extract_first_json('{"a": 1} trailing text') == '{"a": 1}'
        assert _extract_first_json('just text') is None
        assert _extract_first_json('{"formula": "Fe{3}O4"}') == '{"formula": "Fe{3}O4"}'
