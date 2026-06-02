import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import dspy
from pydantic import ValidationError
from contextlib import contextmanager

from ae.core.tasks.config import FieldSpec, TaskConfig
from ae.optimization.contrastive import (
    FieldSpecSummary,
    AnalysisInput,
    EntityObservation,
    FieldObservation,
    DocumentAnalysis,
    VerifiedRule,
    Discrepancy,
    AnalysisResult,
    HumanDecision,
    ReviewSession,
    LocalAnalyzer,
    prepare_analysis_inputs,
    StrictAggregator,
    build_three_level_prompt,
    HumanReviewCLI,
    merge_review_into_result
)


@contextmanager
def mock_dspy_context(*args, **kwargs):
    yield


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.unit
class TestContrastiveModels:
    """Test validation and behavior of Contrastive Pydantic models."""

    def test_verified_rule_evidence_count(self):
        # Valid rule
        rule = VerifiedRule(
            rule_id="r1",
            level="entity",
            rule_text="Always include X",
            evidence_count=2,
            evidence_examples=["doc1", "doc2"]
        )
        assert rule.evidence_count == 2

        # Invalid rule (evidence_count <= 0)
        with pytest.raises(ValidationError):
            VerifiedRule(
                rule_id="r1",
                level="entity",
                rule_text="Always include X",
                evidence_count=0,
                evidence_examples=[]
            )

    def test_discrepancy_consensus_ratio(self):
        # Valid discrepancy
        disc = Discrepancy(
            discrepancy_id="d1",
            level="field",
            field_name="temp",
            problem_description="Conflict on units",
            consensus_ratio=0.8,
            variant_a="Celsius",
            variant_b="Kelvin",
            example_documents=["doc1"]
        )
        assert disc.consensus_ratio == 0.8

        # Invalid discrepancy (consensus_ratio >= 1.0)
        with pytest.raises(ValidationError):
            Discrepancy(
                discrepancy_id="d1",
                level="field",
                field_name="temp",
                problem_description="Conflict on units",
                consensus_ratio=1.0,
                variant_a="Celsius",
                variant_b="Kelvin",
                example_documents=[]
            )

    def test_analysis_result_properties(self):
        rules = [
            VerifiedRule(rule_id="r1", level="entity", rule_text="E1", evidence_count=1, evidence_examples=[]),
            VerifiedRule(rule_id="r2", level="field", field_name="temp", rule_text="F1", evidence_count=1, evidence_examples=[]),
        ]
        result = AnalysisResult(
            task_name="test",
            analyzed_documents=2,
            verified_rules=rules,
            discrepancies=[],
            timestamp="2026-06-02"
        )
        assert len(result.entity_level_rules) == 1
        assert result.entity_level_rules[0].rule_id == "r1"
        assert len(result.field_level_rules) == 1
        assert result.field_level_rules[0].rule_id == "r2"
        assert result.get_rules_for_field("temp")[0].rule_id == "r2"

    def test_serialization_roundtrip(self, tmp_path: Path):
        rules = [VerifiedRule(rule_id="r1", level="entity", rule_text="E1", evidence_count=1, evidence_examples=[])]
        result = AnalysisResult(
            task_name="test",
            analyzed_documents=1,
            verified_rules=rules,
            discrepancies=[],
            timestamp="2026-06-02"
        )
        file_path = tmp_path / "result.json"
        result.to_json(file_path)
        
        loaded = AnalysisResult.from_json(file_path)
        assert loaded.task_name == "test"
        assert len(loaded.verified_rules) == 1


@pytest.mark.unit
class TestLocalAnalyzer:
    """Test LocalAnalyzer execution, caching and retry logic."""

    @pytest.mark.anyio
    async def test_caching_read_through(self, tmp_path: Path):
        task_config = TaskConfig(
            name="test_task",
            experiment_fields={"f1": FieldSpec(type=str, description="F1")},
            compare_fields=["f1"],
            float_tolerance=0.1
        )
        # Create a pre-existing valid cache file
        cache_dir = tmp_path / "analysis"
        cache_dir.mkdir()
        cache_file = cache_dir / "test_task_map_doc1.json"
        
        cached_data = {
            "document_id": "doc1",
            "entity_observations": [],
            "field_observations": [],
            "summary": "cached summary"
        }
        cache_file.write_text(json.dumps(cached_data))

        lm = MagicMock(spec=dspy.LM)
        analyzer = LocalAnalyzer(lm=lm, task_config=task_config, cache_dir=str(cache_dir), rate_limit_delay=0.0)
        
        inp = AnalysisInput(
            document_id="doc1",
            document_text="Hello",
            ground_truth_experiments=[],
            field_specs={}
        )
        result = await analyzer.analyze(inp)
        assert result.summary == "cached summary"
        # Verify LM was not called
        lm.assert_not_called()

    @pytest.mark.anyio
    async def test_retry_loop_with_feedback(self, tmp_path: Path):
        task_config = TaskConfig(
            name="test_task",
            experiment_fields={"f1": FieldSpec(type=str, description="F1")},
            compare_fields=["f1"],
            float_tolerance=0.1
        )
        lm = MagicMock(spec=dspy.LM)
        cache_dir = tmp_path / "analysis"

        analyzer = LocalAnalyzer(lm=lm, task_config=task_config, cache_dir=str(cache_dir), rate_limit_delay=0.0)
        
        # Mock predictor to throw ValidationError on first call and succeed on second
        predictor_mock = MagicMock()
        
        # Second call returns valid output
        valid_prediction = MagicMock()
        valid_prediction.analysis = DocumentAnalysis(
            document_id="doc1",
            entity_observations=[],
            field_observations=[],
            summary="recovered summary"
        )
        
        # Throw validation error on first call
        def side_effect(*args, **kwargs):
            if predictor_mock.call_count == 1:
                raise ValidationError.from_exception_data(
                    title="MockValidationError",
                    line_errors=[{"type": "missing", "loc": ("summary",), "input": {}}]
                )
            return valid_prediction

        predictor_mock.side_effect = side_effect
        analyzer.predictor = predictor_mock

        inp = AnalysisInput(
            document_id="doc1",
            document_text="Hello",
            ground_truth_experiments=[],
            field_specs={}
        )
        
        with patch("dspy.dsp.utils.settings.Settings.context", side_effect=mock_dspy_context):
            result = await analyzer.analyze(inp)
            assert result.summary == "recovered summary"
            assert predictor_mock.call_count == 2
            # Check atomic write
            cache_file = cache_dir / "test_task_map_doc1.json"
            assert cache_file.exists()


@pytest.mark.unit
class TestStrictAggregator:
    """Test StrictAggregator rule compilation and semantic equivalency."""

    def test_semantic_equivalence_consensus(self, tmp_path: Path):
        task_config = TaskConfig(
            name="test_task",
            experiment_fields={"f1": FieldSpec(type=str, description="F1")},
            compare_fields=["f1"],
            float_tolerance=0.1
        )
        lm = MagicMock(spec=dspy.LM)
        aggregator = StrictAggregator(lm=lm, task_config=task_config, cache_dir=str(tmp_path))

        # Setup mock predictor outputs
        agg_output = {
            "verified_rules": [
                {
                    "rule_text": "Temperature rule",
                    "evidence_count": 2,
                    "evidence_examples": ["T1", "T2"]
                }
            ],
            "discrepancies": []
        }
        
        mock_agg_pred = MagicMock()
        mock_agg_pred.return_value.rules_and_discrepancies = json.dumps(agg_output)
        aggregator.agg_predictor = mock_agg_pred

        mock_checker_pred = MagicMock()
        mock_checker_pred.return_value.is_unanimous = True
        mock_checker_pred.return_value.consolidated_rule = "Consolidated temperature rule"
        aggregator.checker_predictor = mock_checker_pred

        analyses = [
            DocumentAnalysis(
                document_id="doc1",
                entity_observations=[],
                field_observations=[
                    FieldObservation(
                        field_name="f1",
                        observation_type="format_applied",
                        description="T1",
                        evidence="e1"
                    )
                ],
                summary="s1"
            )
        ]

        with patch("dspy.dsp.utils.settings.Settings.context", side_effect=mock_dspy_context):
            res = aggregator.aggregate(analyses)
            assert len(res.verified_rules) == 1
            assert res.verified_rules[0].rule_text == "Consolidated temperature rule"


@pytest.mark.unit
class TestPromptBuilder:
    """Test compile behavior of build_three_level_prompt."""

    def test_build_three_level_prompt(self):
        rules = [
            VerifiedRule(rule_id="r1", level="entity", rule_text="E1", evidence_count=1, evidence_examples=[]),
            VerifiedRule(rule_id="r2", level="field", field_name="temp", rule_text="F1", evidence_count=1, evidence_examples=[]),
        ]
        result = AnalysisResult(
            task_name="test",
            analyzed_documents=2,
            verified_rules=rules,
            discrepancies=[],
            timestamp="2026-06-02"
        )
        prompt = build_three_level_prompt(result, meta_rules=["M1"])
        
        assert "[META] (Системные ограничения)" in prompt
        assert "- M1" in prompt
        assert "[ENTITY] (Критерии фильтрации строк)" in prompt
        assert "- E1" in prompt
        assert "[SCHEMA] (Правила валидации полей)" in prompt
        assert "* Поле temp: F1" in prompt
