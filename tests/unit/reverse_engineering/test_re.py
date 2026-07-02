# tests/unit/reverse_engineering/test_re.py
"""Unit tests for the Reverse Engineering (RE) pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import dspy

from ae.reverse_engineering.context import ArtifactManager, RunContext
from ae.reverse_engineering.use_case import (
    ReverseEngineeringUseCase,
    ReverseEngineeringRequest,
    ReverseEngineeringResponse,
)
from ae.reverse_engineering.models import (
    PositiveRowsOutput,
    RowAnalysis,
    PositiveColumnsOutput,
    FieldAnalysis,
    ConsolidatedRowsOutput,
    ConsolidatedColumnsOutput,
    NegativeRowsOutput,
    NegativeColumnsOutput,
    GeneralizedRowsOutput,
    GeneralizedColumnsOutput,
    RowInstructions,
    ColumnInstructions,
    ColumnAnomaly,
    SourceReference,
)

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path

@pytest.fixture
def mock_lm():
    lm = MagicMock(spec=dspy.LM)
    lm.model = "mock-teacher"
    return lm

@pytest.fixture
def sample_gt_csv(temp_dir: Path) -> Path:
    csv_path = temp_dir / "gt.csv"
    # Create simple ground truth CSV with typical columns
    data = {
        "pdf": ["doc1.pdf", "doc1.pdf", "doc2.pdf"],
        "formula": ["Fe3O4", "ZnO", "CuO"],
        "activity": ["peroxidase", "catalase", "oxidase"],
        "temperature": ["25", "37", "RT"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def sample_splits_json(temp_dir: Path) -> Path:
    splits_path = temp_dir / "splits.json"
    splits = {
        "train": ["doc1", "doc2"],
        "test": ["doc3"]
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f)
    return splits_path

@pytest.fixture
def sample_files(temp_dir: Path) -> tuple[Path, Path]:
    baseline_path = temp_dir / "baseline.txt"
    baseline_path.write_text(
        "You are extraction system.\n\n"
        "---\n\n"
        "CRITICAL EXTRACTION POLICY\n"
        "1. Policy 1\n\n"
        "---\n\n"
        "EXPERIMENT SELECTION\n"
        "Select experiments.\n\n"
        "---\n\n"
        "FORMULA\n"
        "Extract formula.\n\n"
        "---\n\n"
        "REASONING STYLE: STRICTLY CONCISE\n"
        "Be concise."
    )
    schema_path = temp_dir / "schema.yaml"
    schema_path.write_text("name: nanozymes\nfields:\n  formula:\n    type: str\n")
    return baseline_path, schema_path

class TestArtifactManager:
    def test_get_path(self, temp_dir):
        am = ArtifactManager(temp_dir / "art")
        path1 = am.get_path("positive", "rows", "doc1.json")
        assert path1 == temp_dir / "art" / "01_positive" / "rows" / "doc1.json"

        path2 = am.get_path("consolidation", "", "rows.json")
        assert path2 == temp_dir / "art" / "02_consolidation" / "rows.json"

    def test_load_or_compute(self, temp_dir):
        am = ArtifactManager(temp_dir / "art", resume=True)
        path = temp_dir / "art" / "test.json"
        
        # 1. Compute and save
        compute_fn = MagicMock(return_value={"a": 1})
        res = am.load_or_compute(path, compute_fn)
        assert res == {"a": 1}
        compute_fn.assert_called_once()
        assert path.exists()

        # 2. Load from cache (resume=True)
        compute_fn_2 = MagicMock()
        res_cached = am.load_or_compute(path, compute_fn_2)
        assert res_cached == {"a": 1}
        compute_fn_2.assert_not_called()

        # 3. Output model validation
        class TestModel(PositiveRowsOutput):
            pass
        
        model_path = temp_dir / "art" / "model.json"
        model_compute = MagicMock(return_value=PositiveRowsOutput(rows=[]))
        res_model = am.load_or_compute(model_path, model_compute, PositiveRowsOutput)
        assert isinstance(res_model, PositiveRowsOutput)
        assert len(res_model.rows) == 0

class TestRunContext:
    def test_lazy_loading(self, temp_dir, sample_files):
        baseline_path, schema_path = sample_files
        context = RunContext(
            task_name="nanozymes",
            doc_ids=["doc1"],
            gt_path=temp_dir / "gt.csv",
            ingestion_dir=temp_dir / "parsed",
            baseline_prompt_path=baseline_path,
            schema_path=schema_path,
            teacher_lm=MagicMock(),
            artifacts_dir=temp_dir / "art",
        )
        assert "You are extraction system." in context.baseline_prompt
        assert "name: nanozymes" in context.schema

    def test_schema_fields(self, temp_dir, sample_files):
        baseline_path, schema_path = sample_files
        context = RunContext(
            task_name="nanozymes",
            doc_ids=["doc1"],
            gt_path=temp_dir / "gt.csv",
            ingestion_dir=temp_dir / "parsed",
            baseline_prompt_path=baseline_path,
            schema_path=schema_path,
            teacher_lm=MagicMock(),
            artifacts_dir=temp_dir / "art",
        )
        assert context.schema_fields == ["formula"]


class TestReverseEngineeringUseCase:
    @patch("dspy.Predict")
    @patch("ae.reverse_engineering.steps.positive_analysis.DocumentRepository")
    @patch("ae.reverse_engineering.steps.negative_analysis.DocumentRepository")
    def test_orchestrator_execution(
        self,
        mock_doc_repo_neg,
        mock_doc_repo_pos,
        mock_predict,
        temp_dir,
        sample_gt_csv,
        sample_files,
        mock_lm,
    ):
        baseline_path, schema_path = sample_files

        # Mock Document Repository to return fake doc text
        mock_pos_repo_inst = MagicMock()
        mock_pos_repo_inst.get.return_value = "Doc 1 text"
        mock_doc_repo_pos.return_value = mock_pos_repo_inst

        mock_neg_repo_inst = MagicMock()
        mock_neg_repo_inst.get.return_value = "Doc 1 text"
        mock_doc_repo_neg.return_value = mock_neg_repo_inst

        # Mock LLM outputs for each predictor
        mock_row_analysis = MagicMock()
        mock_row_analysis.return_value = MagicMock(
            analysis=PositiveRowsOutput(
                rows=[
                    RowAnalysis(
                        row_id="row_1",
                        Row_Instantiation_Trigger_class="Single_Entity",
                        Row_Instantiation_Trigger_description="focuses on Fe3O4",
                        Entity_System_Role_class="Target",
                        Entity_System_Role_description="target catalyst",
                        Entity_Filtration_Boundary_class="Exhaustive_Set",
                        Entity_Filtration_Boundary_description="all included"
                    )
                ]
            )
        )
        
        mock_col_analysis = MagicMock()
        mock_col_analysis.return_value = MagicMock(
            analysis=PositiveColumnsOutput(
                fields=[
                    FieldAnalysis(
                        field_name="formula",
                        raw_text="Fe3O4",
                        ground_truth="Fe3O4",
                        Semantic_Core="iron oxide nanozyme",
                        Analytical_Method="XRD",
                        System_Conditions="Not_Specified",
                        Hierarchy_Level_class="Macro_System",
                        Hierarchy_Level_description="describes the material",
                        Semantic_Binding_class="Syntax_Direct",
                        Semantic_Binding_description="directly linked",
                        Text_Precision_class="Exact_Value",
                        Text_Precision_description="exact name",
                        Value_Multiplicity_class="Single_Instance",
                        Value_Multiplicity_description="one instance",
                        Transformation_description="no changes"
                    )
                ]
            )
        )

        # Phase 2 Consolidation Mock
        mock_row_consolidation = MagicMock()
        mock_row_consolidation.return_value = MagicMock(
            consolidation=ConsolidatedRowsOutput(Row_Instantiation_Trigger_generalized="general rule for rows")
        )
        mock_col_consolidation = MagicMock()
        mock_col_consolidation.return_value = MagicMock(
            consolidation=ConsolidatedColumnsOutput(Semantic_Core_generalized="general core")
        )

        # Phase 3 Negative Analysis Mock
        mock_neg_row = MagicMock()
        mock_neg_row.return_value = MagicMock(
            analysis=NegativeRowsOutput(gap_rows=[])
        )
        mock_neg_col = MagicMock()
        mock_neg_col.return_value = MagicMock(
            analysis=NegativeColumnsOutput(fields=[])
        )

        # Phase 4 Generalization Mock
        mock_row_gen = MagicMock()
        mock_row_gen.return_value = MagicMock(
            generalization=GeneralizedRowsOutput(
                Instructions=RowInstructions(
                    Row_Inclusion_Instructions=["include rule"],
                    Row_Exclusion_Instructions=["exclude rule"],
                    Validation_Rationale="justification"
                ),
                Anomalies=[]
            )
        )
        mock_col_gen = MagicMock()
        mock_col_gen.return_value = MagicMock(
            generalization=GeneralizedColumnsOutput(
                Instructions=ColumnInstructions(
                    Column_Inclusion_Instructions=["inc col"],
                    Column_Exclusion_Instructions=["exc col"],
                    Transformation_Instructions=["trans col"]
                ),
                Anomalies=[
                    ColumnAnomaly(
                        source_reference=SourceReference(
                            document_id="doc1",
                            positive_reference_ids=["row_1"],
                            negative_reference_ids=[]
                        ),
                        anomaly_description="test anomaly"
                    )
                ]
            )
        )

        # Unified side effect for dspy.Predict
        def predict_side_effect(signature):
            name = signature.__name__
            if name == "PositiveRowAnalysis":
                return mock_row_analysis
            elif name == "PositiveColumnAnalysis":
                return mock_col_analysis
            elif name == "RowConsolidation":
                return mock_row_consolidation
            elif name == "ColumnConsolidation":
                return mock_col_consolidation
            elif name == "NegativeRowAnalysis":
                return mock_neg_row
            elif name == "NegativeColumnAnalysis":
                return mock_neg_col
            elif name == "RowGeneralization":
                return mock_row_gen
            elif name == "ColumnGeneralization":
                return mock_col_gen
            raise ValueError(f"Unexpected signature in Predict: {name}")

        mock_predict.side_effect = predict_side_effect

        # Run orchestrator
        request = ReverseEngineeringRequest(
            task_name="nanozymes",
            doc_ids=["doc1"],
            gt_path=sample_gt_csv,
            ingestion_dir=temp_dir / "parsed",
            baseline_prompt_path=baseline_path,
            schema_path=schema_path,
            teacher_lm=mock_lm,
            output_dir=temp_dir / "art",
            resume=False
        )

        # Mock tasks folder paths
        with patch("ae.reverse_engineering.steps.generalization.Path") as mock_path:
            # We want to intercept Path write_text for final generated instruction
            mock_task_dir = MagicMock()
            mock_path.return_value = mock_task_dir
            
            orchestrator = ReverseEngineeringUseCase()
            response = orchestrator.execute(request)

            assert response.success is True
            assert "positive" in response.step_artifacts
            assert "consolidation" in response.step_artifacts
            assert "negative" in response.step_artifacts
            assert "generalization" in response.step_artifacts

            # Verify that only the fields defined in schema.yaml (only 'formula') were processed
            # 1. Check ColumnConsolidation calls
            col_consolidation_calls = mock_col_consolidation.call_args_list
            assert len(col_consolidation_calls) == 1
            assert col_consolidation_calls[0].kwargs["field_name"] == "formula"

            # 2. Check ColumnGeneralization calls
            col_gen_calls = mock_col_gen.call_args_list
            assert len(col_gen_calls) == 1
            assert col_gen_calls[0].kwargs["field_name"] == "formula"
