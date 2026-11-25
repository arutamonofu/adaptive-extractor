# tests/unit/reverse_engineering/test_re.py
"""Unit tests for the Reverse Engineering (RE) pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import dspy

from ae.semantic_optimization.context import ArtifactManager, RunContext
from ae.semantic_optimization.use_case import (
    SemanticOptimizationUseCase,
    SemanticOptimizationRequest,
    SemanticOptimizationResponse,
)
from ae.semantic_optimization.models import (
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


class TestSemanticOptimizationUseCase:
    @patch("dspy.Predict")
    @patch("ae.semantic_optimization.steps.positive_analysis.DocumentRepository")
    @patch("ae.semantic_optimization.steps.negative_analysis.DocumentRepository")
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
        request = SemanticOptimizationRequest(
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
        with patch("ae.semantic_optimization.steps.generalization.Path") as mock_path:
            # We want to intercept Path write_text for final generated instruction
            mock_task_dir = MagicMock()
            mock_path.return_value = mock_task_dir
            
            orchestrator = SemanticOptimizationUseCase()
            response = orchestrator.execute(request)

            assert response.success is True
            assert "positive" in response.step_artifacts
            assert "consolidation" in response.step_artifacts
            assert "negative" in response.step_artifacts
            assert "generalization" in response.step_artifacts

            # Verify that only the fields defined in schema.yaml (only 'formula') were processed
            # 1. Intermediate ColumnConsolidation is bypassed in unified synthesis flow
            col_consolidation_calls = mock_col_consolidation.call_args_list
            assert len(col_consolidation_calls) == 0

            # 2. Check row batching: doc1 has 2 rows in sample_gt_csv, processed in 1 batch call
            assert len(mock_col_analysis.call_args_list) == 1
            assert len(mock_neg_col.call_args_list) == 1
            assert "iron oxide nanozyme" in mock_neg_col.call_args_list[0].kwargs["generalized_core"]

            # 3. Check ColumnGeneralization calls
            col_gen_calls = mock_col_gen.call_args_list
            assert len(col_gen_calls) == 1
            assert col_gen_calls[0].kwargs["field_name"] == "formula"

    @patch("dspy.Predict")
    @patch("ae.semantic_optimization.steps.positive_analysis.DocumentRepository")
    @patch("ae.semantic_optimization.steps.negative_analysis.DocumentRepository")
    def test_prompt_harmonization_and_baseline_preservation(
        self,
        mock_doc_repo_neg,
        mock_doc_repo_pos,
        mock_predict,
        temp_dir,
        sample_gt_csv,
        sample_files,
        mock_lm,
    ):
        from ae.semantic_optimization.models import PromptHarmonizationOutput

        baseline_path, schema_path = sample_files
        mock_doc_repo_pos.return_value.get.return_value = "Doc 1 text"
        mock_doc_repo_neg.return_value.get.return_value = "Doc 1 text"

        mock_row_gen = MagicMock(
            return_value=MagicMock(
                generalization=GeneralizedRowsOutput(
                    Row_Instantiation_Trigger_generalized="Target nanozyme",
                    Entity_Role_and_Filtration_Policy="Target only",
                    Instructions=RowInstructions(
                        Row_Inclusion_Instructions=["Include target nanozymes"],
                        Row_Exclusion_Instructions=["Exclude controls"],
                        Validation_Rationale="rationale",
                    ),
                )
            )
        )
        mock_col_gen = MagicMock(
            return_value=MagicMock(
                generalization=GeneralizedColumnsOutput(
                    Semantic_Core_generalized="Chemical formula",
                    Analytical_and_Binding_Policy="Direct",
                    Instructions=ColumnInstructions(
                        Column_Inclusion_Instructions=["Extract formula"],
                        Column_Exclusion_Instructions=["Exclude precursors"],
                        Transformation_Instructions=["Standardize formula"],
                    ),
                )
            )
        )
        mock_harmonizer = MagicMock(
            return_value=MagicMock(
                harmonization=PromptHarmonizationOutput(
                    harmonized_prompt="HARMONIZED FINAL PROMPT CONTENT",
                    harmonization_notes=["Aligned formula rules with baseline"],
                )
            )
        )

        def predict_side_effect(signature):
            name = signature.__name__
            if name == "PositiveRowAnalysis":
                return MagicMock(return_value=MagicMock(analysis=PositiveRowsOutput(rows=[])))
            if name == "PositiveColumnAnalysis":
                return MagicMock(return_value=MagicMock(analysis=PositiveColumnsOutput(fields=[])))
            if name == "NegativeRowAnalysis":
                return MagicMock(return_value=MagicMock(analysis=NegativeRowsOutput(gap_rows=[])))
            if name == "NegativeColumnAnalysis":
                return MagicMock(return_value=MagicMock(analysis=NegativeColumnsOutput(fields=[])))
            if name == "RowGeneralization":
                return mock_row_gen
            if name == "ColumnGeneralization":
                return mock_col_gen
            if name == "PromptHarmonization":
                return mock_harmonizer
            raise ValueError(f"Unexpected signature: {name}")

        mock_predict.side_effect = predict_side_effect

        request = SemanticOptimizationRequest(
            task_name="nanozymes",
            doc_ids=["doc1"],
            gt_path=sample_gt_csv,
            ingestion_dir=temp_dir / "parsed",
            baseline_prompt_path=baseline_path,
            schema_path=schema_path,
            teacher_lm=mock_lm,
            output_dir=temp_dir / "art_harm",
            harmonize_prompt=True,
            resume=False,
        )

        orchestrator = SemanticOptimizationUseCase()
        response = orchestrator.execute(request)
        assert response.success is True
        mock_harmonizer.assert_called_once()
        iter_prompt = (
            temp_dir / "art_harm" / "iterations" / "iter_1" / "04_generalization" / "candidate_prompt.txt"
        ).read_text(encoding="utf-8")
        assert "HARMONIZED FINAL PROMPT CONTENT" in iter_prompt


def test_unpack_manifests_for_negative_distractor_recall(temp_dir):
    from ae.semantic_optimization.steps.negative_analysis import _unpack_manifests_for_negative
    from langchain_core.messages import AIMessage

    mock_doc_repo = MagicMock()
    mock_doc_repo.get.return_value = "Enriched negative doc text"

    mock_context = MagicMock()
    mock_context.ingestion_dir = temp_dir

    # 1. Document with no unparsed visuals -> immediate return
    res_no_visuals = _unpack_manifests_for_negative(
        norm_doc_id="doc_clean",
        raw_text="No stubs here",
        gt_table_csv="doi\ndoc_clean\n",
        generalized_trigger="trigger",
        generalized_cores_str="{}",
        context=mock_context,
        doc_repo=mock_doc_repo,
    )
    assert res_no_visuals == "No stubs here"

    # 2. Document with unparsed visual stub -> invokes teacher model with distractor prompt
    raw_with_stub = (
        "Text\n\n"
        '<visual id="Scheme_Distractor" type="molecular_structure">\n'
        "Scheme 1. Precursors\n"
        "> Synthesis of starting materials and reference controls\n"
        "</visual>"
    )

    with patch("ae.core.llm.factory.create_chat_model") as mock_create_model, \
         patch("ae.ingestion.service.VisualParsingService") as mock_service_cls, \
         patch("ae.ingestion.tool.make_visual_parsing_tool") as mock_make_tool:

        mock_tool = MagicMock()
        mock_make_tool.return_value = mock_tool

        mock_chat = MagicMock()
        mock_bound = MagicMock()
        mock_chat.bind_tools.return_value = mock_bound
        mock_create_model.return_value = mock_chat

        tool_msg = AIMessage(
            content="",
            tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Scheme_Distractor"]}, "id": "call_1"}],
        )
        mock_bound.invoke.return_value = tool_msg

        res_unpacked = _unpack_manifests_for_negative(
            norm_doc_id="doc_dist",
            raw_text=raw_with_stub,
            gt_table_csv="doi\ndoc_dist\n",
            generalized_trigger="trigger",
            generalized_cores_str="{}",
            context=mock_context,
            doc_repo=mock_doc_repo,
        )

        # Verify teacher prompt specifically targets distractors and exclusion candidates
        bound_args = mock_bound.invoke.call_args[0][0]
        sys_prompt = bound_args[0].content
        assert "distractor entities" in sys_prompt.lower()
        assert "preparatory/intermediate items" in sys_prompt.lower()

        # Verify tool was invoked
        mock_tool.invoke.assert_called_once_with({"visual_ids": ["Scheme_Distractor"]})
        assert res_unpacked == "Enriched negative doc text"


def test_re_visual_parsing_tool_disables_target_hint_and_normalizes_root_ids():
    from ae.ingestion.tool import make_visual_parsing_tool
    from ae.ingestion.parsers.gemini.schemas import VisualParseRecord

    mock_service = MagicMock()
    mock_service.parse_visuals.return_value = {
        "Scheme_2": VisualParseRecord(
            visual_id="Scheme_2",
            status="success",
            mode="composite",
            rendered_markdown="#### Chemical Entities\n| Label | SMILES |\n| :--- | :--- |\n| **6a** | `CCO` |",
        ),
        "Table_1": VisualParseRecord(
            visual_id="Table_1",
            status="success",
            mode="single",
            rendered_markdown="| ID | MIC |\n| --- | --- |\n| 6a | 2.0 |",
        ),
    }

    re_tool = make_visual_parsing_tool("doc_re", service=mock_service, mode="semantic_optimization")
    assert "target_hint" not in re_tool.args
    assert "visual_ids" in re_tool.args

    out = re_tool.invoke({"visual_ids": ["Scheme_2/mol_6a", "Table_1#part1", "Scheme_2"]})
    mock_service.parse_visuals.assert_called_once_with("doc_re", ["Scheme_2", "Table_1"])
    assert '<visual id="Scheme_2">' in out
    assert '<visual id="Table_1">' in out

    # Default inference mode still exposes target_hint
    inf_tool = make_visual_parsing_tool("doc_inf", service=mock_service, mode="inference")
    assert "target_hint" in inf_tool.args


def test_prepare_document_visuals_chained_turns_and_marker_cache(temp_dir):
    from ae.semantic_optimization.steps.visual_prep import prepare_document_visuals
    from langchain_core.messages import AIMessage

    art_mgr = ArtifactManager(temp_dir / "re_out", resume=True)
    mock_context = MagicMock()
    mock_context.ingestion_dir = temp_dir
    mock_context.artifacts = art_mgr
    mock_context.schema = "fields: [compound, smiles, mic]"
    mock_context.baseline_prompt = "Extract antibacterial MIC of synthesized target compounds."

    initial_md = (
        '<visual id="Table_1" type="table">\n'
        "Table 1. Antibacterial activity\n"
        "> MIC values of compounds 6a-6c\n"
        "</visual>\n\n"
        '<visual id="Scheme_2" type="molecular_structure">\n'
        "Scheme 2. Synthesis of 6a-6c from precursor 4\n"
        "> Reaction pathway and chemical structures\n"
        "</visual>\n\n"
        '<visual id="Figure_4" type="chart">\n'
        "Figure 4. 1H NMR spectrum of 6a\n"
        "> Spectral characterization\n"
        "</visual>"
    )
    after_turn1_md = (
        '<visual id="Table_1" type="table">\n'
        "| Compound | MIC |\n| --- | --- |\n| 6a (see Scheme 2) | 1.5 |\n"
        "</visual>\n\n"
        '<visual id="Scheme_2" type="molecular_structure">\n'
        "Scheme 2. Synthesis of 6a-6c from precursor 4\n"
        "> Reaction pathway and chemical structures\n"
        "</visual>\n\n"
        '<visual id="Figure_4" type="chart">\n'
        "Figure 4. 1H NMR spectrum of 6a\n"
        "> Spectral characterization\n"
        "</visual>"
    )
    after_turn2_md = (
        '<visual id="Table_1" type="table">\n'
        "| Compound | MIC |\n| --- | --- |\n| 6a (see Scheme 2) | 1.5 |\n"
        "</visual>\n\n"
        '<visual id="Scheme_2" type="molecular_structure">\n'
        "#### Chemical Entities\n| Label | SMILES |\n| :--- | :--- |\n| **4** | `CC` |\n| **6a** | `CCO` |\n"
        "</visual>\n\n"
        '<visual id="Figure_4" type="chart">\n'
        "Figure 4. 1H NMR spectrum of 6a\n"
        "> Spectral characterization\n"
        "</visual>"
    )

    mock_doc_repo = MagicMock()
    mock_doc_repo.get.side_effect = [after_turn1_md, after_turn2_md, after_turn2_md]

    with patch("ae.core.llm.factory.create_chat_model") as mock_create_model, \
         patch("ae.ingestion.service.VisualParsingService"), \
         patch("ae.ingestion.tool.make_visual_parsing_tool") as mock_make_tool:

        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = ["<visual id='Table_1'>...</visual>", "<visual id='Scheme_2'>...</visual>"]
        mock_make_tool.return_value = mock_tool

        mock_chat = MagicMock()
        mock_bound = MagicMock()
        mock_chat.bind_tools.return_value = mock_bound
        mock_create_model.return_value = mock_chat

        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Table_1"]}, "id": "call_t1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Scheme_2"]}, "id": "call_t2"}],
            ),
            AIMessage(content="READY", tool_calls=[]),
        ]

        res_text = prepare_document_visuals(
            norm_doc_id="doc_chain",
            raw_text=initial_md,
            gt_table_csv="compound,smiles,mic\n6a,CCO,1.5\n",
            context=mock_context,
            doc_repo=mock_doc_repo,
            force_recompute=False,
        )

        assert res_text == after_turn2_md
        assert mock_tool.invoke.call_count == 2
        mock_tool.invoke.assert_any_call({"visual_ids": ["Table_1"]})
        mock_tool.invoke.assert_any_call({"visual_ids": ["Scheme_2"]})

        # Marker file is persisted with unpacked visuals and remaining off-target stub (Figure_4)
        marker_file = art_mgr.get_path("positive", "visual_prep", "doc_chain.json", iteration=None)
        assert marker_file.exists()
        marker_data = json.loads(marker_file.read_text(encoding="utf-8"))
        assert marker_data["unpacked_visual_ids"] == ["Table_1", "Scheme_2"]
        assert marker_data["remaining_stub_ids"] == ["Figure_4"]

        # Subsequent call with resume=True (e.g. during 1B Negative Analysis) must NOT invoke LLM again
        mock_bound.invoke.reset_mock()
        res_cached = prepare_document_visuals(
            norm_doc_id="doc_chain",
            raw_text=after_turn2_md,
            gt_table_csv="compound,smiles,mic\n6a,CCO,1.5\n",
            context=mock_context,
            doc_repo=mock_doc_repo,
            force_recompute=False,
        )
        assert res_cached == after_turn2_md
        mock_bound.invoke.assert_not_called()


def test_select_representative_rows_stratified_sampling():
    from ae.semantic_optimization.steps.positive_analysis import (
        prepare_prompt_table,
        select_representative_rows,
    )

    rows = []
    for i in range(60):
        rows.append(
            {
                "compound_id": f"cpd_{i+1}",
                "target_relation": ">" if i == 42 else "=",
                "target_value": "0,25" if i == 17 else str(10 + i),
                "target_units": "mM" if i == 55 else "ug/mL",
                "bacteria": "S. aureus" if i % 2 == 0 else "E. coli",
                "notes": "special note" if i == 3 else None,
            }
        )
    df_doc = pd.DataFrame(rows)
    df_with_id = prepare_prompt_table(
        df_doc, ["compound_id", "target_relation", "target_value", "target_units", "bacteria", "notes"]
    )
    # Build raw text where verbatim values exist EXCEPT transformed "0,25" (text has "0.25")
    raw_text = "\n".join(
        f"cpd_{i+1} = {10 + i} ug/mL mM S. aureus E. coli special note"
        for i in range(60)
    ).replace("= 27 ug/mL", "= 0.25 ug/mL")

    sampled = select_representative_rows(df_with_id, raw_text, max_rows=20)
    assert len(sampled) == 20
    # Original row_id values must be preserved in ascending order
    sampled_ids = sampled["row_id"].tolist()
    assert sampled_ids == sorted(sampled_ids, key=lambda s: int(s.split("_")[1]))
    # Text-Diff anomaly (row_18 has "0,25" not verbatim in raw_text), rare operator ">" (row_43),
    # rare unit "mM" (row_56), and null-pattern difference (row_4 has non-null notes) must all be captured
    assert "row_18" in sampled_ids
    assert "row_43" in sampled_ids
    assert "row_56" in sampled_ids
    assert "row_4" in sampled_ids


def test_generalization_unsectioned_baseline_uses_neutral_preamble_and_delegates_to_harmonizer(
    temp_dir: Path, mock_lm
):
    from ae.semantic_optimization.models import PromptHarmonizationOutput
    from ae.semantic_optimization.steps.generalization import run_generalization

    gt_path = temp_dir / "gt.csv"
    pd.DataFrame({"pdf": ["doc1"], "field_a": ["val_a"]}).to_csv(gt_path, index=False)

    xml_baseline = (
        "<role>You are an extraction assistant.</role>\n"
        "<instructions>Extract field_a carefully from each article.</instructions>"
    )
    baseline_path = temp_dir / "baseline.txt"
    baseline_path.write_text(xml_baseline, encoding="utf-8")

    schema_path = temp_dir / "schema.yaml"
    schema_path.write_text(
        "name: test\nfields:\n  field_a:\n    type: str\n    description: test field\ncompare_fields:\n  - field_a\n",
        encoding="utf-8",
    )

    ingest_dir = temp_dir / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    gen_prompt_path = temp_dir / "generated_prompt.txt"
    gen_schema_path = temp_dir / "generated_schema.yaml"

    ctx = RunContext(
        task_name="test",
        doc_ids=["doc1"],
        gt_path=gt_path,
        ingestion_dir=ingest_dir,
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        teacher_lm=mock_lm,
        artifacts_dir=temp_dir / "artifacts",
        generated_prompt_path=gen_prompt_path,
        generated_schema_path=gen_schema_path,
        interactive=False,
        harmonize_prompt=True,
        infer_schema_constraints=False,
    )

    row_out = GeneralizedRowsOutput(
        Instructions=RowInstructions(
            Row_Inclusion_Instructions=["Include measured records."],
            Row_Exclusion_Instructions=["Exclude review summaries."],
        ),
        Conflicts=[],
        Anomalies=[],
    )
    col_out = GeneralizedColumnsOutput(
        Instructions=ColumnInstructions(
            Column_Inclusion_Instructions=["Extract verbatim field_a."],
            Column_Exclusion_Instructions=[],
            Transformation_Instructions=[],
        ),
        Conflicts=[],
        Anomalies=[],
    )
    captured_harm_kwargs = {}

    def fake_predict(sig):
        name = getattr(sig, "__name__", str(sig))
        if name == "RowGeneralization":
            return lambda **kwargs: MagicMock(generalization=row_out)
        if name == "ColumnGeneralization":
            return lambda **kwargs: MagicMock(generalization=col_out)
        if name == "PromptHarmonization":
            def _harm(**kwargs):
                captured_harm_kwargs.update(kwargs)
                return MagicMock(
                    harmonization=PromptHarmonizationOutput(
                        harmonized_prompt=kwargs["draft_prompt"],
                        alignment_notes="OK",
                    )
                )
            return _harm
        raise AssertionError(f"Unexpected signature: {name}")

    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=fake_predict):
        needs_recompute = run_generalization(ctx)

    assert needs_recompute is False
    assert captured_harm_kwargs["baseline_prompt"] == xml_baseline
    draft = captured_harm_kwargs["draft_prompt"]
    assert draft.startswith("You are an information extraction system.")
    assert "<role>" not in draft
    assert "Km=5.2" not in draft
    assert "RECORD SELECTION" in draft
    assert "FIELD_A" in draft
    assert "REASONING STYLE" not in draft
    assert gen_prompt_path.exists()


def test_select_representative_rows_balances_cartesian_strata_and_picks_lead_entity():
    from ae.semantic_optimization.steps.positive_analysis import (
        prepare_prompt_table,
        select_representative_rows,
    )

    categories = ["cat_A", "cat_B", "cat_C", "cat_D"]
    rows = []
    for cat in categories:
        for i in range(1, 21):
            is_deep_diff = (cat == "cat_C" and i == 17)
            rows.append(
                {
                    "entity_id": f"ent_{i}",
                    "relation": "=",
                    "value": "999" if is_deep_diff else "0,25",
                    "unit": "mM",
                    "condition": cat,
                }
            )
    df_doc = pd.DataFrame(rows)
    df_with_id = prepare_prompt_table(
        df_doc, ["entity_id", "relation", "value", "unit", "condition"]
    )

    text_lines = [f"ent_{i} = 0.25 mM {cat}" for cat in categories for i in range(1, 21)]
    # Lead entity ent_14 appears frequently across follow-up experiments
    text_lines.extend(["Additional assay for ent_14 with alternative metric IC50 = 1.2 uM."] * 25)
    raw_text = "\n".join(text_lines)

    sampled = select_representative_rows(df_with_id, raw_text, max_rows=20)
    assert len(sampled) == 20
    # Cartesian strata must be balanced across all 4 categories (4-6 per category, not 17/1/1/1)
    cat_counts = sampled["condition"].value_counts().to_dict()
    assert set(cat_counts.keys()) == set(categories)
    assert all(4 <= cnt <= 6 for cnt in cat_counts.values())
    # Lead entity and deep_diff anomaly despite global surface_norm ("0,25" vs "0.25") must be captured
    assert "ent_14" in sampled["entity_id"].tolist()
    assert "999" in sampled["value"].tolist()


def test_select_representative_rows_covers_all_document_tables_and_prose_regions():
    from ae.semantic_optimization.steps.positive_analysis import (
        prepare_prompt_table,
        select_representative_rows,
    )

    rows = []
    for i in range(1, 31):
        rows.append({"entity_id": f"t1_{i}", "value": str(100 + i), "condition": "cond_1"})
    for i in range(1, 3):
        rows.append({"entity_id": f"t2_{i}", "value": str(200 + i), "condition": "cond_1"})
        rows.append({"entity_id": f"p1_{i}", "value": str(300 + i), "condition": "cond_1"})
        rows.append({"entity_id": f"t3_{i}", "value": str(400 + i), "condition": "cond_1"})

    df_with_id = prepare_prompt_table(
        pd.DataFrame(rows), ["entity_id", "value", "condition"]
    )
    raw_text = (
        '<visual id="Table_1" type="table">\n'
        + "\n".join(f"t1_{i} {100+i}" for i in range(1, 31))
        + "\n</visual>\n\n"
        + "In the main text, "
        + " ".join(f"compound p1_{i} showed activity {300+i}" for i in range(1, 3))
        + " under standard testing conditions.\n\n"
        + '<visual id="Table_2" type="table">\n'
        + "\n".join(f"t2_{i} {200+i}" for i in range(1, 3))
        + "\n</visual>\n\n"
        + '<visual id="Table_3" type="table">\n'
        + "\n".join(f"t3_{i} {400+i}" for i in range(1, 3))
        + "\n</visual>"
    )

    sampled = select_representative_rows(df_with_id, raw_text, max_rows=10)
    sampled_entities = set(sampled["entity_id"].tolist())
    assert any(e.startswith("t1_") for e in sampled_entities)
    assert any(e.startswith("t2_") for e in sampled_entities)
    assert any(e.startswith("t3_") for e in sampled_entities)
    assert any(e.startswith("p1_") for e in sampled_entities)


def test_positive_row_analysis_receives_full_gt_and_sampled_targets(temp_dir: Path, sample_files, mock_lm):
    from ae.semantic_optimization.steps.positive_analysis import run_positive_analysis

    baseline_path, schema_path = sample_files
    gt_path = temp_dir / "gt_many.csv"
    pd.DataFrame(
        {
            "pdf": ["doc_big"] * 30,
            "formula": [f"Mat_{i}" for i in range(1, 31)],
        }
    ).to_csv(gt_path, index=False)

    ingest_dir = temp_dir / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "doc_big.md").write_text(
        "\n".join(f"Mat_{i}" for i in range(1, 31)), encoding="utf-8"
    )

    ctx = RunContext(
        task_name="nanozymes",
        doc_ids=["doc_big"],
        gt_path=gt_path,
        ingestion_dir=ingest_dir,
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        teacher_lm=mock_lm,
        artifacts_dir=temp_dir / "art_full_gt",
        max_rows_per_doc=10,
        row_batch_size=10,
        resume=False,
    )

    captured_row_kwargs = {}

    def fake_predict(sig):
        name = getattr(sig, "__name__", str(sig))
        if name == "PositiveRowAnalysis":
            def _row(**kwargs):
                captured_row_kwargs.update(kwargs)
                return MagicMock(analysis=PositiveRowsOutput(rows=[]))
            return _row
        if name == "PositiveColumnAnalysis":
            return lambda **kwargs: MagicMock(analysis=PositiveColumnsOutput(fields=[]))
        raise AssertionError(f"Unexpected signature: {name}")

    with patch("ae.semantic_optimization.steps.positive_analysis.dspy.Predict", side_effect=fake_predict):
        run_positive_analysis(ctx)

    gt_lines = [l for l in captured_row_kwargs["gt_table"].splitlines() if l.strip()]
    target_lines = [l for l in captured_row_kwargs["target_gt_rows"].splitlines() if l.strip()]
    assert len(gt_lines) == 31  # header + all 30 rows
    assert len(target_lines) == 11  # header + 10 sampled rows


def test_resolve_doc_triggers_and_cores_deduplicates_and_includes_boundary(temp_dir: Path, sample_files, mock_lm):
    from ae.semantic_optimization.steps.negative_analysis import _resolve_doc_triggers_and_cores

    baseline_path, schema_path = sample_files
    gt_path = temp_dir / "gt.csv"
    pd.DataFrame({"pdf": ["doc1"], "formula": ["Fe3O4"]}).to_csv(gt_path, index=False)

    ctx = RunContext(
        task_name="nanozymes",
        doc_ids=["doc1"],
        gt_path=gt_path,
        ingestion_dir=temp_dir / "ingest",
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        teacher_lm=mock_lm,
        artifacts_dir=temp_dir / "art_resolve",
        resume=True,
    )

    pos_rows = PositiveRowsOutput(
        rows=[
            RowAnalysis(
                row_id=f"row_{i}",
                Row_Instantiation_Trigger_class="Composite_Vector",
                Row_Instantiation_Trigger_description=f"Unique trigger variant {i % 4}",
                Entity_System_Role_class="Target",
                Entity_System_Role_description="Target material",
                Entity_Filtration_Boundary_class="Exhaustive_Set",
                Entity_Filtration_Boundary_description="All tested variants included without filtering",
            )
            for i in range(1, 9)
        ]
    )
    pos_row_path = ctx.artifacts.get_path("positive", "rows", "doc1.json", iteration=None)
    pos_row_path.parent.mkdir(parents=True, exist_ok=True)
    pos_row_path.write_text(pos_rows.model_dump_json(indent=2), encoding="utf-8")

    for i, core in enumerate(
        ["Chemical formula", "chemical formula", "Molecular formula", "Stoichiometric formula", "Compound formula", "Empirical formula"],
        start=1,
    ):
        col_out = PositiveColumnsOutput(
            fields=[
                FieldAnalysis(
                    row_id=f"row_{i}",
                    field_name="formula",
                    raw_text="Fe3O4",
                    ground_truth="Fe3O4",
                    Semantic_Core=core,
                    Analytical_Method="Not_Specified",
                    System_Conditions="Not_Specified",
                    Hierarchy_Level_class="Macro_System",
                    Hierarchy_Level_description="Whole entity",
                    Semantic_Binding_class="Syntax_Direct",
                    Semantic_Binding_description="Direct",
                    Text_Precision_class="Exact_Value",
                    Text_Precision_description="Exact",
                    Value_Multiplicity_class="Single_Instance",
                    Value_Multiplicity_description="Single",
                    Transformation_description="None",
                )
            ]
        )
        c_path = ctx.artifacts.get_path("positive", "columns", f"doc1_row_{i}.json", iteration=None)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.write_text(col_out.model_dump_json(indent=2), encoding="utf-8")

    trigger_str, cores_json = _resolve_doc_triggers_and_cores(ctx, "doc1", ["formula"])
    assert "Exhaustive_Set" in trigger_str
    assert "Boundary: All tested variants included without filtering" in trigger_str
    # Capped at 2 per (Trigger, Role, Boundary) tuple -> only 2 entries separated by " | "
    assert len(trigger_str.split(" | ")) == 2

    cores_dict = json.loads(cores_json)
    core_parts = cores_dict["formula"].split(" ; ")
    # Case-insensitive duplicate "chemical formula" removed and capped at 3 unique cores
    assert core_parts == ["Chemical formula", "Molecular formula", "Stoichiometric formula"]


def test_are_field_values_equivalent_and_negative_candidate_filtering():
    from ae.semantic_optimization.models import CandidateAnalysis, NegativeFieldResult
    from ae.semantic_optimization.steps.negative_analysis import filter_equivalent_negative_candidates
    from ae.semantic_optimization.steps.positive_analysis import (
        are_field_values_equivalent,
        sanitize_positive_columns_output,
    )

    schema_yaml = (
        "name: test_chem\n"
        "fields:\n"
        "  smiles:\n"
        "    type: str\n"
        "    validators:\n"
        "      - name: smiles\n"
        "        on_error: warn\n"
        "  target_value:\n"
        "    type: str\n"
    )

    # 1. Canonically identical SMILES (different atom ordering & counterion placement) and numeric equivalence
    raw_smiles = "CCCCCn1c(-c2cc(OC)c(O)c3c[n+]4c(cc23)-c2cc3c(cc2CC4)OCO3)nc2ccccc21.[Cl-]"
    gt_smiles = "[Cl-].OC=1C(OC)=CC(C2=NC=3C=CC=CC3N2CCCCC)=C4C=C5C=6C=C7OCOC7=CC6CC[N+]5=CC14"
    diff_smiles = "CCn1c(-c2cc(OC)c(O)c3c[n+]4c(cc23)-c2cc3c(cc2CC4)OCO3)nc2ccccc21.[Cl-]"

    assert are_field_values_equivalent("smiles", raw_smiles, gt_smiles, schema_yaml) is True
    assert are_field_values_equivalent("smiles", diff_smiles, gt_smiles, schema_yaml) is False
    assert are_field_values_equivalent("target_value", "0.030", "0,03", schema_yaml) is True
    assert are_field_values_equivalent("target_value", "0.013", "0,202", schema_yaml) is False
    assert are_field_values_equivalent("target_value", "0.015–0.030", "0,03", schema_yaml) is False

    # 2. Sanitize false Suspected_Ingestion_Or_GT_Issue in PositiveColumnsOutput
    pos_out = PositiveColumnsOutput(
        fields=[
            FieldAnalysis(
                row_id="row_64",
                field_name="smiles",
                raw_text=f"| **4c** | {raw_smiles} |",
                ground_truth=gt_smiles,
                Semantic_Core="SMILES",
                Analytical_Method="Not_Specified",
                System_Conditions="Not_Specified",
                Hierarchy_Level_class="Micro_Component",
                Hierarchy_Level_description="Structure",
                Semantic_Binding_class="Syntax_Direct",
                Semantic_Binding_description="Direct",
                Text_Precision_class="Exact_Value",
                Text_Precision_description="Exact",
                Value_Multiplicity_class="Single_Instance",
                Value_Multiplicity_description="Single",
                Transformation_description="No valid transformation: factual mismatch between ingested text and GT.",
                Suspected_Ingestion_Or_GT_Issue="Incorrect SMILES suspected",
            ),
            FieldAnalysis(
                row_id="row_65",
                field_name="smiles",
                raw_text=diff_smiles,
                ground_truth=gt_smiles,
                Semantic_Core="SMILES",
                Analytical_Method="Not_Specified",
                System_Conditions="Not_Specified",
                Hierarchy_Level_class="Micro_Component",
                Hierarchy_Level_description="Structure",
                Semantic_Binding_class="Syntax_Direct",
                Semantic_Binding_description="Direct",
                Text_Precision_class="Exact_Value",
                Text_Precision_description="Exact",
                Value_Multiplicity_class="Single_Instance",
                Value_Multiplicity_description="Single",
                Transformation_description="No valid transformation: factual mismatch",
                Suspected_Ingestion_Or_GT_Issue="Actual molecule mismatch",
            ),
        ]
    )
    sanitized = sanitize_positive_columns_output(pos_out, schema_yaml)
    assert sanitized.fields[0].Suspected_Ingestion_Or_GT_Issue == "None"
    assert "equivalent" in sanitized.fields[0].Transformation_description.lower()
    assert sanitized.fields[1].Suspected_Ingestion_Or_GT_Issue == "Actual molecule mismatch"

    # 3. Filter equivalent candidates out of NegativeColumnsOutput while keeping genuine alternatives
    def _make_cand(row_id: str, cand_val: str, gt_val: str) -> CandidateAnalysis:
        return CandidateAnalysis(
            row_id=row_id,
            raw_text=cand_val,
            candidate_value=cand_val,
            ground_truth=gt_val,
            Semantic_Core="core",
            Analytical_Method="Not_Specified",
            System_Conditions="Not_Specified",
            Hierarchy_Level_class="Macro_System",
            Hierarchy_Level_description="desc",
            Semantic_Binding_class="Syntax_Direct",
            Semantic_Binding_description="desc",
            Text_Precision_class="Exact_Value",
            Text_Precision_description="desc",
            Value_Multiplicity_class="Single_Instance",
            Value_Multiplicity_description="desc",
        )

    neg_out = NegativeColumnsOutput(
        fields=[
            NegativeFieldResult(
                row_id="row_64",
                field_name="smiles",
                candidates=[
                    _make_cand("row_64", raw_smiles, gt_smiles),
                    _make_cand("row_64", diff_smiles, gt_smiles),
                ],
            ),
            NegativeFieldResult(
                row_id="row_96",
                field_name="target_value",
                candidates=[
                    _make_cand("row_96", "0.030", "0,03"),
                    _make_cand("row_96", "0.015–0.030", "0,03"),
                    _make_cand("row_96", "0.013", "0,202"),
                ],
            ),
        ]
    )
    filtered = filter_equivalent_negative_candidates(neg_out, schema_yaml)
    assert [c.candidate_value for c in filtered.fields[0].candidates] == [diff_smiles]
    assert [c.candidate_value for c in filtered.fields[1].candidates] == ["0.015–0.030", "0.013"]


def test_generalization_carry_forward_and_resume_prompt_cache(temp_dir: Path, mock_lm):
    """Unchanged scopes carry forward from iter_{k-1} without LLM calls, and resume reuses cached candidate_prompt.txt."""
    from ae.semantic_optimization.models import PromptHarmonizationOutput
    from ae.semantic_optimization.steps.generalization import run_generalization

    gt_path = temp_dir / "gt.csv"
    pd.DataFrame(
        {"pdf": ["doc1", "doc2"], "field_a": ["a1", "a2"], "field_b": ["b1", "b2"]}
    ).to_csv(gt_path, index=False)

    baseline_path = temp_dir / "baseline.txt"
    baseline_path.write_text("Base prompt", encoding="utf-8")

    schema_path = temp_dir / "schema.yaml"
    schema_path.write_text(
        "name: test\nfields:\n  field_a:\n    type: str\n  field_b:\n    type: str\ncompare_fields:\n  - field_a\n  - field_b\n",
        encoding="utf-8",
    )
    ingest_dir = temp_dir / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    gen_prompt_path = temp_dir / "generated_prompt.txt"
    gen_schema_path = temp_dir / "generated_schema.yaml"
    art_dir = temp_dir / "artifacts"

    ctx = RunContext(
        task_name="test",
        doc_ids=["doc1", "doc2"],
        gt_path=gt_path,
        ingestion_dir=ingest_dir,
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        teacher_lm=mock_lm,
        artifacts_dir=art_dir,
        generated_prompt_path=gen_prompt_path,
        generated_schema_path=gen_schema_path,
        interactive=False,
        harmonize_prompt=True,
        infer_schema_constraints=False,
        resume=False,
    )

    # Pre-populate iter_1 artifacts
    iter1_gen_dir = ctx.artifacts.get_phase_dir("generalization", iteration=1)
    (iter1_gen_dir / "columns").mkdir(parents=True, exist_ok=True)
    (iter1_gen_dir / "rows.json").write_text(
        GeneralizedRowsOutput(
            Row_Instantiation_Trigger_generalized="Iter 1 row trigger",
            Instructions=RowInstructions(Row_Inclusion_Instructions=["Include A"]),
        ).model_dump_json(),
        encoding="utf-8",
    )
    (iter1_gen_dir / "columns" / "field_a.json").write_text(
        GeneralizedColumnsOutput(
            Semantic_Core_generalized="Iter 1 field_a",
            Instructions=ColumnInstructions(Column_Inclusion_Instructions=["Keep A"]),
        ).model_dump_json(),
        encoding="utf-8",
    )
    (iter1_gen_dir / "columns" / "field_b.json").write_text(
        GeneralizedColumnsOutput(
            Semantic_Core_generalized="Iter 1 field_b",
            Instructions=ColumnInstructions(Column_Inclusion_Instructions=["Keep B"]),
        ).model_dump_json(),
        encoding="utf-8",
    )

    ctx.current_iteration = 2
    called_sigs: list[str] = []

    def fake_predict(sig):
        name = getattr(sig, "__name__", str(sig))
        def _call(**kwargs):
            called_sigs.append(f"{name}:{kwargs.get('field_name', '')}")
            if name == "ColumnGeneralization":
                return MagicMock(
                    generalization=GeneralizedColumnsOutput(
                        Semantic_Core_generalized="Iter 2 updated field_b",
                        Instructions=ColumnInstructions(Column_Inclusion_Instructions=["Keep updated B"]),
                    )
                )
            if name == "PromptHarmonization":
                return MagicMock(
                    harmonization=PromptHarmonizationOutput(
                        harmonized_prompt="HARMONIZED ITER 2\n",
                        harmonization_notes=["ok"],
                    )
                )
            raise AssertionError(f"Unexpected LLM call for carried-forward scope: {name}")
        return _call

    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=fake_predict):
        run_generalization(ctx, changed_scopes={"column:field_b"})

    # Only column:field_b and PromptHarmonization should have invoked LLM; rows & field_a carried forward!
    assert called_sigs == ["ColumnGeneralization:field_b", "PromptHarmonization:"]
    iter2_gen = ctx.artifacts.get_phase_dir("generalization", iteration=2)
    assert "Iter 1 row trigger" in (iter2_gen / "rows.json").read_text(encoding="utf-8")
    assert "Iter 1 field_a" in (iter2_gen / "columns" / "field_a.json").read_text(encoding="utf-8")
    assert "Iter 2 updated field_b" in (iter2_gen / "columns" / "field_b.json").read_text(encoding="utf-8")

    # Resume with cached candidate_prompt.txt must skip PromptHarmonization too
    ctx.resume = True
    ctx.artifacts.resume = True
    called_sigs.clear()
    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=fake_predict):
        run_generalization(ctx, changed_scopes={"column:field_b"})
    assert called_sigs == []

    # 1. If candidate_prompt.txt.sha256 is stale/missing, PromptHarmonization must re-run on resume
    prompt_hash_path = iter2_gen / "candidate_prompt.txt.sha256"
    assert prompt_hash_path.exists()
    prompt_hash_path.write_text("stale_hash", encoding="utf-8")
    called_sigs.clear()
    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=fake_predict):
        run_generalization(ctx, changed_scopes={"column:field_b"})
    assert called_sigs == ["PromptHarmonization:"]

    # 2. If new expert_rulings appear for column:field_a, carry-forward must NOT reuse iter_1/columns/field_a.json
    from ae.semantic_optimization.models import ExpertRuling
    ctx.rulings_store.add_ruling(
        ExpertRuling(
            conflict_id="c_fa",
            scope="column:field_a",
            conflict_type="BASELINE_POLICY_CONFLICT",
            action_type="OVERRIDE_BASELINE",
            decision_text="Always normalize field_a",
            document_id="doc1",
            row_id="row_1",
            field_name="field_a",
        )
    )
    # Simulate fresh iter_3 where field_a is not in changed_scopes
    ctx.current_iteration = 3
    ctx.resume = False
    ctx.artifacts.resume = False
    called_sigs.clear()

    def fake_predict_iter3(sig):
        name = getattr(sig, "__name__", str(sig))
        def _call(**kwargs):
            called_sigs.append(f"{name}:{kwargs.get('field_name', '')}")
            if name == "ColumnGeneralization":
                return MagicMock(
                    generalization=GeneralizedColumnsOutput(
                        Semantic_Core_generalized=f"Iter 3 {kwargs.get('field_name')}",
                        Instructions=ColumnInstructions(Column_Inclusion_Instructions=["Updated rule"]),
                    )
                )
            if name == "PromptHarmonization":
                return MagicMock(
                    harmonization=PromptHarmonizationOutput(
                        harmonized_prompt="HARMONIZED ITER 3\n",
                        harmonization_notes=["ok"],
                    )
                )
            raise AssertionError(f"Unexpected LLM call: {name}")
        return _call

    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=fake_predict_iter3):
        run_generalization(ctx, changed_scopes=set())
    assert "ColumnGeneralization:field_a" in called_sigs
    iter3_cons_fa = ctx.artifacts.get_path("consolidation", "columns", "field_a.json", iteration=3)
    assert iter3_cons_fa.exists()
    assert "Iter 3 field_a" in iter3_cons_fa.read_text(encoding="utf-8")


def test_deterministic_canonical_equivalence_sanitization_and_conflict_filtering():
    """Canonically equivalent values under schema validators clear pseudo-transformations and false conflicts deterministically."""
    pytest.importorskip("rdkit")
    from ae.semantic_optimization.models import FieldAnalysis, PositiveColumnsOutput, REConflict
    from ae.semantic_optimization.steps.positive_analysis import (
        filter_equivalent_conflicts,
        sanitize_positive_columns_output,
    )

    schema_yaml = (
        "name: test\n"
        "fields:\n"
        "  smiles:\n"
        "    type: str\n"
        "    required: false\n"
        "    validators:\n"
        "      - name: smiles\n"
        "  compound_id:\n"
        "    type: str\n"
        "    required: false\n"
    )

    col_out = PositiveColumnsOutput(
        fields=[
            FieldAnalysis(
                row_id="row_1",
                field_name="smiles",
                raw_text="c1ccccc1",
                ground_truth="C1=CC=CC=C1",
                Semantic_Core="SMILES",
                Analytical_Method="Not_Specified",
                System_Conditions="Not_Specified",
                Hierarchy_Level_class="Macro_System",
                Hierarchy_Level_description="Macro",
                Semantic_Binding_class="Syntax_Direct",
                Semantic_Binding_description="Direct",
                Text_Precision_class="Exact_Value",
                Text_Precision_description="Exact",
                Value_Multiplicity_class="Single_Instance",
                Value_Multiplicity_description="Single",
                Transformation_description="Convert aromatic SMILES to Kekule form.",
                Suspected_Ingestion_Or_GT_Issue="Notation mismatch",
            )
        ]
    )
    sanitized = sanitize_positive_columns_output(col_out, schema_yaml)
    assert sanitized.fields[0].Suspected_Ingestion_Or_GT_Issue == "None"
    assert "canonically equivalent" in sanitized.fields[0].Transformation_description.lower()

    false_canon_conflict = REConflict(
        conflict_id="false_kekule",
        scope="column:smiles",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc_1",
        row_id="row_1",
        raw_text_quote="c1ccccc1",
        gt_value="C1=CC=CC=C1",
        conflicting_value_or_rule="Aromatic vs Kekule SMILES",
        explanation="Surface SMILES differs",
        options=[],
    )
    real_non_equiv_conflict = REConflict(
        conflict_id="real_markush",
        scope="column:smiles",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc_2",
        row_id="row_1",
        raw_text_quote="c1ccccc1[*:1]",
        gt_value="Cc1ccccc1",
        conflicting_value_or_rule="Assemble Markush scaffold and R-group vs literal SMILES",
        explanation="Unexpanded Markush placeholder",
        options=[],
    )
    kept = filter_equivalent_conflicts([false_canon_conflict, real_non_equiv_conflict], schema_yaml)
    assert [c.conflict_id for c in kept] == ["real_markush"]


def test_are_field_values_equivalent_and_conflicts_stereo_handling():
    """Unassigned E/Z stereo is equivalent and filtered; explicit (E) vs (Z) conflict is preserved."""
    pytest.importorskip("rdkit")
    from ae.semantic_optimization.models import REConflict
    from ae.semantic_optimization.steps.positive_analysis import (
        are_field_values_equivalent,
        filter_equivalent_conflicts,
    )

    schema_yaml = (
        "name: test_stereo\n"
        "fields:\n"
        "  smiles:\n"
        "    type: str\n"
        "    validators:\n"
        "      - name: smiles\n"
    )

    assert are_field_values_equivalent("smiles", "CC=CC", "C/C=C/C", schema_yaml) is True
    assert are_field_values_equivalent("smiles", "C/C=C/C", "C/C=C\\C", schema_yaml) is False

    unassigned_conflict = REConflict(
        conflict_id="unassigned_ez",
        scope="column:smiles",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="ja.2017.70",
        row_id="row_1",
        raw_text_quote="CC=CC",
        gt_value="C/C=C/C",
        conflicting_value_or_rule="Flat vs E/Z",
        explanation="Missing stereo in VLM",
        options=[],
    )
    explicit_conflict = REConflict(
        conflict_id="explicit_ez_diff",
        scope="column:smiles",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="ja.2017.70",
        row_id="row_2",
        raw_text_quote="C/C=C/C",
        gt_value="C/C=C\\C",
        conflicting_value_or_rule="E vs Z",
        explanation="Stereo mismatch",
        options=[],
    )
    kept = filter_equivalent_conflicts([unassigned_conflict, explicit_conflict], schema_yaml)
    assert [c.conflict_id for c in kept] == ["explicit_ez_diff"]


def test_generalization_hitl_second_round_and_cleanup(temp_dir: Path, mock_lm):
    from types import SimpleNamespace
    from ae.semantic_optimization.hitl import BatchResolutionResult
    from ae.semantic_optimization.models import ExpertRuling, REConflict
    from ae.semantic_optimization.steps.generalization import run_generalization

    gt_path = temp_dir / "gt.csv"
    gt_path.write_text("pdf,field_a\ndoc1,val1\n", encoding="utf-8")
    ing_dir = temp_dir / "ingestion"
    ing_dir.mkdir(parents=True)
    art_dir = temp_dir / "artifacts"
    baseline_path = temp_dir / "baseline.txt"
    baseline_path.write_text("## RECORD SELECTION\nExtract.\n\n---\n\n## FIELD_A\nExtract field_a.", encoding="utf-8")
    schema_path = temp_dir / "schema.yaml"
    schema_path.write_text("name: test\nfields:\n  field_a:\n    type: str\n    description: test\ncompare_fields:\n  - field_a\n", encoding="utf-8")

    ctx = RunContext(
        task_name="test_hitl_rounds",
        doc_ids=["doc1"],
        gt_path=gt_path,
        ingestion_dir=ing_dir,
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        teacher_lm=mock_lm,
        artifacts_dir=art_dir,
        interactive=False,
    )
    ctx.current_iteration = 1

    c1 = REConflict(
        conflict_id="c_round1",
        scope="column:field_a",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc1",
        row_id="row_1",
        explanation="Round 1 conflict",
        options=[],
    )
    c2 = REConflict(
        conflict_id="c_round2_new",
        scope="column:field_a",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc2",
        row_id="row_2",
        explanation="Newly synthesized conflict after round 1 recompute",
        options=[],
    )
    c3 = REConflict(
        conflict_id="c_round3_still_unresolved",
        scope="column:field_a",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc3",
        row_id="row_3",
        explanation="Unresolved conflict after round 2",
        options=[],
    )

    col_calls = {"count": 0}
    resolve_batches: list[list[str]] = []

    class FakeResolver:
        def resolve_batch(self, conflicts):
            resolve_batches.append([c.conflict_id for c in conflicts])
            for c in conflicts:
                ctx.rulings_store.add_ruling(
                    ExpertRuling(
                        conflict_id=c.conflict_id,
                        scope=c.scope,
                        conflict_type=c.conflict_type,
                        action_type="KEEP_BASELINE",
                        decision_text=f"Resolved {c.conflict_id}",
                        document_id=c.document_id,
                        row_id=c.row_id,
                        field_name="field_a",
                    )
                )
            return BatchResolutionResult(
                rulings=[],
                recompute_scopes={"column:field_a"},
                recompute_doc_ids=set(),
            )

    ctx.resolver = FakeResolver()

    class FakePredict:
        def __init__(self, sig):
            self.sig_name = getattr(sig, "__name__", str(sig))

        def __call__(self, **kwargs):
            if "RowGeneralization" in self.sig_name:
                return SimpleNamespace(
                    generalization=GeneralizedRowsOutput(
                        Row_Instantiation_Trigger_generalized="Row trigger",
                        Entity_Role_and_Filtration_Policy="Row policy",
                        Instructions={},
                        Conflicts=[],
                    )
                )
            if "ColumnGeneralization" in self.sig_name:
                col_calls["count"] += 1
                if col_calls["count"] == 1:
                    conflicts = [c1]
                elif col_calls["count"] == 2:
                    # c1 is already resolved, c2 is newly synthesized
                    conflicts = [c1, c2]
                else:
                    # c3 remains after round 2, must not trigger a 3rd round
                    conflicts = [c2, c3]
                return SimpleNamespace(
                    generalization=GeneralizedColumnsOutput(
                        Semantic_Core_generalized="Col core",
                        Analytical_and_Binding_Policy="Col policy",
                        Instructions={},
                        Conflicts=conflicts,
                    )
                )
            return SimpleNamespace(harmonized_prompt="Harmonized prompt")

    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=FakePredict):
        needs_doc_recompute = run_generalization(ctx)

    assert needs_doc_recompute is False
    assert resolve_batches == [["c_round1"], ["c_round2_new"]]
    assert col_calls["count"] == 3

    saved_col = GeneralizedColumnsOutput.model_validate_json(
        ctx.artifacts.get_path("generalization", "columns", "field_a.json", iteration=1).read_text(encoding="utf-8")
    )
    # c2 was resolved in round 2 and cleaned from saved artifact; only c3 remains
    assert [c.conflict_id for c in saved_col.Conflicts] == ["c_round3_still_unresolved"]


def test_re_signatures_and_visual_prep_are_domain_clean_and_harmonized():
    import re
    from ae.semantic_optimization.signatures.columns import (
        ColumnConsolidation,
        ColumnGeneralization,
        NegativeColumnAnalysis,
        PositiveColumnAnalysis,
        PromptHarmonization,
    )
    from ae.semantic_optimization.signatures.residual import ResidualNegativeRowAnalysis
    from ae.semantic_optimization.signatures.rows import (
        NegativeRowAnalysis,
        RowConsolidation,
        RowGeneralization,
    )
    from ae.semantic_optimization.steps.visual_prep import _CONTRASTIVE_VISUAL_PREP_SYSTEM_PROMPT

    leak_re = re.compile(
        r"smiles|rdkit|tio2|cytotoxicity|hemolysis|nmr|hrms|xrd|docking|antibacterial|precursor|morphology",
        re.IGNORECASE,
    )
    targets = [
        PositiveColumnAnalysis.__doc__ or "",
        ColumnConsolidation.__doc__ or "",
        NegativeColumnAnalysis.__doc__ or "",
        ColumnGeneralization.__doc__ or "",
        PromptHarmonization.__doc__ or "",
        RowConsolidation.__doc__ or "",
        NegativeRowAnalysis.__doc__ or "",
        RowGeneralization.__doc__ or "",
        ResidualNegativeRowAnalysis.__doc__ or "",
        _CONTRASTIVE_VISUAL_PREP_SYSTEM_PROMPT,
    ]
    for text in targets:
        assert not leak_re.search(text), f"Domain leak found in: {text[:120]!r}"

    harmonize_doc = (PromptHarmonization.__doc__ or "").lower()
    assert "specificity precedence" in harmonize_doc
    assert "footnote" in harmonize_doc and "takes precedence" in harmonize_doc

    col_gen_doc = (ColumnGeneralization.__doc__ or "").lower()
    assert "cross-checking" in col_gen_doc and "ocr/vlm" in col_gen_doc


def test_visual_prep_logs_to_re_history_with_context(temp_dir):
    """prepare_document_visuals records ChatModel calls into SemanticOptimizationClient re_*.json with doc_id/source."""
    import json
    from unittest.mock import MagicMock, patch
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from ae.core.config.optimization_settings import SemanticOptimizationConfig
    from ae.core.storage.documents import DocumentRepository
    from ae.semantic_optimization.client import SemanticOptimizationClient
    from ae.semantic_optimization.context import RunContext
    from ae.semantic_optimization.steps.visual_prep import prepare_document_visuals

    ingestion_dir = temp_dir / "ingestion"
    doc_dir = ingestion_dir / "doc1"
    doc_dir.mkdir(parents=True)
    raw_text = '<visual id="v1" type="table">Table 1\n> desc</visual>'
    (doc_dir / "document.md").write_text(raw_text, encoding="utf-8")

    re_client = SemanticOptimizationClient(
        config=SemanticOptimizationConfig(model={"provider": "gemini", "model": "test-re-model"})
    )

    def _fake_create_chat_model(cfg, callbacks=None, **kwargs):
        mock_chat = MagicMock()
        def _bind_tools(tools):
            bound = MagicMock()
            def _invoke(messages):
                for cb in callbacks or []:
                    cb.on_chat_model_start(serialized={}, messages=[[]], run_id="vp-1", invocation_params={"model": "test-re-model"})
                    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="READY"))]]), run_id="vp-1")
                return AIMessage(content="READY", tool_calls=[])
            bound.invoke.side_effect = _invoke
            return bound
        mock_chat.bind_tools.side_effect = _bind_tools
        return mock_chat

    from types import SimpleNamespace
    ctx = SimpleNamespace(
        ingestion_dir=ingestion_dir,
        re_client=re_client,
        artifacts=None,
        config_path=None,
        pdf_dir=None,
        schema="",
        baseline_prompt="",
        max_tool_iterations=1,
    )
    doc_repo = DocumentRepository(ingestion_dir)

    with patch("ae.semantic_optimization.client.create_chat_model", side_effect=_fake_create_chat_model):
        prepare_document_visuals(
            norm_doc_id="doc1",
            raw_text=raw_text,
            gt_table_csv="a,b\n1,2",
            context=ctx,
            doc_repo=doc_repo,
        )

    out_dir = temp_dir / "logs"
    saved = re_client.save_history(out_dir, timestamp="vp_ts")
    assert saved == 1
    entries = json.loads((out_dir / "re_vp_ts.json").read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["doc_id"] == "doc1"
    assert entries[0]["source"] == "visual_prep"
    assert entries[0]["stage"] == "visual_prep"
    assert entries[0]["node"] == "prepare_document_visuals"


def test_case_patch_filter_and_contradiction_first_harmonization():
    """CASE_PATCH literals are filtered while GENERALIZED_RULEs survive, and Contradiction-First detects cross-section conflicts."""
    from ae.semantic_optimization.models import RuleAttribution
    from ae.semantic_optimization.steps.generalization import (
        _detect_prompt_contradictions,
        filter_case_patch_instructions,
        parse_prompt_sections,
        replace_prompt_sections,
    )

    instructions = [
        "Extract SMILES from Chemical Entities Registry tables or assemble via assemble_markush_smiles.",
        "In j.ejmech.2016.01.052, expand MRSA to Methicillin-Resistant Staphylococcus aureus N315.",
        "For compound 4d, replace 'CCC' with 'CCN' manually.",
        "Preserve exact precision (e.g., '0.030') when present in source.",
        "For row_22, keep value '0.202' verbatim.",
    ]
    attrs = [
        RuleAttribution(
            rule_text=instructions[0],
            rule_kind="GENERALIZED_RULE",
            supporting_doc_ids=["doc1", "doc2"],
        ),
        RuleAttribution(
            rule_text=instructions[1],
            rule_kind="CASE_PATCH",
            supporting_doc_ids=["j.ejmech.2016.01.052"],
        ),
        RuleAttribution(
            rule_text=instructions[3],
            rule_kind="GENERALIZED_RULE",
            supporting_doc_ids=["doc1", "doc2"],
        ),
    ]
    kept, blocked = filter_case_patch_instructions(
        instructions,
        attrs,
        known_doc_ids={"j.ejmech.2016.01.052", "antibiotics12071220"},
        active_doc_count=2,
        scope="column:smiles",
    )
    assert kept == [instructions[0], instructions[3]]
    assert len(blocked) == 3

    draft = (
        "Preamble\n\n---\n\n"
        "EXPERT RULINGS AND AUTHORITATIVE DECISIONS (HARD CONSTRAINTS):\n"
        "- Record bacterial names in the format they appear without normalization.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Valid SMILES.\n\n---\n\n"
        "BACTERIA\n\nTransformation rules:\n- Apply table footnote expansions for abbreviated headers.\n"
    )
    findings, involved = _detect_prompt_contradictions(
        parse_prompt_sections(draft),
        "Record bacterial names in the format they appear without normalization.",
        ["smiles", "bacteria"],
    )
    assert len(findings) == 1
    assert "BACTERIA" in involved
    assert "SMILES" not in involved

    repaired = replace_prompt_sections(
        target_prompt=draft,
        reference_prompt=(
            "Preamble\n\n---\n\n"
            "SMILES\n\nExtract ONLY if satisfy:\n- Restored SMILES rule.\n\n---\n\n"
            "BACTERIA\n\nTransformation rules:\n- Old bacteria.\n"
        ),
        field_names=["smiles"],
    )
    assert "Restored SMILES rule." in repaired
    assert "Apply table footnote expansions" in repaired


def test_case_patch_normalized_jaccard_benign_tokens_and_generic_contradiction():
    """Verify normalized/Jaccard attribution matching, relaxed operator/unit/assay tokens, and domain-agnostic contradiction audit."""
    from ae.semantic_optimization.models import RuleAttribution
    from ae.semantic_optimization.steps.generalization import (
        _detect_prompt_contradictions,
        filter_case_patch_instructions,
        parse_prompt_sections,
    )

    # 1. Normalized & token-Jaccard fallback catches CASE_PATCH even when rule_text has formatting/minor wording drift
    inst_paraphrased = "Replace 'CCC' with 'CCN' manually for this specific compound."
    attrs = [
        RuleAttribution(
            rule_text="- replace 'CCC' with 'CCN' manually for specific compound",
            rule_kind="CASE_PATCH",
            supporting_doc_ids=["doc1"],
        ),
        # 2. Benign short operators, units, and assay tokens from 1 doc must NOT be falsely filtered
        RuleAttribution(
            rule_text="Normalize operator '<=' and unit 'ug/mL' for 'IC50' values.",
            rule_kind="GENERALIZED_RULE",
            supporting_doc_ids=["doc1"],
        ),
    ]
    kept, blocked = filter_case_patch_instructions(
        [inst_paraphrased, "Normalize operator '<=' and unit 'ug/mL' for 'IC50' values."],
        attrs,
        known_doc_ids={"doc1", "doc2"},
        active_doc_count=2,
        scope="column:target_units",
    )
    assert kept == ["Normalize operator '<=' and unit 'ug/mL' for 'IC50' values."]
    assert len(blocked) == 1
    assert blocked[0]["reason"] == "attributed_as_CASE_PATCH"

    # 3. Domain-agnostic contradiction detection (non-pMIC token such as 'ic50')
    draft = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Include IC50 assays.\n\nExclude:\n- Unrelated controls.\n\n---\n\n"
        "TARGET_TYPE\n\nDO NOT extract:\n- Exclude IC50 measurements.\n"
    )
    findings, involved = _detect_prompt_contradictions(
        parse_prompt_sections(draft),
        "",
        ["target_type"],
    )
    assert len(findings) == 1
    assert "TARGET_TYPE" in involved
    assert "RECORD SELECTION" in involved


def test_run_generalization_never_emits_empty_record_inclusion_section(tmp_path):
    """When Row_Inclusion_Instructions are filtered or empty, RECORD SELECTION falls back cleanly instead of empty Include block."""
    import pandas as pd
    from ae.semantic_optimization.context import RunContext
    from ae.semantic_optimization.models import (
        ColumnInstructions,
        GeneralizedColumnsOutput,
        GeneralizedRowsOutput,
        RowInstructions,
        RuleAttribution,
    )
    from ae.semantic_optimization.steps.generalization import run_generalization

    ingest_dir = tmp_path / "ing"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    gt_path = tmp_path / "gt.csv"
    pd.DataFrame({"pdf": ["doc1", "doc2"], "target_type": ["MIC", "MIC"]}).to_csv(gt_path, index=False)
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfields:\n  target_type:\n    type: str\ncompare_fields:\n  - target_type\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("Base prompt", encoding="utf-8")
    gen_prompt = tmp_path / "generated_instruction.txt"

    ctx = RunContext(
        task_name="test",
        doc_ids=["doc1", "doc2"],
        gt_path=gt_path,
        ingestion_dir=ingest_dir,
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        teacher_lm=MagicMock(),
        artifacts_dir=tmp_path / "art",
        generated_prompt_path=gen_prompt,
        harmonize_prompt=False,
        infer_schema_constraints=False,
        interactive=False,
    )
    ctx.current_iteration = 1

    # Row inclusion contains ONLY a CASE_PATCH that will be stripped
    bad_row_inst = "In doc1, include row_1 only."
    row_out = GeneralizedRowsOutput(
        Row_Instantiation_Trigger_generalized="One record per tested compound-assay pair.",
        Instructions=RowInstructions(
            Row_Inclusion_Instructions=[bad_row_inst],
            Row_Exclusion_Instructions=["Exclude background buffers."],
        ),
        Rule_Attributions=[
            RuleAttribution(
                rule_text=bad_row_inst,
                rule_kind="CASE_PATCH",
                supporting_doc_ids=["doc1"],
            )
        ],
    )
    col_out = GeneralizedColumnsOutput(
        Semantic_Core_generalized="Assay type",
        Instructions=ColumnInstructions(Column_Inclusion_Instructions=["Extract assay type."]),
    )

    def fake_predict(sig):
        name = getattr(sig, "__name__", str(sig))
        if name == "RowGeneralization":
            return lambda **kwargs: MagicMock(generalization=row_out)
        return lambda **kwargs: MagicMock(generalization=col_out)

    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=fake_predict):
        run_generalization(ctx)

    text = gen_prompt.read_text(encoding="utf-8")
    assert "Include ONLY records satisfying:\n- Extract one record per valid target instance matching: One record per tested compound-assay pair." in text
    assert bad_row_inst not in text




