"""Unit tests for SemanticOptimizationGraph (Iterative LangGraph for Semantic Prompt Engineering)."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
import json

from ae.semantic_optimization.graph import SemanticOptimizationGraph, SemanticOptimizationState
from ae.semantic_optimization.context import RunContext


@pytest.fixture
def mock_context(tmp_path):
    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = tmp_path / "artifacts"
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    ctx.artifacts = MagicMock()
    ctx.artifacts.get_phase_dir.side_effect = lambda phase, *sub, **kwargs: tmp_path / "artifacts" / phase
    ctx.schema_path = tmp_path / "schema.yaml"
    ctx.schema_path.write_text("name: test\nfields:\n  - name: test\n", encoding="utf-8")
    ctx.baseline_prompt_path = tmp_path / "baseline.txt"
    ctx.baseline_prompt_path.write_text("baseline", encoding="utf-8")
    ctx.generated_prompt_path = tmp_path / "generated_instruction.txt"
    ctx.generated_schema_path = tmp_path / "generated_schema.yaml"
    ctx.gt_path = tmp_path / "gt.csv"
    ctx.gt_path.write_text("doi,yield\npaper1,80\npaper2,90\n", encoding="utf-8")
    ctx.active_gt_path = ctx.gt_path
    ctx.rulings_store = MagicMock()
    ctx.rulings_store.rulings = {}
    ctx.ingestion_dir = tmp_path / "ingestion"
    ctx.ingestion_dir.mkdir(parents=True, exist_ok=True)
    return ctx


def test_re_graph_early_stopping(mock_context, tmp_path):
    # At iteration 1: evaluate returns f1 = 0.95 (>= 0.90 threshold) -> should early stop!
    def eval_side_effect(state: SemanticOptimizationState):
        return {
            "val_f1_history": state["val_f1_history"] + [0.95],
            "best_f1": 0.95,
            "best_prompt": "prompt_iteration_1",
        }

    # Patch phase runners to avoid LLM calls
    with patch("ae.semantic_optimization.graph.run_positive_analysis"), \
         patch("ae.semantic_optimization.graph.run_negative_analysis"), \
         patch("ae.semantic_optimization.graph.run_generalization", return_value=False), \
         patch.object(SemanticOptimizationGraph, "_evaluate_val", side_effect=eval_side_effect):

        graph = SemanticOptimizationGraph(
            context=mock_context,
            train_doc_ids=["doc_1", "doc_2", "doc_3"],
            val_doc_ids=["val_1"],
            f1_threshold=0.90,
        )

        final_state = graph.run()

        assert final_state["current_iteration"] == 1
        assert final_state["early_stopped"] is True
        assert final_state["best_f1"] == 0.95


def test_re_graph_multi_iteration_exhaustion(mock_context, tmp_path):
    # F1 never reaches 0.95 (0.50 then 0.75)
    eval_mock = MagicMock(side_effect=[
        {"val_f1_history": [0.50], "best_f1": 0.50, "best_prompt": "p1"},
        {"val_f1_history": [0.50, 0.75], "best_f1": 0.75, "best_prompt": "p2"},
    ])

    with patch("ae.semantic_optimization.graph.run_positive_analysis"), \
         patch("ae.semantic_optimization.graph.run_negative_analysis"), \
         patch("ae.semantic_optimization.graph.run_generalization", return_value=False), \
         patch.object(SemanticOptimizationGraph, "_evaluate_val", eval_mock):

        graph = SemanticOptimizationGraph(
            context=mock_context,
            train_doc_ids=["doc_1", "doc_2"],
            val_doc_ids=["val_1"],
            f1_threshold=0.95,
        )

        final_state = graph.run()

        assert final_state["current_iteration"] == 2
        assert final_state["early_stopped"] is False
        assert final_state["best_f1"] == 0.75
        assert len(final_state["val_f1_history"]) == 2


def test_re_graph_configurable_batching(mock_context, tmp_path):
    """Verify initial_docs=2 and step_size=2 process docs in batches without re-running 1A/1B on old docs."""
    pos_batches = []
    neg_batches = []
    gen_active_sets = []

    def track_pos(ctx):
        pos_batches.append(list(ctx.doc_ids))

    def track_neg(ctx):
        neg_batches.append(list(ctx.doc_ids))

    def track_gen(ctx):
        gen_active_sets.append(list(ctx.doc_ids))
        return False

    eval_mock = MagicMock(
        side_effect=[
            {"val_f1_history": [0.60], "best_f1": 0.60, "best_prompt": "p1", "best_iteration": 1},
            {"val_f1_history": [0.60, 0.92], "best_f1": 0.92, "best_prompt": "p2", "best_iteration": 2},
        ]
    )

    with patch("ae.semantic_optimization.graph.run_positive_analysis", side_effect=track_pos), \
         patch("ae.semantic_optimization.graph.run_negative_analysis", side_effect=track_neg), \
         patch("ae.semantic_optimization.graph.run_generalization", side_effect=track_gen), \
         patch.object(SemanticOptimizationGraph, "_evaluate_val", eval_mock):

        graph = SemanticOptimizationGraph(
            context=mock_context,
            train_doc_ids=["doc_1", "doc_2", "doc_3", "doc_4"],
            val_doc_ids=["val_1"],
            f1_threshold=0.90,
            initial_docs=2,
            step_size=2,
        )

        final_state = graph.run()

        assert final_state["current_iteration"] == 2
        assert final_state["early_stopped"] is True
        # Iteration 1 analyzed [doc_1, doc_2]; Iteration 2 analyzed ONLY [doc_3, doc_4]
        assert pos_batches == [["doc_1", "doc_2"], ["doc_3", "doc_4"]]
        assert neg_batches == [["doc_1", "doc_2"], ["doc_3", "doc_4"]]
        # Generalization synthesized rules over active sets [doc_1, doc_2] then [doc_1..doc_4]
        assert gen_active_sets == [["doc_1", "doc_2"], ["doc_1", "doc_2", "doc_3", "doc_4"]]


def test_route_after_generalization_branching(mock_context):
    graph = SemanticOptimizationGraph(context=mock_context, train_doc_ids=["doc_1"])
    state_recompute = {"_gt_modified_in_generalization": True}
    assert graph._route_after_generalization(state_recompute) == "recompute_positive"

    state_proceed = {"_gt_modified_in_generalization": False}
    assert graph._route_after_generalization(state_proceed) == "proceed_to_evaluate"


def test_empty_val_doc_ids_preserves_latest_prompt(mock_context, tmp_path):
    gen_prompt = tmp_path / "gen_prompt.txt"
    gen_prompt.write_text("prompt_iter_2", encoding="utf-8")
    mock_context.generated_prompt_path = gen_prompt

    graph = SemanticOptimizationGraph(context=mock_context, train_doc_ids=["doc_1"])
    state = {
        "val_doc_ids": [],
        "val_f1_history": [0.0],
        "current_iteration": 2,
        "best_prompt": "prompt_iter_1",
        "best_iteration": 1,
        "context": mock_context,
    }
    res = graph._evaluate_val(state)
    assert res["best_iteration"] == 2
    assert res["best_prompt"] == "prompt_iter_2"
    assert res["val_f1_history"] == [0.0, 0.0]


def test_evaluate_val_nan_handling_and_doc_id_matching(mock_context, tmp_path):
    import pandas as pd
    gt_file = tmp_path / "active_gt.csv"
    df = pd.DataFrame({
        "extra_col": [1, 2],
        "document": ["val_1", "val_2"],
        "yield": [85.0, float("nan")],
    })
    df.to_csv(gt_file, index=False)
    mock_context.active_gt_path = gt_file

    gen_prompt = tmp_path / "prompt.txt"
    gen_prompt.write_text("system prompt", encoding="utf-8")
    mock_context.generated_prompt_path = gen_prompt

    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Text with 85% yield", encoding="utf-8")
    (ingest_dir / "val_2.md").write_text("Text with no yield", encoding="utf-8")
    mock_context.ingestion_dir = ingest_dir

    mock_task = MagicMock()
    del mock_task.row_converter
    mock_task.config.metric_config = {"compare_fields": ["yield"]}
    mock_task.config.compare_fields = ["yield"]
    mock_task.config.judge_field_descriptions = {"yield": "reaction yield"}
    mock_task.config.float_tolerance = 1.0

    captured_records = []
    def fake_output_model(records):
        captured_records.extend(records)
        return MagicMock()
    mock_task.output_model = fake_output_model

    graph = SemanticOptimizationGraph(context=mock_context, train_doc_ids=["doc_1"], val_doc_ids=["val_2"])
    state = {
        "val_doc_ids": ["val_2"],
        "val_f1_history": [],
        "current_iteration": 1,
        "best_f1": 0.0,
        "best_prompt": "",
        "best_iteration": 1,
        "context": mock_context,
    }

    with patch("ae.core.schema.load_schema_complete", return_value=mock_task), \
         patch("ae.core.evaluation.TaskMetric", return_value=MagicMock(return_value=1.0)), \
         patch("ae.extraction.agent.UniversalExtractor.forward", return_value=MagicMock()):
        res = graph._evaluate_val(state)

    assert len(captured_records) == 1
    # pandas NaN must be converted to None!
    assert captured_records[0]["yield"] is None


def test_section_cleaner_and_prompt_assembly():
    from ae.semantic_optimization.steps.generalization import _clean_header
    assert _clean_header("## RECORD SELECTION") == "RECORD SELECTION"
    assert _clean_header("**SMILES:**") == "SMILES"
    assert _clean_header("### Reaction Yield (isolated)") == "REACTION YIELD (ISOLATED)"


def test_evaluate_val_updates_best_prompt_on_equal_f1_tie(mock_context, tmp_path):
    gen_prompt = tmp_path / "gen_prompt.txt"
    gen_prompt.write_text("prompt_iter_2", encoding="utf-8")
    mock_context.generated_prompt_path = gen_prompt
    mock_context.config_path = None

    gt_file = tmp_path / "gt_tie.csv"
    gt_file.write_text("doi,yield\nval_1,80\n", encoding="utf-8")
    mock_context.active_gt_path = gt_file

    ingest_dir = tmp_path / "ingest_tie"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("yield is 80", encoding="utf-8")
    mock_context.ingestion_dir = ingest_dir

    mock_task = MagicMock()
    del mock_task.row_converter
    mock_task.config.metric_config = {"compare_fields": ["yield"]}
    mock_task.config.compare_fields = ["yield"]
    mock_task.config.judge_field_descriptions = {"yield": "reaction yield"}
    mock_task.config.float_tolerance = 0.05

    graph = SemanticOptimizationGraph(context=mock_context, train_doc_ids=["doc_1", "doc_2"], val_doc_ids=["val_1"])
    state = {
        "val_doc_ids": ["val_1"],
        "val_f1_history": [0.75],
        "current_iteration": 2,
        "best_f1": 0.75,
        "best_prompt": "prompt_iter_1",
        "best_iteration": 1,
        "context": mock_context,
    }

    with patch("ae.core.schema.load_schema_complete", return_value=mock_task), \
         patch("ae.core.evaluation.TaskMetric", return_value=MagicMock(return_value=0.75)), \
         patch("ae.extraction.agent.UniversalExtractor.forward", return_value=MagicMock()):
        res = graph._evaluate_val(state)

    assert res["best_f1"] == 0.75
    assert res["best_iteration"] == 2
    assert res["best_prompt"] == "prompt_iter_2"


def test_re_graph_hybrid_anchor_to_residual_zero_delta_and_fallback(tmp_path):
    """Verify Anchor (Iter 1) -> Residual Zero-Delta (Iter 2) -> Residual Fallback (Iter 3)."""
    from ae.semantic_optimization.context import ArtifactManager
    from ae.semantic_optimization.steps.residual_analysis import ResidualBatchSummary

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = tmp_path / "artifacts"
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    ctx.artifacts = ArtifactManager(ctx.artifacts_dir, resume=False)
    ctx.schema_path = tmp_path / "schema.yaml"
    ctx.schema_path.write_text("name: test\nfields:\n  a:\n    type: str\ncompare_fields: [a]\n", encoding="utf-8")
    ctx.generated_prompt_path = tmp_path / "generated_instruction.txt"
    ctx.generated_schema_path = tmp_path / "generated_schema.yaml"
    ctx.active_gt_path = tmp_path / "gt.csv"
    ctx.active_gt_path.write_text("pdf,a\ndoc_1,x\ndoc_2,y\ndoc_3,z\n", encoding="utf-8")
    ctx.rulings_store = MagicMock()
    ctx.rulings_store.rulings = {}
    ctx.ingestion_dir = tmp_path / "ingest"
    ctx.ingestion_dir.mkdir(parents=True, exist_ok=True)
    ctx.residual_enabled = True
    ctx.enable_semantic_judge = True

    pos_calls = []
    neg_calls = []
    res_calls = []
    gen_calls = []
    val_calls = []

    def track_pos(c):
        pos_calls.append(list(c.doc_ids))

    def track_neg(c):
        neg_calls.append(list(c.doc_ids))

    def track_gen(c, changed_scopes=None):
        gen_calls.append((list(c.doc_ids), changed_scopes))
        ctx.generated_prompt_path.write_text(f"prompt_iter_{len(gen_calls)}", encoding="utf-8")
        return False

    def track_res(c):
        res_calls.append(list(c.doc_ids))
        if c.doc_ids == ["doc_2"]:
            return ResidualBatchSummary(zero_delta_doc_ids=["doc_2"])
        return ResidualBatchSummary(fallback_doc_ids=["doc_3"])

    orig_evaluate_val = SemanticOptimizationGraph._evaluate_val

    def fake_eval(self_graph, state):
        if state.get("zero_delta_iteration"):
            return orig_evaluate_val(self_graph, state)
        val_calls.append(state["current_iteration"])
        score = 0.70 if state["current_iteration"] == 1 else 0.95
        return {
            "val_f1_history": state["val_f1_history"] + [score],
            "best_f1": score,
            "best_prompt": f"prompt_iter_{state['current_iteration']}",
            "best_iteration": state["current_iteration"],
        }

    with patch("ae.semantic_optimization.graph.run_positive_analysis", side_effect=track_pos), \
         patch("ae.semantic_optimization.graph.run_negative_analysis", side_effect=track_neg), \
         patch("ae.semantic_optimization.graph.run_residual_analysis", side_effect=track_res), \
         patch("ae.semantic_optimization.graph.run_generalization", side_effect=track_gen), \
         patch.object(SemanticOptimizationGraph, "_evaluate_val", new=fake_eval):

        graph = SemanticOptimizationGraph(
            context=ctx,
            train_doc_ids=["doc_1", "doc_2", "doc_3"],
            val_doc_ids=["val_1"],
            f1_threshold=0.90,
            initial_docs=1,
            step_size=1,
        )
        final_state = graph.run()

    # Iter 1: Anchor (doc_1) -> pos, neg, gen, eval (0.70)
    # Iter 2: Residual (doc_2) -> Zero-Delta -> skips pos/neg/gen AND skips extraction in eval (reuses 0.70)
    # Iter 3: Residual (doc_3) -> Alignment Fallback -> pos, neg, gen, eval (0.95) -> early stop!
    assert pos_calls == [["doc_1"], ["doc_3"]]
    assert neg_calls == [["doc_1"], ["doc_3"]]
    assert res_calls == [["doc_2"], ["doc_3"]]
    assert len(gen_calls) == 2
    assert gen_calls[0] == (["doc_1"], None)
    assert gen_calls[1] == (["doc_1", "doc_2", "doc_3"], None)
    assert val_calls == [1, 3]
    assert final_state["val_f1_history"] == [0.70, 0.70, 0.95]
    assert final_state["early_stopped"] is True


def test_evaluate_val_uses_base_schema_for_gt_and_wires_semantic_judge(tmp_path):
    """_evaluate_val must parse GT using base schema (unseen choices safety) and wire student_llm + judge."""
    import pandas as pd
    from ae.semantic_optimization.context import ArtifactManager

    base_schema = tmp_path / "base_schema.yaml"
    base_schema.write_text(
        "name: test\nfields:\n  target_type:\n    type: str\n    required: false\ncompare_fields:\n  - target_type\nfloat_tolerance: 0.05\n",
        encoding="utf-8",
    )
    gen_schema = tmp_path / "generated_schema.yaml"
    gen_schema.write_text(
        "name: test\nfields:\n  target_type:\n    type: str\n    required: false\n    choices:\n      - MIC\ncompare_fields:\n  - target_type\nfloat_tolerance: 0.05\n",
        encoding="utf-8",
    )
    gen_prompt = tmp_path / "generated_instruction.txt"
    gen_prompt.write_text("Prompt v1", encoding="utf-8")

    gt_file = tmp_path / "active_gt.csv"
    # GT contains 'pMIC' which is NOT in generated_schema.yaml choices=['MIC']!
    pd.DataFrame({"pdf": ["val_1"], "target_type": ["pMIC"]}).to_csv(gt_file, index=False)

    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("pMIC value = 5.0", encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = tmp_path / "art"
    ctx.artifacts = ArtifactManager(ctx.artifacts_dir, resume=False)
    ctx.schema_path = base_schema
    ctx.generated_schema_path = gen_schema
    ctx.generated_prompt_path = gen_prompt
    ctx.active_gt_path = gt_file
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.resume = False
    ctx.enable_semantic_judge = True

    mock_student_adapter = MagicMock()
    mock_val_client = MagicMock()
    mock_val_client.get_dspy_adapter.return_value = mock_student_adapter
    ctx.re_client = MagicMock()
    ctx.re_client.get_validation_client.return_value = mock_val_client
    mock_extractor = MagicMock()
    mock_extractor.forward.return_value = MagicMock(extracted_data={"records": [{"target_type": "MIC"}]})
    ctx.re_client.create_validation_extractor.return_value = mock_extractor

    graph = SemanticOptimizationGraph(context=ctx, train_doc_ids=["doc_1"], val_doc_ids=["val_1"])
    state = {
        "val_doc_ids": ["val_1"],
        "val_f1_history": [],
        "current_iteration": 1,
        "best_f1": 0.0,
        "best_prompt": "",
        "best_iteration": 1,
        "context": ctx,
    }

    captured_metric_kwargs = {}

    class SpyTaskMetric:
        def __init__(self, **kwargs):
            captured_metric_kwargs.update(kwargs)

        def __call__(self, example, pred):
            # GT conversion via base_schema must preserve 'pMIC' without failing Literal['MIC'] validation
            assert example.extracted_data.records[0].target_type == "pMIC"
            return 0.88

    with patch("ae.core.evaluation.TaskMetric", SpyTaskMetric):
        res = graph._evaluate_val(state)

    assert res["val_f1_history"] == [0.88]
    assert captured_metric_kwargs["task_config"]["name"] == "test"
    assert captured_metric_kwargs["enable_semantic_judge"] is True
    assert captured_metric_kwargs["student_llm"] is mock_student_adapter
    assert captured_metric_kwargs["field_constraints"] == {"target_type": {"choices": ["MIC"]}}
    assert (ctx.artifacts_dir / "iterations" / "iter_1" / "val_metrics.json").exists()


def test_evaluate_val_invokes_semantic_judge_end_to_end(tmp_path):
    """_evaluate_val must invoke student_llm Semantic Judge end-to-end with real TaskMetric and task_name."""
    import pandas as pd
    from ae.semantic_optimization.context import ArtifactManager

    base_schema = tmp_path / "base_schema.yaml"
    base_schema.write_text(
        "name: benzimidazole_test\n"
        "fields:\n"
        "  compound_id:\n"
        "    type: str\n"
        "    required: false\n"
        "  bacteria:\n"
        "    type: str\n"
        "    required: false\n"
        "    description: Bacterial organism name\n"
        "compare_fields:\n"
        "  - compound_id\n"
        "  - bacteria\n"
        "float_tolerance: 0.05\n",
        encoding="utf-8",
    )
    gen_prompt = tmp_path / "generated_instruction.txt"
    gen_prompt.write_text("Prompt v1", encoding="utf-8")

    gt_file = tmp_path / "active_gt.csv"
    pd.DataFrame(
        {"pdf": ["val_1"], "compound_id": ["5a"], "bacteria": ["S. aureus"]}
    ).to_csv(gt_file, index=False)

    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Compound 5a tested on S. aureus", encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = tmp_path / "art"
    ctx.artifacts = ArtifactManager(ctx.artifacts_dir, resume=False)
    ctx.schema_path = base_schema
    ctx.generated_schema_path = None
    ctx.generated_prompt_path = gen_prompt
    ctx.active_gt_path = gt_file
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.resume = False
    ctx.enable_semantic_judge = True

    judge_calls: list[Any] = []

    def spy_student_llm(messages):
        judge_calls.append(messages)
        return ['{"bacteria": {"reasoning": "standard abbreviation", "match": "YES"}}']

    mock_val_client = MagicMock()
    mock_val_client.get_dspy_adapter.return_value = spy_student_llm
    ctx.re_client = MagicMock()
    ctx.re_client.get_validation_client.return_value = mock_val_client
    mock_extractor = MagicMock()
    mock_extractor.forward.return_value = MagicMock(
        extracted_data={"records": [{"compound_id": "5a", "bacteria": "S.a."}]}
    )
    ctx.re_client.create_validation_extractor.return_value = mock_extractor

    graph = SemanticOptimizationGraph(context=ctx, train_doc_ids=["doc_1"], val_doc_ids=["val_1"])
    state = {
        "val_doc_ids": ["val_1"],
        "val_f1_history": [],
        "current_iteration": 1,
        "best_f1": 0.0,
        "best_prompt": "",
        "best_iteration": 1,
        "context": ctx,
    }

    res = graph._evaluate_val(state)
    assert len(judge_calls) == 1
    assert "Task Domain: benzimidazole_test" in judge_calls[0][0]["content"]
    assert res["val_f1_history"] == [1.0]


def test_re_graph_resume_from_iter1_into_iter2_residual(tmp_path):
    """Simulate resuming from completed Iter 1 artifacts (including resolved stale conflict IDs) into Iter 2 Residual."""
    import pandas as pd
    import dspy
    from ae.semantic_optimization.models import (
        ColumnInstructions,
        ExpertRuling,
        FieldConstraintProposal,
        GeneralizedColumnsOutput,
        GeneralizedRowsOutput,
        NegativeRowsOutput,
        PositiveRowsOutput,
        REConflict,
        RowInstructions,
        SchemaRefinementOutput,
    )

    art_dir = tmp_path / "artifacts"
    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    for d in ("doc_1", "doc_2", "val_1"):
        (ingest_dir / f"{d}.md").write_text(f"Text for {d}: compound 5a MIC 10", encoding="utf-8")

    gt_path = tmp_path / "gt_source.csv"
    pd.DataFrame(
        {
            "pdf": ["doc_1", "doc_2", "val_1"],
            "compound_id": ["5a", "6a", "7a"],
            "target_relation": ["=", "=", "="],
        }
    ).to_csv(gt_path, index=False)

    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfields:\n  compound_id:\n    type: str\n    required: false\n  target_relation:\n    type: str\n    required: false\ncompare_fields:\n  - compound_id\n  - target_relation\nfloat_tolerance: 0.05\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("Base prompt", encoding="utf-8")
    gen_prompt_path = art_dir / "generated_instruction.txt"
    gen_schema_path = art_dir / "generated_schema.yaml"

    mock_re_client = MagicMock()
    mock_re_client.get_lm_provider.return_value = MagicMock()
    mock_val_client = MagicMock()
    mock_val_client.get_dspy_adapter.return_value = None
    mock_re_client.get_validation_client.return_value = mock_val_client

    ctx = RunContext(
        task_name="test",
        doc_ids=["doc_1", "doc_2"],
        gt_path=gt_path,
        ingestion_dir=ingest_dir,
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        re_client=mock_re_client,
        artifacts_dir=art_dir,
        resume=True,
        interactive=True,
        generated_prompt_path=gen_prompt_path,
        generated_schema_path=gen_schema_path,
        initial_docs=1,
        step_size=1,
        enable_semantic_judge=False,
    )

    # Pre-populate Iter 1 artifacts + expert ruling with different conflict_id
    ctx.rulings_store.add_ruling(
        ExpertRuling(
            conflict_id="old_baseline_conflict_id",
            scope="column:compound_id",
            conflict_type="BASELINE_POLICY_CONFLICT",
            action_type="KEEP_BASELINE",
            decision_text="Keep baseline",
            document_id="doc_1",
            row_id="row_1",
            field_name="compound_id",
        )
    )
    (art_dir / "01_positive" / "rows").mkdir(parents=True, exist_ok=True)
    (art_dir / "01_positive" / "rows" / "doc_1.json").write_text(
        PositiveRowsOutput(rows=[]).model_dump_json(), encoding="utf-8"
    )
    (art_dir / "01_positive" / "columns").mkdir(parents=True, exist_ok=True)
    (art_dir / "01_positive" / "columns" / "doc_1_row_1.json").write_text(
        '{"fields": []}', encoding="utf-8"
    )
    (art_dir / "03_negative" / "rows").mkdir(parents=True, exist_ok=True)
    (art_dir / "03_negative" / "rows" / "doc_1.json").write_text(
        NegativeRowsOutput(gap_rows=[]).model_dump_json(), encoding="utf-8"
    )
    (art_dir / "03_negative" / "columns").mkdir(parents=True, exist_ok=True)
    (art_dir / "03_negative" / "columns" / "doc_1_row_1.json").write_text(
        '{"fields": []}', encoding="utf-8"
    )
    iter1_gen = art_dir / "iterations" / "iter_1" / "04_generalization"
    (iter1_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter1_gen / "rows.json").write_text(
        GeneralizedRowsOutput(
            Row_Instantiation_Trigger_generalized="Iter 1 trigger",
            Instructions=RowInstructions(Row_Inclusion_Instructions=["Include compound"]),
        ).model_dump_json(),
        encoding="utf-8",
    )
    (iter1_gen / "columns" / "compound_id.json").write_text(
        GeneralizedColumnsOutput(
            Semantic_Core_generalized="Compound ID",
            Instructions=ColumnInstructions(Column_Inclusion_Instructions=["Extract ID"]),
            Conflicts=[
                REConflict(
                    conflict_id="stale_reemitted_conflict_id",
                    scope="column:compound_id",
                    conflict_type="BASELINE_POLICY_CONFLICT",
                    document_id="doc_1",
                    row_id="row_1",
                    explanation="Already resolved under old_baseline_conflict_id",
                    options=[],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    (iter1_gen / "columns" / "target_relation.json").write_text(
        GeneralizedColumnsOutput(
            Semantic_Core_generalized="Relation",
            Instructions=ColumnInstructions(Column_Inclusion_Instructions=["Extract ="]),
        ).model_dump_json(),
        encoding="utf-8",
    )
    (iter1_gen / "candidate_prompt.txt").write_text("Cached Iter 1 Prompt\n", encoding="utf-8")
    gen_prompt_path.write_text("Cached Iter 1 Prompt\n", encoding="utf-8")

    # Populate .sha256 sidecars matching iter_1 artifacts as produced by run_generalization
    from ae.semantic_optimization.steps.generalization import (
        _compute_col_cache_key,
        _compute_row_cache_key,
        _scope_rulings_hash,
        _write_rulings_hash,
    )
    import hashlib, json
    (iter1_gen / "rows.json.sha256").write_text(
        _compute_row_cache_key(ctx, [], []), encoding="utf-8"
    )
    _write_rulings_hash(iter1_gen / "rows.json", _scope_rulings_hash(ctx, "rows"))
    (iter1_gen / "columns" / "compound_id.json.sha256").write_text(
        _compute_col_cache_key(ctx, "compound_id", [], []), encoding="utf-8"
    )
    _write_rulings_hash(
        iter1_gen / "columns" / "compound_id.json", _scope_rulings_hash(ctx, "column:compound_id")
    )
    (iter1_gen / "columns" / "target_relation.json.sha256").write_text(
        _compute_col_cache_key(ctx, "target_relation", [], []), encoding="utf-8"
    )
    _write_rulings_hash(
        iter1_gen / "columns" / "target_relation.json", _scope_rulings_hash(ctx, "column:target_relation")
    )
    # Match candidate_prompt.txt.sha256
    prompt_cache_payload = {
        "draft_prompt": (
            "Base prompt\n\n---\n\n"
            + (ctx.rulings_store.format_rulings_for_prompt("all", only_policy_rules=True) or "")
        ),
        "general_rulings": ctx.rulings_store.format_rulings_for_prompt("all", only_policy_rules=True) or "",
        "schema": ctx.schema,
        "baseline_prompt": ctx.baseline_prompt,
        "harmonize_prompt": True,
    }
    # Use run_generalization's exact prompt_cache_key by setting harmonize_prompt=False or matching hash;
    # here we disable harmonize_prompt in this resume test so candidate_prompt.txt is not re-harmonized,
    # or write the exact hash. Let's check: ctx.harmonize_prompt is True by default, so we can set it False
    # or let's just set ctx.harmonize_prompt = False? No, test checks PromptHarmonization is not called!
    # We'll compute the exact draft_prompt or set ctx.harmonize_prompt = False.
    ctx.harmonize_prompt = False

    # Extractor returns exact match for doc_2 (Zero-Delta in Iter 2!) and 0.5 F1 on val_1 so Iter 2 executes
    def _fwd_val(document_text):
        if "doc_2" in document_text:
            return dspy.Prediction(extracted_data={"records": [{"compound_id": "6a", "target_relation": "="}]})
        return dspy.Prediction(extracted_data={"records": [{"compound_id": "7a", "target_relation": ">"}]})

    mock_re_client.create_validation_extractor.side_effect = (
        lambda sig: MagicMock(forward=MagicMock(side_effect=_fwd_val))
    )

    called_predictors: list[str] = []

    def fake_predict(sig):
        name = getattr(sig, "__name__", str(sig))
        def _call(**kwargs):
            called_predictors.append(name)
            if name == "SchemaConstraintInference":
                return MagicMock(
                    refinement=SchemaRefinementOutput(
                        fields=[
                            FieldConstraintProposal(
                                field_name="compound_id",
                                description="ID",
                                is_closed_enum=False,
                                choices=None,  # Verify choices: null works cleanly here too!
                            ),
                            FieldConstraintProposal(
                                field_name="target_relation",
                                description="Op",
                                is_closed_enum=True,
                                choices=["=", ">"],
                            ),
                        ]
                    )
                )
            raise AssertionError(f"Unexpected LLM call during resume+Zero-Delta: {name}")
        return _call

    # Also ensure interactive resolver fails if it ever tries to prompt user
    ctx.resolver.input_fn = lambda _: (_ for _ in ()).throw(AssertionError("Must not prompt HITL on resume"))

    with patch("ae.semantic_optimization.steps.generalization.dspy.Predict", side_effect=fake_predict), \
         patch("ae.semantic_optimization.steps.schema_refinement.dspy.Predict", side_effect=fake_predict), \
         patch("ae.semantic_optimization.steps.residual_analysis.dspy.Predict", side_effect=fake_predict):

        graph = SemanticOptimizationGraph(
            context=ctx,
            train_doc_ids=["doc_1", "doc_2"],
            val_doc_ids=["val_1"],
            f1_threshold=0.95,
            initial_docs=1,
            step_size=1,
        )
        final_state = graph.run()

    # Only SchemaConstraintInference ran in Iter 1; Iter 2 ran Residual Zero-Delta (0 LLM calls!)
    assert called_predictors == ["SchemaConstraintInference"]
    assert final_state["current_iteration"] == 2
    assert (art_dir / "01_positive" / "residual" / "doc_2.json").exists()
    assert gen_schema_path.exists()


def test_route_after_generalization_in_residual_mode_routes_to_residual(mock_context, tmp_path):
    import dspy
    from ae.semantic_optimization.context import ArtifactManager
    from ae.semantic_optimization.graph import SemanticOptimizationGraph
    from ae.semantic_optimization.steps.consolidation import _invalidate_doc_artifacts

    prompt_path = tmp_path / "generated_instruction.txt"
    prompt_path.write_text("Extract records.", encoding="utf-8")
    mock_context.generated_prompt_path = prompt_path
    mock_context.residual_enabled = True
    mock_context.artifacts = ArtifactManager(tmp_path / "artifacts", resume=True)

    graph = SemanticOptimizationGraph(
        context=mock_context,
        train_doc_ids=["doc_1"],
        val_doc_ids=[],
        f1_threshold=0.95,
    )
    state_residual = {
        "context": mock_context,
        "iteration_mode": "residual",
        "_gt_modified_in_generalization": True,
    }
    assert graph._route_after_generalization(state_residual) == "recompute_residual"

    res_file = mock_context.artifacts.get_path("positive", "residual", "doc_1.json", iteration=None)
    res_file.parent.mkdir(parents=True, exist_ok=True)
    res_file.write_text("{}", encoding="utf-8")
    _invalidate_doc_artifacts(mock_context, "doc_1")
    assert not res_file.exists()


def test_evaluate_val_cache_survives_inplace_md_edit_when_manifest_exists(tmp_path):
    """val_metrics.json cache on --resume uses canonical manifest text and ignores in-place {doc}.md edits."""
    import dspy
    from ae.semantic_optimization.context import ArtifactManager
    from ae.semantic_optimization.graph import SemanticOptimizationGraph

    ingest_dir = tmp_path / "ingestion"
    doc_dir = ingest_dir / "val_1"
    manifest_dir = doc_dir / "02_manifest_visuals"
    manifest_dir.mkdir(parents=True)
    (doc_dir / "val_1.md").write_text("Initial val text", encoding="utf-8")
    (manifest_dir / "manifest_enriched_text.md").write_text("Canonical val manifest", encoding="utf-8")

    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,compound_id\nval_1,c1\n", encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfloat_tolerance: 0.01\nfields:\n  compound_id:\n    type: str\ncompare_fields:\n  - compound_id\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("Base", encoding="utf-8")
    gen_prompt_path = tmp_path / "generated_prompt.txt"
    gen_prompt_path.write_text("Prompt v1", encoding="utf-8")

    fake_ext = MagicMock()
    fake_ext.forward.return_value = dspy.Prediction(extracted_data={"records": [{"compound_id": "c1"}]})
    fake_re_client = MagicMock()
    fake_re_client.extraction_config = MagicMock(model="test-model")
    fake_re_client.task_config = {"float_tolerance": 0.01}
    fake_re_client.create_validation_extractor.return_value = fake_ext

    ctx = RunContext(
        task_name="test",
        doc_ids=["val_1"],
        gt_path=gt_path,
        ingestion_dir=ingest_dir,
        baseline_prompt_path=baseline_path,
        schema_path=schema_path,
        re_client=fake_re_client,
        artifacts_dir=tmp_path / "artifacts",
        generated_prompt_path=gen_prompt_path,
        enable_semantic_judge=False,
        resume=False,
    )

    graph = SemanticOptimizationGraph(context=ctx, train_doc_ids=[], val_doc_ids=["val_1"], f1_threshold=0.9)
    state = {"context": ctx, "val_doc_ids": ["val_1"], "current_iteration": 1, "val_f1_history": [], "best_f1": 0.0, "best_iteration": 0, "best_prompt": ""}
    res1 = graph._evaluate_val(state)
    assert fake_ext.forward.call_count == 1
    assert res1["val_f1_history"] == [1.0]

    # Modify val_1.md in-place (simulating visual unpacking) while manifest_enriched_text.md is unchanged
    (doc_dir / "val_1.md").write_text("Unpacked visual text", encoding="utf-8")
    ctx.resume = True
    ctx.artifacts.resume = True
    res2 = graph._evaluate_val(state)
    assert fake_ext.forward.call_count == 1  # Loaded from val_metrics.json cache!
    assert res2["val_f1_history"] == [1.0]


def test_per_column_pareto_auto_rollback_restores_regressed_column_only(tmp_path):
    """Per-column Pareto Auto-Rollback restores regressed 'smiles' from iter_1 while keeping 'bacteria' gain in iter_2."""
    from ae.semantic_optimization.context import ArtifactManager

    ingest_dir = tmp_path / "ingestion"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Val 1 text", encoding="utf-8")

    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,smiles,bacteria\nval_1,CCO,S. aureus\n", encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfloat_tolerance: 0.01\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n"
        "compare_fields:\n  - smiles\n  - bacteria\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("Base", encoding="utf-8")
    gen_prompt_path = tmp_path / "generated_instruction.txt"

    iter1_prompt = (
        "Preamble\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Good SMILES rule.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Weak bacteria rule.\n"
    )
    iter2_prompt_broken = (
        "Preamble\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Broken SMILES rule.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Strong bacteria rule.\n"
    )

    art_dir = tmp_path / "artifacts"
    iter1_cols = art_dir / "iterations" / "iter_1" / "04_generalization" / "columns"
    iter2_cols = art_dir / "iterations" / "iter_2" / "04_generalization" / "columns"
    iter1_cols.mkdir(parents=True, exist_ok=True)
    iter2_cols.mkdir(parents=True, exist_ok=True)
    (iter1_cols / "smiles.json").write_text('{"Semantic_Core_generalized": "iter1_smiles"}', encoding="utf-8")
    (iter2_cols / "smiles.json").write_text('{"Semantic_Core_generalized": "iter2_broken"}', encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = art_dir
    ctx.artifacts = ArtifactManager(art_dir, resume=False)
    ctx.schema_path = schema_path
    ctx.generated_schema_path = None
    ctx.generated_prompt_path = gen_prompt_path
    ctx.active_gt_path = gt_path
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.resume = False
    ctx.enable_semantic_judge = False
    ctx.selection_metric = "micro_f1"
    ctx.pareto_regression_threshold = 0.01
    ctx.re_client = MagicMock()
    ctx.re_client.extraction_config = MagicMock(model="test-model")

    graph = SemanticOptimizationGraph(
        context=ctx, train_doc_ids=["d1", "d2"], val_doc_ids=["val_1"], pareto_regression_threshold=0.01
    )

    # Iter 1: smiles=1.0, bacteria=0.5 -> F1=0.75
    gen_prompt_path.write_text(iter1_prompt, encoding="utf-8")
    with patch.object(
        graph,
        "_evaluate_prompt_on_val",
        return_value={
            "f1_score": 0.75,
            "macro_f1": 0.75,
            "micro_f1": 0.75,
            "strict_micro_f1": 0.75,
            "strict_macro_f1": 0.75,
            "per_field_f1": {"smiles": 1.0, "bacteria": 0.5},
            "strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.5},
            "doc_scores": {"val_1": {"f1": 0.75, "gts": 1}},
            "evaluated_count": 1,
        },
    ):
        s1 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 1,
                "val_f1_history": [],
                "val_per_field_history": [],
                "best_f1": 0.0,
                "best_per_field_f1": {},
                "best_iteration": 1,
                "best_prompt": "",
                "pareto_rollback_history": [],
            }
        )

    assert s1["best_f1"] == 0.75
    assert s1["best_per_field_f1"] == {"smiles": 1.0, "bacteria": 0.5}

    # Iter 2: bacteria improves to 1.0, smiles regresses to 0.8 (-0.20 < -0.01) -> triggers rollback of SMILES only!
    gen_prompt_path.write_text(iter2_prompt_broken, encoding="utf-8")
    (art_dir / "iterations" / "iter_2" / "04_generalization" / "candidate_prompt.txt").write_text(
        iter2_prompt_broken, encoding="utf-8"
    )
    eval_side_effects = [
        {
            "f1_score": 0.90,
            "macro_f1": 0.90,
            "micro_f1": 0.90,
            "strict_micro_f1": 0.90,
            "strict_macro_f1": 0.90,
            "per_field_f1": {"smiles": 0.80, "bacteria": 1.0},
            "strict_per_field_f1": {"smiles": 0.80, "bacteria": 1.0},
            "doc_scores": {"val_1": {"f1": 0.90, "gts": 1}},
            "evaluated_count": 1,
        },
        {
            "f1_score": 1.0,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "strict_micro_f1": 1.0,
            "strict_macro_f1": 1.0,
            "per_field_f1": {"smiles": 1.0, "bacteria": 1.0},
            "strict_per_field_f1": {"smiles": 1.0, "bacteria": 1.0},
            "doc_scores": {"val_1": {"f1": 1.0, "gts": 1}},
            "evaluated_count": 1,
        },
    ]
    with patch.object(graph, "_evaluate_prompt_on_val", side_effect=eval_side_effects):
        s2 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 2,
                "val_f1_history": s1["val_f1_history"],
                "val_per_field_history": s1["val_per_field_history"],
                "best_f1": s1["best_f1"],
                "best_per_field_f1": s1["best_per_field_f1"],
                "best_iteration": s1["best_iteration"],
                "best_prompt": s1["best_prompt"],
                "pareto_rollback_history": s1["pareto_rollback_history"],
            }
        )

    assert s2["best_f1"] == 1.0
    assert s2["best_iteration"] == 2
    assert "Good SMILES rule." in s2["best_prompt"]
    assert "Strong bacteria rule." in s2["best_prompt"]
    assert "Broken SMILES rule." not in s2["best_prompt"]
    assert len(s2["pareto_rollback_history"]) == 1
    assert s2["pareto_rollback_history"][0]["regressed_fields"] == ["smiles"]
    assert s2["pareto_rollback_history"][0]["accepted"] is True
    assert "iter1_smiles" in (iter2_cols / "smiles.json").read_text(encoding="utf-8")


def test_evaluate_prompt_on_val_per_field_f1_uses_raw_counts_and_bounded_fallback(tmp_path):
    """per_field_f1 uses raw field_correct/field_total when present and stays <= 1.0 even when n_p != n_g."""
    import dspy

    ingest_dir = tmp_path / "ingestion"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Val 1", encoding="utf-8")
    (ingest_dir / "val_2.md").write_text("Val 2", encoding="utf-8")

    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,smiles,bacteria\nval_1,CCO,S. aureus\nval_2,CCN,E. coli\n", encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfloat_tolerance: 0.01\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n"
        "compare_fields:\n  - smiles\n  - bacteria\n",
        encoding="utf-8",
    )
    gen_prompt = tmp_path / "generated_instruction.txt"
    gen_prompt.write_text("Prompt", encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.schema_path = schema_path
    ctx.generated_prompt_path = gen_prompt
    ctx.active_gt_path = gt_path
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.re_client = MagicMock()
    mock_ext = MagicMock()
    mock_ext.forward.return_value = dspy.Prediction(extracted_data={"records": []})
    ctx.re_client.create_validation_extractor.return_value = mock_ext

    # Case 1: Raw field_correct / field_total provided with n_p != n_g (e.g. 10 preds vs 5 GTs)
    mock_matcher = MagicMock()
    mock_matcher.fields = ["smiles", "bacteria"]
    mock_matcher.get_detailed_report.side_effect = [
        {
            "f1": 0.8,
            "strict_f1": 0.7,
            "counts": {"preds": 10, "gts": 5},
            "fields": {"smiles": 1.0, "bacteria": 0.5},
            "strict_fields": {"smiles": 0.8, "bacteria": 0.5},
            "field_correct": {"smiles": 10, "bacteria": 5},
            "field_total": {"smiles": 10, "bacteria": 10},
            "strict_field_correct": {"smiles": 8, "bacteria": 5},
            "strict_field_total": {"smiles": 10, "bacteria": 10},
        },
        {
            "f1": 0.9,
            "strict_f1": 0.6,
            "counts": {"preds": 2, "gts": 2},
            "fields": {"smiles": 1.0, "bacteria": 1.0},
            "strict_fields": {"smiles": 0.5, "bacteria": 1.0},
            "field_correct": {"smiles": 2, "bacteria": 2},
            "field_total": {"smiles": 2, "bacteria": 2},
            "strict_field_correct": {"smiles": 1, "bacteria": 2},
            "strict_field_total": {"smiles": 2, "bacteria": 2},
        },
    ]
    mock_metric = MagicMock()
    mock_metric.matcher = mock_matcher
    mock_metric._extract_records.return_value = []

    graph = SemanticOptimizationGraph(context=ctx, train_doc_ids=[], val_doc_ids=["val_1", "val_2"])
    with patch("ae.core.evaluation.TaskMetric", return_value=mock_metric):
        out = graph._evaluate_prompt_on_val(
            ctx=ctx,
            val_ids=["val_1", "val_2"],
            eval_schema_path=schema_path,
            generated_path=gen_prompt,
            selection_metric="micro_f1",
            enable_judge=True,
        )

    assert out["per_field_f1"]["smiles"] == 1.0
    assert out["per_field_f1"]["bacteria"] == pytest.approx(7.0 / 12.0, abs=1e-6)
    assert out["strict_per_field_f1"]["smiles"] == pytest.approx(9.0 / 12.0, abs=1e-6)
    assert out["strict_per_field_f1"]["bacteria"] == pytest.approx(7.0 / 12.0, abs=1e-6)

    # Case 2: Fallback without raw counts when n_p=10, n_g=5 must NEVER exceed 1.0
    mock_matcher_legacy = MagicMock()
    mock_matcher_legacy.fields = ["smiles", "bacteria"]
    mock_matcher_legacy.get_detailed_report.return_value = {
        "f1": 0.67,
        "strict_f1": 0.67,
        "counts": {"preds": 10, "gts": 5},
        "fields": {"smiles": 1.0, "bacteria": 0.8},
        "strict_fields": {"smiles": 1.0, "bacteria": 0.6},
    }
    mock_metric_legacy = MagicMock()
    mock_metric_legacy.matcher = mock_matcher_legacy
    mock_metric_legacy._extract_records.return_value = []

    with patch("ae.core.evaluation.TaskMetric", return_value=mock_metric_legacy):
        out_legacy = graph._evaluate_prompt_on_val(
            ctx=ctx,
            val_ids=["val_1"],
            eval_schema_path=schema_path,
            generated_path=gen_prompt,
            selection_metric="micro_f1",
            enable_judge=False,
        )

    assert out_legacy["per_field_f1"]["smiles"] == 1.0
    assert out_legacy["per_field_f1"]["bacteria"] == 0.8
    assert out_legacy["strict_per_field_f1"]["bacteria"] == 0.6


def test_dual_gate_pareto_rollback_and_sync_on_rejected_rollback(tmp_path):
    """Strict per-field regression triggers dual-gate rollback; rejected rollback syncs iter_k artifacts from best_k."""
    from ae.semantic_optimization.context import ArtifactManager

    ingest_dir = tmp_path / "ingestion"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Val 1", encoding="utf-8")

    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,smiles,bacteria\nval_1,CCO,S. aureus\n", encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfloat_tolerance: 0.01\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n"
        "compare_fields:\n  - smiles\n  - bacteria\n",
        encoding="utf-8",
    )
    gen_prompt_path = tmp_path / "generated_instruction.txt"
    gen_schema_path = tmp_path / "generated_schema.yaml"
    gen_schema_path.write_text(schema_path.read_text(encoding="utf-8"), encoding="utf-8")

    iter1_prompt = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Best row rule.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Best SMILES rule.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Best bacteria rule.\n"
    )
    iter2_prompt = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Bad row rule.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Bad SMILES rule.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- New bacteria rule.\n"
    )

    art_dir = tmp_path / "artifacts"
    iter1_gen = art_dir / "iterations" / "iter_1" / "04_generalization"
    iter2_gen = art_dir / "iterations" / "iter_2" / "04_generalization"
    (iter1_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter2_gen / "columns").mkdir(parents=True, exist_ok=True)

    (iter1_gen / "candidate_prompt.txt").write_text(iter1_prompt, encoding="utf-8")
    (iter1_gen / "rows.json").write_text('{"rows": "best_rows"}', encoding="utf-8")
    (iter1_gen / "columns" / "smiles.json").write_text('{"col": "best_smiles"}', encoding="utf-8")
    (iter1_gen / "columns" / "bacteria.json").write_text('{"col": "best_bacteria"}', encoding="utf-8")
    (iter1_gen / "candidate_schema.yaml").write_text("name: best_schema\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n", encoding="utf-8")

    (iter2_gen / "candidate_prompt.txt").write_text(iter2_prompt, encoding="utf-8")
    (iter2_gen / "rows.json").write_text('{"rows": "bad_rows"}', encoding="utf-8")
    (iter2_gen / "columns" / "smiles.json").write_text('{"col": "bad_smiles"}', encoding="utf-8")
    (iter2_gen / "columns" / "bacteria.json").write_text('{"col": "bad_bacteria"}', encoding="utf-8")
    (iter2_gen / "candidate_schema.yaml").write_text("name: bad_schema\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n", encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = art_dir
    ctx.artifacts = ArtifactManager(art_dir, resume=False)
    ctx.schema_path = schema_path
    ctx.generated_schema_path = gen_schema_path
    ctx.generated_prompt_path = gen_prompt_path
    ctx.active_gt_path = gt_path
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.resume = False
    ctx.enable_semantic_judge = True
    ctx.selection_metric = "micro_f1"
    ctx.pareto_regression_threshold = 0.01
    ctx.re_client = MagicMock()
    ctx.re_client.extraction_config = MagicMock(model="test-model")

    graph = SemanticOptimizationGraph(
        context=ctx, train_doc_ids=["d1", "d2"], val_doc_ids=["val_1"], pareto_regression_threshold=0.01
    )

    # Iter 1 sets best_strict_per_field_f1 = {"smiles": 1.0, "bacteria": 0.9}
    gen_prompt_path.write_text(iter1_prompt, encoding="utf-8")
    with patch.object(
        graph,
        "_evaluate_prompt_on_val",
        return_value={
            "f1_score": 0.90,
            "macro_f1": 0.90,
            "micro_f1": 0.90,
            "strict_f1_score": 0.90,
            "strict_micro_f1": 0.90,
            "strict_macro_f1": 0.90,
            "per_field_f1": {"smiles": 1.0, "bacteria": 0.9},
            "strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.9},
            "doc_scores": {"val_1": {"f1": 0.90, "gts": 1}},
            "evaluated_count": 1,
        },
    ):
        s1 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 1,
                "val_f1_history": [],
                "val_per_field_history": [],
                "best_f1": 0.0,
                "best_per_field_f1": {},
                "best_strict_f1": 0.0,
                "best_strict_per_field_f1": {},
                "best_iteration": 1,
                "best_prompt": "",
                "pareto_rollback_history": [],
            }
        )

    # Iter 2: semantic per_field_f1 stays 1.0/0.95 (masked by judge!), but strict_per_field_f1 for smiles drops 1.0 -> 0.50!
    # Post-rollback repaired_f1=0.85 (> pre-rollback 0.70, but < best_f1=0.90) -> must be REJECTED and iter_2 synced from iter_1!
    gen_prompt_path.write_text(iter2_prompt, encoding="utf-8")
    eval_side_effects = [
        {
            "f1_score": 0.70,
            "macro_f1": 0.70,
            "micro_f1": 0.70,
            "strict_f1_score": 0.60,
            "strict_micro_f1": 0.60,
            "strict_macro_f1": 0.60,
            "per_field_f1": {"smiles": 1.0, "bacteria": 0.95},
            "strict_per_field_f1": {"smiles": 0.50, "bacteria": 0.95},
            "doc_scores": {"val_1": {"f1": 0.70, "gts": 1}},
            "evaluated_count": 1,
        },
        {
            "f1_score": 0.85,  # > 0.70 pre-rollback, but < 0.90 best_f1 -> must NOT be accepted!
            "macro_f1": 0.85,
            "micro_f1": 0.85,
            "strict_f1_score": 0.85,
            "strict_micro_f1": 0.85,
            "strict_macro_f1": 0.85,
            "per_field_f1": {"smiles": 1.0, "bacteria": 0.95},
            "strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.95},
            "doc_scores": {"val_1": {"f1": 0.85, "gts": 1}},
            "evaluated_count": 1,
        },
    ]
    with patch.object(graph, "_evaluate_prompt_on_val", side_effect=eval_side_effects):
        s2 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 2,
                "val_f1_history": s1["val_f1_history"],
                "val_per_field_history": s1["val_per_field_history"],
                "best_f1": s1["best_f1"],
                "best_per_field_f1": s1["best_per_field_f1"],
                "best_strict_f1": s1["best_strict_f1"],
                "best_strict_per_field_f1": s1["best_strict_per_field_f1"],
                "best_iteration": s1["best_iteration"],
                "best_prompt": s1["best_prompt"],
                "pareto_rollback_history": s1["pareto_rollback_history"],
            }
        )

    assert len(s2["pareto_rollback_history"]) == 1
    ev = s2["pareto_rollback_history"][0]
    assert ev["regressed_fields"] == ["smiles"]
    assert ev["accepted"] is False
    # Active prompt and all iter_2 generalization artifacts must be synced from iter_1!
    assert gen_prompt_path.read_text(encoding="utf-8") == iter1_prompt
    assert (iter2_gen / "candidate_prompt.txt").read_text(encoding="utf-8") == iter1_prompt
    assert (iter2_gen / "rows.json").read_text(encoding="utf-8") == '{"rows": "best_rows"}'
    assert (iter2_gen / "columns" / "smiles.json").read_text(encoding="utf-8") == '{"col": "best_smiles"}'
    assert (iter2_gen / "columns" / "bacteria.json").read_text(encoding="utf-8") == '{"col": "best_bacteria"}'
    assert "best_schema" in (iter2_gen / "candidate_schema.yaml").read_text(encoding="utf-8")
    assert "best_schema" in gen_schema_path.read_text(encoding="utf-8")


def test_pareto_rollback_rolls_back_rows_json_when_all_columns_regress(tmp_path):
    """When all columns regress, rollback_targets includes 'rows' and rows.json is restored from best_k."""
    from ae.semantic_optimization.context import ArtifactManager

    ingest_dir = tmp_path / "ingestion"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Val 1", encoding="utf-8")
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,smiles,bacteria\nval_1,CCO,S. aureus\n", encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfloat_tolerance: 0.01\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n"
        "compare_fields:\n  - smiles\n  - bacteria\n",
        encoding="utf-8",
    )
    gen_prompt_path = tmp_path / "generated_instruction.txt"

    iter1_prompt = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Best row rule.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Good SMILES.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Good bacteria.\n"
    )
    iter2_prompt = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Broken row rule.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Broken SMILES.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Broken bacteria.\n"
    )

    art_dir = tmp_path / "artifacts"
    iter1_gen = art_dir / "iterations" / "iter_1" / "04_generalization"
    iter2_gen = art_dir / "iterations" / "iter_2" / "04_generalization"
    (iter1_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter2_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter1_gen / "rows.json").write_text('{"rows": "iter1_rows"}', encoding="utf-8")
    (iter2_gen / "rows.json").write_text('{"rows": "iter2_broken_rows"}', encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = art_dir
    ctx.artifacts = ArtifactManager(art_dir, resume=False)
    ctx.schema_path = schema_path
    ctx.generated_schema_path = None
    ctx.generated_prompt_path = gen_prompt_path
    ctx.active_gt_path = gt_path
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.resume = False
    ctx.enable_semantic_judge = False
    ctx.selection_metric = "micro_f1"
    ctx.pareto_regression_threshold = 0.01
    ctx.re_client = MagicMock()
    ctx.re_client.extraction_config = MagicMock(model="test-model")

    graph = SemanticOptimizationGraph(
        context=ctx, train_doc_ids=["d1", "d2"], val_doc_ids=["val_1"], pareto_regression_threshold=0.01
    )
    gen_prompt_path.write_text(iter2_prompt, encoding="utf-8")

    eval_side_effects = [
        {
            "f1_score": 0.50,
            "macro_f1": 0.50,
            "micro_f1": 0.50,
            "strict_f1_score": 0.50,
            "strict_micro_f1": 0.50,
            "strict_macro_f1": 0.50,
            "per_field_f1": {"smiles": 0.5, "bacteria": 0.5},
            "strict_per_field_f1": {"smiles": 0.5, "bacteria": 0.5},
            "doc_scores": {"val_1": {"f1": 0.50, "gts": 1}},
            "evaluated_count": 1,
        },
        {
            "f1_score": 0.90,
            "macro_f1": 0.90,
            "micro_f1": 0.90,
            "strict_f1_score": 0.90,
            "strict_micro_f1": 0.90,
            "strict_macro_f1": 0.90,
            "per_field_f1": {"smiles": 0.9, "bacteria": 0.9},
            "strict_per_field_f1": {"smiles": 0.9, "bacteria": 0.9},
            "doc_scores": {"val_1": {"f1": 0.90, "gts": 1}},
            "evaluated_count": 1,
        },
    ]
    with patch.object(graph, "_evaluate_prompt_on_val", side_effect=eval_side_effects):
        s2 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 2,
                "val_f1_history": [0.90],
                "val_per_field_history": [{"smiles": 0.9, "bacteria": 0.9}],
                "best_f1": 0.90,
                "best_per_field_f1": {"smiles": 0.9, "bacteria": 0.9},
                "best_strict_f1": 0.90,
                "best_strict_per_field_f1": {"smiles": 0.9, "bacteria": 0.9},
                "best_iteration": 1,
                "best_prompt": iter1_prompt,
                "pareto_rollback_history": [],
            }
        )

    assert s2["pareto_rollback_history"][0]["accepted"] is True
    assert "rows" in s2["pareto_rollback_history"][0]["rollback_targets"]
    assert (iter2_gen / "rows.json").read_text(encoding="utf-8") == '{"rows": "iter1_rows"}'
    assert (iter2_gen / "rows.pre_rollback.json").exists()


def test_pareto_strict_f1_gate_tie_breaker_and_cache_recompute(tmp_path):
    """strict_f1_score gates post-rollback acceptance and acts as tie-breaker; full revert records best_f1 & recomputes cache key."""
    from ae.semantic_optimization.context import ArtifactManager

    ingest_dir = tmp_path / "ingestion"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Val 1", encoding="utf-8")
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,smiles,bacteria\nval_1,CCO,S. aureus\n", encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfloat_tolerance: 0.01\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n"
        "compare_fields:\n  - smiles\n  - bacteria\n",
        encoding="utf-8",
    )
    gen_prompt_path = tmp_path / "generated_instruction.txt"
    iter1_prompt = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Best row.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Best SMILES.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Best bacteria.\n"
    )
    iter2_prompt = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Best row.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Broken SMILES.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- New bacteria.\n"
    )

    art_dir = tmp_path / "artifacts"
    iter1_gen = art_dir / "iterations" / "iter_1" / "04_generalization"
    iter2_gen = art_dir / "iterations" / "iter_2" / "04_generalization"
    (iter1_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter2_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter1_gen / "candidate_prompt.txt").write_text(iter1_prompt, encoding="utf-8")
    (iter2_gen / "candidate_prompt.txt").write_text(iter2_prompt, encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = art_dir
    ctx.artifacts = ArtifactManager(art_dir, resume=False)
    ctx.schema_path = schema_path
    ctx.generated_schema_path = None
    ctx.generated_prompt_path = gen_prompt_path
    ctx.active_gt_path = gt_path
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.resume = False
    ctx.enable_semantic_judge = True
    ctx.selection_metric = "micro_f1"
    ctx.pareto_regression_threshold = 0.01
    ctx.re_client = MagicMock()
    ctx.re_client.extraction_config = MagicMock(model="test-model")

    graph = SemanticOptimizationGraph(
        context=ctx, train_doc_ids=["d1"], val_doc_ids=["val_1"], pareto_regression_threshold=0.01
    )

    # 1. Post-rollback has repaired_f1 == best_f1 (0.90) and no per-field regression,
    #    but repaired_strict_f1 dropped 0.88 -> 0.80 (> threshold 0.01) -> MUST BE REJECTED!
    gen_prompt_path.write_text(iter2_prompt, encoding="utf-8")
    eval_effects = [
        {
            "f1_score": 0.80,
            "macro_f1": 0.80,
            "micro_f1": 0.80,
            "strict_f1_score": 0.75,
            "strict_micro_f1": 0.75,
            "strict_macro_f1": 0.75,
            "per_field_f1": {"smiles": 0.70, "bacteria": 0.95},
            "strict_per_field_f1": {"smiles": 0.65, "bacteria": 0.90},
            "doc_scores": {"val_1": {"f1": 0.80, "gts": 1}},
            "evaluated_count": 1,
        },
        {
            "f1_score": 0.90,
            "macro_f1": 0.90,
            "micro_f1": 0.90,
            "strict_f1_score": 0.80,  # Dropped by 0.08 from best_strict_f1=0.88 -> fail strict gate!
            "strict_micro_f1": 0.80,
            "strict_macro_f1": 0.80,
            "per_field_f1": {"smiles": 0.90, "bacteria": 0.95},
            "strict_per_field_f1": {"smiles": 0.895, "bacteria": 0.90},
            "doc_scores": {"val_1": {"f1": 0.90, "gts": 1}},
            "evaluated_count": 1,
        },
    ]
    with patch.object(graph, "_evaluate_prompt_on_val", side_effect=eval_effects):
        s2 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 2,
                "val_f1_history": [0.90],
                "val_per_field_history": [{"smiles": 0.90, "bacteria": 0.90}],
                "best_f1": 0.90,
                "best_per_field_f1": {"smiles": 0.90, "bacteria": 0.90},
                "best_strict_f1": 0.88,
                "best_strict_per_field_f1": {"smiles": 0.90, "bacteria": 0.90},
                "best_iteration": 1,
                "best_prompt": iter1_prompt,
                "pareto_rollback_history": [],
            }
        )

    assert s2["pareto_rollback_history"][0]["accepted"] is False
    # Full revert records best_f1 (0.90) in val_f1_history instead of pre-rollback 0.80
    assert s2["val_f1_history"] == [0.90, 0.90]
    assert gen_prompt_path.read_text(encoding="utf-8") == iter1_prompt

    # Resuming on iter_2 must hit val_metrics.json cache (val_cache_key recomputed for iter1_prompt!)
    ctx.resume = True
    ctx.artifacts.resume = True
    with patch.object(graph, "_evaluate_prompt_on_val", side_effect=AssertionError("Should hit cache")):
        s2_cached = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 2,
                "val_f1_history": [0.90],
                "val_per_field_history": [{"smiles": 0.90, "bacteria": 0.90}],
                "best_f1": 0.90,
                "best_per_field_f1": {"smiles": 0.90, "bacteria": 0.90},
                "best_strict_f1": 0.88,
                "best_strict_per_field_f1": {"smiles": 0.90, "bacteria": 0.90},
                "best_iteration": 1,
                "best_prompt": iter1_prompt,
                "pareto_rollback_history": [],
            }
        )
    assert s2_cached["val_f1_history"] == [0.90, 0.90]

    # 2. Tie-breaker on best_* update: f1_score == best_f1 (0.90), strict_f1_score < best_strict_f1 (0.85 < 0.88) -> keep best_iteration=1
    ctx.resume = False
    ctx.artifacts.resume = False
    iter3_prompt = iter1_prompt.replace("Best bacteria.", "Tied bacteria.")
    gen_prompt_path.write_text(iter3_prompt, encoding="utf-8")
    with patch.object(
        graph,
        "_evaluate_prompt_on_val",
        return_value={
            "f1_score": 0.90,
            "macro_f1": 0.90,
            "micro_f1": 0.90,
            "strict_f1_score": 0.85,
            "strict_micro_f1": 0.85,
            "strict_macro_f1": 0.85,
            "per_field_f1": {"smiles": 0.90, "bacteria": 0.90},
            "strict_per_field_f1": {"smiles": 0.895, "bacteria": 0.895},
            "doc_scores": {"val_1": {"f1": 0.90, "gts": 1}},
            "evaluated_count": 1,
        },
    ):
        s3 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 3,
                "val_f1_history": [0.90, 0.90],
                "val_per_field_history": [{"smiles": 0.90, "bacteria": 0.90}],
                "best_f1": 0.90,
                "best_per_field_f1": {"smiles": 0.90, "bacteria": 0.90},
                "best_strict_f1": 0.88,
                "best_strict_per_field_f1": {"smiles": 0.90, "bacteria": 0.90},
                "best_iteration": 1,
                "best_prompt": iter1_prompt,
                "pareto_rollback_history": [],
            }
        )
    assert s3["best_iteration"] == 1
    assert s3["best_prompt"] == iter1_prompt


def test_pareto_row_only_regression_and_base_drift_sync(tmp_path):
    """Row-only regression targets ['rows'] rollback; small sub-threshold drop with empty regressed_fields syncs from best_k."""
    from ae.semantic_optimization.context import ArtifactManager

    ingest_dir = tmp_path / "ingestion"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    (ingest_dir / "val_1.md").write_text("Val 1", encoding="utf-8")
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,smiles,bacteria\nval_1,CCO,S. aureus\n", encoding="utf-8")
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        "name: test\nfloat_tolerance: 0.01\nfields:\n  smiles:\n    type: str\n  bacteria:\n    type: str\n"
        "compare_fields:\n  - smiles\n  - bacteria\n",
        encoding="utf-8",
    )
    gen_prompt_path = tmp_path / "generated_instruction.txt"
    iter1_prompt = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Best row rule.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Good SMILES.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Weak bacteria.\n"
    )
    iter2_row_broken = (
        "Preamble\n\n---\n\n"
        "RECORD SELECTION\n\nInclude ONLY records satisfying:\n- Overly broad row rule.\n\n---\n\n"
        "SMILES\n\nExtract ONLY if satisfy:\n- Good SMILES.\n\n---\n\n"
        "BACTERIA\n\nExtract ONLY if satisfy:\n- Strong bacteria.\n"
    )

    art_dir = tmp_path / "artifacts"
    iter1_gen = art_dir / "iterations" / "iter_1" / "04_generalization"
    iter2_gen = art_dir / "iterations" / "iter_2" / "04_generalization"
    (iter1_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter2_gen / "columns").mkdir(parents=True, exist_ok=True)
    (iter1_gen / "candidate_prompt.txt").write_text(iter1_prompt, encoding="utf-8")
    (iter1_gen / "rows.json").write_text('{"rows": "iter1_rows"}', encoding="utf-8")
    (iter2_gen / "candidate_prompt.txt").write_text(iter2_row_broken, encoding="utf-8")
    (iter2_gen / "rows.json").write_text('{"rows": "iter2_broken_rows"}', encoding="utf-8")

    ctx = MagicMock(spec=RunContext)
    ctx.artifacts_dir = art_dir
    ctx.artifacts = ArtifactManager(art_dir, resume=False)
    ctx.schema_path = schema_path
    ctx.generated_schema_path = None
    ctx.generated_prompt_path = gen_prompt_path
    ctx.active_gt_path = gt_path
    ctx.ingestion_dir = ingest_dir
    ctx.config_path = None
    ctx.resume = False
    ctx.enable_semantic_judge = False
    ctx.selection_metric = "micro_f1"
    ctx.pareto_regression_threshold = 0.01
    ctx.re_client = MagicMock()
    ctx.re_client.extraction_config = MagicMock(model="test-model")

    graph = SemanticOptimizationGraph(
        context=ctx, train_doc_ids=["d1", "d2"], val_doc_ids=["val_1"], pareto_regression_threshold=0.01
    )

    # Case 1: Row-only regression (per_field_f1 has 0 regressions, but f1_score dropped 0.90 -> 0.70 due to FPs)
    gen_prompt_path.write_text(iter2_row_broken, encoding="utf-8")
    with patch.object(
        graph,
        "_evaluate_prompt_on_val",
        side_effect=[
            {
                "f1_score": 0.70,
                "macro_f1": 0.70,
                "micro_f1": 0.70,
                "strict_f1_score": 0.70,
                "strict_micro_f1": 0.70,
                "strict_macro_f1": 0.70,
                "per_field_f1": {"smiles": 1.0, "bacteria": 0.95},
                "strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.95},
                "doc_scores": {"val_1": {"f1": 0.70, "gts": 1}},
                "evaluated_count": 1,
            },
            {
                "f1_score": 0.95,
                "macro_f1": 0.95,
                "micro_f1": 0.95,
                "strict_f1_score": 0.95,
                "strict_micro_f1": 0.95,
                "strict_macro_f1": 0.95,
                "per_field_f1": {"smiles": 1.0, "bacteria": 0.95},
                "strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.95},
                "doc_scores": {"val_1": {"f1": 0.95, "gts": 1}},
                "evaluated_count": 1,
            },
        ],
    ):
        s2 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 2,
                "val_f1_history": [0.90],
                "val_per_field_history": [{"smiles": 1.0, "bacteria": 0.90}],
                "best_f1": 0.90,
                "best_per_field_f1": {"smiles": 1.0, "bacteria": 0.90},
                "best_strict_f1": 0.90,
                "best_strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.90},
                "best_iteration": 1,
                "best_prompt": iter1_prompt,
                "pareto_rollback_history": [],
            }
        )

    assert s2["pareto_rollback_history"][0]["rollback_targets"] == ["rows"]
    assert s2["pareto_rollback_history"][0]["accepted"] is True
    assert s2["best_f1"] == 0.95
    assert "Best row rule." in s2["best_prompt"]
    assert "Strong bacteria." in s2["best_prompt"]
    assert (iter2_gen / "rows.json").read_text(encoding="utf-8") == '{"rows": "iter1_rows"}'

    # Case 2: Small drop below best_f1 within threshold (0.895 vs 0.90) with regressed_fields == [] -> syncs from best_k to prevent base drift
    iter3_gen = art_dir / "iterations" / "iter_3" / "04_generalization"
    (iter3_gen / "columns").mkdir(parents=True, exist_ok=True)
    iter3_drifted = iter1_prompt.replace("Weak bacteria.", "Slightly worse bacteria.")
    gen_prompt_path.write_text(iter3_drifted, encoding="utf-8")
    (iter3_gen / "candidate_prompt.txt").write_text(iter3_drifted, encoding="utf-8")
    with patch.object(
        graph,
        "_evaluate_prompt_on_val",
        return_value={
            "f1_score": 0.895,
            "macro_f1": 0.895,
            "micro_f1": 0.895,
            "strict_f1_score": 0.895,
            "strict_micro_f1": 0.895,
            "strict_macro_f1": 0.895,
            "per_field_f1": {"smiles": 1.0, "bacteria": 0.895},
            "strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.895},
            "doc_scores": {"val_1": {"f1": 0.895, "gts": 1}},
            "evaluated_count": 1,
        },
    ):
        s3 = graph._evaluate_val(
            {
                "context": ctx,
                "val_doc_ids": ["val_1"],
                "current_iteration": 3,
                "val_f1_history": [0.90],
                "val_per_field_history": [{"smiles": 1.0, "bacteria": 0.90}],
                "best_f1": 0.90,
                "best_per_field_f1": {"smiles": 1.0, "bacteria": 0.90},
                "best_strict_f1": 0.90,
                "best_strict_per_field_f1": {"smiles": 1.0, "bacteria": 0.90},
                "best_iteration": 1,
                "best_prompt": iter1_prompt,
                "pareto_rollback_history": [],
            }
        )

    assert gen_prompt_path.read_text(encoding="utf-8") == iter1_prompt
    assert (iter3_gen / "candidate_prompt.txt").read_text(encoding="utf-8") == iter1_prompt
    assert s3["val_f1_history"] == [0.90, 0.90]



