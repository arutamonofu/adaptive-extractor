"""Unit tests for Reverse Engineering schema constraint inference and refinement."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ae.core.schema.config import FieldSpec, SchemaConfig
from ae.core.schema.loader import load_schema_from_yaml
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.graph import SemanticOptimizationGraph
from ae.semantic_optimization.models import (
    ColumnInstructions,
    FieldConstraintProposal,
    GeneralizedColumnsOutput,
    SchemaRefinementOutput,
)
from ae.semantic_optimization.steps.schema_refinement import (
    apply_schema_refinements,
    build_column_profiles,
    run_schema_refinement,
)


@pytest.mark.unit
class TestSchemaRefinement:
    def test_build_column_profiles_isolates_active_train_docs(self):
        cfg = SchemaConfig(
            name="antibiotics",
            fields={
                "compound_id": FieldSpec(type=str, required=False),
                "target_relation": FieldSpec(type=str, required=False),
                "target_value": FieldSpec(type=float, required=False),
            },
            compare_fields=["compound_id", "target_relation", "target_value"],
        )
        df_gt = pd.DataFrame(
            {
                "pdf": ["doc1.pdf", "doc1.pdf", "doc2.pdf", "val_doc.pdf"],
                "compound_id": ["1a", "1b", "2a", "val_only"],
                "target_relation": ["=", ">", "=", "<"],
                "target_value": [10.0, 20.0, 30.0, 999.0],
            }
        )
        compiled = {
            "target_relation": GeneralizedColumnsOutput(
                Semantic_Core_generalized="Comparison operator for MIC",
                Instructions=ColumnInstructions(
                    Transformation_Instructions=["Use '=' if omitted"]
                ),
            )
        }

        profiles = build_column_profiles(
            df_gt=df_gt,
            schema_config=cfg,
            active_norm_ids={"doc1", "doc2"},
            id_col="pdf",
            compiled_columns=compiled,
        )

        # Must NOT leak 'val_doc' values ('<' or 999.0) into training profile
        assert profiles["target_relation"]["total_rows"] == 3
        assert profiles["target_relation"]["all_observed_values"] == ["=", ">"]
        assert profiles["target_value"]["numeric_max"] == 30.0
        assert profiles["target_relation"]["semantic_core"] == "Comparison operator for MIC"

    def test_apply_schema_refinements_guardrails(self):
        base = SchemaConfig(
            name="antibiotics",
            fields={
                "compound_id": FieldSpec(type=str, required=False),
                "target_relation": FieldSpec(type=str, required=False),
                "target_type": FieldSpec(type=str, required=False, choices=["MIC"]),
            },
            compare_fields=["compound_id", "target_relation", "target_type"],
        )
        profiles = {
            "compound_id": {"all_observed_values": ["1a", "2b"]},
            "target_relation": {"all_observed_values": ["=", ">"]},
            "target_type": {"all_observed_values": ["MIC"]},
        }
        proposal = SchemaRefinementOutput(
            fields=[
                FieldConstraintProposal(
                    field_name="compound_id",
                    description="Compound identifier",
                    is_closed_enum=False,
                    # Invalid regex that fails '1a' should be rejected by guardrails
                    pattern=r"^[A-Z]+$",
                ),
                FieldConstraintProposal(
                    field_name="target_relation",
                    description="Relation operator",
                    is_closed_enum=True,
                    choices=["=", ">", "<"],
                    default_value="=",
                ),
                FieldConstraintProposal(
                    field_name="target_type",
                    description="Measurement type",
                    is_closed_enum=True,
                    # Existing user choices ['MIC'] must NOT be overwritten
                    choices=["MIC", "IC50"],
                ),
            ]
        )

        refined = apply_schema_refinements(base, proposal, profiles)
        assert refined.fields["compound_id"].description == "Compound identifier"
        assert refined.fields["compound_id"].pattern is None
        assert refined.fields["target_relation"].choices == ["=", ">", "<"]
        assert refined.fields["target_relation"].default == "="
        assert refined.fields["target_type"].choices == ["MIC"]

    def test_apply_schema_refinements_rejects_incomplete_enum(self):
        base = SchemaConfig(
            name="test",
            fields={"target_relation": FieldSpec(type=str, required=False)},
            compare_fields=["target_relation"],
        )
        profiles = {"target_relation": {"all_observed_values": ["=", ">"]}}
        proposal = SchemaRefinementOutput(
            fields=[
                FieldConstraintProposal(
                    field_name="target_relation",
                    description="Relation",
                    is_closed_enum=True,
                    # Missing '>' from observed train GT
                    choices=["="],
                )
            ]
        )
        refined = apply_schema_refinements(base, proposal, profiles)
        assert refined.fields["target_relation"].choices is None

    def test_run_schema_refinement_persists_generated_and_iteration_yaml(self, tmp_path):
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(
            "name: antibiotics\n"
            "fields:\n"
            "  target_relation:\n"
            "    type: str\n"
            "    required: false\n"
            "compare_fields:\n"
            "  - target_relation\n"
            "float_tolerance: 0.05\n",
            encoding="utf-8",
        )
        baseline_path = tmp_path / "baseline.txt"
        baseline_path.write_text("target_relation: '=', '>', or '<'. Default '='.", encoding="utf-8")
        gt_path = tmp_path / "gt.csv"
        pd.DataFrame(
            {
                "pdf": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
                "target_relation": ["=", ">", "=", ">", "=", ">", "=", ">"],
            }
        ).to_csv(gt_path, index=False)
        gen_prompt_path = tmp_path / "generated_instruction.txt"
        gen_prompt_path.write_text("Generated prompt", encoding="utf-8")
        gen_schema_path = tmp_path / "generated_schema.yaml"

        ctx = RunContext(
            task_name="antibiotics",
            doc_ids=["doc1", "doc2", "doc3", "doc4"],
            gt_path=gt_path,
            ingestion_dir=tmp_path / "ing",
            baseline_prompt_path=baseline_path,
            schema_path=schema_path,
            teacher_lm=MagicMock(),
            artifacts_dir=tmp_path / "art",
            generated_prompt_path=gen_prompt_path,
            generated_schema_path=gen_schema_path,
            interactive=False,
        )
        ctx.current_iteration = 1

        mock_predictor = MagicMock(
            return_value=MagicMock(
                refinement=SchemaRefinementOutput(
                    fields=[
                        FieldConstraintProposal(
                            field_name="target_relation",
                            description="Comparison operator",
                            is_closed_enum=True,
                            choices=["=", ">", "<"],
                            default_value="=",
                        )
                    ]
                )
            )
        )

        with patch("ae.semantic_optimization.steps.schema_refinement.dspy.Predict", return_value=mock_predictor):
            out_path = run_schema_refinement(
                context=ctx,
                df_gt=pd.read_csv(gt_path),
                id_col="pdf",
                compiled_columns={},
                final_prompt="Generated prompt",
                active_norm_ids={"doc1", "doc2", "doc3", "doc4"},
            )

        assert out_path == gen_schema_path.resolve()
        assert gen_schema_path.exists()
        reloaded = load_schema_from_yaml(gen_schema_path)
        assert reloaded.fields["target_relation"].choices == ["=", ">", "<"]
        assert reloaded.fields["target_relation"].default == "="
        assert reloaded.fields["target_relation"].description == "Comparison operator"

        iter_candidate = ctx.artifacts.get_path("generalization", "", "candidate_schema.yaml", iteration=1)
        assert iter_candidate.exists()

        # Resume with identical inputs must use cached schema_refinement.json (no extra LLM call)
        ctx.resume = True
        ctx.artifacts.resume = True
        mock_predictor.reset_mock()
        with patch("ae.semantic_optimization.steps.schema_refinement.dspy.Predict", return_value=mock_predictor):
            run_schema_refinement(
                context=ctx,
                df_gt=pd.read_csv(gt_path),
                id_col="pdf",
                compiled_columns={},
                final_prompt="Generated prompt",
                active_norm_ids={"doc1", "doc2", "doc3", "doc4"},
            )
        assert mock_predictor.call_count == 0

        # Changing final_prompt on resume must invalidate cache and recompute
        with patch("ae.semantic_optimization.steps.schema_refinement.dspy.Predict", return_value=mock_predictor):
            run_schema_refinement(
                context=ctx,
                df_gt=pd.read_csv(gt_path),
                id_col="pdf",
                compiled_columns={},
                final_prompt="Updated generated prompt",
                active_norm_ids={"doc1", "doc2", "doc3", "doc4"},
            )
        assert mock_predictor.call_count == 1

    def test_re_graph_finalize_copies_best_candidate_schema(self, tmp_path):
        art_dir = tmp_path / "art"
        best_gen_dir = art_dir / "iterations" / "iter_1" / "04_generalization"
        best_gen_dir.mkdir(parents=True, exist_ok=True)
        (best_gen_dir / "candidate_schema.yaml").write_text("name: best_iter_schema\n", encoding="utf-8")

        gen_schema_out = tmp_path / "generated_schema.yaml"

        ctx = MagicMock(spec=RunContext)
        ctx.artifacts_dir = art_dir
        ctx.generated_schema_path = gen_schema_out
        ctx.generated_prompt_path = tmp_path / "generated_instruction.txt"
        ctx.active_gt_path = tmp_path / "gt.csv"
        ctx.rulings_store = MagicMock()
        ctx.rulings_store.rulings = {}
        ctx.artifacts = MagicMock()
        ctx.artifacts.get_phase_dir.side_effect = (
            lambda phase, *sub, iteration=None: (
                art_dir / "iterations" / f"iter_{iteration}" / ("04_generalization" if phase == "generalization" else phase)
                if iteration is not None
                else art_dir / phase
            )
        )

        graph = SemanticOptimizationGraph(context=ctx, train_doc_ids=["doc1"])
        graph._finalize(
            {
                "current_iteration": 1,
                "best_iteration": 1,
                "best_f1": 0.9,
                "best_prompt": "best prompt",
                "val_f1_history": [0.9],
                "train_doc_ids": ["doc1"],
                "val_doc_ids": [],
                "f1_threshold": 0.85,
                "context": ctx,
            }
        )

        assert gen_schema_out.exists()
        assert "best_iter_schema" in gen_schema_out.read_text(encoding="utf-8")

    def test_schema_refinement_output_coerces_null_choices_and_metadata(self):
        """LLM outputs with choices: null on open-vocabulary fields must validate cleanly."""
        raw_json = """
        {
          "fields": [
            {
              "field_name": "compound_id",
              "description": null,
              "is_closed_enum": null,
              "choices": null,
              "default_value": null,
              "pattern": null,
              "min_value": null,
              "max_value": null,
              "rationale": null
            },
            {
              "field_name": "target_relation",
              "description": "Relation operator",
              "is_closed_enum": true,
              "choices": ["=", ">", "<"],
              "default_value": "=",
              "rationale": "Closed operator vocabulary"
            }
          ]
        }
        """
        parsed = SchemaRefinementOutput.model_validate_json(raw_json)
        assert len(parsed.fields) == 2
        assert parsed.fields[0].choices == []
        assert parsed.fields[0].description == ""
        assert parsed.fields[0].rationale == ""
        assert parsed.fields[0].is_closed_enum is False
        assert parsed.fields[1].choices == ["=", ">", "<"]

    def test_doc_support_gate_blocks_low_doc_count_and_singleton_values(self):
        """doc_support gate blocks is_closed_enum when active_docs_count < 4 or any value has doc_count == 1."""
        base = SchemaConfig(
            name="antibiotics",
            fields={
                "target_units": FieldSpec(type=str, required=False),
                "target_type": FieldSpec(type=str, required=False),
            },
            compare_fields=["target_units", "target_type"],
        )
        proposal = SchemaRefinementOutput(
            fields=[
                FieldConstraintProposal(
                    field_name="target_units",
                    description="Concentration units",
                    is_closed_enum=True,
                    choices=["µg/mL", "mM", "µM"],
                ),
                FieldConstraintProposal(
                    field_name="target_type",
                    description="Measurement type",
                    is_closed_enum=True,
                    choices=["MIC", "pMIC"],
                ),
            ]
        )

        # Case 1: iter_1 with 2 docs -> active_docs_count < 4 blocks both enums
        df_iter1 = pd.DataFrame(
            {
                "pdf": ["doc1", "doc2"],
                "target_units": ["µg/mL", "mM"],
                "target_type": ["MIC", "MIC"],
            }
        )
        profiles_iter1 = build_column_profiles(
            df_gt=df_iter1,
            schema_config=base,
            active_norm_ids={"doc1", "doc2"},
            id_col="pdf",
            compiled_columns={},
        )
        assert profiles_iter1["target_units"]["active_docs_count"] == 2
        refined_iter1 = apply_schema_refinements(base, proposal, profiles_iter1)
        assert refined_iter1.fields["target_units"].choices is None
        assert refined_iter1.fields["target_type"].choices is None

        # Case 2: iter_2 with 6 docs -> target_units has singleton doc_count == 1 ('mM'), target_type has doc_count == 6
        df_iter2 = pd.DataFrame(
            {
                "pdf": ["d1", "d2", "d3", "d4", "d5", "d6"],
                "target_units": ["µg/mL", "µg/mL", "µg/mL", "µg/mL", "µg/mL", "mM"],
                "target_type": ["MIC", "MIC", "MIC", "MIC", "MIC", "MIC"],
            }
        )
        profiles_iter2 = build_column_profiles(
            df_gt=df_iter2,
            schema_config=base,
            active_norm_ids={"d1", "d2", "d3", "d4", "d5", "d6"},
            id_col="pdf",
            compiled_columns={},
        )
        assert profiles_iter2["target_units"]["value_doc_support"]["mM"] == 1
        assert profiles_iter2["target_type"]["value_doc_support"]["MIC"] == 6
        refined_iter2 = apply_schema_refinements(base, proposal, profiles_iter2)
        assert refined_iter2.fields["target_units"].choices is None
        assert refined_iter2.fields["target_type"].choices == ["MIC", "pMIC"]

    def test_distinct_count_gt_50_blocks_enum_and_regex(self):
        """High-cardinality fields (distinct_count > 50 / all_observed_values is None) cannot be closed or regex-gated."""
        base = SchemaConfig(
            name="antibiotics",
            fields={
                "compound_id": FieldSpec(type=str, required=False),
                "smiles": FieldSpec(type=str, required=False),
            },
            compare_fields=["compound_id", "smiles"],
        )
        df_high = pd.DataFrame(
            {
                "pdf": [f"doc{i % 6}" for i in range(60)],
                "compound_id": [f"cmp_{i}" for i in range(60)],
                "smiles": [f"C{'C' * (i % 10)}O{i}" for i in range(60)],
            }
        )
        profiles = build_column_profiles(
            df_gt=df_high,
            schema_config=base,
            active_norm_ids={f"doc{i}" for i in range(6)},
            id_col="pdf",
            compiled_columns={},
        )
        assert profiles["compound_id"]["distinct_count"] == 60
        assert "all_observed_values" not in profiles["compound_id"]

        proposal = SchemaRefinementOutput(
            fields=[
                FieldConstraintProposal(
                    field_name="compound_id",
                    description="Compound identifier",
                    is_closed_enum=True,
                    choices=["cmp_0", "cmp_1"],
                    pattern=r"^cmp_\d+$",
                ),
                FieldConstraintProposal(
                    field_name="smiles",
                    description="SMILES string",
                    is_closed_enum=False,
                    pattern=r"^C+O\d+$",
                ),
            ]
        )
        refined = apply_schema_refinements(base, proposal, profiles)
        assert refined.fields["compound_id"].choices is None
        assert refined.fields["compound_id"].pattern is None
        assert refined.fields["smiles"].pattern is None

    def test_proportional_enum_doc_support_prompt_backed_singleton_and_pattern_default_gating(self):
        """Enum allows >=80% multi-doc ratio or prompt-backed singletons; pattern/default respect min_enum_doc_support."""
        base = SchemaConfig(
            name="antibiotics",
            fields={
                "target_relation": FieldSpec(type=str, required=False),
                "assay_type": FieldSpec(type=str, required=False),
            },
            compare_fields=["target_relation", "assay_type"],
        )
        proposal = SchemaRefinementOutput(
            fields=[
                FieldConstraintProposal(
                    field_name="target_relation",
                    description="Relation operator",
                    is_closed_enum=True,
                    choices=["=", ">", "<", ">=", "<="],
                    default_value="=",
                    pattern=r"^(?:=|>|<|>=|<=)$",
                ),
                FieldConstraintProposal(
                    field_name="assay_type",
                    description="Assay type",
                    is_closed_enum=True,
                    choices=["MIC", "IC50"],
                ),
            ]
        )

        # Case A: active_docs_count=2 < 4 blocks default_value and pattern in addition to enum
        df_low_docs = pd.DataFrame(
            {
                "pdf": ["d1", "d2"],
                "target_relation": ["=", ">"],
                "assay_type": ["MIC", "MIC"],
            }
        )
        profiles_low = build_column_profiles(
            df_gt=df_low_docs,
            schema_config=base,
            active_norm_ids={"d1", "d2"},
            id_col="pdf",
            compiled_columns={},
        )
        refined_low = apply_schema_refinements(base, proposal, profiles_low, min_enum_doc_support=4)
        assert refined_low.fields["target_relation"].choices is None
        assert refined_low.fields["target_relation"].default is None
        assert refined_low.fields["target_relation"].pattern is None

        # Case B: 5 distinct values across 4 docs, 4 with doc_count >= 2 and 1 rare ('<=' with doc_count=1) -> 4/5 = 0.80 >= 0.80
        # Meanwhile assay_type has 2 values ('MIC' doc_count=4, 'IC50' doc_count=1) -> 1/2 = 0.50, blocked WITHOUT prompt mention,
        # but allowed WITH explicit prompt mention!
        df_rich = pd.DataFrame(
            {
                "pdf": ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4", "d4"],
                "target_relation": ["=", ">", "=", "<", ">=", "<", ">", ">=", "<="],
                "assay_type": ["MIC", "MIC", "MIC", "MIC", "MIC", "MIC", "MIC", "MIC", "IC50"],
            }
        )
        profiles_rich = build_column_profiles(
            df_gt=df_rich,
            schema_config=base,
            active_norm_ids={"d1", "d2", "d3", "d4"},
            id_col="pdf",
            compiled_columns={},
        )
        refined_no_prompt = apply_schema_refinements(base, proposal, profiles_rich, min_enum_doc_support=4)
        assert refined_no_prompt.fields["target_relation"].choices == ["=", ">", "<", ">=", "<="]
        assert refined_no_prompt.fields["target_relation"].default == "="
        assert refined_no_prompt.fields["target_relation"].pattern == r"^(?:=|>|<|>=|<=)$"
        assert refined_no_prompt.fields["assay_type"].choices is None

        refined_with_prompt = apply_schema_refinements(
            base,
            proposal,
            profiles_rich,
            min_enum_doc_support=4,
            prompt_text="Extract assay_type as MIC or IC50.",
        )
        assert refined_with_prompt.fields["assay_type"].choices == ["MIC", "IC50"]

    def test_numeric_bounds_doc_support_gate_and_prompt_grounded_iter1_constraints(self):
        """Numeric bounds respect doc_support gate, while explicitly prompt-grounded defaults/patterns work on iter 1."""
        base = SchemaConfig(
            name="antibiotics",
            fields={
                "target_relation": FieldSpec(type=str, required=False),
                "target_value": FieldSpec(type=float, required=False),
            },
            compare_fields=["target_relation", "target_value"],
        )
        df_iter1 = pd.DataFrame(
            {
                "pdf": ["d1", "d2"],
                "target_relation": ["=", ">"],
                "target_value": [1.0, 50.0],
            }
        )
        profiles_iter1 = build_column_profiles(
            df_gt=df_iter1,
            schema_config=base,
            active_norm_ids={"d1", "d2"},
            id_col="pdf",
            compiled_columns={},
        )
        proposal = SchemaRefinementOutput(
            fields=[
                FieldConstraintProposal(
                    field_name="target_relation",
                    description="Relation operator",
                    is_closed_enum=False,
                    default_value="=",
                    pattern=r"^(?:=|>|<)$",
                ),
                FieldConstraintProposal(
                    field_name="target_value",
                    description="Numeric measurement",
                    is_closed_enum=False,
                    min_value=0.0,
                    max_value=100.0,
                ),
            ]
        )

        # On iter 1 (2 < 4) WITH explicit prompt grounding for target_relation default & pattern:
        # default_value and pattern are allowed, while numeric bounds (no prompt gate bypass) are blocked!
        prompt = "target_relation: '=', '>', or '<'. If omitted, default '='."
        refined_iter1 = apply_schema_refinements(
            base,
            proposal,
            profiles_iter1,
            min_enum_doc_support=4,
            prompt_text=prompt,
        )
        assert refined_iter1.fields["target_relation"].default == "="
        assert refined_iter1.fields["target_relation"].pattern == r"^(?:=|>|<)$"
        assert refined_iter1.fields["target_value"].min_value is None
        assert refined_iter1.fields["target_value"].max_value is None

        # On 4 docs (>= 4), numeric bounds are applied
        df_iter2 = pd.DataFrame(
            {
                "pdf": ["d1", "d2", "d3", "d4"],
                "target_relation": ["=", ">", "<", "="],
                "target_value": [1.0, 50.0, 10.0, 25.0],
            }
        )
        profiles_iter2 = build_column_profiles(
            df_gt=df_iter2,
            schema_config=base,
            active_norm_ids={"d1", "d2", "d3", "d4"},
            id_col="pdf",
            compiled_columns={},
        )
        refined_iter2 = apply_schema_refinements(
            base,
            proposal,
            profiles_iter2,
            min_enum_doc_support=4,
            prompt_text=prompt,
        )
        assert refined_iter2.fields["target_value"].min_value == 0.0
        assert refined_iter2.fields["target_value"].max_value == 100.0

    def test_run_schema_refinement_passes_dynamic_min_enum_doc_support(self, tmp_path):
        """run_schema_refinement passes context.min_enum_doc_support into SchemaConstraintInference."""
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(
            "name: test\nfloat_tolerance: 0.05\nfields:\n  target_relation:\n    type: str\ncompare_fields: [target_relation]\n",
            encoding="utf-8",
        )
        baseline_path = tmp_path / "baseline.txt"
        baseline_path.write_text("Base", encoding="utf-8")
        gt_path = tmp_path / "gt.csv"
        pd.DataFrame({"pdf": ["d1", "d2"], "target_relation": ["=", ">"]}).to_csv(gt_path, index=False)

        ctx = RunContext(
            task_name="test",
            doc_ids=["d1", "d2"],
            gt_path=gt_path,
            ingestion_dir=tmp_path / "ing",
            baseline_prompt_path=baseline_path,
            schema_path=schema_path,
            teacher_lm=MagicMock(),
            artifacts_dir=tmp_path / "art",
            generated_prompt_path=tmp_path / "gen.txt",
            generated_schema_path=tmp_path / "gen.yaml",
            min_enum_doc_support=2,
            interactive=False,
        )
        ctx.current_iteration = 1

        mock_predictor = MagicMock(return_value=MagicMock(refinement=SchemaRefinementOutput(fields=[])))
        with patch("ae.semantic_optimization.steps.schema_refinement.dspy.Predict", return_value=mock_predictor):
            run_schema_refinement(
                context=ctx,
                df_gt=pd.read_csv(gt_path),
                id_col="pdf",
                compiled_columns={},
                final_prompt="Final prompt",
                active_norm_ids={"d1", "d2"},
            )

        assert mock_predictor.call_args.kwargs["min_enum_doc_support"] == 2


