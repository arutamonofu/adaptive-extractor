"""Unit tests for Error-Driven / Residual Semantic Analysis (Iteration >= 2)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pandas as pd
import pytest

from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.models import (
    CandidateAnalysis,
    FieldAnalysis,
    GapRowAnalysis,
    NegativeColumnsOutput,
    NegativeFieldResult,
    NegativeRowsOutput,
    PositiveColumnsOutput,
    PositiveRowsOutput,
    ResidualColumnsOutput,
    ResidualDocReport,
    RowAnalysis,
)
from ae.semantic_optimization.steps.residual_analysis import run_residual_analysis


def _make_schema(tmp_path: Path, choices_target_type: list[str] | None = None) -> Path:
    schema_path = tmp_path / "schema.yaml"
    choices_block = (
        "    choices:\n" + "".join(f"      - {c}\n" for c in choices_target_type)
        if choices_target_type
        else ""
    )
    schema_path.write_text(
        "name: antibiotics\n"
        "fields:\n"
        "  compound_id:\n"
        "    type: str\n"
        "    required: false\n"
        "  smiles:\n"
        "    type: str\n"
        "    required: false\n"
        "    validators:\n"
        "      - name: smiles\n"
        "        on_error: warn\n"
        "  target_type:\n"
        "    type: str\n"
        "    required: false\n"
        f"{choices_block}"
        "  target_relation:\n"
        "    type: str\n"
        "    required: false\n"
        "  target_value:\n"
        "    type: str\n"
        "    required: false\n"
        "  target_units:\n"
        "    type: str\n"
        "    required: false\n"
        "  bacteria:\n"
        "    type: str\n"
        "    required: false\n"
        "compare_fields:\n"
        "  - compound_id\n"
        "  - smiles\n"
        "  - target_type\n"
        "  - target_relation\n"
        "  - target_value\n"
        "  - target_units\n"
        "  - bacteria\n"
        "float_tolerance: 0.05\n",
        encoding="utf-8",
    )
    return schema_path


def _setup_residual_context(
    tmp_path: Path,
    rows: list[dict],
    doc_texts: dict[str, str],
    generated_schema_choices: list[str] | None = None,
    **ctx_kwargs,
) -> RunContext:
    gt_path = tmp_path / "gt.csv"
    pd.DataFrame(rows).to_csv(gt_path, index=False)

    ingest_dir = tmp_path / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    for d_id, txt in doc_texts.items():
        (ingest_dir / f"{d_id}.md").write_text(txt, encoding="utf-8")

    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("Base prompt", encoding="utf-8")

    base_schema = _make_schema(tmp_path)
    gen_prompt_path = tmp_path / "generated_instruction.txt"
    gen_prompt_path.write_text("Current optimized v1 prompt", encoding="utf-8")

    gen_schema_path = tmp_path / "generated_schema.yaml"
    if generated_schema_choices is not None:
        sub_dir = tmp_path / "gen_schema"
        sub_dir.mkdir(parents=True, exist_ok=True)
        strict_schema = _make_schema(sub_dir, choices_target_type=generated_schema_choices)
        gen_schema_path.write_text(strict_schema.read_text(encoding="utf-8"), encoding="utf-8")

    mock_re_client = MagicMock()
    mock_re_client.get_lm_provider.return_value = MagicMock()
    mock_val_client = MagicMock()
    mock_val_client.get_dspy_adapter.return_value = None
    mock_re_client.get_validation_client.return_value = mock_val_client

    return RunContext(
        task_name="antibiotics",
        doc_ids=list(doc_texts.keys()),
        gt_path=gt_path,
        ingestion_dir=ingest_dir,
        baseline_prompt_path=baseline_path,
        schema_path=base_schema,
        re_client=mock_re_client,
        artifacts_dir=tmp_path / "artifacts",
        generated_prompt_path=gen_prompt_path,
        generated_schema_path=gen_schema_path,
        interactive=False,
        enable_semantic_judge=False,
        **ctx_kwargs,
    )


def _field_analysis(row_id: str, field_name: str, raw_text: str, gt_val: str) -> FieldAnalysis:
    return FieldAnalysis(
        row_id=row_id,
        field_name=field_name,
        raw_text=raw_text,
        ground_truth=gt_val,
        Semantic_Core=f"Core of {field_name}",
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
        Transformation_description=f"Normalize {field_name}",
        Suspected_Ingestion_Or_GT_Issue="None",
    )


def _candidate_analysis(row_id: str, cand: str, gt: str) -> CandidateAnalysis:
    return CandidateAnalysis(
        row_id=row_id,
        raw_text=cand,
        candidate_value=cand,
        ground_truth=gt,
        Semantic_Core="Alternative candidate",
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
    )


def test_residual_zero_delta_fast_forward_skips_reasoning_llm(tmp_path: Path):
    """Perfect / >=0.98 F1 extraction (including equivalent canonical SMILES) skips all reasoning LLM calls."""
    pytest.importorskip("rdkit")
    rows = [
        {
            "pdf": "doc_perfect",
            "compound_id": "5a",
            "smiles": "C1=CC=C2C(=C1)NC=N2",
            "target_type": "MIC",
            "target_relation": "=",
            "target_value": "62.5",
            "target_units": "µg/mL",
            "bacteria": "E. coli",
        }
    ]
    ctx = _setup_residual_context(tmp_path, rows, {"doc_perfect": "Compound 5a MIC 62.5"})

    fake_extractor = MagicMock()
    # Aromatic SMILES equivalent to Kekule GT SMILES
    fake_extractor.forward.return_value = dspy.Prediction(
        extracted_data={
            "records": [
                {
                    "compound_id": "5a",
                    "smiles": "c1ccc2[nH]cnc2c1",
                    "target_type": "MIC",
                    "target_relation": "=",
                    "target_value": "62.5",
                    "target_units": "µg/mL",
                    "bacteria": "E. coli",
                }
            ]
        }
    )
    ctx.re_client.create_validation_extractor.return_value = fake_extractor

    with patch("ae.semantic_optimization.steps.residual_analysis.dspy.Predict") as mock_predict:
        summary = run_residual_analysis(ctx)

    assert summary.is_zero_delta is True
    assert summary.zero_delta_doc_ids == ["doc_perfect"]
    assert summary.fallback_doc_ids == []
    assert summary.residual_doc_ids == []
    assert mock_predict.return_value.call_count == 0

    report_path = ctx.artifacts.get_path("positive", "residual", "doc_perfect.json")
    assert report_path.exists()
    report = ResidualDocReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.mode == "zero_delta_skip"
    assert report.f1 == pytest.approx(1.0)


def test_residual_alignment_fallback_on_low_f1(tmp_path: Path):
    """Documents with F1 < 0.35 or Recall < 0.35 trigger alignment_fallback to full stratified analysis."""
    rows = [
        {
            "pdf": "doc_novel",
            "compound_id": f"c{i}",
            "smiles": "c1ccccc1",
            "target_type": "MIC",
            "target_relation": "=",
            "target_value": str(10 * i),
            "target_units": "µg/mL",
            "bacteria": "S. aureus",
        }
        for i in range(1, 5)
    ]
    ctx = _setup_residual_context(tmp_path, rows, {"doc_novel": "Novel review layout"})

    fake_extractor = MagicMock()
    fake_extractor.forward.return_value = dspy.Prediction(extracted_data={"records": []})
    ctx.re_client.create_validation_extractor.return_value = fake_extractor

    summary = run_residual_analysis(ctx)
    assert summary.fallback_doc_ids == ["doc_novel"]
    assert summary.is_zero_delta is False
    assert summary.doc_reports[0].mode == "alignment_fallback"


def test_residual_cache_input_hash_resume_and_invalidation(tmp_path: Path):
    """input_hash is persisted for skip/fallback, survives in-place md edits when manifest exists, and invalidates on schema/judge change."""
    rows = [
        {
            "pdf": "doc_cache",
            "compound_id": "5a",
            "smiles": "c1ccccc1",
            "target_type": "MIC",
            "target_relation": "=",
            "target_value": "62.5",
            "target_units": "µg/mL",
            "bacteria": "E. coli",
        }
    ]
    ctx = _setup_residual_context(tmp_path, rows, {"doc_cache": "Initial text"})
    # Create canonical manifest alongside nested doc_cache.md
    doc_dir = ctx.ingestion_dir / "doc_cache"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (ctx.ingestion_dir / "doc_cache.md").unlink()
    (doc_dir / "doc_cache.md").write_text("Initial text", encoding="utf-8")
    manifest_dir = doc_dir / "02_manifest_visuals"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest_enriched_text.md").write_text("Canonical manifest", encoding="utf-8")

    # 1. Zero-delta skip writes input_hash and resumes without calling extractor
    fake_extractor = MagicMock()
    fake_extractor.forward.return_value = dspy.Prediction(
        extracted_data={
            "records": [
                {
                    "compound_id": "5a",
                    "smiles": "c1ccccc1",
                    "target_type": "MIC",
                    "target_relation": "=",
                    "target_value": "62.5",
                    "target_units": "µg/mL",
                    "bacteria": "E. coli",
                }
            ]
        }
    )
    ctx.re_client.create_validation_extractor.return_value = fake_extractor
    s1 = run_residual_analysis(ctx)
    assert s1.doc_reports[0].input_hash != ""
    assert fake_extractor.forward.call_count == 1

    # Simulate in-place visual unpacking modifying doc_cache.md while manifest_enriched_text.md is unchanged
    (doc_dir / "doc_cache.md").write_text("Modified in-place with unpacked visual", encoding="utf-8")
    ctx.resume = True
    ctx.artifacts.resume = True
    s2 = run_residual_analysis(ctx)
    assert s2.zero_delta_doc_ids == ["doc_cache"]
    assert fake_extractor.forward.call_count == 1  # Still 1 (loaded from cache!)

    # Changing schema or enable_semantic_judge invalidates cache
    ctx.enable_semantic_judge = True
    s3 = run_residual_analysis(ctx)
    assert fake_extractor.forward.call_count == 2
    assert s3.doc_reports[0].input_hash != s1.doc_reports[0].input_hash

    # 2. Extractor error fallback also persists input_hash and resumes from cache
    ctx.artifacts.resume = False
    ctx.resume = False
    err_extractor = MagicMock()
    err_extractor.forward.side_effect = RuntimeError("boom")
    ctx.re_client.create_validation_extractor.return_value = err_extractor
    s4 = run_residual_analysis(ctx)
    assert s4.doc_reports[0].mode == "alignment_fallback"
    assert s4.doc_reports[0].input_hash != ""
    assert err_extractor.forward.call_count == 1

    ctx.resume = True
    ctx.artifacts.resume = True
    s5 = run_residual_analysis(ctx)
    assert s5.fallback_doc_ids == ["doc_cache"]
    assert err_extractor.forward.call_count == 1


def test_residual_targeted_fn_fp_split_and_signature_grouping(tmp_path: Path):
    """Verify 0-discriminative pair splitting (FP+FN), signature grouping cap (<=3), and unseen GT enum safety."""
    # 6 GT rows: rows 1-5 have wrong bacteria abbreviation ("S.a." vs "S. aureus") -> 1 mismatch signature (cap 3)
    # row 6 ("4h") is missed by extractor, while extractor hallucinates "Penicillin" (same MIC/=/µg/mL constants!)
    rows = []
    for i in range(1, 6):
        rows.append(
            {
                "pdf": "doc_mixed",
                "compound_id": f"5{chr(96 + i)}",
                "smiles": "c1ccccc1",
                "target_type": "pMIC" if i == 1 else "MIC",  # pMIC outside generated_schema choices=["MIC"]!
                "target_relation": "=",
                "target_value": str(10 * i),
                "target_units": "µg/mL",
                "bacteria": "S. aureus",
            }
        )
    rows.append(
        {
            "pdf": "doc_mixed",
            "compound_id": "4h",
            "smiles": "Cc1ccccc1",
            "target_type": "MIC",
            "target_relation": "=",
            "target_value": "128",
            "target_units": "µg/mL",
            "bacteria": "E. coli",
        }
    )

    ctx = _setup_residual_context(
        tmp_path,
        rows,
        {"doc_mixed": "Table 1 compounds 5a-5e, 4h, and Penicillin control."},
        generated_schema_choices=["MIC"],
        residual_max_examples_per_sig=3,
        row_batch_size=5,
    )

    preds = [
        {
            "compound_id": f"5{chr(96 + i)}",
            "smiles": "c1ccccc1",
            "target_type": "pMIC" if i == 1 else "MIC",
            "target_relation": "=",
            "target_value": str(10 * i),
            "target_units": "µg/mL",
            "bacteria": "S.a.",
        }
        for i in range(1, 6)
    ]
    # Add hallucinated control "Penicillin" sharing MIC, =, µg/mL with missed "4h"
    preds.append(
        {
            "compound_id": "Penicillin",
            "smiles": "CC(=O)N",
            "target_type": "MIC",
            "target_relation": "=",
            "target_value": "1.56",
            "target_units": "µg/mL",
            "bacteria": "B. subtilis",
        }
    )

    fake_extractor = MagicMock()
    fake_extractor.forward.return_value = dspy.Prediction(extracted_data={"records": preds})
    ctx.re_client.create_validation_extractor.return_value = fake_extractor

    recorded_calls: dict[str, list[dict]] = {}

    def fake_predict(sig):
        sig_name = getattr(sig, "__name__", str(sig))

        def _invoke(**kwargs):
            recorded_calls.setdefault(sig_name, []).append(kwargs)
            if sig_name == "ResidualPositiveRowAnalysis":
                return MagicMock(
                    analysis=PositiveRowsOutput(
                        rows=[
                            RowAnalysis(
                                row_id="row_6",
                                Row_Instantiation_Trigger_class="Composite_Vector",
                                Row_Instantiation_Trigger_description="Missed 4h",
                                Entity_System_Role_class="Target",
                                Entity_System_Role_description="Target compound",
                                Entity_Filtration_Boundary_class="Exhaustive_Set",
                                Entity_Filtration_Boundary_description="All included",
                            )
                        ]
                    )
                )
            if sig_name == "PositiveColumnAnalysis":
                return MagicMock(
                    analysis=PositiveColumnsOutput(
                        fields=[_field_analysis("row_6", "compound_id", "4h", "4h")]
                    )
                )
            if sig_name == "ResidualNegativeRowAnalysis":
                return MagicMock(
                    analysis=NegativeRowsOutput(
                        gap_rows=[
                            GapRowAnalysis(
                                gap_row_id="gap_row_1",
                                raw_text="Penicillin control",
                                gap_entity_description="Standard reference drug cohort",
                                Row_Instantiation_Trigger_class="Single_Entity",
                                Row_Instantiation_Trigger_description="Control drug",
                                Entity_System_Role_class="Reference",
                                Entity_System_Role_description="Reference benchmark",
                                Entity_Filtration_Boundary_class="Taxonomic_Class",
                                Entity_Filtration_Boundary_description="Excluded control",
                            )
                        ]
                    )
                )
            if sig_name == "ResidualColumnAnalysis":
                payload = json.loads(kwargs["mismatched_rows_context"])
                pos_f = [
                    _field_analysis(item["row_id"], "bacteria", "S.a.", "S. aureus")
                    for item in payload
                ]
                neg_f = [
                    NegativeFieldResult(
                        row_id=item["row_id"],
                        field_name="bacteria",
                        candidates=[_candidate_analysis(item["row_id"], "S.a.", "S. aureus")],
                    )
                    for item in payload
                ]
                return MagicMock(
                    analysis=ResidualColumnsOutput(positive_fields=pos_f, negative_fields=neg_f)
                )
            raise AssertionError(f"Unexpected signature: {sig_name}")

        return _invoke

    with patch("ae.semantic_optimization.steps.residual_analysis.dspy.Predict", side_effect=fake_predict):
        summary = run_residual_analysis(ctx)

    assert summary.residual_doc_ids == ["doc_mixed"]
    assert summary.fallback_doc_ids == []
    assert "rows" in summary.affected_scopes
    assert "column:bacteria" in summary.affected_scopes

    # Zero-discriminative pair ("Penicillin" vs "4h") must have split into FN (row_6) + FP ("Penicillin")
    assert "ResidualPositiveRowAnalysis" in recorded_calls
    assert "row_6" in recorded_calls["ResidualPositiveRowAnalysis"][0]["missed_gt_rows"]
    assert "ResidualNegativeRowAnalysis" in recorded_calls
    fp_payload = json.loads(recorded_calls["ResidualNegativeRowAnalysis"][0]["false_positive_rows"])
    assert fp_payload[0]["compound_id"] == "Penicillin"

    # 5 identical bacteria mismatch rows must be capped to <= 3 in ResidualColumnAnalysis with occurrence_count_in_doc=5
    assert len(recorded_calls["ResidualColumnAnalysis"]) == 1
    col_payload = json.loads(recorded_calls["ResidualColumnAnalysis"][0]["mismatched_rows_context"])
    assert len(col_payload) == 3
    assert col_payload[0]["mismatches"][0]["occurrence_count_in_doc"] == 5

    # Standard 01_positive and 03_negative artifacts written cleanly for run_generalization
    assert ctx.artifacts.get_path("positive", "rows", "doc_mixed.json").exists()
    assert ctx.artifacts.get_path("negative", "rows", "doc_mixed.json").exists()
    assert ctx.artifacts.get_path("positive", "columns", "doc_mixed_row_6.json").exists()
    assert ctx.artifacts.get_path("positive", "columns", "doc_mixed_row_1.json").exists()
    assert ctx.artifacts.get_path("negative", "columns", "doc_mixed_row_1.json").exists()


def test_residual_classifies_and_groups_stereo_diff_and_invalidates_on_eval_schema_change(tmp_path: Path):
    """_classify_field_mismatch preserves stereo_diff reason and eval_schema changes invalidate residual cache."""
    from ae.semantic_optimization.steps.residual_analysis import (
        _classify_field_mismatch,
        _group_mismatch_pairs,
    )

    assert _classify_field_mismatch("C/C=C/C", "C/C=C\\C", "stereo_diff") == "stereo_diff"

    items = [
        {
            "row_id": "row_1",
            "row_context": {"smiles": "C/C=C\\C"},
            "mismatched_fields": {
                "smiles": {"pred": "C/C=C/C", "gold": "C/C=C\\C", "reason": "stereo_diff"}
            },
        }
    ]
    selected, sig_counts = _group_mismatch_pairs(items, max_per_sig=3)
    assert len(selected) == 1
    assert sig_counts == {"smiles:stereo_diff": 1}

    rows = [
        {
            "pdf": "doc_eval_schema",
            "compound_id": "5a",
            "smiles": "c1ccccc1",
            "target_type": "MIC",
            "target_relation": "=",
            "target_value": "62.5",
            "target_units": "µg/mL",
            "bacteria": "E. coli",
        }
    ]
    ctx = _setup_residual_context(
        tmp_path,
        rows,
        {"doc_eval_schema": "Compound 5a MIC 62.5"},
        generated_schema_choices=["MIC"],
    )
    fake_extractor = MagicMock()
    fake_extractor.forward.return_value = dspy.Prediction(
        extracted_data={
            "records": [
                {
                    "compound_id": "5a",
                    "smiles": "c1ccccc1",
                    "target_type": "MIC",
                    "target_relation": "=",
                    "target_value": "62.5",
                    "target_units": "µg/mL",
                    "bacteria": "E. coli",
                }
            ]
        }
    )
    ctx.re_client.create_validation_extractor.return_value = fake_extractor

    s1 = run_residual_analysis(ctx)
    assert fake_extractor.forward.call_count == 1

    # Updating generated_schema.yaml must invalidate residual cache on resume even when base context.schema is unchanged
    sub_dir2 = tmp_path / "gen_schema_2"
    sub_dir2.mkdir(parents=True, exist_ok=True)
    updated_schema = _make_schema(sub_dir2, choices_target_type=["MIC", "IC50"])
    Path(ctx.generated_schema_path).write_text(updated_schema.read_text(encoding="utf-8"), encoding="utf-8")

    ctx.resume = True
    ctx.artifacts.resume = True
    s2 = run_residual_analysis(ctx)
    assert fake_extractor.forward.call_count == 2
    assert s2.doc_reports[0].input_hash != s1.doc_reports[0].input_hash
