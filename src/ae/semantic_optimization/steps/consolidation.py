# src/ae/reverse_engineering/steps/consolidation.py
"""Phase 2: Positive Consolidation step implementation with HITL conflict handling."""

import contextlib
import json
import logging
from pathlib import Path
import pandas as pd
import dspy

from ae.core.utils import normalize_document_key
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.models import (
    ConsolidatedColumnsOutput,
    ConsolidatedRowsOutput,
    PositiveColumnsOutput,
    PositiveRowsOutput,
)
from ae.semantic_optimization.signatures import ColumnConsolidation, RowConsolidation
from ae.semantic_optimization.hitl import _get_id_column
from ae.semantic_optimization.steps.positive_analysis import filter_equivalent_conflicts

logger = logging.getLogger(__name__)


def _invalidate_doc_artifacts(context: RunContext, doc_id: str) -> None:
    """Invalidate cached positive and negative artifacts for doc_id after GT/markdown changes."""
    norm_doc = normalize_document_key(doc_id)
    # Remove visual prep marker
    prep_path = context.artifacts.get_path("positive", "visual_prep", f"{norm_doc}.json", iteration=None)
    if prep_path.exists():
        prep_path.unlink()
        logger.info(f"Invalidated cached visual prep marker: {prep_path}")

    # Remove residual report artifact
    residual_path = context.artifacts.get_path("positive", "residual", f"{norm_doc}.json", iteration=None)
    if residual_path.exists():
        residual_path.unlink()
        logger.info(f"Invalidated cached residual report artifact: {residual_path}")

    # Remove positive rows artifact
    pos_row_path = context.artifacts.get_path("positive", "rows", f"{norm_doc}.json", iteration=None)
    if pos_row_path.exists():
        pos_row_path.unlink()
        logger.info(f"Invalidated cached positive rows artifact: {pos_row_path}")

    # Remove positive columns artifacts
    pos_cols_dir = context.artifacts.get_phase_dir("positive", "columns", iteration=None)
    if pos_cols_dir.exists():
        for f in pos_cols_dir.glob(f"{norm_doc}_row_*.json"):
            f.unlink()
            logger.info(f"Invalidated cached positive column artifact: {f}")

    # Remove negative rows artifact
    neg_row_path = context.artifacts.get_path("negative", "rows", f"{norm_doc}.json", iteration=None)
    if neg_row_path.exists():
        neg_row_path.unlink()
        logger.info(f"Invalidated cached negative rows artifact: {neg_row_path}")

    # Remove negative columns artifacts
    neg_cols_dir = context.artifacts.get_phase_dir("negative", "columns", iteration=None)
    if neg_cols_dir.exists():
        for f in neg_cols_dir.glob(f"{norm_doc}_row_*.json"):
            f.unlink()
            logger.info(f"Invalidated cached negative column artifact: {f}")


def run_consolidation(context: RunContext) -> bool:
    """Execute Phase 2.1 and 2.2 Positive Consolidation.

    Returns:
        needs_recompute (bool): True if human expert modified Ground Truth or .md during conflict resolution.
    """
    logger.info("Starting Phase 2: Positive Consolidation")

    needs_recompute_total = False
    gt_path = context.active_gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"Active Ground Truth CSV not found at {gt_path}")
    df_gt = pd.read_csv(gt_path)
    df_gt.columns = df_gt.columns.str.lower().str.strip()
    id_col = _get_id_column(df_gt)

    schema_fields = context.schema_fields
    fields = [col for col in df_gt.columns if col in schema_fields]

    active_norm_ids = {normalize_document_key(d) for d in context.doc_ids} if context.doc_ids is not None else set()

    # 1. Phase 2.1: Row Consolidation
    logger.info("Executing Phase 2.1: Row Consolidation")
    iter_num = context.current_iteration
    row_consolidated_path = context.artifacts.get_path("consolidation", "", "rows.json", iteration=iter_num)

    all_row_analyses = []
    rows_dir = context.artifacts.get_phase_dir("positive", "rows", iteration=None)
    if rows_dir.exists():
        for file_path in sorted(rows_dir.glob("*.json")):
            stem = file_path.stem
            if active_norm_ids and stem not in active_norm_ids:
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                analysis = PositiveRowsOutput.model_validate_json(content)
                for r in analysis.rows:
                    if context.rulings_store.is_ignored(stem, r.row_id):
                        continue
                    r_dict = r.model_dump()
                    r_dict["document_id"] = stem
                    all_row_analyses.append(r_dict)
            except Exception as e:
                logger.warning(f"Failed to read row analysis from {file_path}: {e}")

    def compute_row_consolidation():
        if not all_row_analyses:
            logger.info("No row analyses found. Creating default consolidated rows output.")
            return ConsolidatedRowsOutput(
                Row_Instantiation_Trigger_generalized="Single target entity extraction",
                Entity_Role_and_Filtration_Policy="Extract target objects with confirmed properties.",
            )

        row_predictor = dspy.Predict(RowConsolidation)
        analyses_str = json.dumps(all_row_analyses, ensure_ascii=False, indent=2)
        expert_rulings = context.rulings_store.format_rulings_for_prompt("rows")

        logger.info("Calling RowConsolidation LLM")
        response = row_predictor(
            row_analyses=analyses_str,
            baseline_prompt=context.baseline_prompt,
            extraction_schema=context.schema,
            expert_rulings=expert_rulings,
        )
        return response.consolidation

    force_recompute = not getattr(context, "resume", False)
    tier_ctx_fn = getattr(context, "tier_4a_context", None)
    tier_ctx = tier_ctx_fn() if callable(tier_ctx_fn) else contextlib.nullcontext()

    with tier_ctx:
        consolidated_rows = context.artifacts.load_or_compute(
            row_consolidated_path, compute_row_consolidation, ConsolidatedRowsOutput, force_recompute=force_recompute
        )

        all_conflicts = list(consolidated_rows.Conflicts)

        # 2. Phase 2.2: Column Consolidation
        logger.info("Executing Phase 2.2: Column Consolidation")
        col_predictor = dspy.Predict(ColumnConsolidation)

        all_field_analyses = []
        cols_dir = context.artifacts.get_phase_dir("positive", "columns", iteration=None)
        if cols_dir.exists():
            for file_path in sorted(cols_dir.glob("*.json")):
                stem = file_path.stem
                doc_part = stem.rsplit("_row_", 1)[0] if "_row_" in stem else stem
                row_id_part = f"row_{stem.rsplit('_row_', 1)[1]}" if "_row_" in stem else "row_1"
                if active_norm_ids and doc_part not in active_norm_ids:
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8")
                    analysis = PositiveColumnsOutput.model_validate_json(content)
                    for f in analysis.fields:
                        if context.rulings_store.is_ignored(doc_part, row_id_part, f.field_name):
                            continue
                        f_dict = f.model_dump()
                        f_dict["document_id"] = doc_part
                        f_dict["row_id"] = row_id_part
                        all_field_analyses.append(f_dict)
                except Exception as e:
                    logger.warning(f"Failed to read column analysis from {file_path}: {e}")

        for field in fields:
            field_lower = field.lower().strip()
            field_entries = [
                fa for fa in all_field_analyses if fa.get("field_name", "").lower().strip() == field_lower
            ]

            col_consolidated_path = context.artifacts.get_path(
                "consolidation", "columns", f"{field}.json", iteration=iter_num
            )

            def compute_col_consolidation(f_name=field, entries=field_entries):
                if not entries:
                    logger.info(f"No field analyses found for field {f_name}. Using default.")
                    return ConsolidatedColumnsOutput(
                        Semantic_Core_generalized="Not specified in training data",
                        Analytical_and_Binding_Policy="Direct extraction if mentioned.",
                    )

                entries_str = json.dumps(entries, ensure_ascii=False, indent=2)
                expert_rulings = context.rulings_store.format_rulings_for_prompt(f"column:{f_name}")
                logger.info(f"Calling ColumnConsolidation LLM for field {f_name}")
                response = col_predictor(
                    field_analyses=entries_str,
                    baseline_prompt=context.baseline_prompt,
                    extraction_schema=context.schema,
                    field_name=f_name,
                    expert_rulings=expert_rulings,
                )
                return response.consolidation

            consolidated_col = context.artifacts.load_or_compute(
                col_consolidated_path, compute_col_consolidation, ConsolidatedColumnsOutput, force_recompute=force_recompute
            )

            if consolidated_col.Conflicts:
                all_conflicts.extend(consolidated_col.Conflicts)

    all_conflicts = filter_equivalent_conflicts(all_conflicts, context.schema)
    if all_conflicts:
        logger.info(f"Consolidation produced {len(all_conflicts)} conflicts to resolve in batch.")
        if hasattr(context.resolver, "resolve_batch"):
            batch_res = context.resolver.resolve_batch(all_conflicts)
            if batch_res.needs_doc_recompute:
                for doc_id in batch_res.recompute_doc_ids:
                    _invalidate_doc_artifacts(context, doc_id)
                    if hasattr(context.gt_editor, "reindex_doc_rows"):
                        context.gt_editor.reindex_doc_rows(doc_id)
                needs_recompute_total = True
        else:
            for conflict in all_conflicts:
                _, recompute = context.resolver.resolve(conflict)
                if recompute:
                    _invalidate_doc_artifacts(context, conflict.document_id)
                    if hasattr(context.gt_editor, "reindex_doc_rows"):
                        context.gt_editor.reindex_doc_rows(conflict.document_id)
                    needs_recompute_total = True

    logger.info("Finished Phase 2: Positive Consolidation")
    return needs_recompute_total
