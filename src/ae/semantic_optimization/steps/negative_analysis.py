# src/ae/reverse_engineering/steps/negative_analysis.py
"""Phase 3: Negative Analysis (Rows and Columns) step implementation."""

import contextlib
import json
import logging
import re
from pathlib import Path
import pandas as pd
import dspy

from ae.core.storage.documents import DocumentRepository
from ae.core.utils import normalize_document_key
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.models import (
    ConsolidatedColumnsOutput,
    ConsolidatedRowsOutput,
    NegativeColumnsOutput,
    NegativeRowsOutput,
    PositiveColumnsOutput,
    PositiveRowsOutput,
)
from ae.semantic_optimization.signatures import NegativeColumnAnalysis, NegativeRowAnalysis
from ae.semantic_optimization.steps.positive_analysis import (
    are_field_values_equivalent,
    prepare_prompt_table,
    select_representative_rows,
)
from ae.semantic_optimization.steps.visual_prep import prepare_document_visuals
from ae.semantic_optimization.hitl import _get_id_column

logger = logging.getLogger(__name__)


def filter_equivalent_negative_candidates(
    output: NegativeColumnsOutput,
    schema_yaml: str = "",
) -> NegativeColumnsOutput:
    """Drop negative column candidates that normalize or canonicalize to the exact GT value."""
    for field_res in output.fields:
        kept = []
        for cand in field_res.candidates:
            cand_val = (cand.candidate_value or "").strip() or (cand.raw_text or "").strip()
            if are_field_values_equivalent(
                field_res.field_name,
                cand_val,
                cand.ground_truth,
                schema_yaml,
            ):
                continue
            kept.append(cand)
        field_res.candidates = kept
    return output


def _unpack_manifests_for_negative(
    norm_doc_id: str,
    raw_text: str,
    gt_table_csv: str,
    generalized_trigger: str,
    generalized_cores_str: str,
    context: RunContext,
    doc_repo: DocumentRepository,
    force_recompute: bool = False,
) -> str:
    """Ensure Tier 1 visuals are unpacked in full for Negative Analysis (no-op if already prepared in 1A)."""
    extra_ctx = (
        f"Generalized Row Trigger:\n{generalized_trigger}\n\n"
        f"Generalized Column Cores:\n{generalized_cores_str}"
    )
    return prepare_document_visuals(
        norm_doc_id=norm_doc_id,
        raw_text=raw_text,
        gt_table_csv=gt_table_csv,
        context=context,
        doc_repo=doc_repo,
        force_recompute=force_recompute,
        extra_context=extra_ctx,
    )


def _resolve_doc_triggers_and_cores(
    context: RunContext, norm_doc_id: str, fields: list[str]
) -> tuple[str, str]:
    """Resolve row trigger and column semantic cores for negative analysis directly from step 1A positive artifacts (with fallback to consolidated/generalized artifacts if present)."""
    iter_num = context.current_iteration

    # 1. Resolve row trigger & filtration boundary from 1A positive row analysis (grouped & deduplicated)
    trigger_parts: list[str] = []
    pos_row_path = context.artifacts.get_path("positive", "rows", f"{norm_doc_id}.json", iteration=None)
    if pos_row_path.exists():
        try:
            pos_rows = PositiveRowsOutput.model_validate_json(pos_row_path.read_text(encoding="utf-8"))
            grouped_triggers: dict[tuple[str, str, str], list[str]] = {}
            seen_trigger_norms: dict[tuple[str, str, str], set[str]] = {}
            for r in pos_rows.rows:
                cls_key = (
                    r.Row_Instantiation_Trigger_class,
                    r.Entity_System_Role_class,
                    r.Entity_Filtration_Boundary_class,
                )
                t_desc = (r.Row_Instantiation_Trigger_description or "").strip()
                b_desc = (r.Entity_Filtration_Boundary_description or "").strip()
                norm_pair = f"{t_desc.lower()}||{b_desc.lower()}"
                bucket = grouped_triggers.setdefault(cls_key, [])
                seen = seen_trigger_norms.setdefault(cls_key, set())
                if norm_pair in seen or len(bucket) >= 2:
                    continue
                seen.add(norm_pair)
                cls_label = " / ".join(cls_key)
                entry = f"[{cls_label}] Trigger: {t_desc}"
                if b_desc:
                    entry += f"; Boundary: {b_desc}"
                bucket.append(entry)

            for bucket in grouped_triggers.values():
                trigger_parts.extend(bucket)
        except Exception as e:
            logger.debug(f"Could not read positive rows for {norm_doc_id}: {e}")

    # Optional fallback to consolidated/generalized rows if available
    if not trigger_parts:
        for phase in ("consolidation", "generalization"):
            cand_path = context.artifacts.get_path(phase, "", "rows.json", iteration=iter_num)
            if not cand_path.exists():
                cand_path = context.artifacts.get_path(phase, "", "rows.json", iteration=None)
            if cand_path.exists():
                try:
                    data = json.loads(cand_path.read_text(encoding="utf-8"))
                    trig = data.get("Row_Instantiation_Trigger_generalized", "")
                    if trig:
                        trigger_parts.append(trig)
                        break
                except Exception:
                    pass

    generalized_trigger = (
        " | ".join(trigger_parts)
        if trigger_parts
        else "Target entity extraction according to schema and baseline prompt"
    )

    # 2. Resolve column semantic cores from 1A positive column analyses (deduplicated & capped)
    doc_cores: dict[str, list[str]] = {f: [] for f in fields}
    seen_core_norms: dict[str, set[str]] = {f: set() for f in fields}
    pos_cols_dir = context.artifacts.get_phase_dir("positive", "columns", iteration=None)
    if pos_cols_dir.exists():
        for col_file in sorted(pos_cols_dir.glob(f"{norm_doc_id}_row_*.json")):
            try:
                pos_col = PositiveColumnsOutput.model_validate_json(col_file.read_text(encoding="utf-8"))
                for fa in pos_col.fields:
                    f_key = fa.field_name.lower().strip()
                    core = (fa.Semantic_Core or "").strip()
                    if not core or f_key not in doc_cores:
                        continue
                    core_norm = core.lower()
                    if core_norm not in seen_core_norms[f_key] and len(doc_cores[f_key]) < 3:
                        seen_core_norms[f_key].add(core_norm)
                        doc_cores[f_key].append(core)
            except Exception as e:
                logger.debug(f"Could not read positive column file {col_file}: {e}")

    generalized_cores: dict[str, str] = {}
    for field in fields:
        if doc_cores.get(field):
            generalized_cores[field] = " ; ".join(doc_cores[field])
            continue

        # Fallback to consolidated/generalized column artifact if available
        found_fallback = ""
        for phase in ("consolidation", "generalization"):
            col_path = context.artifacts.get_path(phase, "columns", f"{field}.json", iteration=iter_num)
            if not col_path.exists():
                col_path = context.artifacts.get_path(phase, "columns", f"{field}.json", iteration=None)
            if col_path.exists():
                try:
                    c_data = json.loads(col_path.read_text(encoding="utf-8"))
                    found_fallback = c_data.get("Semantic_Core_generalized", "")
                    if found_fallback:
                        break
                except Exception:
                    pass
        generalized_cores[field] = found_fallback or "Defined by extraction schema and baseline prompt"

    return generalized_trigger, json.dumps(generalized_cores, ensure_ascii=False, indent=2)


def run_negative_analysis(context: RunContext, force_doc_ids: list[str] | None = None) -> None:
    """Execute Phase 1B (formerly Phase 3) Negative Analysis right after Positive Analysis."""
    logger.info("Starting Negative Analysis (Document-Level Contrastive Pass 1B)")

    doc_repo = DocumentRepository(ingestion_dir=context.ingestion_dir)

    gt_path = context.active_gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"Active Ground Truth CSV not found at {gt_path}")
    df_gt = pd.read_csv(gt_path)
    df_gt.columns = df_gt.columns.str.lower().str.strip()
    id_col = _get_id_column(df_gt)

    schema_fields = context.schema_fields
    keep_cols = [col for col in df_gt.columns if col == id_col or col in schema_fields]
    df_gt = df_gt[keep_cols]

    # Normalize ID column for matching
    df_gt[id_col] = df_gt[id_col].apply(lambda x: normalize_document_key(str(x)) if pd.notna(x) else "")

    fields = [col for col in df_gt.columns if col in schema_fields]

    row_predictor = dspy.Predict(NegativeRowAnalysis)
    col_predictor = dspy.Predict(NegativeColumnAnalysis)

    tier_ctx_fn = getattr(context, "tier_4a_context", None)
    tier_ctx = tier_ctx_fn() if callable(tier_ctx_fn) else contextlib.nullcontext()

    target_doc_ids = force_doc_ids if force_doc_ids is not None else context.doc_ids
    raw_batch_size = getattr(context, "row_batch_size", 5)
    batch_size = max(1, raw_batch_size if isinstance(raw_batch_size, int) and raw_batch_size >= 1 else 5)
    raw_max_rows = getattr(context, "max_rows_per_doc", 20)
    max_rows = raw_max_rows if isinstance(raw_max_rows, int) and raw_max_rows >= 0 else 20

    with tier_ctx:
        for doc_id in target_doc_ids:
            norm_doc_id = normalize_document_key(doc_id)
            force_recompute = force_doc_ids is not None and doc_id in force_doc_ids
            logger.info(f"Processing negative analysis for document: {norm_doc_id} (force_recompute={force_recompute})")

            raw_text = doc_repo.get(norm_doc_id)
            if not raw_text:
                logger.warning(f"Parsed text for document {norm_doc_id} not found in {context.ingestion_dir}. Skipping.")
                continue

            df_doc = df_gt[df_gt[id_col] == norm_doc_id].copy()
            generalized_trigger, generalized_cores_str = _resolve_doc_triggers_and_cores(
                context, norm_doc_id, fields
            )

            # Step 1B.1: Negative Row Analysis (Gaps - always uses FULL gt_table_csv)
            row_artifact_path = context.artifacts.get_path("negative", "rows", f"{norm_doc_id}.json", iteration=None)

            df_doc_with_id = prepare_prompt_table(df_doc, schema_fields)
            gt_table_csv = df_doc_with_id.to_csv(index=False)

            def compute_negative_rows(trig=generalized_trigger, g_cores=generalized_cores_str):
                fresh_row_text = _unpack_manifests_for_negative(
                    norm_doc_id,
                    raw_text,
                    gt_table_csv,
                    trig,
                    g_cores,
                    context,
                    doc_repo,
                    force_recompute=False,
                )
                logger.info(f"Calling NegativeRowAnalysis LLM for {norm_doc_id}")
                response = row_predictor(
                    extraction_schema=context.schema,
                    baseline_prompt=context.baseline_prompt,
                    raw_text=fresh_row_text,
                    gt_table=gt_table_csv,
                    generalized_trigger=trig,
                )
                return response.analysis

            context.artifacts.load_or_compute(
                row_artifact_path, compute_negative_rows, NegativeRowsOutput, force_recompute=force_recompute
            )
            raw_text = doc_repo.get(norm_doc_id) or raw_text

            # Step 1B.2: Negative Column Analysis (with stratified sampling, row batching & prefix-cache field ordering)
            if df_doc.empty:
                continue

            raw_text = _unpack_manifests_for_negative(
                norm_doc_id,
                raw_text,
                gt_table_csv,
                generalized_trigger,
                generalized_cores_str,
                context,
                doc_repo,
                force_recompute=False,
            )
            sampled_cols_df = select_representative_rows(
                df_doc_with_id, raw_text, max_rows=max_rows
            )
            if len(sampled_cols_df) < len(df_doc_with_id):
                logger.info(
                    f"NegativeColumnAnalysis sampling for {norm_doc_id}: "
                    f"{len(sampled_cols_df)}/{len(df_doc_with_id)} representative rows selected"
                )

            pending_positions: list[int] = []
            for pos_idx in range(len(sampled_cols_df)):
                row_id = str(sampled_cols_df.iloc[pos_idx]["row_id"])
                col_artifact_path = context.artifacts.get_path(
                    "negative", "columns", f"{norm_doc_id}_{row_id}.json", iteration=None
                )
                if not force_recompute and context.artifacts.resume and col_artifact_path.exists():
                    context.artifacts.load_or_compute(
                        col_artifact_path,
                        lambda: NegativeColumnsOutput(fields=[]),
                        NegativeColumnsOutput,
                        force_recompute=False,
                    )
                else:
                    pending_positions.append(pos_idx)

            for start_idx in range(0, len(pending_positions), batch_size):
                batch_positions = pending_positions[start_idx : start_idx + batch_size]
                batch_df = sampled_cols_df.iloc[batch_positions]
                batch_row_ids = [str(r) for r in batch_df["row_id"].tolist()]
                target_gt_rows_csv = batch_df.to_csv(index=False)

                if len(batch_positions) == 1:
                    single_row_id = batch_row_ids[0]
                    col_artifact_path = context.artifacts.get_path(
                        "negative", "columns", f"{norm_doc_id}_{single_row_id}.json", iteration=None
                    )

                    def compute_single_negative_column(
                        t_row=target_gt_rows_csv, r_id=single_row_id, g_cores=generalized_cores_str
                    ):
                        fresh_col_text = doc_repo.get(norm_doc_id) or raw_text
                        logger.info(f"Calling NegativeColumnAnalysis LLM for {norm_doc_id} {r_id}")
                        response = col_predictor(
                            extraction_schema=context.schema,
                            baseline_prompt=context.baseline_prompt,
                            raw_text=fresh_col_text,
                            gt_table=gt_table_csv,
                            generalized_core=g_cores,
                            target_gt_row=t_row,
                        )
                        return filter_equivalent_negative_candidates(response.analysis, context.schema)

                    context.artifacts.load_or_compute(
                        col_artifact_path,
                        compute_single_negative_column,
                        NegativeColumnsOutput,
                        force_recompute=True,
                    )
                else:
                    fresh_col_text = doc_repo.get(norm_doc_id) or raw_text
                    logger.info(
                        f"Calling NegativeColumnAnalysis LLM for {norm_doc_id} batch {batch_row_ids}"
                    )
                    response = col_predictor(
                        extraction_schema=context.schema,
                        baseline_prompt=context.baseline_prompt,
                        raw_text=fresh_col_text,
                        gt_table=gt_table_csv,
                        generalized_core=generalized_cores_str,
                        target_gt_row=target_gt_rows_csv,
                    )
                    batch_neg_output: NegativeColumnsOutput = filter_equivalent_negative_candidates(
                        response.analysis, context.schema
                    )

                    fields_by_row: dict[str, list] = {r_id: [] for r_id in batch_row_ids}
                    for nf in batch_neg_output.fields:
                        raw_r_id = getattr(nf, "row_id", "") or ""
                        if raw_r_id in fields_by_row:
                            target_r_id = raw_r_id
                        else:
                            m = re.search(r"(\d+)", str(raw_r_id))
                            if m:
                                idx = int(m.group(1))
                                if f"row_{idx}" in fields_by_row:
                                    target_r_id = f"row_{idx}"
                                elif 1 <= idx <= len(batch_row_ids):
                                    target_r_id = batch_row_ids[idx - 1]
                                else:
                                    target_r_id = batch_row_ids[0]
                            else:
                                target_r_id = batch_row_ids[0]
                        fields_by_row[target_r_id].append(nf)

                    for r_id in batch_row_ids:
                        col_artifact_path = context.artifacts.get_path(
                            "negative", "columns", f"{norm_doc_id}_{r_id}.json", iteration=None
                        )
                        row_neg_output = NegativeColumnsOutput(fields=fields_by_row[r_id])
                        context.artifacts.load_or_compute(
                            col_artifact_path,
                            lambda out=row_neg_output: out,
                            NegativeColumnsOutput,
                            force_recompute=True,
                        )

    logger.info("Finished Negative Analysis")
