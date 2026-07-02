# src/ae/reverse_engineering/steps/negative_analysis.py
"""Phase 3: Negative Analysis (Rows and Columns) step implementation."""

import logging
import json
import pandas as pd
import dspy
from pathlib import Path

from ae.core.utils import normalize_document_key
from ae.core.storage.documents import DocumentRepository
from ae.reverse_engineering.context import RunContext
from ae.reverse_engineering.models import (
    ConsolidatedRowsOutput,
    ConsolidatedColumnsOutput,
    NegativeRowsOutput,
    NegativeColumnsOutput,
)
from ae.reverse_engineering.signatures import NegativeRowAnalysis, NegativeColumnAnalysis
from ae.reverse_engineering.steps.positive_analysis import _get_id_column, prepare_prompt_table

logger = logging.getLogger(__name__)

def run_negative_analysis(context: RunContext) -> None:
    """Execute Phase 3.1 and 3.2 Negative Analysis."""
    logger.info("Starting Phase 3: Negative Analysis")

    doc_repo = DocumentRepository(ingestion_dir=context.ingestion_dir)

    # Load and normalize ground truth CSV
    if not context.gt_path.exists():
        raise FileNotFoundError(f"Ground Truth CSV not found at {context.gt_path}")
    df_gt = pd.read_csv(context.gt_path)
    df_gt.columns = df_gt.columns.str.lower().str.strip()
    id_col = _get_id_column(df_gt)

    # Filter columns to only include id_col and those defined in the schema
    schema_fields = context.schema_fields
    keep_cols = [col for col in df_gt.columns if col == id_col or col in schema_fields]
    df_gt = df_gt[keep_cols]
    fields = [col for col in df_gt.columns if col in schema_fields]

    # Normalize ID column for matching
    df_gt[id_col] = df_gt[id_col].apply(lambda x: normalize_document_key(str(x)) if pd.notna(x) else "")

    # Load generalized row instantiation trigger
    rows_consolidated_path = context.artifacts.get_path("consolidation", "", "rows.json")
    if not rows_consolidated_path.exists():
        raise FileNotFoundError(f"Consolidated rows artifact not found at {rows_consolidated_path}. Please run Phase 2 first.")
    
    consolidated_rows = ConsolidatedRowsOutput.model_validate_json(
        rows_consolidated_path.read_text(encoding="utf-8")
    )
    generalized_trigger = consolidated_rows.Row_Instantiation_Trigger_generalized

    # Load generalized column cores for all fields
    generalized_cores = {}
    for field in fields:
        col_consolidated_path = context.artifacts.get_path("consolidation", "columns", f"{field}.json")
        if col_consolidated_path.exists():
            try:
                consolidated_col = ConsolidatedColumnsOutput.model_validate_json(
                    col_consolidated_path.read_text(encoding="utf-8")
                )
                generalized_cores[field] = consolidated_col.Semantic_Core_generalized
            except Exception as e:
                logger.warning(f"Failed to read consolidated column for {field}: {e}")
        else:
            generalized_cores[field] = "Not specified in training data"

    generalized_cores_str = json.dumps(generalized_cores, ensure_ascii=False, indent=2)

    # Initialize predictors
    row_predictor = dspy.Predict(NegativeRowAnalysis)
    col_predictor = dspy.Predict(NegativeColumnAnalysis)

    for doc_id in context.doc_ids:
        norm_doc_id = normalize_document_key(doc_id)
        logger.info(f"Processing negative analysis for document: {norm_doc_id}")

        # Get document text
        raw_text = doc_repo.get(norm_doc_id)
        if not raw_text:
            logger.warning(f"Parsed text for document {norm_doc_id} not found in {context.ingestion_dir}. Skipping.")
            continue

        # Get GT rows for this document
        df_doc = df_gt[df_gt[id_col] == norm_doc_id].copy()

        # Step 3.1: Negative Row Analysis (Gaps)
        row_artifact_path = context.artifacts.get_path("negative", "rows", f"{norm_doc_id}.json")

        # Prepare doc GT with row_id and only schema fields
        df_doc_with_id = prepare_prompt_table(df_doc, schema_fields)
        gt_table_csv = df_doc_with_id.to_csv(index=False)

        def compute_negative_rows():
            logger.info(f"Calling NegativeRowAnalysis LLM for {norm_doc_id}")
            response = row_predictor(
                raw_text=raw_text,
                gt_table=gt_table_csv,
                baseline_prompt=context.baseline_prompt,
                extraction_schema=context.schema,
                generalized_trigger=generalized_trigger
            )
            return response.analysis

        context.artifacts.load_or_compute(
            row_artifact_path, compute_negative_rows, NegativeRowsOutput
        )

        # Step 3.2: Negative Column Analysis
        if df_doc.empty:
            continue

        for i in range(len(df_doc_with_id)):
            row_id = f"row_{i+1}"
            col_artifact_path = context.artifacts.get_path("negative", "columns", f"{norm_doc_id}_{row_id}.json")

            target_row_df = df_doc_with_id.iloc[[i]]
            target_gt_row = target_row_df.to_csv(index=False)

            def compute_negative_columns():
                logger.info(f"Calling NegativeColumnAnalysis LLM for {norm_doc_id} row {row_id}")
                response = col_predictor(
                    raw_text=raw_text,
                    gt_table=gt_table_csv,
                    target_gt_row=target_gt_row,
                    baseline_prompt=context.baseline_prompt,
                    extraction_schema=context.schema,
                    generalized_core=generalized_cores_str
                )
                return response.analysis

            context.artifacts.load_or_compute(
                col_artifact_path, compute_negative_columns, NegativeColumnsOutput
            )

    logger.info("Finished Phase 3: Negative Analysis")
