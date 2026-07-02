# src/ae/reverse_engineering/steps/positive_analysis.py
"""Phase 1: Positive Analysis (Rows and Columns) step implementation."""

import logging
import pandas as pd
import dspy
from pathlib import Path

from ae.core.utils import normalize_document_key
from ae.core.storage.documents import DocumentRepository
from ae.reverse_engineering.context import RunContext
from ae.reverse_engineering.models import PositiveRowsOutput, PositiveColumnsOutput
from ae.reverse_engineering.signatures import PositiveRowAnalysis, PositiveColumnAnalysis

logger = logging.getLogger(__name__)

def _get_id_column(df: pd.DataFrame) -> str:
    """Identify the document ID column in Ground Truth DataFrame."""
    for col in ["pdf", "filename", "source", "doi", "document"]:
        if col in df.columns:
            return col
    raise ValueError(f"No valid document ID column found. Columns: {list(df.columns)}")

def prepare_prompt_table(df_doc: pd.DataFrame, schema_fields: list[str]) -> pd.DataFrame:
    """Prepare ground truth dataframe for prompts by keeping only schema fields and prepending row_id."""
    schema_cols = [col for col in df_doc.columns if col in schema_fields]
    df_clean = df_doc[schema_cols].copy()
    df_clean.insert(0, "row_id", [f"row_{i+1}" for i in range(len(df_clean))])
    return df_clean

def run_positive_analysis(context: RunContext) -> None:
    """Execute Phase 1.1 and 1.2 Positive Analysis for all train documents."""
    logger.info("Starting Phase 1: Positive Analysis")

    # Initialize document repository to load source text
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

    # Normalize ID column for matching
    df_gt[id_col] = df_gt[id_col].apply(lambda x: normalize_document_key(str(x)) if pd.notna(x) else "")

    # Initialize predictors
    row_predictor = dspy.Predict(PositiveRowAnalysis)
    col_predictor = dspy.Predict(PositiveColumnAnalysis)

    for doc_id in context.doc_ids:
        norm_doc_id = normalize_document_key(doc_id)
        logger.info(f"Processing positive analysis for document: {norm_doc_id}")

        # Get document text
        raw_text = doc_repo.get(norm_doc_id)
        if not raw_text:
            logger.warning(f"Parsed text for document {norm_doc_id} not found in {context.ingestion_dir}. Skipping.")
            continue

        # Get GT rows for this document
        df_doc = df_gt[df_gt[id_col] == norm_doc_id].copy()

        # Step 1.1: Positive Row Analysis
        row_artifact_path = context.artifacts.get_path("positive", "rows", f"{norm_doc_id}.json")

        def compute_rows():
            if df_doc.empty:
                logger.info(f"No Ground Truth rows found for document {norm_doc_id}. Creating empty RowAnalysis.")
                return PositiveRowsOutput(rows=[])

            # Prepare doc GT with row_id and only schema fields
            df_doc_with_id = prepare_prompt_table(df_doc, schema_fields)
            gt_table_csv = df_doc_with_id.to_csv(index=False)

            logger.info(f"Calling PositiveRowAnalysis LLM for {norm_doc_id}")
            response = row_predictor(
                raw_text=raw_text,
                gt_table=gt_table_csv,
                baseline_prompt=context.baseline_prompt,
                extraction_schema=context.schema
            )
            return response.analysis

        # Load from cache or call LLM
        row_analysis = context.artifacts.load_or_compute(
            row_artifact_path, compute_rows, PositiveRowsOutput
        )

        # Step 1.2: Positive Column (Cell) Analysis
        if df_doc.empty:
            continue

        # Prepare doc GT with row_id and only schema fields
        df_doc_with_id = prepare_prompt_table(df_doc, schema_fields)
        gt_table_csv = df_doc_with_id.to_csv(index=False)

        for i in range(len(df_doc_with_id)):
            row_id = f"row_{i+1}"
            col_artifact_path = context.artifacts.get_path("positive", "columns", f"{norm_doc_id}_{row_id}.json")

            target_row_df = df_doc_with_id.iloc[[i]]
            target_gt_row = target_row_df.to_csv(index=False)

            # Skip if row doesn't have any non-empty fields besides row_id
            non_empty_cols = [c for c in target_row_df.columns if c != "row_id" and pd.notna(target_row_df[c].values[0])]
            if not non_empty_cols:
                logger.info(f"Row {row_id} in document {norm_doc_id} has no populated fields. Skipping column analysis.")
                continue

            def compute_columns():
                logger.info(f"Calling PositiveColumnAnalysis LLM for {norm_doc_id} row {row_id}")
                response = col_predictor(
                    raw_text=raw_text,
                    gt_table=gt_table_csv,
                    target_gt_row=target_gt_row,
                    baseline_prompt=context.baseline_prompt,
                    extraction_schema=context.schema
                )
                return response.analysis

            context.artifacts.load_or_compute(
                col_artifact_path, compute_columns, PositiveColumnsOutput
            )

    logger.info("Finished Phase 1: Positive Analysis")
