# src/ae/reverse_engineering/steps/consolidation.py
"""Phase 2: Positive Consolidation step implementation."""

import logging
import json
import pandas as pd
import dspy
from pathlib import Path

from ae.reverse_engineering.context import RunContext
from ae.reverse_engineering.models import (
    PositiveRowsOutput,
    PositiveColumnsOutput,
    ConsolidatedRowsOutput,
    ConsolidatedColumnsOutput,
)
from ae.reverse_engineering.signatures import RowConsolidation, ColumnConsolidation
from ae.reverse_engineering.steps.positive_analysis import _get_id_column

logger = logging.getLogger(__name__)

def run_consolidation(context: RunContext) -> None:
    """Execute Phase 2.1 and 2.2 Positive Consolidation."""
    logger.info("Starting Phase 2: Positive Consolidation")

    # 1. Load Ground Truth columns to identify fields
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

    # 2. Phase 2.1: Row Consolidation
    logger.info("Executing Phase 2.1: Row Consolidation")
    row_consolidated_path = context.artifacts.get_path("consolidation", "", "rows.json")

    # Load all positive row analysis results
    all_row_analyses = []
    rows_dir = context.artifacts.get_phase_dir("positive", "rows")
    if rows_dir.exists():
        for file_path in rows_dir.glob("*.json"):
            try:
                content = file_path.read_text(encoding="utf-8")
                analysis = PositiveRowsOutput.model_validate_json(content)
                all_row_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to read row analysis from {file_path}: {e}")

    # Gather trigger descriptions
    trigger_descriptions = []
    for analysis in all_row_analyses:
        for row in analysis.rows:
            trigger_descriptions.append(row.Row_Instantiation_Trigger_description)

    def compute_row_consolidation():
        if not trigger_descriptions:
            logger.info("No row trigger descriptions found. Creating default consolidated rows output.")
            return ConsolidatedRowsOutput(Row_Instantiation_Trigger_generalized="Single target entity extraction")

        row_predictor = dspy.Predict(RowConsolidation)
        trigger_analyses_str = json.dumps(trigger_descriptions, ensure_ascii=False, indent=2)

        logger.info("Calling RowConsolidation LLM")
        response = row_predictor(
            row_trigger_analyses=trigger_analyses_str,
            baseline_prompt=context.baseline_prompt,
            extraction_schema=context.schema
        )
        return response.consolidation

    # Run consolidation and load/save
    consolidated_rows = context.artifacts.load_or_compute(
        row_consolidated_path, compute_row_consolidation, ConsolidatedRowsOutput
    )

    # 3. Phase 2.2: Column Consolidation
    logger.info("Executing Phase 2.2: Column Consolidation")
    col_predictor = dspy.Predict(ColumnConsolidation)

    # Load all positive column analysis files
    all_field_analyses = []
    cols_dir = context.artifacts.get_phase_dir("positive", "columns")
    if cols_dir.exists():
        for file_path in cols_dir.glob("*.json"):
            try:
                content = file_path.read_text(encoding="utf-8")
                analysis = PositiveColumnsOutput.model_validate_json(content)
                all_field_analyses.extend(analysis.fields)
            except Exception as e:
                logger.warning(f"Failed to read column analysis from {file_path}: {e}")

    for field in fields:
        # Filter analyses for this field
        field_cores = [
            fa.Semantic_Core for fa in all_field_analyses if fa.field_name.lower().strip() == field.lower().strip()
        ]

        col_consolidated_path = context.artifacts.get_path("consolidation", "columns", f"{field}.json")

        def compute_col_consolidation(f_name=field, cores=field_cores):
            if not cores:
                logger.info(f"No semantic cores found for field {f_name}. Using default.")
                return ConsolidatedColumnsOutput(Semantic_Core_generalized="Not specified in training data")

            cores_str = json.dumps(cores, ensure_ascii=False, indent=2)
            logger.info(f"Calling ColumnConsolidation LLM for field {f_name}")
            response = col_predictor(
                semantic_core_analyses=cores_str,
                baseline_prompt=context.baseline_prompt,
                extraction_schema=context.schema,
                field_name=f_name
            )
            return response.consolidation

        context.artifacts.load_or_compute(
            col_consolidated_path, compute_col_consolidation, ConsolidatedColumnsOutput
        )

    logger.info("Finished Phase 2: Positive Consolidation")
