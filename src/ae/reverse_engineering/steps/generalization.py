# src/ae/reverse_engineering/steps/generalization.py
"""Phase 4: Generalization step implementation."""

import logging
import json
import pandas as pd
import dspy
from pathlib import Path

from ae.reverse_engineering.context import RunContext
from ae.reverse_engineering.models import (
    GeneralizedRowsOutput,
    GeneralizedColumnsOutput,
)
from ae.reverse_engineering.signatures import RowGeneralization, ColumnGeneralization
from ae.reverse_engineering.steps.positive_analysis import _get_id_column

logger = logging.getLogger(__name__)

def run_generalization(context: RunContext) -> None:
    """Execute Phase 4.1 and 4.2 Generalization."""
    logger.info("Starting Phase 4: Generalization")

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

    # 1. Step 4.1: Row Generalization
    logger.info("Executing Phase 4.1: Row Generalization")
    row_rules_path = context.artifacts.get_path("generalization", "", "rows.json")

    # Load positive row analyses
    pos_rows = []
    pos_rows_dir = context.artifacts.get_phase_dir("positive", "rows")
    if pos_rows_dir.exists():
        for file_path in pos_rows_dir.glob("*.json"):
            try:
                doc_id = file_path.stem
                content = json.loads(file_path.read_text(encoding="utf-8"))
                for row in content.get("rows", []):
                    row["document"] = doc_id
                    pos_rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to read row analysis {file_path}: {e}")

    # Load negative row analyses
    neg_rows = []
    neg_rows_dir = context.artifacts.get_phase_dir("negative", "rows")
    if neg_rows_dir.exists():
        for file_path in neg_rows_dir.glob("*.json"):
            try:
                doc_id = file_path.stem
                content = json.loads(file_path.read_text(encoding="utf-8"))
                for row in content.get("gap_rows", []):
                    row["document"] = doc_id
                    neg_rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to read negative row analysis {file_path}: {e}")

    def compute_row_generalization():
        row_predictor = dspy.Predict(RowGeneralization)
        pos_str = json.dumps(pos_rows, ensure_ascii=False, indent=2)
        neg_str = json.dumps(neg_rows, ensure_ascii=False, indent=2)

        logger.info("Calling RowGeneralization LLM")
        response = row_predictor(
            positive_rows_analyses=pos_str,
            negative_rows_analyses=neg_str,
            baseline_prompt=context.baseline_prompt,
            extraction_schema=context.schema
        )
        return response.generalization

    row_data = context.artifacts.load_or_compute(
        row_rules_path, compute_row_generalization, GeneralizedRowsOutput
    )

    # 2. Step 4.2: Column Generalization
    logger.info("Executing Phase 4.2: Column Generalization")
    col_predictor = dspy.Predict(ColumnGeneralization)

    # Load positive column analyses
    pos_cols = []
    pos_cols_dir = context.artifacts.get_phase_dir("positive", "columns")
    if pos_cols_dir.exists():
        for file_path in pos_cols_dir.glob("*.json"):
            try:
                # filename format: {doc_id}_{row_id}.json
                stem = file_path.stem
                if "_" in stem:
                    parts = stem.split("_")
                    doc_id = "_".join(parts[:-1])
                    row_id = parts[-1]
                else:
                    doc_id = stem
                    row_id = "unknown"

                content = json.loads(file_path.read_text(encoding="utf-8"))
                for field_analysis in content.get("fields", []):
                    field_analysis["document"] = doc_id
                    field_analysis["row_id"] = row_id
                    pos_cols.append(field_analysis)
            except Exception as e:
                logger.warning(f"Failed to read positive column analysis {file_path}: {e}")

    # Load negative column analyses
    neg_cols = []
    neg_cols_dir = context.artifacts.get_phase_dir("negative", "columns")
    if neg_cols_dir.exists():
        for file_path in neg_cols_dir.glob("*.json"):
            try:
                stem = file_path.stem
                if "_" in stem:
                    parts = stem.split("_")
                    doc_id = "_".join(parts[:-1])
                    row_id = parts[-1]
                else:
                    doc_id = stem
                    row_id = "unknown"

                content = json.loads(file_path.read_text(encoding="utf-8"))
                for field_result in content.get("fields", []):
                    field_name = field_result.get("field_name", "")
                    for candidate in field_result.get("candidates", []):
                        candidate["field_name"] = field_name
                        candidate["document"] = doc_id
                        candidate["row_id"] = row_id
                        neg_cols.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to read negative column analysis {file_path}: {e}")

    # Process each field
    compiled_columns = {}
    for field in fields:
        field_lower = field.lower().strip()
        # Filter positive analyses for this field
        pos_field_analyses = [pa for pa in pos_cols if pa.get("field_name", "").lower().strip() == field_lower]
        # Filter negative candidates for this field
        neg_field_candidates = [na for na in neg_cols if na.get("field_name", "").lower().strip() == field_lower]

        col_rules_path = context.artifacts.get_path("generalization", "columns", f"{field}.json")

        def compute_column_generalization(f_name=field, pos=pos_field_analyses, neg=neg_field_candidates):
            pos_str = json.dumps(pos, ensure_ascii=False, indent=2)
            neg_str = json.dumps(neg, ensure_ascii=False, indent=2)

            logger.info(f"Calling ColumnGeneralization LLM for field {f_name}")
            response = col_predictor(
                positive_columns_analyses=pos_str,
                negative_columns_analyses=neg_str,
                baseline_prompt=context.baseline_prompt,
                extraction_schema=context.schema,
                field_name=f_name
            )
            return response.generalization

        col_data = context.artifacts.load_or_compute(
            col_rules_path, compute_column_generalization, GeneralizedColumnsOutput
        )
        compiled_columns[field] = col_data

    # 3. Final Prompt Assembly (formerly Phase 5.3)
    logger.info("Executing Final Prompt Assembly")

    # Read baseline instruction text
    baseline_text = context.baseline_prompt

    # Try parsing baseline text into sections split by "---"
    sections = baseline_text.split("---")

    preamble = ""
    reasoning_style = ""

    # Locate preamble (everything before the first "---" and the policy section)
    if len(sections) >= 2:
        preamble = sections[0].strip() + "\n\n---\n\n" + sections[1].strip()
    else:
        # Fallback if baseline text is unexpected
        preamble = (
            "You are an information extraction system specialized in nanozyme research. "
            "Your task is to extract structured experimental data from scientific articles.\n\n"
            "Each experiment must be represented as a separate JSON object."
        )

    # Locate reasoning style section if it exists in the baseline
    for section in sections:
        if "REASONING STYLE" in section:
            reasoning_style = section.strip()
            break

    if not reasoning_style:
        # Fallback reasoning style
        reasoning_style = (
            "REASONING STYLE: STRICTLY CONCISE\n"
            "1. No rule repetition: Do NOT restate or explain the extraction policies. Assume they are understood.\n"
            "2. Direct extraction: Go straight to the values found. Do not perform a \"self-check\" commentary.\n"
            "3. Format: For each experiment, use a short bulleted list:\n"
            "   - Source: (e.g., \"Table 1\")\n"
            "   - Key Values: (e.g., \"Km=5.2, Vmax=10. Activity=oxidase.\")\n"
            "   - Nulls: (e.g., \"Temperature: null (not mentioned)\")\n"
            "4. Brevity: Limit reasoning to max 3-4 lines per experiment."
        )

    # Reconstruct the prompt
    assembled_prompt_parts = []

    # Add preamble
    assembled_prompt_parts.append(preamble)

    # Add compiled Experiment Selection rules
    assembled_prompt_parts.append(
        "EXPERIMENT SELECTION\n\n"
        "Include ONLY experiments satisfying:\n" +
        "\n".join(f"- {inst}" for inst in row_data.Instructions.Row_Inclusion_Instructions) +
        "\n\nExclude:\n" +
        "\n".join(f"- {inst}" for inst in row_data.Instructions.Row_Exclusion_Instructions)
    )

    # Add compiled rules for each field
    for field in fields:
        if field not in compiled_columns:
            continue
        cc = compiled_columns[field]

        field_section = f"{field.upper()}\n\n"

        if cc.Instructions.Column_Inclusion_Instructions:
            field_section += "Extract ONLY if satisfy:\n" + \
                             "\n".join(f"- {inst}" for inst in cc.Instructions.Column_Inclusion_Instructions) + "\n\n"

        if cc.Instructions.Column_Exclusion_Instructions:
            field_section += "DO NOT extract:\n" + \
                             "\n".join(f"- {inst}" for inst in cc.Instructions.Column_Exclusion_Instructions) + "\n\n"

        if cc.Instructions.Transformation_Instructions:
            field_section += "Transformation rules:\n" + \
                             "\n".join(f"- {inst}" for inst in cc.Instructions.Transformation_Instructions) + "\n"

        assembled_prompt_parts.append(field_section.strip())

    # Add postamble (reasoning style)
    assembled_prompt_parts.append(reasoning_style)

    # Join everything with "---" separators
    final_prompt = "\n\n---\n\n".join(assembled_prompt_parts) + "\n"

    from ae.core.config.settings import Settings
    settings = Settings.load(load_env_file=False)
    generated_instruction_path = Path(settings.paths.generated_prompt_file).resolve()
    generated_instruction_path.parent.mkdir(parents=True, exist_ok=True)
    generated_instruction_path.write_text(final_prompt, encoding="utf-8")

    logger.info(f"Successfully assembled and saved generated instruction prompt to {generated_instruction_path}")

    # Generate and save agent JSON to data/processed/agents
    try:
        from ae.core.schema import load_schema_complete
        from ae.extraction.agent import UniversalExtractor
        from ae.core.storage.agents import save_agent, AgentMetadata
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"engineered_{timestamp}.json"

        logger.info(f"Generating agent JSON with the generated instruction: {filename}...")
        task = load_schema_complete(
            yaml_path=context.schema_path,
            instruction_path=generated_instruction_path,
        )

        agent = UniversalExtractor(task.signature)
        
        # Serialize the agent using a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            agent.save(temp_path)

        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                agent_dict = json.load(f)
        finally:
            import os
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        schema_hash = task.config.get_schema_hash() if hasattr(task.config, "get_schema_hash") else None
        instruction_hash = task.config.get_instruction_hash() if hasattr(task.config, "get_instruction_hash") else None
        
        metadata = AgentMetadata(
            created_at=datetime.now(),
            model_version=str(context.teacher_lm.model) if hasattr(context.teacher_lm, "model") else "unknown",
            metrics={},
            config_snapshot={},
            description=f"Generated agent via RE pipeline",
            initial_instruction_file=str(context.baseline_prompt_path),
            instruction_hash=instruction_hash,
            schema_hash=schema_hash,
        )

        agents_dir = Path(settings.paths.agents_dir)
        saved_agent_path = save_agent(
            agent=agent_dict,
            agents_dir=agents_dir,
            metadata=metadata,
            filename=filename,
        )
        logger.info(f"Successfully generated and saved agent JSON to {saved_agent_path}")
    except Exception as e:
        logger.error(f"Failed to generate and save agent JSON: {e}", exc_info=True)

    logger.info("Finished Phase 4: Generalization")

