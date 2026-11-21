# src/ae/reverse_engineering/steps/positive_analysis.py
"""Phase 1: Positive Analysis (Rows and Columns) step implementation."""

import logging
import math
import re
from pathlib import Path
import pandas as pd
import dspy
import yaml

from ae.core.schema.validators import (
    FieldValidatorSpec,
    compare_field_values_via_validator,
    normalize_validator_spec,
    supports_value_comparison,
)
from ae.core.utils import normalize_document_key
from ae.core.storage.documents import DocumentRepository
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.models import PositiveRowsOutput, PositiveColumnsOutput, REConflict
from ae.semantic_optimization.signatures import PositiveRowAnalysis, PositiveColumnAnalysis
from ae.semantic_optimization.hitl import _get_id_column
from ae.semantic_optimization.steps.visual_prep import prepare_document_visuals

logger = logging.getLogger(__name__)


def _unpack_manifests_for_positive(
    norm_doc_id: str,
    raw_text: str,
    gt_table_csv: str,
    context: RunContext,
    doc_repo: DocumentRepository,
    force_recompute: bool = False,
) -> str:
    """Prepare Tier 1 visuals in full for Contrastive Reverse Engineering (1A + 1B)."""
    return prepare_document_visuals(
        norm_doc_id=norm_doc_id,
        raw_text=raw_text,
        gt_table_csv=gt_table_csv,
        context=context,
        doc_repo=doc_repo,
        force_recompute=force_recompute,
    )


def prepare_prompt_table(df_doc: pd.DataFrame, schema_fields: list[str]) -> pd.DataFrame:
    """Prepare ground truth dataframe for prompts by keeping only schema fields and prepending row_id."""
    schema_cols = [col for col in df_doc.columns if col in schema_fields]
    df_clean = df_doc[schema_cols].copy()
    df_clean.insert(0, "row_id", [f"row_{i+1}" for i in range(len(df_clean))])
    return df_clean


def _normalize_cell_for_match(val: str) -> str:
    """Normalize surface formatting differences (quotes, unicode signs, decimal commas) for matching."""
    s = val.strip().strip("'").strip('"').strip()
    s = s.replace("−", "-").replace("–", "-").replace("⁻¹", "-1")
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    return s.lower()


def _is_numeric_like(val_norm: str) -> bool:
    """Return True if a normalized cell value represents a number with optional relation/error bounds."""
    return bool(
        re.fullmatch(
            r"[<>≤≥=~\s]*[-+]?\d+(?:\.\d+)?(?:\s*[±+-]\s*\d+(?:\.\d+)?)?",
            val_norm.strip(),
        )
    )


def _clean_surface_markup(val: str) -> str:
    """Strip common Markdown/HTML wrappers and surrounding quotes from a raw cell fragment."""
    s = re.sub(r"<[^>]+>", "", val.strip())
    s = s.strip().strip("`").strip("*").strip("'").strip('"').strip()
    return s


def _extract_field_validators_from_yaml(field_name: str, schema_yaml: str) -> list[FieldValidatorSpec]:
    """Parse declared validators for ``field_name`` from raw schema YAML string."""
    if not schema_yaml or not field_name:
        return []
    try:
        parsed = yaml.safe_load(schema_yaml) or {}
    except Exception:
        return []
    fields_dict = parsed.get("fields", {})
    if not isinstance(fields_dict, dict):
        return []
    target_key = field_name.lower().strip()
    field_cfg = None
    for k, v in fields_dict.items():
        if str(k).lower().strip() == target_key:
            field_cfg = v
            break
    if not isinstance(field_cfg, dict):
        return []
    raw_validators = field_cfg.get("validators") or []
    specs: list[FieldValidatorSpec] = []
    for item in raw_validators:
        try:
            specs.append(normalize_validator_spec(item))
        except Exception:
            continue
    return specs


def are_field_values_equivalent(
    field_name: str,
    raw_or_candidate: str,
    gt_value: str,
    schema_yaml: str = "",
) -> bool:
    """Return True if two field values represent the same underlying value after normalization or schema canonicalization."""
    if not raw_or_candidate or not gt_value:
        return False

    cand_clean = _clean_surface_markup(str(raw_or_candidate))
    gt_clean = _clean_surface_markup(str(gt_value))
    if not cand_clean or not gt_clean:
        return False

    cand_norm = _normalize_cell_for_match(cand_clean)
    gt_norm = _normalize_cell_for_match(gt_clean)
    if cand_norm and cand_norm == gt_norm:
        return True

    # Plain numeric equivalence (e.g. "0.030" == "0,03")
    num_pat = r"[-+]?\d+(?:[.,]\d+)?"
    if re.fullmatch(num_pat, cand_clean) and re.fullmatch(num_pat, gt_clean):
        try:
            if abs(float(cand_clean.replace(",", ".")) - float(gt_clean.replace(",", "."))) <= 1e-9:
                return True
        except ValueError:
            pass

    # Schema-driven validator canonicalization equivalence (e.g. canonical SMILES comparison)
    validator_specs = _extract_field_validators_from_yaml(field_name, schema_yaml)
    if validator_specs:
        cand_tokens = [cand_clean]
        if "|" in cand_clean:
            for part in cand_clean.split("|"):
                cleaned_part = _clean_surface_markup(part)
                if cleaned_part and cleaned_part not in cand_tokens:
                    cand_tokens.append(cleaned_part)
        elif " " in cand_clean:
            space_parts = [_clean_surface_markup(p) for p in cand_clean.split() if _clean_surface_markup(p)]
            if 1 < len(space_parts) <= 6:
                for sp in space_parts:
                    if sp not in cand_tokens:
                        cand_tokens.append(sp)

        for spec in validator_specs:
            if not supports_value_comparison(spec):
                continue
            for token in cand_tokens:
                try:
                    match, _ = compare_field_values_via_validator(spec, token, gt_clean)
                except Exception:
                    continue
                if match is True:
                    return True

    return False


def has_canonicalizing_validator(field_name: str, schema_yaml: str = "") -> bool:
    """Return True if ``field_name`` declares a schema validator supporting value comparison."""
    for spec in _extract_field_validators_from_yaml(field_name, schema_yaml):
        if supports_value_comparison(spec):
            return True
    return False


def sanitize_positive_columns_output(
    output: PositiveColumnsOutput,
    schema_yaml: str = "",
) -> PositiveColumnsOutput:
    """Normalize false discrepancy flags and pseudo-transformations for equivalent values."""
    for f_item in output.fields:
        has_canon_val = has_canonicalizing_validator(f_item.field_name, schema_yaml)
        is_equiv = are_field_values_equivalent(
            f_item.field_name,
            f_item.raw_text,
            f_item.ground_truth,
            schema_yaml,
        )
        if has_canon_val and is_equiv:
            f_item.Suspected_Ingestion_Or_GT_Issue = "None"
            f_item.Transformation_description = (
                "None (canonically equivalent under schema validator; no transformation required)."
            )
            continue

        issue = (getattr(f_item, "Suspected_Ingestion_Or_GT_Issue", "None") or "None").strip()
        if issue.lower() not in ("", "none") and is_equiv:
            f_item.Suspected_Ingestion_Or_GT_Issue = "None"
            if "no valid transformation" in (f_item.Transformation_description or "").lower():
                f_item.Transformation_description = (
                    "Canonicalization / surface formatting normalization (verified equivalent)."
                )
    return output


def filter_equivalent_conflicts(
    conflicts: list[REConflict],
    schema_yaml: str = "",
) -> list[REConflict]:
    """Filter out false column discrepancy conflicts whose values are canonically equivalent."""
    if not conflicts or not schema_yaml:
        return list(conflicts)

    filtered: list[REConflict] = []
    for conflict in conflicts:
        scope = (conflict.scope or "").strip()
        if (
            scope.startswith("column:")
            and conflict.conflict_type
            in ("BASELINE_POLICY_CONFLICT", "TEXT_VS_GT_MISMATCH", "SUSPECTED_GT_ERROR")
            and conflict.raw_text_quote
            and conflict.gt_value
        ):
            field_name = scope.split("column:", 1)[1].strip()
            if has_canonicalizing_validator(field_name, schema_yaml) and are_field_values_equivalent(
                field_name,
                conflict.raw_text_quote,
                conflict.gt_value,
                schema_yaml,
            ):
                logger.info(
                    f"Skipping false conflict '{conflict.conflict_id}' ({scope}): "
                    f"raw_text_quote and gt_value are canonically equivalent under schema validator."
                )
                continue
        filtered.append(conflict)
    return filtered


def _extract_document_regions(raw_text: str) -> list[tuple[str, str]]:
    """Split document markdown into structural regions (visual blocks, tables, prose sections)."""
    if not raw_text:
        return [("doc:0", "")]

    regions: list[tuple[str, str]] = []
    visual_matches = list(
        re.finditer(r'<visual\s+[^>]*id="([^"]+)"[^>]*>(.*?)</visual>', raw_text, flags=re.S)
    )
    if visual_matches:
        last_end = 0
        prose_idx = 0
        for m in visual_matches:
            if m.start() > last_end:
                chunk = raw_text[last_end : m.start()].strip()
                if len(chunk) > 80:
                    regions.append((f"prose:{prose_idx}", chunk.lower()))
                    prose_idx += 1
            regions.append((f"visual:{m.group(1)}", m.group(2).lower()))
            last_end = m.end()
        if last_end < len(raw_text):
            chunk = raw_text[last_end:].strip()
            if len(chunk) > 80:
                regions.append((f"prose:{prose_idx}", chunk.lower()))
        if regions:
            return regions

    # Fallback for plain Markdown without <visual> blocks: split by markdown tables / headings
    last_end = 0
    tbl_idx = 0
    sec_idx = 0
    for m in re.finditer(r"(?:^\|.*\|$\n?)+", raw_text, flags=re.M):
        if m.start() > last_end:
            chunk = raw_text[last_end : m.start()].strip()
            if len(chunk) > 80:
                regions.append((f"prose:{sec_idx}", chunk.lower()))
                sec_idx += 1
        regions.append((f"table:{tbl_idx}", m.group(0).lower()))
        tbl_idx += 1
        last_end = m.end()
    if last_end < len(raw_text):
        chunk = raw_text[last_end:].strip()
        if len(chunk) > 80:
            regions.append((f"prose:{sec_idx}", chunk.lower()))

    return regions or [("doc:0", raw_text.lower())]


def select_representative_rows(
    df_doc_with_id: pd.DataFrame,
    raw_text: str,
    max_rows: int = 20,
) -> pd.DataFrame:
    """Select a domain-independent, document-aware stratified subset of GT rows.

    Preserves original ``row_id`` values while greedily optimizing submodular coverage across:
    1) 3-tier text transformation differences (``verbatim`` / ``surface_norm`` / ``deep_diff``),
    2) null-pattern signatures,
    3) structural document regions (tables / visual blocks / prose sections),
    4) balanced low-cardinality categorical strata (excluding continuous numeric columns and IDs),
    5) entity diversity weighted by text mention prominence (capturing lead entities for negative analysis), and
    6) positional dispersion across the table.
    """
    n_rows = len(df_doc_with_id)
    if max_rows <= 0 or n_rows <= max_rows or n_rows == 0:
        return df_doc_with_id

    schema_cols = [c for c in df_doc_with_id.columns if c != "row_id"]
    if not schema_cols:
        step_indices = sorted(
            {round(k * (n_rows - 1) / max(1, max_rows - 1)) for k in range(max_rows)}
        )
        return df_doc_with_id.iloc[step_indices].copy()

    raw_lower = (raw_text or "").lower()
    raw_norm = _normalize_cell_for_match(raw_text or "")
    regions = _extract_document_regions(raw_text or "")
    region_norms = [(r_name, r_text, _normalize_cell_for_match(r_text)) for r_name, r_text in regions]

    row_null_sigs: list[tuple[bool, ...]] = []
    row_values: list[dict[str, str]] = []
    row_match_types: list[dict[str, str]] = []
    distinct_by_col: dict[str, set[str]] = {c: set() for c in schema_cols}
    val_freq_by_col: dict[str, dict[str, int]] = {c: {} for c in schema_cols}
    non_empty_count: dict[str, int] = {c: 0 for c in schema_cols}
    numeric_count: dict[str, int] = {c: 0 for c in schema_cols}
    max_len_by_col: dict[str, int] = {c: 0 for c in schema_cols}

    for idx in range(n_rows):
        row = df_doc_with_id.iloc[idx]
        null_flags: list[bool] = []
        vals_map: dict[str, str] = {}
        m_types: dict[str, str] = {}

        for col in schema_cols:
            raw_val = row[col]
            is_pop = pd.notna(raw_val) and str(raw_val).strip() != ""
            null_flags.append(is_pop)
            if not is_pop:
                continue
            val_str = str(raw_val).strip()
            vals_map[col] = val_str
            non_empty_count[col] += 1
            distinct_by_col[col].add(val_str)
            val_freq_by_col[col][val_str] = val_freq_by_col[col].get(val_str, 0) + 1
            max_len_by_col[col] = max(max_len_by_col[col], len(val_str))

            val_norm = _normalize_cell_for_match(val_str)
            if _is_numeric_like(val_norm):
                numeric_count[col] += 1

            if val_str.lower() in raw_lower:
                m_types[col] = "verbatim"
            elif val_norm and val_norm in raw_norm:
                m_types[col] = "surface_norm"
            else:
                m_types[col] = "deep_diff"

        row_null_sigs.append(tuple(null_flags))
        row_values.append(vals_map)
        row_match_types.append(m_types)

    # Separate low-cardinality categorical strata from short entity-identifier columns and continuous numbers
    stratum_cols: list[str] = []
    entity_cols: list[str] = []
    stratum_cap = min(max_rows // 2 + 2, max(2, n_rows // 3))
    for col in schema_cols:
        ne = non_empty_count[col]
        if ne == 0:
            continue
        uniq = len(distinct_by_col[col])
        is_mostly_num = (numeric_count[col] / ne) >= 0.8 and uniq > 5
        if not is_mostly_num and 1 <= uniq <= stratum_cap:
            stratum_cols.append(col)
        elif uniq > 1 and max_len_by_col[col] <= 35 and not is_mostly_num:
            entity_cols.append(col)

    if not entity_cols:
        entity_cols = [
            c
            for c in schema_cols
            if len(distinct_by_col[c]) > 1
            and max_len_by_col[c] <= 35
            and (non_empty_count[c] == 0 or (numeric_count[c] / non_empty_count[c]) < 0.8)
        ]

    # Link each row to an anchor entity, its text mention frequency, and its best matching document region
    row_entity: list[str] = []
    row_mentions: list[int] = []
    row_region: list[str] = []
    mention_cache: dict[str, int] = {}
    anchor_patterns: dict[str, re.Pattern[str]] = {}
    anchor_freq_limit = max(4, n_rows // 4)

    for idx in range(n_rows):
        vals = row_values[idx]
        ent_parts = [vals[c] for c in entity_cols if c in vals]
        row_entity.append("|".join(ent_parts) if ent_parts else f"row_{idx}")

        anchor: str | None = None
        for c in entity_cols:
            v = vals.get(c)
            if v and val_freq_by_col[c].get(v, 0) <= anchor_freq_limit:
                anchor = v
                break

        if anchor:
            if anchor not in anchor_patterns:
                anchor_patterns[anchor] = re.compile(
                    r"(?<![A-Za-z0-9])" + re.escape(anchor.lower()) + r"(?![A-Za-z0-9])"
                )
            pat = anchor_patterns[anchor]
            if anchor not in mention_cache:
                mention_cache[anchor] = len(pat.findall(raw_lower)) if raw_lower else 0
            row_mentions.append(mention_cache[anchor])

            best_reg = f"pos_bin:{(idx * 5) // n_rows}"
            best_reg_score = (0, 0)
            other_vals = [
                _normalize_cell_for_match(v)
                for c, v in vals.items()
                if c not in entity_cols and len(v) <= 30
            ]
            for r_name, r_text, r_norm in region_norms:
                if pat.search(r_text):
                    val_hits = sum(1 for ov in other_vals if ov and ov in r_norm)
                    is_structured = 1 if r_name.startswith(("visual:", "table:")) else 0
                    sc = (1 + val_hits, is_structured)
                    if sc > best_reg_score:
                        best_reg_score = sc
                        best_reg = r_name
            row_region.append(best_reg)
        else:
            row_mentions.append(0)
            row_region.append(f"pos_bin:{(idx * 5) // n_rows}")

    max_mentions = max(row_mentions) if any(row_mentions) else 1

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_null_sigs: set[tuple[bool, ...]] = set()
    covered_transforms: set[tuple[str, str]] = set()
    covered_deep_vals: dict[str, set[str]] = {c: set() for c in schema_cols}
    stratum_counts: dict[str, dict[str, int]] = {c: {} for c in stratum_cols}
    region_counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}

    def _add_row(idx: int) -> None:
        selected.append(idx)
        selected_set.add(idx)
        covered_null_sigs.add(row_null_sigs[idx])
        for c, mt in row_match_types[idx].items():
            covered_transforms.add((c, mt))
            if mt == "deep_diff":
                covered_deep_vals[c].add(row_values[idx].get(c, ""))
        for c in stratum_cols:
            val = row_values[idx].get(c)
            if val is not None:
                stratum_counts[c][val] = stratum_counts[c].get(val, 0) + 1
        reg = row_region[idx]
        region_counts[reg] = region_counts.get(reg, 0) + 1
        ent = row_entity[idx]
        entity_counts[ent] = entity_counts.get(ent, 0) + 1

    while len(selected) < max_rows:
        best_idx = -1
        best_score = -1e9

        for idx in range(n_rows):
            if idx in selected_set:
                continue
            gain = 0.0

            # 1. Text transformation / anomaly diversity (3-tier)
            for c, mt in row_match_types[idx].items():
                if (c, mt) not in covered_transforms:
                    gain += 2.0 if mt == "deep_diff" else (1.0 if mt == "surface_norm" else 0.3)
                elif (
                    mt == "deep_diff"
                    and len(covered_deep_vals[c]) < 4
                    and row_values[idx].get(c, "") not in covered_deep_vals[c]
                ):
                    gain += 0.6

            # 2. Null-pattern diversity
            if row_null_sigs[idx] not in covered_null_sigs:
                gain += 1.5

            # 3. Document region (table / visual / prose section) coverage
            reg = row_region[idx]
            r_cnt = region_counts.get(reg, 0)
            gain += 2.5 if r_cnt == 0 else (0.25 / (1 + r_cnt))

            # 4. Categorical stratum coverage and proportional balance
            for c in stratum_cols:
                val = row_values[idx].get(c)
                if val is not None:
                    s_cnt = stratum_counts[c].get(val, 0)
                    if s_cnt == 0:
                        gain += 3.0
                    else:
                        gain += (1.0 / max(1, len(distinct_by_col[c]))) / (1 + s_cnt)

            # 5. Entity diversity & text prominence (surfaces lead entities with rich distractor context)
            ent = row_entity[idx]
            e_cnt = entity_counts.get(ent, 0)
            if e_cnt == 0:
                gain += 0.8 + 0.7 * (math.log1p(row_mentions[idx]) / math.log1p(max_mentions))
            else:
                gain -= 0.5 * e_cnt

            # 6. Positional dispersion across the table
            if selected:
                min_dist = min(abs(idx - s_idx) for s_idx in selected) / n_rows
                gain += 0.4 * min_dist
            else:
                gain += 0.1 * (1.0 - idx / n_rows)

            if gain > best_score:
                best_score = gain
                best_idx = idx

        if best_idx < 0:
            break
        _add_row(best_idx)

    return df_doc_with_id.iloc[sorted(selected)].copy()


def run_positive_analysis(context: RunContext, force_doc_ids: list[str] | None = None) -> None:
    """Execute Phase 1.1 and 1.2 Positive Analysis for train documents."""
    logger.info("Starting Phase 1: Positive Analysis")

    doc_repo = DocumentRepository(ingestion_dir=context.ingestion_dir)

    # Use active (corrected) GT path
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

    row_predictor = dspy.Predict(PositiveRowAnalysis)
    col_predictor = dspy.Predict(PositiveColumnAnalysis)

    target_doc_ids = force_doc_ids if force_doc_ids is not None else context.doc_ids

    for doc_id in target_doc_ids:
        norm_doc_id = normalize_document_key(doc_id)
        force_recompute = force_doc_ids is not None and doc_id in force_doc_ids
        logger.info(f"Processing positive analysis for document: {norm_doc_id} (force_recompute={force_recompute})")

        raw_text = doc_repo.get(norm_doc_id)
        if not raw_text:
            logger.warning(f"Parsed text for document {norm_doc_id} not found in {context.ingestion_dir}. Skipping.")
            continue

        df_doc = df_gt[df_gt[id_col] == norm_doc_id].copy()

        raw_max_rows = getattr(context, "max_rows_per_doc", 20)
        max_rows = raw_max_rows if isinstance(raw_max_rows, int) and raw_max_rows >= 0 else 20

        # Step 1.1: Positive Row Analysis
        row_artifact_path = context.artifacts.get_path(
            "positive", "rows", f"{norm_doc_id}.json", iteration=None
        )

        def compute_rows():
            if df_doc.empty:
                logger.info(f"No Ground Truth rows found for document {norm_doc_id}. Creating empty RowAnalysis.")
                return PositiveRowsOutput(rows=[])

            df_doc_with_id = prepare_prompt_table(df_doc, schema_fields)
            gt_table_csv = df_doc_with_id.to_csv(index=False)
            fresh_text = _unpack_manifests_for_positive(
                norm_doc_id,
                raw_text,
                gt_table_csv,
                context,
                doc_repo,
                force_recompute=force_recompute,
            )
            sampled_rows_df = select_representative_rows(
                df_doc_with_id, fresh_text, max_rows=max_rows
            )
            if len(sampled_rows_df) < len(df_doc_with_id):
                logger.info(
                    f"PositiveRowAnalysis sampling for {norm_doc_id}: "
                    f"{len(sampled_rows_df)}/{len(df_doc_with_id)} representative rows selected"
                )

            logger.info(f"Calling PositiveRowAnalysis LLM for {norm_doc_id}")
            response = row_predictor(
                extraction_schema=context.schema,
                baseline_prompt=context.baseline_prompt,
                raw_text=fresh_text,
                gt_table=gt_table_csv,
                target_gt_rows=sampled_rows_df.to_csv(index=False),
            )
            return response.analysis

        row_analysis = context.artifacts.load_or_compute(
            row_artifact_path,
            compute_rows,
            PositiveRowsOutput,
            force_recompute=force_recompute,
        )
        raw_text = doc_repo.get(norm_doc_id) or raw_text

        # Step 1.2: Positive Column Analysis (with stratified sampling, row batching & prefix-cache field ordering)
        if df_doc.empty:
            continue

        df_doc_with_id = prepare_prompt_table(df_doc, schema_fields)
        gt_table_csv = df_doc_with_id.to_csv(index=False)
        raw_text = _unpack_manifests_for_positive(
            norm_doc_id,
            raw_text,
            gt_table_csv,
            context,
            doc_repo,
            force_recompute=False,
        )
        sampled_cols_df = select_representative_rows(
            df_doc_with_id, raw_text, max_rows=max_rows
        )
        if len(sampled_cols_df) < len(df_doc_with_id):
            logger.info(
                f"PositiveColumnAnalysis sampling for {norm_doc_id}: "
                f"{len(sampled_cols_df)}/{len(df_doc_with_id)} representative rows selected"
            )

        raw_batch_size = getattr(context, "row_batch_size", 5)
        batch_size = max(1, raw_batch_size if isinstance(raw_batch_size, int) and raw_batch_size >= 1 else 5)

        # Collect sampled row positions that have populated fields and require computation
        pending_positions: list[int] = []
        for pos_idx in range(len(sampled_cols_df)):
            target_row_df = sampled_cols_df.iloc[[pos_idx]]
            row_id = str(target_row_df["row_id"].values[0])
            non_empty_cols = [
                c
                for c in target_row_df.columns
                if c != "row_id"
                and pd.notna(target_row_df[c].values[0])
                and str(target_row_df[c].values[0]).strip() != ""
            ]
            if not non_empty_cols:
                logger.info(f"Row {row_id} in document {norm_doc_id} has no populated fields. Skipping column analysis.")
                continue

            col_artifact_path = context.artifacts.get_path(
                "positive", "columns", f"{norm_doc_id}_{row_id}.json", iteration=None
            )
            if not force_recompute and context.artifacts.resume and col_artifact_path.exists():
                context.artifacts.load_or_compute(
                    col_artifact_path,
                    lambda: PositiveColumnsOutput(fields=[]),
                    PositiveColumnsOutput,
                    force_recompute=False,
                )
            else:
                pending_positions.append(pos_idx)

        # Process pending rows in batches of `batch_size`
        for start_idx in range(0, len(pending_positions), batch_size):
            batch_positions = pending_positions[start_idx : start_idx + batch_size]
            batch_df = sampled_cols_df.iloc[batch_positions]
            batch_row_ids = [str(r) for r in batch_df["row_id"].tolist()]
            target_gt_rows_csv = batch_df.to_csv(index=False)

            if len(batch_positions) == 1:
                single_row_id = batch_row_ids[0]
                col_artifact_path = context.artifacts.get_path(
                    "positive", "columns", f"{norm_doc_id}_{single_row_id}.json", iteration=None
                )

                def compute_single_column(t_row=target_gt_rows_csv, r_id=single_row_id):
                    fresh_col_text = doc_repo.get(norm_doc_id) or raw_text
                    logger.info(f"Calling PositiveColumnAnalysis LLM for {norm_doc_id} {r_id}")
                    response = col_predictor(
                        extraction_schema=context.schema,
                        baseline_prompt=context.baseline_prompt,
                        raw_text=fresh_col_text,
                        gt_table=gt_table_csv,
                        target_gt_row=t_row,
                    )
                    return sanitize_positive_columns_output(response.analysis, context.schema)

                context.artifacts.load_or_compute(
                    col_artifact_path,
                    compute_single_column,
                    PositiveColumnsOutput,
                    force_recompute=True,
                )
            else:
                fresh_col_text = doc_repo.get(norm_doc_id) or raw_text
                logger.info(
                    f"Calling PositiveColumnAnalysis LLM for {norm_doc_id} batch {batch_row_ids}"
                )
                response = col_predictor(
                    extraction_schema=context.schema,
                    baseline_prompt=context.baseline_prompt,
                    raw_text=fresh_col_text,
                    gt_table=gt_table_csv,
                    target_gt_row=target_gt_rows_csv,
                )
                batch_analysis: PositiveColumnsOutput = sanitize_positive_columns_output(
                    response.analysis, context.schema
                )

                # Partition fields by row_id and save per-row artifacts
                fields_by_row: dict[str, list] = {r_id: [] for r_id in batch_row_ids}
                for f_item in batch_analysis.fields:
                    raw_r_id = getattr(f_item, "row_id", "") or ""
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
                    fields_by_row[target_r_id].append(f_item)

                for r_id in batch_row_ids:
                    col_artifact_path = context.artifacts.get_path(
                        "positive", "columns", f"{norm_doc_id}_{r_id}.json", iteration=None
                    )
                    row_output = PositiveColumnsOutput(fields=fields_by_row[r_id])
                    context.artifacts.load_or_compute(
                        col_artifact_path,
                        lambda out=row_output: out,
                        PositiveColumnsOutput,
                        force_recompute=True,
                    )

    logger.info("Finished Phase 1: Positive Analysis")
