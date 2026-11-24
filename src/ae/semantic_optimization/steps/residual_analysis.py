"""Error-Driven / Residual Semantic Analysis step for Iterations >= 2."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy
import pandas as pd

from ae.core.config.settings import Settings
from ae.core.evaluation import TaskMetric
from ae.core.schema import load_schema_complete
from ae.core.schema.loader import create_row_converter, load_schema_with_models
from ae.core.storage.documents import DocumentRepository
from ae.core.utils import normalize_document_key
from ae.extraction.agent import UniversalExtractor
from ae.ingestion.parsers.gemini.stages.render_markdown import has_unparsed_visuals
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.hitl import _get_id_column
from ae.semantic_optimization.models import (
    FieldAnalysis,
    NegativeColumnsOutput,
    NegativeFieldResult,
    NegativeRowsOutput,
    PositiveColumnsOutput,
    PositiveRowsOutput,
    ResidualColumnsOutput,
    ResidualDocReport,
)
from ae.semantic_optimization.signatures import (
    PositiveColumnAnalysis,
    ResidualColumnAnalysis,
    ResidualNegativeRowAnalysis,
    ResidualPositiveRowAnalysis,
)
from ae.semantic_optimization.steps.negative_analysis import (
    filter_equivalent_negative_candidates,
)
from ae.semantic_optimization.steps.positive_analysis import (
    _is_numeric_like,
    _normalize_cell_for_match,
    are_field_values_equivalent,
    prepare_prompt_table,
    sanitize_positive_columns_output,
    select_representative_rows,
)
from ae.semantic_optimization.steps.visual_prep import prepare_document_visuals

logger = logging.getLogger(__name__)


@dataclass
class ResidualBatchSummary:
    """Aggregated outcome of running Residual Analysis over a batch of new train docs."""

    doc_reports: list[ResidualDocReport] = field(default_factory=list)
    fallback_doc_ids: list[str] = field(default_factory=list)
    zero_delta_doc_ids: list[str] = field(default_factory=list)
    residual_doc_ids: list[str] = field(default_factory=list)
    affected_scopes: set[str] = field(default_factory=set)

    @property
    def is_zero_delta(self) -> bool:
        """True when no processed documents required fallback or residual rule updates."""
        return not self.fallback_doc_ids and not self.residual_doc_ids


def _is_value_na(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() == ""


def _classify_field_mismatch(
    pred_val: Any,
    gt_val: Any,
    reason: Optional[str] = None,
) -> str:
    """Classify a column discrepancy into a coarse mismatch signature bucket."""
    p_na = _is_value_na(pred_val)
    g_na = _is_value_na(gt_val)
    if p_na and not g_na:
        return "missing_pred"
    if not p_na and g_na:
        return "spurious_pred"
    if reason == "stereo_diff":
        return "stereo_diff"
    p_str = str(pred_val).strip()
    g_str = str(gt_val).strip()
    if _is_numeric_like(_normalize_cell_for_match(p_str)) and _is_numeric_like(
        _normalize_cell_for_match(g_str)
    ):
        return "numeric_diff"
    p_low = p_str.lower()
    g_low = g_str.lower()
    if p_low in g_low or g_low in p_low:
        return "format_or_substring_diff"
    return "lexical_diff"


def _split_zero_discriminative_pairs(
    pair_results: list[dict[str, Any]],
    df_doc: pd.DataFrame,
    compare_fields: list[str],
) -> list[dict[str, Any]]:
    """Split aligned mismatch pairs that share zero discriminative fields into FP + FN."""
    weights: dict[str, float] = {}
    for f in compare_fields:
        if f not in df_doc.columns:
            continue
        series = df_doc[f].dropna().astype(str).str.strip()
        non_null = len(series)
        nunique = series.nunique()
        if non_null >= 2 and nunique > 1:
            weights[f] = (nunique - 1) / (non_null - 1)

    total_disc_weight = sum(weights.values())
    if total_disc_weight <= 0.0:
        return pair_results

    adjusted: list[dict[str, Any]] = []
    for pair in pair_results:
        if pair["status"] != "field_mismatch":
            adjusted.append(pair)
            continue
        matched_disc_weight = sum(weights.get(f, 0.0) for f in pair.get("matched_fields", []))
        if (matched_disc_weight / total_disc_weight) >= 0.25:
            adjusted.append(pair)
            continue

        # Zero discriminative fields matched -> split forced Hungarian pair into FP + FN
        adjusted.append(
            {
                "status": "fp",
                "pred_index": pair["pred_index"],
                "gt_index": None,
                "pred": pair["pred"],
                "gold": None,
                "matched_fields": [],
                "mismatched_fields": {},
            }
        )
        adjusted.append(
            {
                "status": "fn",
                "pred_index": None,
                "gt_index": pair["gt_index"],
                "pred": None,
                "gold": pair["gold"],
                "matched_fields": [],
                "mismatched_fields": {},
            }
        )
    return adjusted


def _group_mismatch_pairs(
    mismatch_items: list[dict[str, Any]],
    max_per_sig: int,
) -> Tuple[list[dict[str, Any]], dict[str, int]]:
    """Group aligned mismatch items by error signature and cap diverse examples per group."""
    buckets: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for item in mismatch_items:
        sig = tuple(
            (fname, _classify_field_mismatch(info["pred"], info["gold"], info.get("reason")))
            for fname, info in sorted(item["mismatched_fields"].items())
        )
        buckets.setdefault(sig, []).append(item)

    selected_items: list[dict[str, Any]] = []
    sig_counts: dict[str, int] = {}

    for sig, group in buckets.items():
        sig_label = ", ".join(f"{f}:{mtype}" for f, mtype in sig)
        sig_counts[sig_label] = len(group)

        # Pick diverse examples within the signature group by distinct (pred, gold) tuples
        chosen: list[dict[str, Any]] = []
        seen_value_tuples: set[tuple[tuple[str, str, str], ...]] = set()
        for cand in group:
            if len(chosen) >= max_per_sig:
                break
            val_sig = tuple(
                (
                    f,
                    "" if _is_value_na(info["pred"]) else str(info["pred"]).strip(),
                    "" if _is_value_na(info["gold"]) else str(info["gold"]).strip(),
                )
                for f, info in sorted(cand["mismatched_fields"].items())
            )
            if val_sig not in seen_value_tuples:
                seen_value_tuples.add(val_sig)
                chosen.append(cand)

        if len(chosen) < max_per_sig:
            chosen_ids = {id(c) for c in chosen}
            for cand in group:
                if len(chosen) >= max_per_sig:
                    break
                if id(cand) not in chosen_ids:
                    chosen.append(cand)

        for cand in chosen:
            enriched = dict(cand)
            enriched["occurrence_count_in_doc"] = len(group)
            selected_items.append(enriched)

    return selected_items, sig_counts


def _record_to_dict(record: Any, fields: list[str]) -> dict[str, Any]:
    return {
        f: (None if _is_value_na(getattr(record, f, None)) else str(getattr(record, f)).strip())
        for f in fields
    }


def _score_fp_pu_candidates(
    fp_pairs: list[dict[str, Any]],
    gt_records: list[Any],
    fields: list[str],
    field_constraints: Optional[dict[str, dict[str, Any]]] = None,
    pu_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Score unmatched FP predictions under Positive-Unlabeled (PU) heuristics.

    Identifies `high_confidence_unlabeled` records that structurally match Positive GT
    records (potential GT omissions) while separating POT `duplicate_of_tp` / `row_split`.
    """
    if not fp_pairs:
        return []
    if not gt_records:
        return [
            {
                **p,
                "pu_status": p.get("diagnostic_tag") or "unlabeled",
                "pu_score": 0.0,
                "pu_reason": "no_gt_reference",
            }
            for p in fp_pairs
        ]

    gt_dicts = [_record_to_dict(g, fields) for g in gt_records]
    gt_null_sigs = {tuple(d[f] is not None for f in fields) for d in gt_dicts}
    gt_vals_by_field: dict[str, set[str]] = {f: set() for f in fields}
    for d in gt_dicts:
        for f in fields:
            if d[f] is not None:
                gt_vals_by_field[f].add(_normalize_cell_for_match(d[f]))

    cat_cap = min(8, max(1, len(gt_dicts) // 3)) if len(gt_dicts) >= 3 else 1
    cat_fields = [
        f
        for f in fields
        if 1 <= len(gt_vals_by_field[f]) <= cat_cap
        and not (
            len(gt_dicts) >= 2
            and len(gt_vals_by_field[f]) == len(gt_dicts)
            and all(_is_numeric_like(v) for v in gt_vals_by_field[f])
        )
    ]
    anchor_fields = [
        f
        for f in fields
        if f not in cat_fields
        and len(gt_vals_by_field[f]) > 1
        and not all(_is_numeric_like(v) for v in gt_vals_by_field[f])
    ]

    constraints = field_constraints or {}
    scored: list[dict[str, Any]] = []
    for p in fp_pairs:
        entry = dict(p)
        diag_tag = p.get("diagnostic_tag")
        if diag_tag == "duplicate_of_tp":
            entry["pu_status"] = "duplicate_of_tp"
            entry["pu_score"] = 0.0
            entry["pu_reason"] = "POT detected near-duplicate of an already matched TP row"
            scored.append(entry)
            continue

        pred = p.get("pred")
        if pred is None:
            entry["pu_status"] = "unlabeled"
            entry["pu_score"] = 0.0
            entry["pu_reason"] = "empty_prediction"
            scored.append(entry)
            continue

        pred_dict = _record_to_dict(pred, fields)
        violates = False
        for f in fields:
            val = pred_dict[f]
            if val is None:
                continue
            fc = constraints.get(f, {})
            choices = fc.get("choices")
            if choices and val not in choices:
                violates = True
                break
            pattern = fc.get("pattern")
            if pattern and not re.search(pattern, val):
                violates = True
                break

        if violates:
            entry["pu_status"] = "constraint_violation"
            entry["pu_score"] = 0.0
            entry["pu_reason"] = "violates_schema_constraints"
            scored.append(entry)
            continue

        pred_null_sig = tuple(pred_dict[f] is not None for f in fields)
        null_match = 1.0 if pred_null_sig in gt_null_sigs else 0.0

        pop_cats = [f for f in cat_fields if pred_dict[f] is not None]
        if pop_cats:
            cat_hits = sum(
                1 for f in pop_cats if _normalize_cell_for_match(pred_dict[f]) in gt_vals_by_field[f]
            )
            cat_ratio = cat_hits / len(pop_cats)
        else:
            cat_ratio = 0.0

        anchor_hit = any(
            pred_dict[f] is not None
            and _normalize_cell_for_match(pred_dict[f]) in gt_vals_by_field[f]
            for f in anchor_fields
        )

        pu_score = (0.35 * null_match) + (0.45 * cat_ratio) + (0.20 if anchor_hit else 0.0)
        pu_score = round(min(1.0, pu_score), 4)

        if null_match >= 1.0 and pu_score >= pu_threshold:
            entry["pu_status"] = "high_confidence_unlabeled"
            reasons = ["matches_gt_null_pattern"]
            if cat_ratio >= 0.8:
                reasons.append(f"cat_support={cat_ratio:.2f}")
            if anchor_hit:
                reasons.append("shared_gt_anchor_entity")
            entry["pu_reason"] = ", ".join(reasons)
        elif diag_tag == "row_split":
            entry["pu_status"] = "row_split"
            entry["pu_reason"] = "POT detected split fragment of a GT row"
        else:
            entry["pu_status"] = "unlabeled"
            entry["pu_reason"] = f"pu_score={pu_score:.2f}"
        entry["pu_score"] = pu_score
        scored.append(entry)

    return scored


def _group_fp_rows(
    fp_pairs: list[dict[str, Any]],
    fields: list[str],
    max_per_sig: int,
    max_total: int,
) -> list[dict[str, Any]]:
    """Group False Positive rows by structural/repeating signature, prioritizing PU/POT triggers."""
    if not fp_pairs:
        return []

    fp_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for p in fp_pairs:
        if p.get("pred") is None:
            continue
        fp_rows.append((_record_to_dict(p["pred"], fields), p))
    if not fp_rows:
        return []

    # Identify fields with repeated values across >= 2 FPs (good distractor cohort keys)
    cohort_fields = []
    for f in fields:
        vals = [d[f] for d, _ in fp_rows if d[f] is not None]
        if len(vals) >= 2 and len(set(vals)) < len(vals):
            cohort_fields.append(f)

    buckets: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for d, p in fp_rows:
        null_sig = tuple(d[f] is not None for f in fields)
        val_sig = tuple((f, d[f]) for f in cohort_fields[:3])
        pu_tag = p.get("pu_status") or p.get("diagnostic_tag") or ""
        buckets.setdefault((pu_tag, null_sig, val_sig), []).append((d, p))

    def _bucket_sort_key(item: tuple[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]]) -> tuple[int, float]:
        (pu_tag, _, _), group = item
        if pu_tag == "high_confidence_unlabeled":
            rank = 0
        elif pu_tag in ("duplicate_of_tp", "row_split"):
            rank = 1
        else:
            rank = 2
        max_pu = max(float(p.get("pu_score") or 0.0) for _, p in group)
        return (rank, -max_pu)

    grouped_payload: list[dict[str, Any]] = []
    for _, group in sorted(buckets.items(), key=_bucket_sort_key):
        if len(grouped_payload) >= max_total:
            break
        take = min(max_per_sig, max_total - len(grouped_payload), len(group))
        for idx in range(take):
            d, p = group[idx]
            entry = dict(d)
            entry["cohort_count"] = len(group)
            if p.get("pu_status"):
                entry["pu_status"] = p["pu_status"]
                entry["pu_score"] = float(p.get("pu_score") or 0.0)
            if p.get("pu_reason"):
                entry["pu_reason"] = p["pu_reason"]
            if p.get("diagnostic_tag"):
                entry["diagnostic_tag"] = p["diagnostic_tag"]
            grouped_payload.append(entry)

    return grouped_payload


def _partition_items_by_row(
    items: list[Any],
    batch_row_ids: list[str],
) -> dict[str, list[Any]]:
    """Partition DSPy output items by row_id into buckets keyed by batch_row_ids."""
    by_row: dict[str, list[Any]] = {r_id: [] for r_id in batch_row_ids}
    if not batch_row_ids:
        return by_row
    for item in items:
        raw_r_id = getattr(item, "row_id", "") or ""
        if raw_r_id in by_row:
            target_r_id = raw_r_id
        else:
            m = re.search(r"(\d+)", str(raw_r_id))
            if m:
                idx = int(m.group(1))
                if f"row_{idx}" in by_row:
                    target_r_id = f"row_{idx}"
                elif 1 <= idx <= len(batch_row_ids):
                    target_r_id = batch_row_ids[idx - 1]
                else:
                    target_r_id = batch_row_ids[0]
            else:
                target_r_id = batch_row_ids[0]
        if hasattr(item, "row_id"):
            item.row_id = target_r_id
        by_row[target_r_id].append(item)
    return by_row


def _cached_residual_artifacts_exist(
    context: RunContext,
    norm_doc_id: str,
    report: ResidualDocReport,
) -> bool:
    """Verify that expected scope analysis artifacts exist on disk for a cached residual report."""
    if report.mode != "residual":
        return True

    scopes = set(report.affected_scopes or [])
    if "rows" in scopes:
        if report.analyzed_fn_row_ids:
            pos_row_path = context.artifacts.get_path(
                "positive", "rows", f"{norm_doc_id}.json", iteration=None
            )
            if not pos_row_path.exists():
                return False
        if report.fp_rows > 0:
            neg_row_path = context.artifacts.get_path(
                "negative", "rows", f"{norm_doc_id}.json", iteration=None
            )
            if not neg_row_path.exists():
                return False

    if any(s.startswith("column:") for s in scopes):
        if report.analyzed_mismatch_row_ids:
            for r_id in report.analyzed_mismatch_row_ids:
                pos_col = context.artifacts.get_path(
                    "positive", "columns", f"{norm_doc_id}_{r_id}.json", iteration=None
                )
                neg_col = context.artifacts.get_path(
                    "negative", "columns", f"{norm_doc_id}_{r_id}.json", iteration=None
                )
                if not pos_col.exists() and not neg_col.exists():
                    return False
        elif report.analyzed_fn_row_ids:
            for r_id in report.analyzed_fn_row_ids:
                pos_col = context.artifacts.get_path(
                    "positive", "columns", f"{norm_doc_id}_{r_id}.json", iteration=None
                )
                if not pos_col.exists():
                    return False

    return True


def _save_merged_positive_columns(
    context: RunContext,
    norm_doc_id: str,
    row_id: str,
    new_fields: list[FieldAnalysis],
    overwrite_existing: bool,
) -> None:
    """Save or merge per-row PositiveColumnsOutput artifacts."""
    path = context.artifacts.get_path(
        "positive", "columns", f"{norm_doc_id}_{row_id}.json", iteration=None
    )
    sanitized = sanitize_positive_columns_output(
        PositiveColumnsOutput(fields=list(new_fields)), context.schema
    )
    merged: list[FieldAnalysis] = []
    seen_fields: dict[str, int] = {}

    if not overwrite_existing and path.exists():
        try:
            existing = PositiveColumnsOutput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for fa in existing.fields:
                key = fa.field_name.lower().strip()
                seen_fields[key] = len(merged)
                merged.append(fa)
        except Exception:
            pass

    for fa in sanitized.fields:
        fa.row_id = row_id
        key = fa.field_name.lower().strip()
        if key in seen_fields:
            merged[seen_fields[key]] = fa
        else:
            seen_fields[key] = len(merged)
            merged.append(fa)

    out = PositiveColumnsOutput(fields=merged)
    context.artifacts.load_or_compute(
        path, lambda o=out: o, PositiveColumnsOutput, force_recompute=True
    )


def _save_merged_negative_columns(
    context: RunContext,
    norm_doc_id: str,
    row_id: str,
    new_results: list[NegativeFieldResult],
    overwrite_existing: bool,
) -> None:
    """Save or merge per-row NegativeColumnsOutput artifacts."""
    path = context.artifacts.get_path(
        "negative", "columns", f"{norm_doc_id}_{row_id}.json", iteration=None
    )
    filtered = filter_equivalent_negative_candidates(
        NegativeColumnsOutput(fields=list(new_results)), context.schema
    )
    merged: list[NegativeFieldResult] = []
    seen_fields: dict[str, int] = {}

    if not overwrite_existing and path.exists():
        try:
            existing = NegativeColumnsOutput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for nf in existing.fields:
                key = nf.field_name.lower().strip()
                seen_fields[key] = len(merged)
                merged.append(nf)
        except Exception:
            pass

    for nf in filtered.fields:
        if not nf.candidates:
            continue
        nf.row_id = row_id
        for cand in nf.candidates:
            cand.row_id = row_id
        key = nf.field_name.lower().strip()
        if key in seen_fields:
            merged[seen_fields[key]] = nf
        else:
            seen_fields[key] = len(merged)
            merged.append(nf)

    out = NegativeColumnsOutput(fields=merged)
    context.artifacts.load_or_compute(
        path, lambda o=out: o, NegativeColumnsOutput, force_recompute=True
    )


def _resolve_generated_prompt_path(context: RunContext) -> Path:
    if getattr(context, "generated_prompt_path", None) is not None:
        return Path(context.generated_prompt_path).resolve()
    raw_cfg = getattr(context, "config_path", None)
    cfg_path = raw_cfg if isinstance(raw_cfg, (str, Path)) else None
    settings = Settings.load(config_path=cfg_path, load_env_file=False)
    return Path(settings.paths.generated_prompt_file).resolve()


def _resolve_student_llm(context: RunContext, enable_judge: bool) -> Optional[Any]:
    if not enable_judge:
        return None
    re_client = getattr(context, "re_client", None)
    if re_client is not None and hasattr(re_client, "get_validation_client"):
        try:
            val_client = re_client.get_validation_client()
            if val_client is not None and hasattr(val_client, "get_dspy_adapter"):
                return val_client.get_dspy_adapter()
        except Exception as exc:
            logger.warning(f"Could not resolve validation student_llm for semantic judge: {exc}")
    return None


def run_residual_analysis(
    context: RunContext,
    doc_ids: Optional[list[str]] = None,
) -> ResidualBatchSummary:
    """Run Error-Driven / Residual analysis for Iteration >= 2 documents."""
    target_doc_ids = list(doc_ids if doc_ids is not None else (context.doc_ids or []))
    summary = ResidualBatchSummary()
    if not target_doc_ids:
        return summary

    prompt_path = _resolve_generated_prompt_path(context)
    if not prompt_path.exists() or not prompt_path.read_text(encoding="utf-8").strip():
        logger.info("No generated_instruction.txt found for residual mode; falling back to full analysis.")
        summary.fallback_doc_ids.extend(target_doc_ids)
        return summary

    current_prompt = prompt_path.read_text(encoding="utf-8")

    eval_schema_path = context.schema_path
    gen_schema_cand = getattr(context, "generated_schema_path", None)
    if gen_schema_cand is not None and Path(gen_schema_cand).exists():
        eval_schema_path = Path(gen_schema_cand)

    eval_schema_text = (
        Path(eval_schema_path).read_text(encoding="utf-8")
        if eval_schema_path and Path(eval_schema_path).exists()
        else ""
    )

    extract_task = load_schema_complete(
        yaml_path=eval_schema_path,
        instruction_path=prompt_path,
    )
    if Path(eval_schema_path).resolve() != Path(context.schema_path).resolve():
        gt_cfg, gt_rec_model, _ = load_schema_with_models(
            yaml_path=context.schema_path,
            instruction_path=None,
        )
        gt_row_converter = create_row_converter(gt_cfg, gt_rec_model)
    else:
        gt_row_converter = getattr(extract_task, "row_converter", None)

    re_client = getattr(context, "re_client", None)
    if re_client is not None and hasattr(re_client, "create_validation_extractor"):
        import inspect

        try:
            if "ingestion_dir" in inspect.signature(re_client.create_validation_extractor).parameters:
                extractor = re_client.create_validation_extractor(
                    extract_task.signature, ingestion_dir=context.ingestion_dir
                )
            else:
                extractor = re_client.create_validation_extractor(extract_task.signature)
        except (TypeError, ValueError):
            extractor = re_client.create_validation_extractor(extract_task.signature)
    else:
        extractor = UniversalExtractor(
            extract_task.signature, ingestion_dir=context.ingestion_dir
        )

    enable_judge = bool(getattr(context, "enable_semantic_judge", True))
    student_llm = _resolve_student_llm(context, enable_judge)

    metric_config = getattr(extract_task.config, "metric_config", {})
    if not metric_config and hasattr(extract_task.config, "model_dump"):
        metric_config = extract_task.config.model_dump()
    if "compare_fields" not in metric_config:
        metric_config["compare_fields"] = list(extract_task.config.compare_fields)
    if not metric_config.get("name"):
        metric_config["name"] = (
            getattr(extract_task.config, "name", None)
            or getattr(context, "task_name", None)
            or "unknown"
        )

    descriptions = extract_task.config.judge_field_descriptions or {
        name: spec.description for name, spec in extract_task.config.fields.items()
    }
    raw_cfg = getattr(context, "config_path", None)
    cfg_path = raw_cfg if isinstance(raw_cfg, (str, Path)) else None
    try:
        settings = Settings.load(config_path=cfg_path, load_env_file=False)
        flt_tol = getattr(
            settings.syntactic_optimization,
            "float_tolerance",
            extract_task.config.float_tolerance,
        )
    except Exception:
        flt_tol = extract_task.config.float_tolerance

    if enable_judge and student_llm is not None:
        logger.info(
            f"Residual evaluation: Semantic Judge ENABLED "
            f"(model={getattr(student_llm, 'model', type(student_llm).__name__)})"
        )
    elif enable_judge:
        logger.warning(
            "Residual evaluation: enable_semantic_judge=True, but student_llm could not be resolved; "
            "falling back to strict matching."
        )
    else:
        logger.info("Residual evaluation: Semantic Judge DISABLED (strict matching only).")

    metric = TaskMetric(
        task_config=metric_config,
        float_tolerance=flt_tol,
        student_llm=student_llm,
        field_descriptions=descriptions,
        enable_semantic_judge=enable_judge,
        field_constraints=getattr(extract_task.config, "field_constraints", None),
        field_validators=getattr(extract_task.config, "field_validators", None),
    )

    doc_repo = DocumentRepository(ingestion_dir=context.ingestion_dir)
    gt_path = context.active_gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"Active Ground Truth CSV not found at {gt_path}")
    df_gt = pd.read_csv(gt_path)
    df_gt.columns = df_gt.columns.str.lower().str.strip()
    id_col = _get_id_column(df_gt)
    df_gt[id_col] = df_gt[id_col].apply(
        lambda x: normalize_document_key(str(x)) if pd.notna(x) else ""
    )

    schema_fields = context.schema_fields
    compare_fields = list(metric.matcher.fields)

    fallback_thresh = float(getattr(context, "residual_fallback_threshold", 0.35))
    skip_f1 = float(getattr(context, "residual_skip_f1", 0.98))
    max_per_sig = int(getattr(context, "residual_max_examples_per_sig", 3))
    max_fn_rows = int(getattr(context, "residual_max_fn_rows", 8))
    max_fp_rows = int(getattr(context, "residual_max_fp_rows", 10))
    raw_batch_size = getattr(context, "row_batch_size", 5)
    batch_size = max(1, raw_batch_size if isinstance(raw_batch_size, int) and raw_batch_size >= 1 else 5)

    def _equiv_fn(field_name: str, pred_val: Any, gt_val: Any) -> bool:
        return are_field_values_equivalent(
            field_name, str(pred_val), str(gt_val), context.schema
        )

    res_pos_row_pred = dspy.Predict(ResidualPositiveRowAnalysis)
    res_neg_row_pred = dspy.Predict(ResidualNegativeRowAnalysis)
    res_col_pred = dspy.Predict(ResidualColumnAnalysis)
    pos_col_pred = dspy.Predict(PositiveColumnAnalysis)

    for doc_id in target_doc_ids:
        norm_doc_id = normalize_document_key(doc_id)
        report_path = context.artifacts.get_path(
            "positive", "residual", f"{norm_doc_id}.json", iteration=None
        )

        raw_text = doc_repo.get(norm_doc_id)
        if not raw_text:
            logger.warning(
                f"Parsed text for document {norm_doc_id} not found in {context.ingestion_dir}. Skipping."
            )
            continue

        df_doc = df_gt[df_gt[id_col] == norm_doc_id].copy()
        df_doc_with_id = prepare_prompt_table(df_doc, schema_fields)
        gt_table_csv = df_doc_with_id.to_csv(index=False)

        canonical_text = doc_repo.get_canonical_manifest_text(norm_doc_id) or raw_text
        ext_model_id = ""
        re_client = getattr(context, "re_client", None)
        ext_cfg = getattr(re_client, "extraction_config", None) if re_client is not None else None
        if ext_cfg is not None:
            ext_model_id = str(getattr(ext_cfg, "model", ""))

        doc_input_hash = hashlib.sha256(
            json.dumps(
                {
                    "prompt": current_prompt,
                    "gt_csv": gt_table_csv,
                    "md_hash": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
                    "schema_hash": hashlib.sha256(
                        f"{context.schema or ''}\n{eval_schema_text}".encode("utf-8")
                    ).hexdigest(),
                    "enable_judge": bool(enable_judge),
                    "ext_model": ext_model_id,
                    "fallback_thresh": float(fallback_thresh),
                    "skip_f1": float(skip_f1),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        if getattr(context.artifacts, "resume", False) and report_path.exists():
            try:
                cached_report = ResidualDocReport.model_validate_json(
                    report_path.read_text(encoding="utf-8")
                )
                if (
                    cached_report.input_hash == doc_input_hash
                    and _cached_residual_artifacts_exist(context, norm_doc_id, cached_report)
                ):
                    summary.doc_reports.append(cached_report)
                    if cached_report.mode == "alignment_fallback":
                        summary.fallback_doc_ids.append(doc_id)
                    elif cached_report.mode == "zero_delta_skip":
                        summary.zero_delta_doc_ids.append(doc_id)
                    else:
                        summary.residual_doc_ids.append(doc_id)
                        summary.affected_scopes.update(cached_report.affected_scopes)
                    logger.info(
                        f"Loaded cached residual report for {norm_doc_id} (mode={cached_report.mode})"
                    )
                    continue
                logger.info(
                    f"Residual report cache stale for {norm_doc_id} (input hash changed); recomputing."
                )
            except Exception as exc:
                logger.warning(f"Failed to load cached residual report {report_path}: {exc}")

        gt_records: list[Any] = []
        gt_row_ids: list[str] = []
        for idx_row, (_, row_series) in enumerate(df_doc.iterrows()):
            clean_rec = {k: (None if pd.isna(v) else v) for k, v in row_series.to_dict().items()}
            if callable(gt_row_converter):
                converted = gt_row_converter(clean_rec)
            else:
                converted = clean_rec
            if converted is not None:
                gt_records.append(converted)
                gt_row_ids.append(f"row_{idx_row + 1}")

        doc_header = f"<!-- document_id: {norm_doc_id} -->\n\n"
        eval_text = raw_text if norm_doc_id in raw_text else doc_header + raw_text

        try:
            pred = extractor.forward(document_text=eval_text)
        except Exception as exc:
            logger.warning(
                f"Extractor failed on train document {norm_doc_id} ({exc}); falling back to full analysis."
            )
            fallback_report = ResidualDocReport(
                doc_id=norm_doc_id,
                mode="alignment_fallback",
                gt_count=len(gt_records),
                fallback_reason=f"extractor_error: {exc}",
                input_hash=doc_input_hash,
            )
            context.artifacts.load_or_compute(
                report_path, lambda r=fallback_report: r, ResidualDocReport, force_recompute=True
            )
            summary.doc_reports.append(fallback_report)
            summary.fallback_doc_ids.append(doc_id)
            continue

        pred_records = metric._extract_records(pred)
        raw_text = doc_repo.get(norm_doc_id) or raw_text

        eval_res = metric.matcher.evaluate_pairs(
            pred_records,
            gt_records,
            task_name=context.task_name,
            field_equivalence_fn=_equiv_fn,
        )
        doc_f1 = float(eval_res["f1"])
        doc_prec = float(eval_res["precision"])
        doc_rec = float(eval_res["recall"])

        adjusted_pairs = _split_zero_discriminative_pairs(
            eval_res["pair_results"], df_doc, compare_fields
        )
        pot_diagnostics = eval_res.get("diagnostics") or {}
        tp_pairs = [p for p in adjusted_pairs if p["status"] == "tp"]
        fn_pairs = [p for p in adjusted_pairs if p["status"] == "fn"]
        raw_fp_pairs = [p for p in adjusted_pairs if p["status"] == "fp"]
        fp_pairs = _score_fp_pu_candidates(
            raw_fp_pairs,
            gt_records,
            compare_fields,
            field_constraints=getattr(extract_task.config, "field_constraints", None),
        )
        pu_high_conf_rows = [
            {
                **_record_to_dict(p["pred"], compare_fields),
                "pu_score": p.get("pu_score", 0.0),
                "pu_reason": p.get("pu_reason", ""),
            }
            for p in fp_pairs
            if p.get("pu_status") == "high_confidence_unlabeled" and p.get("pred") is not None
        ][:max_fp_rows]
        dup_of_tp_cnt = len(pot_diagnostics.get("duplicate_of_tp", []))
        row_merge_cnt = len(pot_diagnostics.get("row_merge", []))
        row_split_cnt = len(pot_diagnostics.get("row_split", []))
        pu_high_cnt = sum(1 for p in fp_pairs if p.get("pu_status") == "high_confidence_unlabeled")
        mismatch_pairs = [p for p in adjusted_pairs if p["status"] == "field_mismatch"]

        # Guardrail 1: Alignment Fallback when F1 or Recall is too low
        if gt_records and (doc_f1 < fallback_thresh or doc_rec < fallback_thresh):
            reason = (
                f"Alignment unreliable on {norm_doc_id}: "
                f"F1={doc_f1:.3f}, Recall={doc_rec:.3f} < {fallback_thresh:.2f}"
            )
            logger.info(f"{reason}. Falling back to full stratified analysis.")
            fallback_report = ResidualDocReport(
                doc_id=norm_doc_id,
                mode="alignment_fallback",
                f1=doc_f1,
                precision=doc_prec,
                recall=doc_rec,
                pred_count=len(pred_records),
                gt_count=len(gt_records),
                tp_rows=len(tp_pairs),
                fn_rows=len(fn_pairs),
                fp_rows=len(fp_pairs),
                mismatch_pairs=len(mismatch_pairs),
                duplicate_of_tp_count=dup_of_tp_cnt,
                row_merge_count=row_merge_cnt,
                row_split_count=row_split_cnt,
                high_confidence_unlabeled_count=pu_high_cnt,
                pu_high_confidence_rows=pu_high_conf_rows,
                pot_diagnostics=pot_diagnostics,
                fallback_reason=reason,
                input_hash=doc_input_hash,
            )
            context.artifacts.load_or_compute(
                report_path, lambda r=fallback_report: r, ResidualDocReport, force_recompute=True
            )
            summary.doc_reports.append(fallback_report)
            summary.fallback_doc_ids.append(doc_id)
            continue

        # Guardrail 2: Zero-Delta Fast-Forward
        no_discrepancies = not fn_pairs and not fp_pairs and not mismatch_pairs
        if no_discrepancies or doc_f1 >= skip_f1:
            logger.info(
                f"Zero-Delta Fast-Forward for {norm_doc_id}: F1={doc_f1:.3f} "
                f"(TP={len(tp_pairs)}, FN={len(fn_pairs)}, FP={len(fp_pairs)}, Mismatch={len(mismatch_pairs)}). "
                "Skipping reasoning LLM calls."
            )
            skip_report = ResidualDocReport(
                doc_id=norm_doc_id,
                mode="zero_delta_skip",
                f1=doc_f1,
                precision=doc_prec,
                recall=doc_rec,
                pred_count=len(pred_records),
                gt_count=len(gt_records),
                tp_rows=len(tp_pairs),
                fn_rows=len(fn_pairs),
                fp_rows=len(fp_pairs),
                mismatch_pairs=len(mismatch_pairs),
                duplicate_of_tp_count=dup_of_tp_cnt,
                row_merge_count=row_merge_cnt,
                row_split_count=row_split_cnt,
                high_confidence_unlabeled_count=pu_high_cnt,
                pu_high_confidence_rows=pu_high_conf_rows,
                pot_diagnostics=pot_diagnostics,
                input_hash=doc_input_hash,
            )
            context.artifacts.load_or_compute(
                report_path, lambda r=skip_report: r, ResidualDocReport, force_recompute=True
            )
            summary.doc_reports.append(skip_report)
            summary.zero_delta_doc_ids.append(doc_id)
            continue

        # Targeted Residual Analysis
        logger.info(
            f"Residual Analysis for {norm_doc_id}: F1={doc_f1:.3f}, Recall={doc_rec:.3f} | "
            f"TP={len(tp_pairs)}, FN={len(fn_pairs)}, FP={len(fp_pairs)}, Mismatch={len(mismatch_pairs)}"
        )
        if (fn_pairs or mismatch_pairs) and has_unparsed_visuals(raw_text):
            raw_text = prepare_document_visuals(
                norm_doc_id=norm_doc_id,
                raw_text=raw_text,
                gt_table_csv=gt_table_csv,
                context=context,
                doc_repo=doc_repo,
                force_recompute=False,
            )

        doc_scopes: set[str] = set()
        written_pos_rows: set[str] = set()
        written_neg_rows: set[str] = set()
        analyzed_fn_row_ids: list[str] = []
        analyzed_mismatch_row_ids: list[str] = []
        grouped_signatures: dict[str, int] = {}

        # A. False Negative Rows -> ResidualPositiveRowAnalysis + PositiveColumnAnalysis
        if fn_pairs:
            fn_row_ids = [
                gt_row_ids[p["gt_index"]]
                for p in fn_pairs
                if p.get("gt_index") is not None and 0 <= p["gt_index"] < len(gt_row_ids)
            ]
            fn_df = df_doc_with_id[df_doc_with_id["row_id"].isin(fn_row_ids)].copy()
            sampled_fn_df = select_representative_rows(fn_df, raw_text, max_rows=max_fn_rows)
            analyzed_fn_row_ids = [str(r) for r in sampled_fn_df["row_id"].tolist()]

            tp_row_ids = [
                gt_row_ids[p["gt_index"]]
                for p in tp_pairs
                if p.get("gt_index") is not None and 0 <= p["gt_index"] < len(gt_row_ids)
            ][:2]
            contrast_tp_df = df_doc_with_id[df_doc_with_id["row_id"].isin(tp_row_ids)].copy()
            contrast_csv = contrast_tp_df.to_csv(index=False) if not contrast_tp_df.empty else ""

            if not sampled_fn_df.empty:
                logger.info(
                    f"Calling ResidualPositiveRowAnalysis for {norm_doc_id} on {len(sampled_fn_df)} FN row(s)"
                )
                pos_row_resp = res_pos_row_pred(
                    extraction_schema=context.schema,
                    current_prompt=current_prompt,
                    raw_text=raw_text,
                    gt_table=gt_table_csv,
                    missed_gt_rows=sampled_fn_df.to_csv(index=False),
                    contrast_tp_rows=contrast_csv,
                )
                pos_row_path = context.artifacts.get_path(
                    "positive", "rows", f"{norm_doc_id}.json", iteration=None
                )
                context.artifacts.load_or_compute(
                    pos_row_path,
                    lambda out=pos_row_resp.analysis: out,
                    PositiveRowsOutput,
                    force_recompute=True,
                )
                doc_scopes.add("rows")

                # Populate column analyses for missed FN rows only when no TP rows exist or column already has mismatches
                mismatched_col_names = {
                    fname.lower().strip()
                    for p in mismatch_pairs
                    for fname in p.get("mismatched_fields", {}).keys()
                }
                if not tp_pairs or mismatched_col_names:
                    non_empty_fn_indices: list[int] = []
                    for pos_i in range(len(sampled_fn_df)):
                        row_slice = sampled_fn_df.iloc[[pos_i]]
                        pop_cols = [
                            c
                            for c in row_slice.columns
                            if c != "row_id"
                            and pd.notna(row_slice[c].values[0])
                            and str(row_slice[c].values[0]).strip() != ""
                            and (not tp_pairs or c.lower().strip() in mismatched_col_names)
                        ]
                        if pop_cols:
                            non_empty_fn_indices.append(pos_i)
                            for c in pop_cols:
                                doc_scopes.add(f"column:{c}")

                    for start_i in range(0, len(non_empty_fn_indices), batch_size):
                        b_pos = non_empty_fn_indices[start_i : start_i + batch_size]
                        batch_df = sampled_fn_df.iloc[b_pos]
                        batch_row_ids = [str(r) for r in batch_df["row_id"].tolist()]
                        logger.info(
                            f"Calling PositiveColumnAnalysis for missed FN rows {norm_doc_id} {batch_row_ids}"
                        )
                        fn_col_resp = pos_col_pred(
                            extraction_schema=context.schema,
                            baseline_prompt=current_prompt,
                            raw_text=raw_text,
                            gt_table=gt_table_csv,
                            target_gt_row=batch_df.to_csv(index=False),
                        )
                        partitioned = _partition_items_by_row(
                            fn_col_resp.analysis.fields, batch_row_ids
                        )
                        for r_id in batch_row_ids:
                            _save_merged_positive_columns(
                                context,
                                norm_doc_id,
                                r_id,
                                partitioned[r_id],
                                overwrite_existing=(r_id not in written_pos_rows),
                            )
                            written_pos_rows.add(r_id)

        # B. False Positive Rows -> ResidualNegativeRowAnalysis
        if fp_pairs:
            grouped_fps = _group_fp_rows(
                fp_pairs,
                compare_fields,
                max_per_sig=max_per_sig,
                max_total=max_fp_rows,
            )
            if grouped_fps:
                logger.info(
                    f"Calling ResidualNegativeRowAnalysis for {norm_doc_id} on {len(grouped_fps)} grouped FP(s)"
                )
                neg_row_resp = res_neg_row_pred(
                    extraction_schema=context.schema,
                    current_prompt=current_prompt,
                    raw_text=raw_text,
                    gt_table=gt_table_csv,
                    false_positive_rows=json.dumps(grouped_fps, ensure_ascii=False, indent=2),
                )
                neg_row_path = context.artifacts.get_path(
                    "negative", "rows", f"{norm_doc_id}.json", iteration=None
                )
                context.artifacts.load_or_compute(
                    neg_row_path,
                    lambda out=neg_row_resp.analysis: out,
                    NegativeRowsOutput,
                    force_recompute=True,
                )
                doc_scopes.add("rows")

        # C. Aligned Field Mismatches -> ResidualColumnAnalysis (grouped by error signature)
        if mismatch_pairs:
            enriched_mismatches: list[dict[str, Any]] = []
            for p in mismatch_pairs:
                gt_idx = p.get("gt_index")
                if gt_idx is None or not (0 <= gt_idx < len(gt_row_ids)):
                    continue
                r_id = gt_row_ids[gt_idx]
                row_ctx = _record_to_dict(p["gold"], compare_fields)
                enriched_mismatches.append(
                    {
                        "row_id": r_id,
                        "row_context": row_ctx,
                        "mismatched_fields": p["mismatched_fields"],
                    }
                )

            selected_mismatches, grouped_signatures = _group_mismatch_pairs(
                enriched_mismatches, max_per_sig=max_per_sig
            )
            analyzed_mismatch_row_ids = [item["row_id"] for item in selected_mismatches]

            for item in selected_mismatches:
                for fname in item["mismatched_fields"].keys():
                    doc_scopes.add(f"column:{fname}")

            for start_i in range(0, len(selected_mismatches), batch_size):
                batch_items = selected_mismatches[start_i : start_i + batch_size]
                batch_row_ids = [item["row_id"] for item in batch_items]
                payload = []
                for item in batch_items:
                    occ = item["occurrence_count_in_doc"]
                    m_list = [
                        {
                            "field_name": fname,
                            "predicted_value": (
                                None
                                if _is_value_na(info["pred"])
                                else str(info["pred"]).strip()
                            ),
                            "gt_value": (
                                None
                                if _is_value_na(info["gold"])
                                else str(info["gold"]).strip()
                            ),
                            "mismatch_type": _classify_field_mismatch(
                                info["pred"], info["gold"], info.get("reason")
                            ),
                            "occurrence_count_in_doc": occ,
                        }
                        for fname, info in sorted(item["mismatched_fields"].items())
                    ]
                    payload.append(
                        {
                            "row_id": item["row_id"],
                            "row_context": item["row_context"],
                            "mismatches": m_list,
                        }
                    )

                logger.info(
                    f"Calling ResidualColumnAnalysis for {norm_doc_id} batch {batch_row_ids}"
                )
                col_resp = res_col_pred(
                    extraction_schema=context.schema,
                    current_prompt=current_prompt,
                    raw_text=raw_text,
                    gt_table=gt_table_csv,
                    mismatched_rows_context=json.dumps(payload, ensure_ascii=False, indent=2),
                )
                res_out: ResidualColumnsOutput = col_resp.analysis

                pos_by_row = _partition_items_by_row(res_out.positive_fields, batch_row_ids)
                neg_by_row = _partition_items_by_row(res_out.negative_fields, batch_row_ids)

                for r_id in batch_row_ids:
                    if pos_by_row[r_id]:
                        _save_merged_positive_columns(
                            context,
                            norm_doc_id,
                            r_id,
                            pos_by_row[r_id],
                            overwrite_existing=(r_id not in written_pos_rows),
                        )
                        written_pos_rows.add(r_id)
                    if neg_by_row[r_id]:
                        _save_merged_negative_columns(
                            context,
                            norm_doc_id,
                            r_id,
                            neg_by_row[r_id],
                            overwrite_existing=(r_id not in written_neg_rows),
                        )
                        written_neg_rows.add(r_id)

        doc_report = ResidualDocReport(
            doc_id=norm_doc_id,
            mode="residual",
            f1=doc_f1,
            precision=doc_prec,
            recall=doc_rec,
            pred_count=len(pred_records),
            gt_count=len(gt_records),
            tp_rows=len(tp_pairs),
            fn_rows=len(fn_pairs),
            fp_rows=len(fp_pairs),
            mismatch_pairs=len(mismatch_pairs),
            duplicate_of_tp_count=dup_of_tp_cnt,
            row_merge_count=row_merge_cnt,
            row_split_count=row_split_cnt,
            high_confidence_unlabeled_count=pu_high_cnt,
            pu_high_confidence_rows=pu_high_conf_rows,
            pot_diagnostics=pot_diagnostics,
            analyzed_fn_row_ids=analyzed_fn_row_ids,
            analyzed_mismatch_row_ids=analyzed_mismatch_row_ids,
            grouped_signatures=grouped_signatures,
            affected_scopes=sorted(doc_scopes),
            input_hash=doc_input_hash,
        )
        context.artifacts.load_or_compute(
            report_path, lambda r=doc_report: r, ResidualDocReport, force_recompute=True
        )
        summary.doc_reports.append(doc_report)
        summary.residual_doc_ids.append(doc_id)
        summary.affected_scopes.update(doc_scopes)

    return summary
