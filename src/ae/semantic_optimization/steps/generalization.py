# src/ae/reverse_engineering/steps/generalization.py
"""Unified Rule Synthesis, Batch HITL Triage, and Prompt Harmonization step."""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional
import pandas as pd
import dspy

from ae.core.utils import normalize_document_key
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.models import (
    ConsolidatedColumnsOutput,
    ConsolidatedRowsOutput,
    GeneralizedColumnsOutput,
    GeneralizedRowsOutput,
    REConflict,
)
from ae.semantic_optimization.signatures import (
    ColumnGeneralization,
    PromptHarmonization,
    RowGeneralization,
)
from ae.semantic_optimization.steps.consolidation import _invalidate_doc_artifacts
from ae.semantic_optimization.steps.positive_analysis import filter_equivalent_conflicts
from ae.semantic_optimization.hitl import _get_id_column

logger = logging.getLogger(__name__)


def _clean_header(text: str) -> str:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    return re.sub(r"^[#*\s]+|[:*\s]+$", "", first_line).upper()


def _strip_cjk_noise(text: str) -> str:
    """Remove accidental CJK/Han characters injected by multilingual reasoning models while preserving line structure."""
    cleaned_lines = []
    for line in text.splitlines():
        no_cjk = re.sub(r"[\u4e00-\u9fff]+", "", line)
        no_cjk = re.sub(r" {2,}", " ", no_cjk).rstrip()
        cleaned_lines.append(no_cjk)
    trailing_nl = "\n" if text.endswith("\n") else ""
    return "\n".join(cleaned_lines) + trailing_nl


def _collect_pos_neg_rows(
    context: RunContext, active_norm_ids: set[str]
) -> tuple[list[dict], list[dict]]:
    """Collect positive and negative row analyses across active documents, skipping ignored examples."""
    pos_rows: list[dict] = []
    pos_rows_dir = context.artifacts.get_phase_dir("positive", "rows", iteration=None)
    if pos_rows_dir.exists():
        for file_path in sorted(pos_rows_dir.glob("*.json")):
            doc_id = file_path.stem
            if active_norm_ids and doc_id not in active_norm_ids:
                continue
            try:
                content = json.loads(file_path.read_text(encoding="utf-8"))
                for row in content.get("rows", []):
                    if context.rulings_store.is_ignored(doc_id, row.get("row_id", "")):
                        continue
                    row["document"] = doc_id
                    pos_rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to read row analysis {file_path}: {e}")

    neg_rows: list[dict] = []
    neg_rows_dir = context.artifacts.get_phase_dir("negative", "rows", iteration=None)
    if neg_rows_dir.exists():
        for file_path in sorted(neg_rows_dir.glob("*.json")):
            doc_id = file_path.stem
            if active_norm_ids and doc_id not in active_norm_ids:
                continue
            try:
                content = json.loads(file_path.read_text(encoding="utf-8"))
                for row in content.get("gap_rows", []):
                    if context.rulings_store.is_ignored(doc_id, row.get("gap_row_id", "")):
                        continue
                    row["document"] = doc_id
                    neg_rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to read negative row analysis {file_path}: {e}")

    return pos_rows, neg_rows


def _compact_positive_col_entries(entries: list[dict], max_per_sig: int = 2) -> list[dict]:
    """Deduplicate homogeneous positive column analyses per document while preserving all anomaly/issue entries."""
    if len(entries) <= 6:
        return entries

    anomalous: list[dict] = []
    buckets: dict[tuple[str, ...], list[dict]] = {}
    for entry in entries:
        issue = str(entry.get("Suspected_Ingestion_Or_GT_Issue", "None") or "None").strip()
        if issue.lower() not in ("", "none"):
            anomalous.append(entry)
            continue
        sig = (
            str(entry.get("document", "")).strip(),
            str(entry.get("Semantic_Core", "")).strip().lower(),
            str(entry.get("Analytical_Method", "")).strip().lower(),
            str(entry.get("Hierarchy_Level_class", "")).strip(),
            str(entry.get("Semantic_Binding_class", "")).strip(),
            str(entry.get("Transformation_description", "")).strip().lower(),
        )
        buckets.setdefault(sig, []).append(entry)

    compacted: list[dict] = list(anomalous)
    for group in buckets.values():
        for item in group[:max_per_sig]:
            copied = dict(item)
            if len(group) > max_per_sig:
                copied["occurrence_count"] = len(group)
            compacted.append(copied)
    return compacted


def _collect_pos_neg_cols(
    context: RunContext, active_norm_ids: set[str]
) -> tuple[list[dict], list[dict]]:
    """Collect positive and negative column analyses across active documents, skipping ignored examples."""
    pos_cols: list[dict] = []
    pos_cols_dir = context.artifacts.get_phase_dir("positive", "columns", iteration=None)
    if pos_cols_dir.exists():
        for file_path in sorted(pos_cols_dir.glob("*.json")):
            try:
                stem = file_path.stem
                if "_row_" in stem:
                    doc_id, row_num = stem.rsplit("_row_", 1)
                    row_id = f"row_{row_num}"
                else:
                    doc_id = stem
                    row_id = "row_1"

                if active_norm_ids and doc_id not in active_norm_ids:
                    continue

                content = json.loads(file_path.read_text(encoding="utf-8"))
                for field_analysis in content.get("fields", []):
                    f_name = field_analysis.get("field_name", "")
                    if context.rulings_store.is_ignored(doc_id, row_id, f_name):
                        continue
                    field_analysis["document"] = doc_id
                    field_analysis["row_id"] = row_id
                    pos_cols.append(field_analysis)
            except Exception as e:
                logger.warning(f"Failed to read positive column analysis {file_path}: {e}")

    neg_cols: list[dict] = []
    neg_cols_dir = context.artifacts.get_phase_dir("negative", "columns", iteration=None)
    if neg_cols_dir.exists():
        for file_path in sorted(neg_cols_dir.glob("*.json")):
            try:
                stem = file_path.stem
                if "_row_" in stem:
                    doc_id, row_num = stem.rsplit("_row_", 1)
                    row_id = f"row_{row_num}"
                else:
                    doc_id = stem
                    row_id = "row_1"

                if active_norm_ids and doc_id not in active_norm_ids:
                    continue

                content = json.loads(file_path.read_text(encoding="utf-8"))
                for field_result in content.get("fields", []):
                    field_name = field_result.get("field_name", "")
                    for candidate in field_result.get("candidates", []):
                        if context.rulings_store.is_ignored(doc_id, row_id, field_name):
                            continue
                        candidate["field_name"] = field_name
                        candidate["document"] = doc_id
                        candidate["row_id"] = row_id
                        neg_cols.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to read negative column analysis {file_path}: {e}")

    return pos_cols, neg_cols


def _is_conflict_resolved(context: RunContext, conflict: REConflict) -> bool:
    """Check whether conflict is already covered by RulingsStore."""
    store = getattr(context, "rulings_store", None)
    if store is None:
        return False
    if hasattr(store, "has_matching_ruling"):
        return bool(store.has_matching_ruling(conflict))
    if hasattr(store, "has_ruling"):
        return bool(store.has_ruling(conflict.conflict_id))
    return False


def _resolve_consolidation_policy(
    context: RunContext,
    category: str,
    filename: str,
    iter_num: Optional[int],
) -> tuple[str, Path]:
    """Resolve consolidated policy JSON string (with iter_{k-1} and root fallback) plus target iter path."""
    iter_target_path = context.artifacts.get_path(
        "consolidation", category, filename, iteration=iter_num
    )
    candidates: list[Path] = [iter_target_path]
    if iter_num is not None and iter_num > 1:
        candidates.append(
            context.artifacts.get_path(
                "consolidation", category, filename, iteration=iter_num - 1
            )
        )
    candidates.append(
        context.artifacts.get_path("consolidation", category, filename, iteration=None)
    )
    for cand in candidates:
        if cand.exists():
            return cand.read_text(encoding="utf-8"), iter_target_path
    return "{}", iter_target_path


def _scope_rulings_hash(context: RunContext, scope_key: str) -> str:
    """Compute a deterministic hash of expert rulings and ignored examples for a scope."""
    rulings_text = context.rulings_store.format_rulings_for_prompt(scope_key)
    all_ignored = sorted(getattr(context.rulings_store, "ignored_examples", set()) or [])
    if scope_key.startswith("column:"):
        field_lower = scope_key.split(":", 1)[1].lower().strip()
        relevant_ignored = [
            item for item in all_ignored
            if item.rsplit(":", 1)[-1].lower().strip() == field_lower
        ]
    else:
        relevant_ignored = all_ignored
    return hashlib.sha256(
        json.dumps(
            {
                "rulings_prompt": rulings_text,
                "ignored_examples": relevant_ignored,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _empty_scope_rulings_hash() -> str:
    """Hash corresponding to an empty rulings_store for a scope."""
    return hashlib.sha256(
        json.dumps(
            {"rulings_prompt": "", "ignored_examples": []},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_rulings_hash(artifact_path: Path, rulings_hash: str) -> None:
    """Persist the scope rulings hash sidecar next to a generalization artifact."""
    hash_path = artifact_path.with_suffix(artifact_path.suffix + ".rulings.sha256")
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(rulings_hash, encoding="utf-8")


def _can_carry_forward_rulings(prev_path: Path, context: RunContext, scope_key: str) -> bool:
    """Return True if expert_rulings for scope_key have not changed since prev_path was created."""
    curr_hash = _scope_rulings_hash(context, scope_key)
    prev_hash_path = prev_path.with_suffix(prev_path.suffix + ".rulings.sha256")
    if prev_hash_path.exists():
        return prev_hash_path.read_text(encoding="utf-8").strip() == curr_hash
    # Backward compatibility when sidecar is absent: allow carry-forward only if no rulings exist
    return curr_hash == _empty_scope_rulings_hash()


def _compute_row_cache_key(
    context: RunContext,
    pos_rows: list,
    neg_rows: list,
) -> str:
    """Compute deterministic cache key for rows.json generalization artifact."""
    return hashlib.sha256(
        json.dumps(
            {
                "schema": context.schema,
                "baseline_prompt": context.baseline_prompt,
                "pos_rows": pos_rows,
                "neg_rows": neg_rows,
                "expert_rulings": context.rulings_store.format_rulings_for_prompt("rows"),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _compute_col_cache_key(
    context: RunContext,
    field: str,
    pos_field_analyses: list,
    neg_field_candidates: list,
) -> str:
    """Compute deterministic cache key for columns/{field}.json generalization artifact."""
    return hashlib.sha256(
        json.dumps(
            {
                "schema": context.schema,
                "baseline_prompt": context.baseline_prompt,
                "field": field,
                "pos_cols": pos_field_analyses,
                "neg_cols": neg_field_candidates,
                "expert_rulings": context.rulings_store.format_rulings_for_prompt(f"column:{field}"),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def run_generalization(
    context: RunContext,
    changed_scopes: Optional[set[str]] = None,
) -> bool:
    """Execute Unified Rule Synthesis, Batch HITL Triage, and Prompt Harmonization.

    Args:
        context: Active semantic optimization RunContext.
        changed_scopes: Optional set of scopes (e.g. ``{"rows", "column:bacteria"}``)
            that received new residual/fallback examples in ``iter_num > 1``. When provided,
            unchanged scopes are carried forward directly from ``iter_{k-1}``.

    Returns:
        needs_doc_recompute (bool): True if human expert patched .md or added a GT row requiring 1A/1B rerun.
    """
    logger.info("Starting Unified Rule Synthesis & Generalization")

    gt_path = context.active_gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"Active Ground Truth CSV not found at {gt_path}")
    df_gt = pd.read_csv(gt_path)
    df_gt.columns = df_gt.columns.str.lower().str.strip()
    id_col = _get_id_column(df_gt)

    schema_fields = context.schema_fields
    fields = [col for col in df_gt.columns if col in schema_fields]

    active_norm_ids = {normalize_document_key(d) for d in context.doc_ids} if context.doc_ids is not None else set()
    iter_num = context.current_iteration
    force_recompute = not getattr(context, "resume", False)
    can_carry_forward = (
        iter_num is not None and iter_num > 1 and changed_scopes is not None
    )

    # 1. Step 2.1: Row Rule Synthesis
    logger.info("Executing Row Rule Synthesis")
    row_rules_path = context.artifacts.get_path("generalization", "", "rows.json", iteration=iter_num)
    pos_rows, neg_rows = _collect_pos_neg_rows(context, active_norm_ids)

    cons_policy_str, rows_cons_path = _resolve_consolidation_policy(
        context, "", "rows.json", iter_num
    )

    row_predictor = dspy.Predict(RowGeneralization)

    def compute_row_generalization(p_rows=pos_rows, n_rows=neg_rows):
        pos_str = json.dumps(p_rows, ensure_ascii=False, indent=2)
        neg_str = json.dumps(n_rows, ensure_ascii=False, indent=2)
        expert_rulings = context.rulings_store.format_rulings_for_prompt("rows")

        logger.info("Calling RowGeneralization LLM")
        response = row_predictor(
            extraction_schema=context.schema,
            baseline_prompt=context.baseline_prompt,
            positive_rows_analyses=pos_str,
            negative_rows_analyses=neg_str,
            consolidated_policy=cons_policy_str,
            expert_rulings=expert_rulings,
        )
        return response.generalization

    carried_row_data: Optional[GeneralizedRowsOutput] = None
    if (
        can_carry_forward
        and changed_scopes is not None
        and "rows" not in changed_scopes
        and not (getattr(context, "resume", False) and row_rules_path.exists())
    ):
        assert iter_num is not None
        prev_row_path = context.artifacts.get_path(
            "generalization", "", "rows.json", iteration=iter_num - 1
        )
        if prev_row_path.exists() and _can_carry_forward_rulings(prev_row_path, context, "rows"):
            try:
                carried_row_data = GeneralizedRowsOutput.model_validate_json(
                    prev_row_path.read_text(encoding="utf-8")
                )
                logger.info(
                    f"Iter {iter_num}: Carrying forward unchanged 'rows' rules from iter_{iter_num - 1}."
                )
            except Exception as exc:
                logger.warning(f"Failed to carry forward rows from {prev_row_path}: {exc}")
        elif prev_row_path.exists():
            logger.info(
                f"Iter {iter_num}: Skipping carry-forward for 'rows' because expert_rulings changed."
            )

    row_cache_key = _compute_row_cache_key(context, pos_rows, neg_rows)
    if carried_row_data is not None:
        row_data = context.artifacts.load_or_compute(
            row_rules_path,
            lambda out=carried_row_data: out,
            GeneralizedRowsOutput,
            force_recompute=True,
            cache_key=row_cache_key,
        )
    else:
        row_data = context.artifacts.load_or_compute(
            row_rules_path,
            compute_row_generalization,
            GeneralizedRowsOutput,
            force_recompute=force_recompute,
            cache_key=row_cache_key,
        )
    _write_rulings_hash(row_rules_path, _scope_rulings_hash(context, "rows"))

    # 2. Step 2.2: Column Rule Synthesis
    logger.info("Executing Column Rule Synthesis")
    col_predictor = dspy.Predict(ColumnGeneralization)
    pos_cols, neg_cols = _collect_pos_neg_cols(context, active_norm_ids)

    compiled_columns: dict[str, GeneralizedColumnsOutput] = {}
    col_cons_targets: dict[str, Path] = {}
    col_policy_map: dict[str, str] = {}
    for field in fields:
        field_lower = field.lower().strip()
        pos_field_analyses = _compact_positive_col_entries(
            [pa for pa in pos_cols if pa.get("field_name", "").lower().strip() == field_lower]
        )
        neg_field_candidates = [na for na in neg_cols if na.get("field_name", "").lower().strip() == field_lower]

        col_policy_str, col_cons_target = _resolve_consolidation_policy(
            context, "columns", f"{field}.json", iter_num
        )
        col_cons_targets[field] = col_cons_target
        col_policy_map[field] = col_policy_str

        col_rules_path = context.artifacts.get_path("generalization", "columns", f"{field}.json", iteration=iter_num)

        def compute_column_generalization(
            f_name=field, pos=pos_field_analyses, neg=neg_field_candidates, pol=col_policy_str
        ):
            pos_str = json.dumps(pos, ensure_ascii=False, indent=2)
            neg_str = json.dumps(neg, ensure_ascii=False, indent=2)
            expert_rulings = context.rulings_store.format_rulings_for_prompt(f"column:{f_name}")

            logger.info(f"Calling ColumnGeneralization LLM for field {f_name}")
            response = col_predictor(
                extraction_schema=context.schema,
                baseline_prompt=context.baseline_prompt,
                field_name=f_name,
                positive_columns_analyses=pos_str,
                negative_columns_analyses=neg_str,
                consolidated_policy=pol,
                expert_rulings=expert_rulings,
            )
            return response.generalization

        scope_key = f"column:{field}"
        carried_col_data: Optional[GeneralizedColumnsOutput] = None
        if (
            can_carry_forward
            and changed_scopes is not None
            and scope_key not in changed_scopes
            and not (getattr(context, "resume", False) and col_rules_path.exists())
        ):
            assert iter_num is not None
            prev_col_path = context.artifacts.get_path(
                "generalization", "columns", f"{field}.json", iteration=iter_num - 1
            )
            if prev_col_path.exists() and _can_carry_forward_rulings(prev_col_path, context, scope_key):
                try:
                    carried_col_data = GeneralizedColumnsOutput.model_validate_json(
                        prev_col_path.read_text(encoding="utf-8")
                    )
                    logger.info(
                        f"Iter {iter_num}: Carrying forward unchanged '{scope_key}' rules from iter_{iter_num - 1}."
                    )
                except Exception as exc:
                    logger.warning(f"Failed to carry forward {field} from {prev_col_path}: {exc}")
            elif prev_col_path.exists():
                logger.info(
                    f"Iter {iter_num}: Skipping carry-forward for '{scope_key}' because expert_rulings changed."
                )

        col_cache_key = _compute_col_cache_key(
            context, field, pos_field_analyses, neg_field_candidates
        )
        if carried_col_data is not None:
            col_data = context.artifacts.load_or_compute(
                col_rules_path,
                lambda out=carried_col_data: out,
                GeneralizedColumnsOutput,
                force_recompute=True,
                cache_key=col_cache_key,
            )
        else:
            col_data = context.artifacts.load_or_compute(
                col_rules_path,
                compute_column_generalization,
                GeneralizedColumnsOutput,
                force_recompute=force_recompute,
                cache_key=col_cache_key,
            )
        _write_rulings_hash(col_rules_path, _scope_rulings_hash(context, scope_key))
        compiled_columns[field] = col_data

    # 3. Unified Batch HITL Conflict Digest
    def _clean_and_collect_conflicts() -> list[REConflict]:
        nonlocal row_data
        collected: list[REConflict] = []

        orig_row_conflicts = list(row_data.Conflicts)
        eq_row_conflicts = filter_equivalent_conflicts(orig_row_conflicts, context.schema)
        unresolved_row_conflicts = [
            c for c in eq_row_conflicts if not _is_conflict_resolved(context, c)
        ]
        if [c.conflict_id for c in unresolved_row_conflicts] != [
            c.conflict_id for c in orig_row_conflicts
        ]:
            row_data.Conflicts = unresolved_row_conflicts
            row_data = context.artifacts.load_or_compute(
                row_rules_path,
                lambda out=row_data: out,
                GeneralizedRowsOutput,
                force_recompute=True,
                cache_key=_compute_row_cache_key(context, pos_rows, neg_rows),
            )
            _write_rulings_hash(row_rules_path, _scope_rulings_hash(context, "rows"))
        collected.extend(row_data.Conflicts)

        for field in fields:
            if field not in compiled_columns:
                continue
            col_out = compiled_columns[field]
            orig_col_conflicts = list(col_out.Conflicts)
            if not orig_col_conflicts:
                continue
            eq_col_conflicts = filter_equivalent_conflicts(orig_col_conflicts, context.schema)
            unresolved_col_conflicts = [
                c for c in eq_col_conflicts if not _is_conflict_resolved(context, c)
            ]
            if [c.conflict_id for c in unresolved_col_conflicts] != [
                c.conflict_id for c in orig_col_conflicts
            ]:
                col_out.Conflicts = unresolved_col_conflicts
                field_lower = field.lower().strip()
                pos_f = _compact_positive_col_entries(
                    [pa for pa in pos_cols if pa.get("field_name", "").lower().strip() == field_lower]
                )
                neg_f = [na for na in neg_cols if na.get("field_name", "").lower().strip() == field_lower]
                col_rules_path = context.artifacts.get_path(
                    "generalization", "columns", f"{field}.json", iteration=iter_num
                )
                col_out = context.artifacts.load_or_compute(
                    col_rules_path,
                    lambda out=col_out: out,
                    GeneralizedColumnsOutput,
                    force_recompute=True,
                    cache_key=_compute_col_cache_key(context, field, pos_f, neg_f),
                )
                _write_rulings_hash(col_rules_path, _scope_rulings_hash(context, f"column:{field}"))
                compiled_columns[field] = col_out
            collected.extend(compiled_columns[field].Conflicts)

        return collected

    did_rule_recompute = False
    max_hitl_rounds = 2
    for hitl_round in range(1, max_hitl_rounds + 1):
        all_conflicts = _clean_and_collect_conflicts()
        if not all_conflicts:
            break

        logger.info(
            f"Unified HITL Triage (round {hitl_round}/{max_hitl_rounds}): "
            f"resolving batch of {len(all_conflicts)} conflict(s)."
        )
        if hasattr(context.resolver, "resolve_batch"):
            batch_res = context.resolver.resolve_batch(all_conflicts)
            if batch_res.needs_doc_recompute:
                for doc_id in batch_res.recompute_doc_ids:
                    _invalidate_doc_artifacts(context, doc_id)
                    if hasattr(context.gt_editor, "reindex_doc_rows"):
                        context.gt_editor.reindex_doc_rows(doc_id)
                if row_rules_path.exists():
                    row_rules_path.unlink()
                for field in fields:
                    c_path = context.artifacts.get_path(
                        "generalization", "columns", f"{field}.json", iteration=iter_num
                    )
                    if c_path.exists():
                        c_path.unlink()
                setattr(context, "_recompute_doc_ids", sorted(list(batch_res.recompute_doc_ids)))
                return True

            if batch_res.needs_rule_recompute:
                did_rule_recompute = True
                logger.info(
                    f"Recomputing targeted rule synthesis after HITL rulings (round {hitl_round}) "
                    f"for scopes: {batch_res.recompute_scopes}"
                )
                pos_rows, neg_rows = _collect_pos_neg_rows(context, active_norm_ids)
                pos_cols, neg_cols = _collect_pos_neg_cols(context, active_norm_ids)

                if "rows" in batch_res.recompute_scopes:
                    recomputed_row_cache_key = _compute_row_cache_key(
                        context, pos_rows, neg_rows
                    )
                    row_data = context.artifacts.load_or_compute(
                        row_rules_path,
                        lambda: compute_row_generalization(pos_rows, neg_rows),
                        GeneralizedRowsOutput,
                        force_recompute=True,
                        cache_key=recomputed_row_cache_key,
                    )
                    _write_rulings_hash(row_rules_path, _scope_rulings_hash(context, "rows"))

                for field in fields:
                    scope_key = f"column:{field}"
                    if scope_key in batch_res.recompute_scopes:
                        field_lower = field.lower().strip()
                        pos_f = _compact_positive_col_entries(
                            [pa for pa in pos_cols if pa.get("field_name", "").lower().strip() == field_lower]
                        )
                        neg_f = [na for na in neg_cols if na.get("field_name", "").lower().strip() == field_lower]
                        col_rules_path = context.artifacts.get_path(
                            "generalization", "columns", f"{field}.json", iteration=iter_num
                        )
                        recomputed_col_cache_key = _compute_col_cache_key(
                            context, field, pos_f, neg_f
                        )
                        recomputed_col = context.artifacts.load_or_compute(
                            col_rules_path,
                            lambda f_name=field, pos=pos_f, neg=neg_f, pol=col_policy_map.get(field, "{}"): compute_column_generalization(
                                f_name=f_name, pos=pos, neg=neg, pol=pol
                            ),
                            GeneralizedColumnsOutput,
                            force_recompute=True,
                            cache_key=recomputed_col_cache_key,
                        )
                        _write_rulings_hash(col_rules_path, _scope_rulings_hash(context, scope_key))
                        compiled_columns[field] = recomputed_col

                remaining_conflicts = _clean_and_collect_conflicts()
                if hitl_round == max_hitl_rounds:
                    if remaining_conflicts:
                        logger.warning(
                            f"HITL Triage reached max_hitl_rounds={max_hitl_rounds} with "
                            f"{len(remaining_conflicts)} unresolved conflict(s) remaining."
                        )
                    break
                continue

            _clean_and_collect_conflicts()
            break
        else:
            # Fallback if custom resolver without resolve_batch is injected
            needs_recompute_total = False
            for conflict in all_conflicts:
                _, recompute = context.resolver.resolve(conflict)
                if recompute:
                    _invalidate_doc_artifacts(context, conflict.document_id)
                    if hasattr(context.gt_editor, "reindex_doc_rows"):
                        context.gt_editor.reindex_doc_rows(conflict.document_id)
                    needs_recompute_total = True
            if needs_recompute_total:
                if row_rules_path.exists():
                    row_rules_path.unlink()
                for field in fields:
                    c_path = context.artifacts.get_path(
                        "generalization", "columns", f"{field}.json", iteration=iter_num
                    )
                    if c_path.exists():
                        c_path.unlink()
                return True
            _clean_and_collect_conflicts()
            break

    # Persist consolidated artifacts for this iteration so they always match final row_data & compiled_columns
    if row_data.Row_Instantiation_Trigger_generalized:
        cons_row_out = ConsolidatedRowsOutput(
            Row_Instantiation_Trigger_generalized=row_data.Row_Instantiation_Trigger_generalized,
            Entity_Role_and_Filtration_Policy=row_data.Entity_Role_and_Filtration_Policy,
            Conflicts=[],
        )
        context.artifacts.load_or_compute(
            rows_cons_path, lambda out=cons_row_out: out, ConsolidatedRowsOutput, force_recompute=True
        )

    for field in fields:
        if field in compiled_columns and compiled_columns[field].Semantic_Core_generalized:
            col_cons_target = col_cons_targets.get(
                field, context.artifacts.get_path("consolidation", "columns", f"{field}.json", iteration=iter_num)
            )
            cc_data = compiled_columns[field]
            cons_col_out = ConsolidatedColumnsOutput(
                Semantic_Core_generalized=cc_data.Semantic_Core_generalized,
                Analytical_and_Binding_Policy=cc_data.Analytical_and_Binding_Policy,
                Conflicts=[],
            )
            context.artifacts.load_or_compute(
                col_cons_target, lambda out=cons_col_out: out, ConsolidatedColumnsOutput, force_recompute=True
            )

    # 4. Final Prompt Assembly & Harmonization
    logger.info("Executing Prompt Assembly")
    baseline_text = context.baseline_prompt
    sections = [s.strip() for s in baseline_text.split("---") if s.strip()]

    preamble = ""
    reasoning_style = ""
    preserved_policy_sections: list[str] = []
    upper_fields = {f.upper() for f in fields}

    candidate_sections = list(sections)
    if len(candidate_sections) > 1:
        preamble_parts = [candidate_sections[0]]
        candidate_sections = candidate_sections[1:]
        if candidate_sections:
            sec_header = _clean_header(candidate_sections[0])
            if (
                sec_header not in upper_fields
                and "RECORD SELECTION" not in sec_header
                and "EXPERIMENT SELECTION" not in sec_header
                and "REASONING STYLE" not in sec_header
            ):
                preamble_parts.append(candidate_sections[0])
                candidate_sections = candidate_sections[1:]
        preamble = "\n\n---\n\n".join(preamble_parts)

        for sec in candidate_sections:
            sec_header = _clean_header(sec)
            if "REASONING STYLE" in sec_header or "REASONING STYLE" in sec.upper():
                reasoning_style = sec
            elif (
                sec_header not in upper_fields
                and "EXPERIMENT SELECTION" not in sec_header
                and "RECORD SELECTION" not in sec_header
            ):
                preserved_policy_sections.append(sec)
    else:
        # Baseline prompt is unsectioned (arbitrary format such as XML, Markdown, or free-form text);
        # use a neutral preamble here and let PromptHarmonization integrate non-conflicting guidance.
        preamble = (
            "You are an information extraction system. "
            "Your task is to extract structured data from scientific articles.\n\n"
            "Each record must be represented as a separate JSON object."
        )

    assembled_prompt_parts = [preamble]
    assembled_prompt_parts.extend(preserved_policy_sections)

    # Add authoritative rulings if any exist
    general_rulings = context.rulings_store.format_rulings_for_prompt("all", only_policy_rules=True)
    if general_rulings:
        assembled_prompt_parts.append(general_rulings)

    # Record Selection rules
    assembled_prompt_parts.append(
        "RECORD SELECTION\n\n"
        "Include ONLY records satisfying:\n"
        + "\n".join(f"- {inst}" for inst in row_data.Instructions.Row_Inclusion_Instructions)
        + "\n\nExclude:\n"
        + "\n".join(f"- {inst}" for inst in row_data.Instructions.Row_Exclusion_Instructions)
    )

    # Field specific rules
    for field in fields:
        if field not in compiled_columns:
            continue
        cc = compiled_columns[field]
        field_section = f"{field.upper()}\n\n"

        if cc.Instructions.Column_Inclusion_Instructions:
            field_section += (
                "Extract ONLY if satisfy:\n"
                + "\n".join(f"- {inst}" for inst in cc.Instructions.Column_Inclusion_Instructions)
                + "\n\n"
            )

        if cc.Instructions.Column_Exclusion_Instructions:
            field_section += (
                "DO NOT extract:\n"
                + "\n".join(f"- {inst}" for inst in cc.Instructions.Column_Exclusion_Instructions)
                + "\n\n"
            )

        if cc.Instructions.Transformation_Instructions:
            field_section += (
                "Transformation rules:\n"
                + "\n".join(f"- {inst}" for inst in cc.Instructions.Transformation_Instructions)
                + "\n"
            )

        assembled_prompt_parts.append(field_section.strip())

    if reasoning_style:
        assembled_prompt_parts.append(reasoning_style)
    draft_prompt = "\n\n---\n\n".join(assembled_prompt_parts) + "\n"
    final_prompt = draft_prompt

    prompt_cache_key = hashlib.sha256(
        json.dumps(
            {
                "draft_prompt": draft_prompt,
                "general_rulings": general_rulings or "",
                "schema": context.schema,
                "baseline_prompt": context.baseline_prompt,
                "harmonize_prompt": bool(getattr(context, "harmonize_prompt", True)),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    cached_iter_prompt_path: Optional[Path] = None
    cached_iter_prompt_hash_path: Optional[Path] = None
    if iter_num is not None:
        cached_iter_prompt_path = context.artifacts.get_path(
            "generalization", "", "candidate_prompt.txt", iteration=iter_num
        )
        cached_iter_prompt_hash_path = cached_iter_prompt_path.with_suffix(".txt.sha256")

    reused_cached_prompt = False
    if (
        getattr(context, "resume", False)
        and not did_rule_recompute
        and cached_iter_prompt_path is not None
        and cached_iter_prompt_path.exists()
        and cached_iter_prompt_hash_path is not None
        and cached_iter_prompt_hash_path.exists()
        and cached_iter_prompt_hash_path.read_text(encoding="utf-8").strip() == prompt_cache_key
    ):
        cached_text = cached_iter_prompt_path.read_text(encoding="utf-8")
        if cached_text.strip():
            final_prompt = cached_text
            reused_cached_prompt = True
            logger.info(
                f"Iter {iter_num}: Reusing cached candidate prompt from {cached_iter_prompt_path} on resume."
            )

    # Optional LLM Prompt Harmonization step
    if not reused_cached_prompt and getattr(context, "harmonize_prompt", True):
        try:
            harmonizer = dspy.Predict(PromptHarmonization)
            harm_resp = harmonizer(
                extraction_schema=context.schema,
                baseline_prompt=context.baseline_prompt,
                expert_rulings=general_rulings or "None",
                draft_prompt=draft_prompt,
            )
            if (
                hasattr(harm_resp, "harmonization")
                and getattr(harm_resp.harmonization, "harmonized_prompt", "")
                and harm_resp.harmonization.harmonized_prompt.strip()
            ):
                final_prompt = harm_resp.harmonization.harmonized_prompt.strip() + "\n"
                logger.info("Successfully harmonized final prompt across all columns and baseline rules.")
        except Exception as e:
            logger.warning(f"PromptHarmonization skipped or fell back to assembled draft prompt: {e}")

    final_prompt = _strip_cjk_noise(final_prompt)

    if getattr(context, "generated_prompt_path", None) is not None:
        generated_instruction_path = Path(context.generated_prompt_path).resolve()
    else:
        from ae.core.config.settings import Settings

        raw_cfg = getattr(context, "config_path", None)
        cfg_path = raw_cfg if isinstance(raw_cfg, (str, Path)) else None
        settings = Settings.load(config_path=cfg_path, load_env_file=False)
        generated_instruction_path = Path(settings.paths.generated_prompt_file).resolve()
    generated_instruction_path.parent.mkdir(parents=True, exist_ok=True)
    generated_instruction_path.write_text(final_prompt, encoding="utf-8")

    if iter_num is not None:
        iter_prompt_path = context.artifacts.get_path(
            "generalization", "", "candidate_prompt.txt", iteration=iter_num
        )
        iter_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        iter_prompt_path.write_text(final_prompt, encoding="utf-8")
        iter_prompt_path.with_suffix(".txt.sha256").write_text(prompt_cache_key, encoding="utf-8")

    logger.info(f"Successfully assembled and saved generated instruction prompt to {generated_instruction_path}")

    if getattr(context, "infer_schema_constraints", True):
        try:
            from ae.semantic_optimization.steps.schema_refinement import run_schema_refinement

            refined_schema_path = run_schema_refinement(
                context=context,
                df_gt=df_gt,
                id_col=id_col,
                compiled_columns=compiled_columns,
                final_prompt=final_prompt,
                active_norm_ids=active_norm_ids,
            )
            if refined_schema_path is not None:
                logger.info(f"Saved refined schema configuration to {refined_schema_path}")
        except Exception as e:
            logger.warning(f"Schema constraint inference skipped or fell back to base schema: {e}")

    logger.info("Finished Unified Rule Synthesis & Generalization")
    return False
