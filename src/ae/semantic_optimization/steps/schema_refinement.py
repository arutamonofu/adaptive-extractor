"""Schema constraint inference and safe refinement step for Reverse Engineering."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
import pandas as pd

from ae.core.schema.config import SchemaConfig
from ae.core.schema.loader import load_schema_from_yaml, save_schema_to_yaml
from ae.core.utils import normalize_document_key
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.models import (
    FieldConstraintProposal,
    GeneralizedColumnsOutput,
    SchemaRefinementOutput,
)
from ae.semantic_optimization.signatures.schema import SchemaConstraintInference

logger = logging.getLogger(__name__)

_MAX_ENUM_CHOICES = 25
_DEFAULT_MIN_ENUM_DOC_SUPPORT = 4
_MAX_EXHAUSTIVE_DISTINCT_VALUES = 50


def _is_value_explicit_in_prompt(value: str, prompt_text: str) -> bool:
    """Return True if a discrete value appears explicitly in the extraction prompt."""
    if not prompt_text or not value:
        return False
    val = value.strip()
    if not val:
        return False
    if len(val) == 1:
        if val.isalnum():
            return f"'{val}'" in prompt_text or f'"{val}"' in prompt_text
        return val in prompt_text
    if re.fullmatch(r"[A-Za-z0-9_.-]+", val):
        return bool(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(val)}(?![A-Za-z0-9_])", prompt_text)
        )
    return val in prompt_text


def build_column_profiles(
    df_gt: pd.DataFrame,
    schema_config: SchemaConfig,
    active_norm_ids: set[str],
    id_col: str,
    compiled_columns: Dict[str, GeneralizedColumnsOutput],
) -> Dict[str, Dict[str, Any]]:
    """Build per-column empirical profiles strictly from active training documents."""
    if active_norm_ids and id_col in df_gt.columns:
        doc_keys = df_gt[id_col].astype(str).apply(normalize_document_key)
        train_df = df_gt[doc_keys.isin(active_norm_ids)]
    else:
        train_df = df_gt

    profiles: Dict[str, Dict[str, Any]] = {}
    total_rows = int(len(train_df))
    has_id_col = id_col in train_df.columns and not train_df.empty
    train_doc_keys = (
        train_df[id_col].astype(str).apply(normalize_document_key).tolist()
        if has_id_col
        else []
    )
    active_docs_count = (
        len(active_norm_ids)
        if active_norm_ids
        else (len(set(train_doc_keys)) if train_doc_keys else 0)
    )

    for field_name, spec in schema_config.fields.items():
        col_key = field_name.lower().strip()
        observed: List[str] = []
        value_counts: Dict[str, int] = {}
        value_docs: Dict[str, set[str]] = {}
        null_count = total_rows

        if col_key in train_df.columns:
            series = train_df[col_key]
            for row_idx, raw_val in enumerate(series.tolist()):
                if pd.isna(raw_val):
                    continue
                val = str(raw_val).strip()
                if not val:
                    continue
                observed.append(val)
                value_counts[val] = value_counts.get(val, 0) + 1
                if train_doc_keys:
                    value_docs.setdefault(val, set()).add(train_doc_keys[row_idx])
            null_count = total_rows - len(observed)

        sorted_counts = sorted(value_counts.items(), key=lambda x: (-x[1], x[0]))
        distinct_count = len(value_counts)
        non_null_count = len(observed)
        uniqueness_ratio = round(distinct_count / non_null_count, 4) if non_null_count > 0 else 0.0

        profile: Dict[str, Any] = {
            "declared_type": getattr(spec.type, "__name__", str(spec.type)),
            "required": spec.required,
            "active_docs_count": active_docs_count,
            "total_rows": total_rows,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "distinct_count": distinct_count,
            "uniqueness_ratio": uniqueness_ratio,
            "single_doc_values_count": sum(1 for docs in value_docs.values() if len(docs) == 1),
            "multi_doc_values_count": sum(1 for docs in value_docs.values() if len(docs) >= 2),
            "top_values": [
                {
                    "value": v,
                    "count": c,
                    "doc_count": len(value_docs.get(v, set())),
                }
                for v, c in sorted_counts[:20]
            ],
        }

        if distinct_count <= 50:
            profile["all_observed_values"] = [v for v, _ in sorted_counts]
            if value_docs:
                profile["value_doc_support"] = {
                    v: len(value_docs.get(v, set())) for v, _ in sorted_counts
                }

        if observed:
            lengths = [len(v) for v in observed]
            profile["min_length"] = min(lengths)
            profile["max_length"] = max(lengths)

            if spec.type in (int, float):
                nums: List[float] = []
                for v in observed:
                    try:
                        nums.append(float(v))
                    except ValueError:
                        pass
                if len(nums) == len(observed):
                    profile["numeric_min"] = min(nums)
                    profile["numeric_max"] = max(nums)

        cc = compiled_columns.get(field_name) or compiled_columns.get(col_key)
        if cc is not None:
            if cc.Semantic_Core_generalized:
                profile["semantic_core"] = cc.Semantic_Core_generalized
            if cc.Instructions.Transformation_Instructions:
                profile["transformation_instructions"] = list(cc.Instructions.Transformation_Instructions)

        profiles[field_name] = profile

    return profiles


def _extract_field_prompt_scope(field_name: str, prompt_text: str) -> str:
    """Return only the target field's section from a sectioned prompt (or whole text if unsectioned)."""
    if not prompt_text or not prompt_text.strip():
        return ""
    from ae.semantic_optimization.steps.generalization import _clean_header, parse_prompt_sections

    sections = parse_prompt_sections(prompt_text)
    target_key = field_name.strip().upper()
    matched = [
        sec_text
        for key, sec_text in sections
        if key == target_key or _clean_header(sec_text) == target_key
    ]
    if matched:
        return "\n\n".join(matched)
    # If the prompt is sectioned (contains '---' or multiple sections), do NOT fall back
    # to other fields' sections or preamble!
    if "---" in prompt_text or len(sections) > 1:
        return ""
    return prompt_text


def _passes_enum_doc_support_gate(
    field_name: str,
    profile: Dict[str, Any],
    min_enum_doc_support: int,
    prompt_text: str = "",
) -> bool:
    """Return True if empirical document support justifies closing a string field into an enum."""
    distinct_count = profile.get("distinct_count")
    if (
        profile.get("all_observed_values") is None
        or (isinstance(distinct_count, int) and distinct_count > _MAX_EXHAUSTIVE_DISTINCT_VALUES)
    ):
        logger.warning(
            f"Skipping proposed closed enum for '{field_name}': "
            f"distinct_count={distinct_count} exceeds {_MAX_EXHAUSTIVE_DISTINCT_VALUES} or observed values unavailable"
        )
        return False

    active_docs = profile.get("active_docs_count")
    if active_docs is not None and active_docs < min_enum_doc_support:
        logger.warning(
            f"Skipping proposed closed enum for '{field_name}': "
            f"active_docs_count={active_docs} < {min_enum_doc_support} (doc_support gate)"
        )
        return False

    field_prompt = _extract_field_prompt_scope(field_name, prompt_text)
    value_doc_support = profile.get("value_doc_support")
    if isinstance(value_doc_support, dict) and value_doc_support:
        unbacked_singletons = [
            v
            for v, dc in value_doc_support.items()
            if dc == 1 and not _is_value_explicit_in_prompt(v, field_prompt)
        ]
        if unbacked_singletons:
            logger.warning(
                f"Skipping proposed closed enum for '{field_name}': "
                f"unbacked doc_count==1 value(s) ({unbacked_singletons[:5]})"
            )
            return False

    return True


def _is_quoted_or_word_explicit_in_prompt(value: str, prompt_text: str) -> bool:
    """Return True if a literal value appears explicitly (quoted if 1-char) in prompt_text."""
    if not prompt_text or not value:
        return False
    val = value.strip()
    if not val:
        return False
    if len(val) == 1:
        return f"'{val}'" in prompt_text or f'"{val}"' in prompt_text
    if re.fullmatch(r"[A-Za-z0-9_.-]+", val):
        return bool(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(val)}(?![A-Za-z0-9_])", prompt_text)
        )
    return val in prompt_text


def _is_default_prompt_grounded(field_name: str, default_val: str, prompt_text: str) -> bool:
    """Return True when default_val is explicitly grounded in the target field's section of prompt_text."""
    field_prompt = _extract_field_prompt_scope(field_name, prompt_text)
    if not field_prompt or not default_val:
        return False
    val = default_val.strip()
    if not val:
        return False
    if len(val) == 1:
        if not (f"'{val}'" in field_prompt or f'"{val}"' in field_prompt):
            return False
    elif not _is_quoted_or_word_explicit_in_prompt(val, field_prompt):
        return False
    low_prompt = field_prompt.lower()
    return (
        field_name.lower() in low_prompt
        or "default" in low_prompt
        or "omitted" in low_prompt
        or "not stated" in low_prompt
        or "if absent" in low_prompt
        or "if missing" in low_prompt
    )


def _is_pattern_prompt_grounded(field_name: str, pattern: str, prompt_text: str) -> bool:
    """Return True when regex pattern (or its explicit closed alternatives) is grounded in target field's section."""
    field_prompt = _extract_field_prompt_scope(field_name, prompt_text)
    if not field_prompt or not pattern:
        return False
    pat = pattern.strip()
    if not pat:
        return False
    core = re.sub(r"^\^|\$$", "", pat)
    if (len(pat) >= 4 and pat in field_prompt) or (len(core) >= 4 and core in field_prompt):
        return True
    m = re.fullmatch(r"\^?\((?:\?:)?([^()]+)\)\$?", pat)
    if m and "|" in m.group(1):
        branches = [b.strip() for b in m.group(1).split("|") if b.strip()]
        if branches and all(
            _is_quoted_or_word_explicit_in_prompt(b.replace("\\", ""), field_prompt)
            for b in branches
        ):
            return field_name.lower() in field_prompt.lower() or all(
                f"'{b}'" in field_prompt or f'"{b}"' in field_prompt
                for b in branches
                if len(b) == 1
            )
    return False


def _passes_constraint_doc_support_gate(
    field_name: str,
    profile: Dict[str, Any],
    min_doc_support: int,
    constraint_kind: str,
) -> bool:
    """Return True if active training document count meets min_doc_support for pattern/default constraints."""
    active_docs = profile.get("active_docs_count")
    if active_docs is not None and active_docs < min_doc_support:
        logger.warning(
            f"Skipping proposed {constraint_kind} for '{field_name}': "
            f"active_docs_count={active_docs} < {min_doc_support} (doc_support gate)"
        )
        return False
    return True


def apply_schema_refinements(
    base_config: SchemaConfig,
    refinement: SchemaRefinementOutput,
    column_profiles: Dict[str, Dict[str, Any]],
    min_enum_doc_support: int = _DEFAULT_MIN_ENUM_DOC_SUPPORT,
    prompt_text: str = "",
) -> SchemaConfig:
    """Safely merge proposed field constraints into a copy of SchemaConfig with guardrails."""
    refined = copy.deepcopy(base_config)
    proposals: Dict[str, FieldConstraintProposal] = {
        p.field_name.strip(): p for p in refinement.fields if p.field_name and p.field_name.strip()
    }

    for field_name, spec in refined.fields.items():
        proposal = proposals.get(field_name) or proposals.get(field_name.lower())
        if proposal is None:
            continue

        profile = column_profiles.get(field_name, {})
        observed_values: Optional[List[str]] = profile.get("all_observed_values")
        distinct_count = profile.get("distinct_count")
        exceeds_exhaustive = (
            observed_values is None
            or (isinstance(distinct_count, int) and distinct_count > _MAX_EXHAUSTIVE_DISTINCT_VALUES)
        )

        # 1. Description (only fill when missing)
        if not spec.description and proposal.description and proposal.description.strip():
            spec.description = proposal.description.strip()

        # 2. Closed enum choices (only for str fields without existing choices + doc_support gate)
        if (
            spec.type is str
            and not spec.choices
            and proposal.is_closed_enum
            and proposal.choices
        ):
            cleaned_choices = list(
                dict.fromkeys(str(c).strip() for c in proposal.choices if str(c).strip())
            )
            if (
                not exceeds_exhaustive
                and observed_values is not None
                and 1 <= len(cleaned_choices) <= _MAX_ENUM_CHOICES
                and _passes_enum_doc_support_gate(
                    field_name, profile, min_enum_doc_support, prompt_text=prompt_text
                )
            ):
                if all(v in cleaned_choices for v in observed_values):
                    spec.choices = cleaned_choices
                else:
                    missing = [v for v in observed_values if v not in cleaned_choices]
                    logger.warning(
                        f"Skipping proposed choices for '{field_name}': missing observed train GT values {missing}"
                    )

        # 3. Default value (only for optional fields without existing default + doc_support gate)
        if not spec.required and spec.default is None and proposal.default_value is not None:
            cand_default = str(proposal.default_value).strip()
            if cand_default:
                default_min_support = (
                    1
                    if _is_default_prompt_grounded(field_name, cand_default, prompt_text)
                    else min_enum_doc_support
                )
                if _passes_constraint_doc_support_gate(
                    field_name, profile, default_min_support, "default_value"
                ):
                    if not spec.choices or cand_default in spec.choices:
                        spec.default = cand_default
                    else:
                        logger.warning(
                            f"Skipping proposed default {cand_default!r} for '{field_name}': not in choices {spec.choices}"
                        )

        # 4. Regex pattern (conservative: doc_support gate, full observed coverage, no >50 bypass)
        if spec.type is str and not spec.pattern and proposal.pattern:
            pattern_min_support = (
                1
                if _is_pattern_prompt_grounded(field_name, proposal.pattern, prompt_text)
                else min_enum_doc_support
            )
            if exceeds_exhaustive or not observed_values:
                logger.warning(
                    f"Skipping proposed pattern {proposal.pattern!r} for '{field_name}': "
                    "cannot verify regex against full observed values (empty or distinct_count > 50)"
                )
            elif _passes_constraint_doc_support_gate(
                field_name, profile, pattern_min_support, "pattern"
            ):
                try:
                    compiled_re = re.compile(proposal.pattern)
                    if all(compiled_re.search(v) for v in observed_values):
                        spec.pattern = proposal.pattern
                    else:
                        logger.warning(
                            f"Skipping proposed pattern {proposal.pattern!r} for '{field_name}': "
                            "does not match all observed train GT values"
                        )
                except re.error as exc:
                    logger.warning(f"Skipping invalid regex pattern {proposal.pattern!r} for '{field_name}': {exc}")

        # 5. Numeric bounds (only for int/float fields, gated by doc_support and consistent with observed values)
        if spec.type in (int, float) and (
            (spec.min_value is None and proposal.min_value is not None)
            or (spec.max_value is None and proposal.max_value is not None)
        ):
            if _passes_constraint_doc_support_gate(
                field_name, profile, min_enum_doc_support, "numeric_bounds"
            ):
                obs_min = profile.get("numeric_min")
                obs_max = profile.get("numeric_max")
                cand_min = proposal.min_value if spec.min_value is None else spec.min_value
                cand_max = proposal.max_value if spec.max_value is None else spec.max_value

                if cand_min is not None and (obs_min is None or cand_min <= obs_min):
                    if cand_max is None or cand_min <= cand_max:
                        spec.min_value = cand_min
                if cand_max is not None and (obs_max is None or cand_max >= obs_max):
                    if spec.min_value is None or spec.min_value <= cand_max:
                        spec.max_value = cand_max

    return refined


def run_schema_refinement(
    context: RunContext,
    df_gt: pd.DataFrame,
    id_col: str,
    compiled_columns: Dict[str, GeneralizedColumnsOutput],
    final_prompt: str,
    active_norm_ids: Optional[set[str]] = None,
) -> Optional[Path]:
    """Infer and persist refined schema constraints without mutating source schema.yaml."""
    if not getattr(context, "infer_schema_constraints", True):
        return None

    base_config = load_schema_from_yaml(
        context.schema_path,
        instruction_path=context.generated_prompt_path,
    )
    norm_ids = active_norm_ids if active_norm_ids is not None else (
        {normalize_document_key(d) for d in context.doc_ids} if context.doc_ids else set()
    )
    column_profiles = build_column_profiles(
        df_gt=df_gt,
        schema_config=base_config,
        active_norm_ids=norm_ids,
        id_col=id_col,
        compiled_columns=compiled_columns,
    )

    iter_num = context.current_iteration
    force_recompute = not getattr(context, "resume", False)
    refinement_artifact = context.artifacts.get_path(
        "generalization", "", "schema_refinement.json", iteration=iter_num
    )

    raw_min_support = getattr(context, "min_enum_doc_support", _DEFAULT_MIN_ENUM_DOC_SUPPORT)
    min_support = (
        raw_min_support
        if isinstance(raw_min_support, int) and raw_min_support >= 1
        else _DEFAULT_MIN_ENUM_DOC_SUPPORT
    )

    def _compute_refinement() -> SchemaRefinementOutput:
        predictor = dspy.Predict(SchemaConstraintInference)
        profiles_json = json.dumps(column_profiles, ensure_ascii=False, indent=2)
        resp = predictor(
            extraction_schema=context.schema,
            baseline_prompt=context.baseline_prompt,
            generated_prompt=final_prompt,
            column_profiles=profiles_json,
            min_enum_doc_support=min_support,
        )
        return resp.refinement

    cache_key = hashlib.sha256(
        json.dumps(
            {
                "extraction_schema": context.schema,
                "baseline_prompt": context.baseline_prompt,
                "generated_prompt": final_prompt,
                "column_profiles": column_profiles,
                "min_enum_doc_support": min_support,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    refinement: SchemaRefinementOutput = context.artifacts.load_or_compute(
        refinement_artifact,
        _compute_refinement,
        SchemaRefinementOutput,
        force_recompute=force_recompute,
        cache_key=cache_key,
    )

    combined_prompt_text = "\n\n---\n\n".join(
        p for p in (getattr(context, "baseline_prompt", "") or "", final_prompt or "") if p
    )
    refined_config = apply_schema_refinements(
        base_config,
        refinement,
        column_profiles,
        min_enum_doc_support=min_support,
        prompt_text=combined_prompt_text,
    )
    output_schema_path = Path(context.generated_schema_path).resolve()
    save_schema_to_yaml(refined_config, output_schema_path)

    if iter_num is not None:
        iter_schema_path = context.artifacts.get_path(
            "generalization", "", "candidate_schema.yaml", iteration=iter_num
        )
        save_schema_to_yaml(refined_config, iter_schema_path)

    return output_schema_path
