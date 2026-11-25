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
from ae.semantic_optimization.hitl import (
    _get_id_column,
    classify_conflict_priority,
    sort_conflicts_by_priority,
)
from ae.semantic_optimization.models import (
    ConsolidatedColumnsOutput,
    ConsolidatedRowsOutput,
    GeneralizedColumnsOutput,
    GeneralizedRowsOutput,
    REConflict,
    ResidualDocReport,
    RuleAttribution,
)
from ae.semantic_optimization.signatures import (
    ColumnGeneralization,
    PromptHarmonization,
    RowGeneralization,
)
from ae.semantic_optimization.steps.consolidation import _invalidate_doc_artifacts
from ae.semantic_optimization.steps.positive_analysis import filter_equivalent_conflicts

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


_EXAMPLE_PAREN_RE = re.compile(r"\((?:e\.g\.|for example|such as|i\.e\.)[^)]*\)", re.IGNORECASE)
_ROW_REF_RE = re.compile(r"\b(?:gap_)?row_\d+\b", re.IGNORECASE)
_INSTANCE_PATCH_RE = re.compile(
    r"(?:\bfor\s+(?:compound|entry|sample|record)\s+[A-Za-z0-9_-]{1,12}\b"
    r"|\bin\s+(?:this\s+(?:article|paper|study)|document|paper)\s+[A-Za-z0-9._-]{4,}\b"
    r"|\boverride\s+(?:only\s+)?for\s+(?:compound|row)\b)",
    re.IGNORECASE,
)
_QUOTED_LITERAL_RE = re.compile(r"['\"]([^'\"]{2,80})['\"]")
_BENIGN_QUOTED_TOKENS = frozenset(
    {
        "nan",
        "mic",
        "pmic",
        "none",
        "=",
        ">",
        "<",
        "≥",
        "≤",
        "<=",
        ">=",
        "==",
        "!=",
        "~=",
        "<td>",
        "</td>",
        "<th>",
        "</th>",
        "<b>",
        "</b>",
        "<i>",
        "</i>",
        "<em>",
        "</em>",
        "µg/ml",
        "ug/ml",
        "mg/l",
        "mg/ml",
        "ng/ml",
        "pg/ml",
        "mmol/l",
        "µmol/l",
        "umol/l",
        "mm",
        "µm",
        "um",
        "nm",
        "pm",
        "iu/ml",
        "cfu/ml",
        "±",
        "ic50",
        "ec50",
        "cc50",
        "gi50",
        "lc50",
        "ld50",
        "ki",
        "kd",
        "mic50",
        "mic90",
        "mbc",
        "mfc",
        "zoi",
        "auc",
        "nd",
        "na",
        "n/a",
        "null",
        "true",
        "false",
        "not detected",
        "not tested",
        "inactive",
    }
)
_OPERATOR_LITERAL_RE = re.compile(r"^[<>≤≥=~!±/%]+$")
_UNIT_LITERAL_RE = re.compile(
    r"^(?:[µuμnmkp]?(?:g|l|m|mol)|iu|cfu|ppm|ppb|hz|khz|mhz)(?:/[µuμnmk]?[gl]|/h|/min)?$",
    re.IGNORECASE,
)
_ASSAY_LITERAL_RE = re.compile(
    r"^(?:ic|ec|cc|gi|lc|ld|mic|mbc|mfc|ed|td)\d{1,3}$",
    re.IGNORECASE,
)


def _is_benign_quoted_literal(token: str) -> bool:
    """Return True if a quoted token is a generic operator, unit, or standard assay symbol."""
    cleaned = token.strip()
    low = cleaned.lower()
    if low in _BENIGN_QUOTED_TOKENS:
        return True
    if _OPERATOR_LITERAL_RE.fullmatch(cleaned):
        return True
    if _UNIT_LITERAL_RE.fullmatch(cleaned):
        return True
    if _ASSAY_LITERAL_RE.fullmatch(cleaned):
        return True
    return False


def _normalize_rule_text(text: str) -> str:
    """Normalize an instruction string for resilient attribution lookup."""
    s = (text or "").strip().lower()
    s = re.sub(r"^(?:[-*•]+|\d+[.)])\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".;:, ")


def _rule_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", _normalize_rule_text(text)))


def _match_rule_attribution(
    inst: str,
    attr_by_text: dict[str, RuleAttribution],
    attr_by_norm: dict[str, RuleAttribution],
    attr_tokens_list: list[tuple[set[str], RuleAttribution]],
) -> Optional[RuleAttribution]:
    """Match an instruction to its RuleAttribution using exact, normalized, then token-Jaccard fallback."""
    exact = attr_by_text.get(inst.strip().lower())
    if exact is not None:
        return exact

    norm_inst = _normalize_rule_text(inst)
    norm_match = attr_by_norm.get(norm_inst)
    if norm_match is not None:
        return norm_match

    inst_tokens = _rule_tokens(inst)
    if not inst_tokens:
        return None

    best_score = 0.0
    best_attr: Optional[RuleAttribution] = None
    min_overlap = 2 if len(inst_tokens) >= 2 else 1
    for cand_tokens, cand_attr in attr_tokens_list:
        if not cand_tokens:
            continue
        overlap = len(inst_tokens & cand_tokens)
        if overlap < min_overlap:
            continue
        union = len(inst_tokens | cand_tokens)
        jaccard = overlap / union if union else 0.0
        if jaccard >= 0.70 and (
            jaccard > best_score
            or (jaccard == best_score and cand_attr.rule_kind == "CASE_PATCH")
        ):
            best_score = jaccard
            best_attr = cand_attr
    return best_attr


def _strip_parenthetical_examples(text: str) -> str:
    return _EXAMPLE_PAREN_RE.sub("", text)


def _is_case_patch_instruction(
    inst: str,
    attr: Optional[RuleAttribution],
    known_doc_ids: set[str],
    active_doc_count: int,
) -> tuple[bool, str]:
    """Return (is_case_patch, reason) if an instruction is a single-document/instance patch."""
    clean = (inst or "").strip()
    if not clean:
        return True, "empty_rule"

    if attr is not None and attr.rule_kind == "CASE_PATCH":
        return True, "attributed_as_CASE_PATCH"

    lower_clean = clean.lower()
    for d_id in known_doc_ids:
        d_norm = d_id.strip().lower()
        if len(d_norm) >= 4 and re.search(
            rf"(?<![a-z0-9]){re.escape(d_norm)}(?![a-z0-9])", lower_clean
        ):
            return True, f"mentions_document_id:{d_norm}"

    if _ROW_REF_RE.search(clean):
        return True, "mentions_row_id"

    stripped = _strip_parenthetical_examples(clean)
    if _INSTANCE_PATCH_RE.search(stripped):
        return True, "instance_specific_patch_phrase"

    if (
        attr is not None
        and active_doc_count >= 2
        and len({normalize_document_key(d) for d in attr.supporting_doc_ids if d}) == 1
    ):
        quoted = [
            q.strip()
            for q in _QUOTED_LITERAL_RE.findall(stripped)
            if not _is_benign_quoted_literal(q)
        ]
        has_specific_number = bool(re.search(r"\b\d+\.\d{2,}\b|\b\d{4,}\b", stripped))
        if quoted or has_specific_number:
            return True, "single_doc_support_with_hardcoded_literal"

    return False, ""


def filter_case_patch_instructions(
    instructions: list[str],
    rule_attributions: list[RuleAttribution],
    known_doc_ids: set[str],
    active_doc_count: int,
    scope: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Filter single-article CASE_PATCH literals out of generalized rule lists."""
    attr_by_text: dict[str, RuleAttribution] = {}
    attr_by_norm: dict[str, RuleAttribution] = {}
    attr_tokens_list: list[tuple[set[str], RuleAttribution]] = []
    for a in rule_attributions:
        if not a.rule_text:
            continue
        raw_k = a.rule_text.strip().lower()
        norm_k = _normalize_rule_text(a.rule_text)
        attr_by_text.setdefault(raw_k, a)
        if norm_k:
            attr_by_norm.setdefault(norm_k, a)
        toks = _rule_tokens(a.rule_text)
        if toks:
            attr_tokens_list.append((toks, a))

    kept: list[str] = []
    blocked: list[dict[str, str]] = []
    for inst in instructions:
        attr = _match_rule_attribution(inst, attr_by_text, attr_by_norm, attr_tokens_list)
        is_patch, reason = _is_case_patch_instruction(
            inst, attr, known_doc_ids, active_doc_count
        )
        if is_patch:
            blocked.append(
                {
                    "scope": scope,
                    "rule_text": inst,
                    "reason": reason,
                    "supporting_doc_ids": ", ".join(attr.supporting_doc_ids) if attr else "",
                }
            )
            logger.warning(f"Filtered CASE_PATCH rule from {scope} ({reason}): {inst}")
        else:
            kept.append(inst)
    return kept, blocked


def _sanitize_row_output_case_patches(
    row_data: GeneralizedRowsOutput,
    known_doc_ids: set[str],
    active_doc_count: int,
) -> tuple[GeneralizedRowsOutput, list[dict[str, str]]]:
    blocked_all: list[dict[str, str]] = []
    kept_inc, b_inc = filter_case_patch_instructions(
        list(row_data.Instructions.Row_Inclusion_Instructions),
        row_data.Rule_Attributions,
        known_doc_ids,
        active_doc_count,
        "rows:inclusion",
    )
    kept_exc, b_exc = filter_case_patch_instructions(
        list(row_data.Instructions.Row_Exclusion_Instructions),
        row_data.Rule_Attributions,
        known_doc_ids,
        active_doc_count,
        "rows:exclusion",
    )
    blocked_all.extend(b_inc)
    blocked_all.extend(b_exc)
    if blocked_all:
        row_data.Instructions.Row_Inclusion_Instructions = kept_inc
        row_data.Instructions.Row_Exclusion_Instructions = kept_exc
    return row_data, blocked_all


def _sanitize_col_output_case_patches(
    field_name: str,
    col_data: GeneralizedColumnsOutput,
    known_doc_ids: set[str],
    active_doc_count: int,
) -> tuple[GeneralizedColumnsOutput, list[dict[str, str]]]:
    blocked_all: list[dict[str, str]] = []
    kept_inc, b_inc = filter_case_patch_instructions(
        list(col_data.Instructions.Column_Inclusion_Instructions),
        col_data.Rule_Attributions,
        known_doc_ids,
        active_doc_count,
        f"column:{field_name}:inclusion",
    )
    kept_exc, b_exc = filter_case_patch_instructions(
        list(col_data.Instructions.Column_Exclusion_Instructions),
        col_data.Rule_Attributions,
        known_doc_ids,
        active_doc_count,
        f"column:{field_name}:exclusion",
    )
    kept_trn, b_trn = filter_case_patch_instructions(
        list(col_data.Instructions.Transformation_Instructions),
        col_data.Rule_Attributions,
        known_doc_ids,
        active_doc_count,
        f"column:{field_name}:transformation",
    )
    blocked_all.extend(b_inc)
    blocked_all.extend(b_exc)
    blocked_all.extend(b_trn)
    if blocked_all:
        col_data.Instructions.Column_Inclusion_Instructions = kept_inc
        col_data.Instructions.Column_Exclusion_Instructions = kept_exc
        col_data.Instructions.Transformation_Instructions = kept_trn
    return col_data, blocked_all


def parse_prompt_sections(prompt_text: str) -> list[tuple[str, str]]:
    """Split a sectioned extraction prompt on `---` into ordered (header_key, section_text) pairs."""
    raw_sections = [s.strip() for s in (prompt_text or "").split("---") if s.strip()]
    parsed: list[tuple[str, str]] = []
    for idx, sec in enumerate(raw_sections):
        header = _clean_header(sec)
        if idx == 0 and header not in ("RECORD SELECTION", "EXPERIMENT SELECTION"):
            parsed.append(("PREAMBLE", sec))
        elif "EXPERT RULINGS" in header:
            parsed.append(("EXPERT RULINGS", sec))
        elif "RECORD SELECTION" in header or "EXPERIMENT SELECTION" in header:
            parsed.append(("RECORD SELECTION", sec))
        elif "REASONING STYLE" in header:
            parsed.append(("REASONING STYLE", sec))
        elif "GENERAL EXTRACTION GUIDELINES" in header:
            parsed.append(("GENERAL EXTRACTION GUIDELINES", sec))
        else:
            parsed.append((header, sec))
    return parsed


def replace_prompt_sections(
    target_prompt: str,
    reference_prompt: str,
    field_names: list[str],
) -> str:
    """Replace specific field sections (or RECORD SELECTION) in target_prompt from reference_prompt."""
    target_secs = parse_prompt_sections(target_prompt)
    ref_map = {k: v for k, v in parse_prompt_sections(reference_prompt)}
    replace_keys = {
        "RECORD SELECTION" if f.lower().strip() == "rows" else f.strip().upper()
        for f in field_names
        if f and f.strip()
    }
    if not replace_keys or not target_secs:
        return target_prompt

    updated_parts: list[str] = []
    seen_keys: set[str] = set()
    for key, sec_text in target_secs:
        if key in replace_keys and key in ref_map:
            updated_parts.append(ref_map[key])
            seen_keys.add(key)
        else:
            updated_parts.append(sec_text)
            seen_keys.add(key)

    return "\n\n---\n\n".join(updated_parts).strip() + "\n"


_GENERIC_EXCLUDED_STOPWORDS = frozenset(
    {
        "any",
        "all",
        "the",
        "this",
        "that",
        "such",
        "other",
        "non",
        "records",
        "record",
        "rows",
        "row",
        "values",
        "value",
        "entries",
        "entry",
        "items",
        "item",
        "fields",
        "field",
        "data",
        "null",
        "empty",
        "missing",
    }
)


def _detect_prompt_contradictions(
    sections: list[tuple[str, str]],
    general_rulings: str,
    fields: list[str],
) -> tuple[list[str], set[str]]:
    """Detect cross-section contradictions before PromptHarmonization (Contradiction-First)."""
    sec_map = {k: v for k, v in sections}
    findings: list[str] = []
    involved_sections: set[str] = set()

    rulings_low = (general_rulings or "").lower()
    has_general_no_norm = any(
        tok in rulings_low
        for tok in ("without normalization", "preserve source", "format they appear", "do not normalize")
    )

    rec_sel_low = sec_map.get("RECORD SELECTION", "").lower()
    if "\nexclude:" in rec_sel_low:
        rec_inc_low, rec_exc_low = rec_sel_low.split("\nexclude:", 1)
    else:
        rec_inc_low, rec_exc_low = rec_sel_low, ""

    for field in fields:
        f_key = field.upper()
        f_text = sec_map.get(f_key, "")
        f_low = f_text.lower()
        if not f_text:
            continue
        if has_general_no_norm and any(
            tok in f_low for tok in ("footnote", "superscript", "legend", "expand")
        ):
            findings.append(
                f"CONFLICT [EXPERT RULINGS vs {f_key}]: General ruling preserves raw source format, "
                f"while {f_key} requires resolving table footnotes/superscripts/legends. "
                "Apply specificity precedence: local footnote/legend/caption definition takes precedence "
                "over standalone format preservation; separate both scopes explicitly."
            )
            involved_sections.add(f_key)
            involved_sections.add("EXPERT RULINGS")

        if rec_inc_low:
            excluded_terms = re.findall(
                r"\b(?:exclude|do not extract|omit|skip)\s+([a-z0-9][a-z0-9_/-]{1,40})\b",
                f_low,
            )
            for term in excluded_terms:
                if term in _GENERIC_EXCLUDED_STOPWORDS:
                    continue
                term_pat = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
                if re.search(term_pat, rec_inc_low) and not re.search(term_pat, rec_exc_low):
                    findings.append(
                        f"CONFLICT [RECORD SELECTION vs {f_key}]: RECORD SELECTION includes '{term}' "
                        f"while {f_key} excludes it."
                    )
                    involved_sections.add(f_key)
                    involved_sections.add("RECORD SELECTION")
                    break

    return findings, involved_sections


def _load_residual_doc_diagnostics(context: RunContext) -> dict[str, ResidualDocReport]:
    """Load persisted ResidualDocReport artifacts keyed by normalized doc_id."""
    reports: dict[str, ResidualDocReport] = {}
    res_dir = context.artifacts.get_phase_dir("positive", "residual", iteration=None)
    if not res_dir.exists():
        return reports
    for p in sorted(res_dir.glob("*.json")):
        try:
            rep = ResidualDocReport.model_validate_json(p.read_text(encoding="utf-8"))
            reports[normalize_document_key(rep.doc_id)] = rep
        except Exception:
            pass
    return reports


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
    all_gt_doc_ids = (
        {
            normalize_document_key(str(d))
            for d in df_gt[id_col].dropna().tolist()
            if str(d).strip()
        }
        if id_col in df_gt.columns
        else set()
    )
    known_doc_ids = set(active_norm_ids) | all_gt_doc_ids
    active_doc_count = len(active_norm_ids) if active_norm_ids else len(all_gt_doc_ids)
    blocked_case_patches: list[dict[str, str]] = []
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
    residual_reports_by_doc = _load_residual_doc_diagnostics(context)

    def _enrich_conflict_diagnostics(conflict: REConflict) -> REConflict:
        norm_d = normalize_document_key(conflict.document_id) if conflict.document_id else ""
        rep = residual_reports_by_doc.get(norm_d)
        updates: dict[str, str] = {}
        if rep is not None and not conflict.diagnostic_tag:
            text_low = (
                f"{conflict.conflict_id} {conflict.explanation} {conflict.conflicting_value_or_rule}"
            ).lower()
            if rep.duplicate_of_tp_count > 0 and (
                "duplicate" in text_low or "repeat" in text_low or "table_" in text_low
            ):
                updates["diagnostic_tag"] = "duplicate_of_tp"
            elif rep.row_merge_count > 0 and "merge" in text_low:
                updates["diagnostic_tag"] = "row_merge"
            elif (
                conflict.conflict_type == "SUSPECTED_GT_OMISSION"
                and rep.high_confidence_unlabeled_count > 0
            ):
                updates["diagnostic_tag"] = "high_confidence_unlabeled"
        cand = conflict.model_copy(update=updates) if updates else conflict
        _, _, tier = classify_conflict_priority(cand)
        if cand.priority != tier:
            cand = cand.model_copy(update={"priority": tier})
        return cand

    def _clean_and_collect_conflicts() -> list[REConflict]:
        nonlocal row_data
        collected: list[REConflict] = []

        orig_row_conflicts = list(row_data.Conflicts)
        eq_row_conflicts = filter_equivalent_conflicts(orig_row_conflicts, context.schema)
        unresolved_row_conflicts = [
            _enrich_conflict_diagnostics(c)
            for c in eq_row_conflicts
            if not _is_conflict_resolved(context, c)
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
        else:
            row_data.Conflicts = unresolved_row_conflicts
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
                _enrich_conflict_diagnostics(c)
                for c in eq_col_conflicts
                if not _is_conflict_resolved(context, c)
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
            else:
                col_out.Conflicts = unresolved_col_conflicts
            collected.extend(compiled_columns[field].Conflicts)

        return sort_conflicts_by_priority(collected)

    did_rule_recompute = False
    max_hitl_rounds = 2
    triage_audit_log: list[dict[str, object]] = []
    for hitl_round in range(1, max_hitl_rounds + 1):
        all_conflicts = _clean_and_collect_conflicts()
        if not all_conflicts:
            break

        triage_audit_log.append(
            {
                "round": hitl_round,
                "queue": [
                    {
                        "conflict_id": c.conflict_id,
                        "priority": c.priority or classify_conflict_priority(c)[2],
                        "conflict_type": c.conflict_type,
                        "scope": c.scope,
                        "document_id": c.document_id,
                        "row_id": c.row_id,
                        "diagnostic_tag": c.diagnostic_tag,
                    }
                    for c in all_conflicts
                ],
            }
        )
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

    # Persist HITL triage queue audit if any conflicts were processed
    if triage_audit_log:
        triage_path = context.artifacts.get_path(
            "generalization", "", "hitl_triage_queue.json", iteration=iter_num
        )
        triage_path.parent.mkdir(parents=True, exist_ok=True)
        triage_path.write_text(
            json.dumps(triage_audit_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Filter CASE_PATCH literals out of generalized row and column instructions before persistence
    row_data, row_blocked = _sanitize_row_output_case_patches(
        row_data, known_doc_ids, active_doc_count
    )
    if row_blocked:
        blocked_case_patches.extend(row_blocked)
        row_data = context.artifacts.load_or_compute(
            row_rules_path,
            lambda out=row_data: out,
            GeneralizedRowsOutput,
            force_recompute=True,
            cache_key=_compute_row_cache_key(context, pos_rows, neg_rows),
        )
        _write_rulings_hash(row_rules_path, _scope_rulings_hash(context, "rows"))

    for field in fields:
        if field not in compiled_columns:
            continue
        sanitized_col, col_blocked = _sanitize_col_output_case_patches(
            field, compiled_columns[field], known_doc_ids, active_doc_count
        )
        if col_blocked:
            blocked_case_patches.extend(col_blocked)
            field_lower = field.lower().strip()
            pos_f = _compact_positive_col_entries(
                [pa for pa in pos_cols if pa.get("field_name", "").lower().strip() == field_lower]
            )
            neg_f = [na for na in neg_cols if na.get("field_name", "").lower().strip() == field_lower]
            c_path = context.artifacts.get_path(
                "generalization", "columns", f"{field}.json", iteration=iter_num
            )
            sanitized_col = context.artifacts.load_or_compute(
                c_path,
                lambda out=sanitized_col: out,
                GeneralizedColumnsOutput,
                force_recompute=True,
                cache_key=_compute_col_cache_key(context, field, pos_f, neg_f),
            )
            _write_rulings_hash(c_path, _scope_rulings_hash(context, f"column:{field}"))
        compiled_columns[field] = sanitized_col

    case_patch_audit_path = context.artifacts.get_path(
        "generalization", "", "case_patch_audit.json", iteration=iter_num
    )
    case_patch_audit_path.parent.mkdir(parents=True, exist_ok=True)
    case_patch_audit_path.write_text(
        json.dumps(
            {
                "iteration": iter_num,
                "active_docs_count": active_doc_count,
                "blocked_count": len(blocked_case_patches),
                "blocked_rules": blocked_case_patches,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

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

    # Record Selection rules (never emit an empty Include ONLY records satisfying: block)
    row_inc_list = [
        inst.strip()
        for inst in row_data.Instructions.Row_Inclusion_Instructions
        if inst and inst.strip()
    ]
    if not row_inc_list:
        trig = (row_data.Row_Instantiation_Trigger_generalized or "").strip()
        if trig:
            row_inc_list = [f"Extract one record per valid target instance matching: {trig}"]
        else:
            row_inc_list = [
                "Extract all target records that match the extraction schema and non-conflicting baseline guidelines."
            ]
    row_exc_list = [
        inst.strip()
        for inst in row_data.Instructions.Row_Exclusion_Instructions
        if inst and inst.strip()
    ]
    record_selection_text = (
        "RECORD SELECTION\n\n"
        "Include ONLY records satisfying:\n"
        + "\n".join(f"- {inst}" for inst in row_inc_list)
    )
    if row_exc_list:
        record_selection_text += "\n\nExclude:\n" + "\n".join(
            f"- {inst}" for inst in row_exc_list
        )
    assembled_prompt_parts.append(record_selection_text)

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

    # Check for carried-forward harmonized sections and run Contradiction-First audit
    prev_sections_map: dict[str, str] = {}
    if iter_num is not None and iter_num > 1 and can_carry_forward:
        prev_prompt_path = context.artifacts.get_path(
            "generalization", "", "candidate_prompt.txt", iteration=iter_num - 1
        )
        if prev_prompt_path.exists():
            prev_sections_map = {
                k: v
                for k, v in parse_prompt_sections(prev_prompt_path.read_text(encoding="utf-8"))
            }

    draft_prompt = "\n\n---\n\n".join(assembled_prompt_parts) + "\n"
    contradictions, contradiction_sections = _detect_prompt_contradictions(
        parse_prompt_sections(draft_prompt), general_rulings or "", fields
    )

    locked_sections: dict[str, str] = {}
    if prev_sections_map and changed_scopes is not None:
        for field in fields:
            f_upper = field.upper()
            scope_key = f"column:{field}"
            col_rules_prev = context.artifacts.get_path(
                "generalization", "columns", f"{field}.json", iteration=iter_num - 1
            )
            if (
                scope_key not in changed_scopes
                and f_upper not in contradiction_sections
                and f_upper in prev_sections_map
                and col_rules_prev.exists()
                and _can_carry_forward_rulings(col_rules_prev, context, scope_key)
            ):
                locked_sections[f_upper] = prev_sections_map[f_upper]

        if locked_sections:
            updated_parts: list[str] = []
            for key, sec_txt in parse_prompt_sections(draft_prompt):
                if key in locked_sections:
                    updated_parts.append(locked_sections[key])
                else:
                    updated_parts.append(sec_txt)
            draft_prompt = "\n\n---\n\n".join(updated_parts) + "\n"

    final_prompt = draft_prompt

    audit_parts: list[str] = []
    if general_rulings:
        audit_parts.append(general_rulings)
    if contradictions:
        audit_parts.append(
            "[CONTRADICTION-FIRST AUDIT — RESOLVE FIRST]\n"
            + "\n".join(f"- {item}" for item in contradictions)
        )
    if locked_sections:
        audit_parts.append(
            "[LOCKED UNCHANGED SECTIONS — PRESERVE VERBATIM]\n"
            f"Sections unchanged since iter_{iter_num - 1}: {', '.join(sorted(locked_sections.keys()))}"
        )
    harmonize_rulings_input = "\n\n".join(audit_parts) if audit_parts else "None"

    prompt_cache_key = hashlib.sha256(
        json.dumps(
            {
                "draft_prompt": draft_prompt,
                "general_rulings": harmonize_rulings_input,
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

    # Optional LLM Prompt Harmonization step (Contradiction-First + Sectional Isolation)
    if not reused_cached_prompt and getattr(context, "harmonize_prompt", True):
        try:
            harmonizer = dspy.Predict(PromptHarmonization)
            harm_resp = harmonizer(
                extraction_schema=context.schema,
                baseline_prompt=context.baseline_prompt,
                expert_rulings=harmonize_rulings_input,
                draft_prompt=draft_prompt,
            )
            if (
                hasattr(harm_resp, "harmonization")
                and getattr(harm_resp.harmonization, "harmonized_prompt", "")
                and harm_resp.harmonization.harmonized_prompt.strip()
            ):
                final_prompt = harm_resp.harmonization.harmonized_prompt.strip() + "\n"
                if locked_sections and "---" in final_prompt:
                    final_secs = parse_prompt_sections(final_prompt)
                    final_map = {k: v for k, v in final_secs}
                    restored_any = False
                    rebuilt: list[str] = []
                    for key, sec_txt in final_secs:
                        if key in locked_sections and sec_txt.strip() != locked_sections[key].strip():
                            rebuilt.append(locked_sections[key])
                            restored_any = True
                        else:
                            rebuilt.append(sec_txt)
                    # Also restore any locked section accidentally dropped
                    for l_key, l_txt in locked_sections.items():
                        if l_key not in final_map:
                            rebuilt.append(l_txt)
                            restored_any = True
                    if restored_any:
                        final_prompt = "\n\n---\n\n".join(rebuilt).strip() + "\n"
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
