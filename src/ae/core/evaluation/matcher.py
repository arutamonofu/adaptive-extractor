"""Evaluation engine for comparing extracted structured records against ground truth."""

import ast
import difflib
import functools
import hashlib
import json
import logging
import math
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union

from pydantic import BaseModel
from tabulate import tabulate  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_SEMANTIC_JUDGE_EXAMPLE = (
    '{\n'
    '  "field_a": {"reasoning": "1.5e3 is mathematically equivalent to 1500.", "match": "YES"},\n'
    '  "field_b": {"reasoning": "GT lists 3 items, Pred extracted only 2. Incomplete list.", "match": "NO"},\n'
    '  "field_c": {"reasoning": "Standard domain abbreviation matches full canonical term.", "match": "YES"}\n'
    '}'
)

_SEMANTIC_JUDGE_BATCH_EXAMPLE = (
    '{\n'
    '  "0": {\n'
    '    "field_a": {"reasoning": "1.5e3 is mathematically equivalent to 1500.", "match": "YES"}\n'
    '  },\n'
    '  "1": {\n'
    '    "field_b": {"reasoning": "GT lists 3 items, Pred extracted only 2. Incomplete list.", "match": "NO"}\n'
    '  }\n'
    '}'
)

RecordEntity: TypeAlias = Union[BaseModel, Any]
ExperimentEntity: TypeAlias = RecordEntity


def _extract_first_json(text: str) -> Optional[str]:
    """Extract the first valid JSON object from text by brace balancing.

    Handles nested objects, strings (ignores braces inside quotes),
    and trailing content after the first JSON object.

    Args:
        text: Input text potentially containing JSON.

    Returns:
        The first JSON substring if found, None otherwise.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Unbalanced braces — fallback: try to find a single-level JSON object
    fallback_match = re.search(r"\{[^{}]*\}", text)
    if fallback_match:
        return fallback_match.group(0)

    return None


def _clean_judge_response(response_text: Any) -> str:
    """Strip thinking/reasoning tags (paired or dangling) from LLM judge output."""
    text = str(response_text or "")
    cleaned = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if re.search(r"</think(?:ing)?>", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(r"</think(?:ing)?>", cleaned, flags=re.IGNORECASE)[-1]
    cleaned = re.sub(r"<think(?:ing)?>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _parse_judge_json(cleaned_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from cleaned judge output."""
    json_str = _extract_first_json(cleaned_text)
    if json_str is None:
        return None

    try:
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        parsed = json.loads(no_trailing_commas)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    try:
        normalized = re.sub(r"\bnull\b", "None", no_trailing_commas)
        normalized = re.sub(r"\btrue\b", "True", normalized)
        normalized = re.sub(r"\bfalse\b", "False", normalized)
        parsed = ast.literal_eval(normalized.strip())
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError):
        return None


def _extract_field_verdict(raw: Any) -> str:
    """Normalize a raw field verdict (dict, bool, or str) to 'YES' or 'NO'."""
    if raw is None:
        return "NO"
    if isinstance(raw, dict):
        raw = raw.get("match", "NO")
    if isinstance(raw, bool):
        return "YES" if raw else "NO"
    return "YES" if str(raw).strip().upper() in ("YES", "TRUE") else "NO"


def _extract_field_reasoning(raw: Any) -> str:
    """Extract concise judge reasoning text from a raw field verdict object."""
    if isinstance(raw, dict):
        return str(raw.get("reasoning") or "").strip()
    return ""


_RE_STRICT_CLEAN = re.compile(r"\s+")
_DASH_MAP = str.maketrans({"−": "-", "–": "-", "—": "-"})


@functools.lru_cache(maxsize=4096)
def _normalize_text_cached(val_str: str) -> str:
    """Helper to normalize text with caching."""
    return _RE_STRICT_CLEAN.sub("", val_str.translate(_DASH_MAP))


class RecordMatcher:
    """Evaluation engine for comparing extracted structured records against ground truth.

    - Strings: Normalized Exact Match (removes spaces, standardizes dashes).
    - Floats: Exact relative tolerance comparison with optional semantic judge fallback.
    """

    # Pre-compiled regex for performance
    _RE_STRICT_CLEAN = _RE_STRICT_CLEAN

    # Dash normalization mapping
    _DASH_MAP = _DASH_MAP

    def __init__(
        self,
        fields_to_compare: List[str],
        float_tolerance: float,
        student_llm: Optional[Any] = None,
        field_descriptions: Optional[Dict[str, str]] = None,
        enable_semantic_judge: bool = True,
        field_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        field_validators: Optional[Dict[str, List[Any]]] = None,
        max_judge_batch_size: int = 15,
        min_alignment_score: float = 0.25,
    ):
        """Initialize the RecordMatcher.

        Args:
            fields_to_compare: List of field names to compare between entities.
            float_tolerance: Tolerance for float comparisons (0.0 to 1.0).
            student_llm: DSPy LLM object for semantic judgment (optional).
            field_descriptions: Dictionary of field descriptions (optional).
            enable_semantic_judge: Flag to enable/disable semantic judge (default: True).
            field_constraints: Optional schema field constraints (choices, pattern, min/max).
            field_validators: Optional declarative validators per field (canonical equivalence).
            max_judge_batch_size: Max aligned pairs to evaluate in a single batched LLM call.
            min_alignment_score: Minimum similarity score in [0.0, 1.0] to keep a Hungarian pair.

        Raises:
            ValueError: If fields_to_compare is empty or float_tolerance is invalid.
        """
        if not fields_to_compare:
            raise ValueError("fields_to_compare cannot be empty")
        if not 0 <= float_tolerance <= 1:
            raise ValueError("float_tolerance must be between 0 and 1")
        if not 0.0 <= min_alignment_score <= 1.0:
            raise ValueError("min_alignment_score must be between 0 and 1")

        self.fields = fields_to_compare
        self.tolerance = float_tolerance
        self.student_llm = student_llm
        self.field_descriptions = field_descriptions or {}
        self.enable_semantic_judge = enable_semantic_judge
        self.field_constraints = field_constraints or {}
        self.field_validators: Dict[str, List[Any]] = {}
        if field_validators:
            from ae.core.schema.validators import normalize_validator_spec

            for fname, specs in field_validators.items():
                if specs:
                    self.field_validators[fname] = [
                        normalize_validator_spec(s) for s in specs
                    ]
        self.max_judge_batch_size = max(1, int(max_judge_batch_size))
        self.min_alignment_score = float(min_alignment_score)
        self._schema_constraints_hash = self._compute_schema_constraints_hash()
        self._cache_lock = threading.Lock()
        self._judge_counter_local = threading.local()
        self._verdict_cache: Dict[Tuple[Any, ...], Dict[str, str]] = {}
        self._reasoning_cache: Dict[Tuple[Any, ...], Dict[str, str]] = {}
        self._validator_compare_cache: Dict[
            Tuple[str, str, str], Tuple[Optional[bool], Optional[str]]
        ] = {}

    def _compute_schema_constraints_hash(self) -> str:
        """Compute deterministic hash of fields, tolerance, constraints, descriptions, and validators."""
        payload = {
            "fields": list(self.fields),
            "float_tolerance": self.tolerance,
            "field_constraints": self.field_constraints,
            "field_descriptions": self.field_descriptions,
            "field_validators": {
                fname: [spec.to_canonical_dict() for spec in specs]
                for fname, specs in sorted(self.field_validators.items())
            },
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _get_judge_model_id(self) -> str:
        """Return stable identifier for the configured SemanticJudge LLM."""
        if self.student_llm is None:
            return "none"
        model_id = getattr(self.student_llm, "model", None)
        if model_id:
            return str(model_id)
        return type(self.student_llm).__name__

    def _get_judge_config_signature(self) -> Tuple[Any, ...]:
        """Return (model, provider, temperature, thinking_config) signature for cache pinning."""
        if self.student_llm is None:
            return ("none", "none", None, "")
        inner = getattr(self.student_llm, "provider", None)
        target = inner if inner is not None and not isinstance(inner, str) else self.student_llm
        provider_val = getattr(target, "provider", None)
        provider_name = (
            str(provider_val)
            if isinstance(provider_val, str) and provider_val
            else type(target).__name__
        )
        temperature = getattr(self.student_llm, "temperature", getattr(target, "temperature", None))
        thinking_payload = {
            "thinking_level": getattr(target, "thinking_level", None),
            "thinking_budget": getattr(target, "thinking_budget", None),
            "reasoning": getattr(target, "reasoning", None),
        }
        thinking_repr = json.dumps(thinking_payload, sort_keys=True, default=str)
        return (self._get_judge_model_id(), provider_name, temperature, thinking_repr)

    def _record_judge_counter(self, key: str, delta: int = 1) -> None:
        """Increment an active per-evaluation judge error/parse counter if tracking is enabled."""
        counters = getattr(self._judge_counter_local, "counters", None)
        if counters is not None and key in counters:
            counters[key] += delta

    def clear_cache(self) -> None:
        """Clear the in-memory semantic judge verdict, reasoning, and validator caches."""
        with self._cache_lock:
            self._verdict_cache.clear()
            self._reasoning_cache.clear()
            self._validator_compare_cache.clear()

    def _is_na(self, val: Any) -> bool:
        """Return True if value is None or NaN."""
        return val is None or (isinstance(val, float) and math.isnan(val))

    def _is_empty_or_na(self, val: Any) -> bool:
        """Return True if value is None, NaN, or empty/whitespace-only string."""
        return self._is_na(val) or self._normalize_text(val) == ""

    def _violates_field_constraints(self, field_name: str, val_pred: Any) -> bool:
        """Return True if predicted value violates explicit schema constraints."""
        if self._is_na(val_pred):
            return False
        constraints = self.field_constraints.get(field_name)
        if not constraints:
            return False

        pred_str = str(val_pred).strip()
        choices = constraints.get("choices")
        if choices and pred_str not in choices:
            return True

        pattern = constraints.get("pattern")
        if pattern and not re.search(pattern, pred_str):
            return True

        min_val = constraints.get("min_value")
        max_val = constraints.get("max_value")
        if min_val is not None or max_val is not None:
            try:
                num_val = float(pred_str)
                if math.isfinite(num_val):
                    if min_val is not None and num_val < min_val:
                        return True
                    if max_val is not None and num_val > max_val:
                        return True
            except (ValueError, TypeError):
                pass

        return False

    def _normalize_text(self, value: Any) -> str:
        """Normalize input values for comparison.

        Handles dash artifacts and whitespace. Case-sensitive.

        Args:
            value: Input value to normalize.

        Returns:
            Normalized string value (case-preserved).
        """
        if value is None:
            return ""

        # Convert to string, normalize dashes, remove whitespace
        # NOTE: No .lower() - case-sensitive comparison
        return _normalize_text_cached(str(value))

    def _compare_floats(self, val_pred: float, val_gold: float) -> bool:
        """Compare two float values using configured relative tolerance."""
        return math.isclose(val_pred, val_gold, rel_tol=max(self.tolerance, 1e-9))

    def _is_match(self, pred: Any, gold: Any) -> bool:
        """Check if two values match according to strict rules.

        Args:
            pred: Predicted value.
            gold: Ground truth value.

        Returns:
            bool: True if values match, False otherwise.
        """
        gold_is_na = gold is None or (isinstance(gold, float) and math.isnan(gold))
        pred_is_na = pred is None or (isinstance(pred, float) and math.isnan(pred))
        if gold_is_na:
            return pred_is_na
        if pred_is_na:
            return False

        # Numerical comparison (only when schema typed the field as numeric)
        if isinstance(gold, (int, float)) and not isinstance(gold, bool):
            try:
                return self._compare_floats(float(pred), float(gold))
            except (ValueError, TypeError):
                pass

        # String comparison
        return self._normalize_text(pred) == self._normalize_text(gold)

    def _compare_via_validators(
        self, field_name: str, val_a: Any, val_b: Any
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Compare two field values using configured declarative validators if supported."""
        specs = self.field_validators.get(field_name)
        if not specs or self._is_na(val_a) or self._is_na(val_b):
            return None, None

        cache_key = (field_name, str(val_a), str(val_b))
        with self._cache_lock:
            if cache_key in self._validator_compare_cache:
                return self._validator_compare_cache[cache_key]

        from ae.core.schema.validators import (
            FieldValidatorSpec,
            compare_field_values_via_validator,
            supports_value_comparison,
        )

        outcome: Tuple[Optional[bool], Optional[str]] = (None, None)
        for spec in specs:
            if not supports_value_comparison(spec):
                continue
            effective_spec = spec
            if (
                spec.name == "numeric_quantity"
                and "rel_tol" not in spec.params
                and self.tolerance > 0
            ):
                effective_spec = FieldValidatorSpec(
                    name=spec.name,
                    params={**spec.params, "rel_tol": max(self.tolerance, 1e-9)},
                    on_error=spec.on_error,
                    message=spec.message,
                    callable_path=spec.callable_path,
                    ignore_values=spec.ignore_values,
                )
            try:
                match_val, reason = compare_field_values_via_validator(
                    effective_spec, val_a, val_b
                )
            except Exception:
                continue
            if match_val is True:
                outcome = (True, None)
                break
            if match_val is False:
                outcome = (False, reason)
                break

        with self._cache_lock:
            self._validator_compare_cache[cache_key] = outcome
        return outcome

    def _is_field_match(
        self,
        field_name: Optional[str],
        pred: Any,
        gold: Any,
        field_equivalence_fn: Optional[Callable[[str, Any, Any], bool]] = None,
    ) -> bool:
        """Check strict equality, declarative validator equivalence, or custom equivalence."""
        if self._is_match(pred, gold):
            return True
        if self._is_na(pred) or self._is_na(gold):
            return False

        if field_name and field_name in self.field_validators:
            match_val, _ = self._compare_via_validators(field_name, pred, gold)
            if match_val is True:
                return True

        if field_name and field_equivalence_fn is not None:
            try:
                if field_equivalence_fn(field_name, pred, gold):
                    return True
            except Exception as exc:
                logger.debug(f"field_equivalence_fn failed for '{field_name}': {exc}")

        return False

    def _field_alignment_similarity(
        self, val_p: Any, val_g: Any, field_name: Optional[str] = None
    ) -> float:
        """Compute field similarity in [0.0, 1.0] for Hungarian alignment."""
        p_na = self._is_na(val_p)
        g_na = self._is_na(val_g)
        if g_na and p_na:
            return 0.1  # Neutral shared-null weight, strictly below real value matches
        if g_na or p_na:
            return 0.0
        if self._is_field_match(field_name, val_p, val_g):
            return 1.0

        norm_p = self._normalize_text(val_p)
        norm_g = self._normalize_text(val_g)
        if not norm_p or not norm_g:
            return 0.0

        lp = norm_p.lower()
        lg = norm_g.lower()
        if lp == lg:
            return 0.45

        # Numeric proximity (tolerance interval or metric/SI scaling 10^3, 10^6, 10^9)
        try:
            fp_val = float(str(val_p).strip())
            fg_val = float(str(val_g).strip())
            if math.isfinite(fp_val) and math.isfinite(fg_val):
                if math.isclose(fp_val, fg_val, rel_tol=max(self.tolerance, 1e-9)):
                    return 0.45
                if fp_val != 0.0 and fg_val != 0.0:
                    log_ratio = abs(math.log10(abs(fp_val / fg_val)))
                    if any(math.isclose(log_ratio, k, abs_tol=1e-2) for k in (3.0, 6.0, 9.0)):
                        return 0.35
        except (ValueError, TypeError):
            pass

        if lp in lg or lg in lp:
            return min(0.45, 0.15 + 0.30 * (min(len(lp), len(lg)) / max(len(lp), len(lg))))

        return difflib.SequenceMatcher(None, lp, lg).ratio() * 0.35

    def _classify_soft_validator_reason(
        self, field_name: str, val_p: Any, val_g: Any
    ) -> Tuple[str, str]:
        """Classify the validator mechanism and reason for a non-exact field match."""
        specs = self.field_validators.get(field_name, ())
        for spec in specs:
            if spec.name == "smiles":
                from ae.core.schema.domain_validators.chemistry import (
                    _STEREO_TOKENS,
                    _canonicalize_smiles_cached,
                )

                clean_p = str(val_p).strip()
                clean_g = str(val_g).strip()
                canon_taut = bool(spec.params.get("canonicalize_tautomers", True))
                isomeric = bool(spec.params.get("isomeric", True))
                allow_dummy = bool(spec.params.get("allow_dummy_atoms", False))
                allow_frags = bool(spec.params.get("allow_fragments", True))
                if isomeric and bool(spec.params.get("ignore_unassigned_stereo", True)):
                    iso_p, _ = _canonicalize_smiles_cached(
                        clean_p, allow_dummy, allow_frags, canon_taut, True
                    )
                    iso_g, _ = _canonicalize_smiles_cached(
                        clean_g, allow_dummy, allow_frags, canon_taut, True
                    )
                    if iso_p and iso_g and iso_p != iso_g:
                        has_p = any(t in iso_p for t in _STEREO_TOKENS)
                        has_g = any(t in iso_g for t in _STEREO_TOKENS)
                        if not has_p or not has_g:
                            return "validator:smiles", "ignore_unassigned_stereo"
                return "validator:smiles", "canonical_equivalence"
            if spec.name == "unit":
                return "validator:unit", "pint_unit_equivalence"
            if spec.name == "numeric_quantity":
                return "validator:numeric_quantity", "numeric_quantity_equivalence"
            return f"validator:{spec.name}", "canonical_equivalence"
        return "validator", "canonical_equivalence"

    def align_pairs_with_indices(
        self,
        preds: List[ExperimentEntity],
        gts: List[ExperimentEntity],
        min_alignment_score: Optional[float] = None,
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """Align prediction indices to ground truth indices via Hungarian algorithm.

        Applies a discriminative column-weighted cutoff threshold so unrelated FP and FN
        records that only share constant columns are split instead of forced into pairs.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        if not preds and not gts:
            return []
        if not preds:
            return [(None, j) for j in range(len(gts))]
        if not gts:
            return [(i, None) for i in range(len(preds))]

        threshold = (
            self.min_alignment_score
            if min_alignment_score is None
            else float(min_alignment_score)
        )

        # Compute discriminative column weights from GT distribution to avoid gluing
        # unrelated FP/FN records on constant columns (e.g., target_type='MIC').
        weights: Dict[str, float] = {}
        if len(gts) >= 2:
            for field in self.fields:
                vals = [
                    self._normalize_text(getattr(g, field, None))
                    for g in gts
                    if not self._is_na(getattr(g, field, None))
                ]
                non_null = len(vals)
                nunique = len(set(vals))
                if non_null >= 2 and nunique > 1:
                    weights[field] = (nunique - 1) / (non_null - 1)
                else:
                    weights[field] = 0.0
        total_disc_weight = sum(weights.values())

        n_preds, n_gts = len(preds), len(gts)
        cost_matrix = np.zeros((n_preds, n_gts))
        disc_score_matrix = np.zeros((n_preds, n_gts)) if total_disc_weight > 0.0 else None
        n_fields = len(self.fields)

        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                sim_sum = 0.0
                disc_sum = 0.0
                for field in self.fields:
                    f_sim = self._field_alignment_similarity(
                        getattr(p, field, None), getattr(g, field, None), field_name=field
                    )
                    sim_sum += f_sim
                    if disc_score_matrix is not None:
                        disc_sum += weights[field] * f_sim
                score = sim_sum / n_fields if n_fields else 0.0
                cost_matrix[i, j] = 1.0 - score
                if disc_score_matrix is not None:
                    disc_score_matrix[i, j] = disc_sum / total_disc_weight

        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        matched_pred_indices: set[int] = set()
        matched_gt_indices: set[int] = set()
        indexed_pairs: List[Tuple[Optional[int], Optional[int]]] = []

        for r_raw, c_raw in zip(row_inds, col_inds):
            r, c = int(r_raw), int(c_raw)
            if threshold > 0.0:
                pair_score = (
                    float(disc_score_matrix[r, c])
                    if disc_score_matrix is not None
                    else float(1.0 - cost_matrix[r, c])
                )
                if pair_score < threshold:
                    continue
            matched_pred_indices.add(r)
            matched_gt_indices.add(c)
            indexed_pairs.append((r, c))

        indexed_pairs.extend((i, None) for i in range(n_preds) if i not in matched_pred_indices)
        indexed_pairs.extend((None, j) for j in range(n_gts) if j not in matched_gt_indices)
        return indexed_pairs

    def align_pairs(
        self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]
    ) -> List[Tuple[Optional[ExperimentEntity], Optional[ExperimentEntity]]]:
        """Align prediction objects to ground truth objects to maximize total similarity
        using the Hungarian Algorithm.

        Args:
            preds: List of predicted record entities.
            gts: List of ground truth record entities.

        Returns:
            List of aligned pairs (pred, gt), with None for unaligned entities.
        """
        indexed = self.align_pairs_with_indices(preds, gts)
        return [
            (
                preds[pi] if pi is not None else None,
                gts[gi] if gi is not None else None,
            )
            for pi, gi in indexed
        ]

    def _build_judge_system_prompt(self, task_name: str, is_batch: bool = False) -> str:
        """Build static system prompt for semantic judge containing instructions and schema.

        Args:
            task_name: Name of the task (e.g., "clinical_trials", "materials").
            is_batch: Whether to format output instructions for multi-pair batch judging.

        Returns:
            Formatted system prompt string.
        """
        schema_lines = []
        all_fields = list(
            dict.fromkeys(
                list(self.fields)
                + list(self.field_descriptions.keys())
                + list(self.field_constraints.keys())
            )
        )
        for field_name in all_fields:
            desc = str(self.field_descriptions.get(field_name, "")).strip()
            constraints = self.field_constraints.get(field_name, {})
            constraint_parts = []
            if constraints.get("choices"):
                constraint_parts.append(f"allowed choices: {constraints['choices']}")
            if constraints.get("pattern"):
                constraint_parts.append(f"required regex pattern: {constraints['pattern']}")
            if constraints.get("min_value") is not None or constraints.get("max_value") is not None:
                constraint_parts.append(
                    f"allowed range: [{constraints.get('min_value', '-inf')}, {constraints.get('max_value', '+inf')}]"
                )
            constraint_str = f" [{'; '.join(constraint_parts)}]" if constraint_parts else ""
            if desc or constraint_str:
                schema_lines.append(f"- {field_name}: {desc}{constraint_str}".rstrip())
        schema_context = "\n".join(schema_lines) if schema_lines else "- (No additional field descriptions provided)"

        if is_batch:
            output_section = (
                "[OUTPUT FORMAT]\n"
                'Return exactly ONE JSON object keyed by the string pair IDs ("0", "1", ...) — nothing before or after it. '
                "Do not include markdown blocks like ```json.\n"
                "For EACH pair ID and EACH listed discrepancy field in that pair, provide a nested object containing "
                'brief mathematical/logical "reasoning" (max 15 words) and the final "match" verdict ("YES" or "NO").\n\n'
                f"Example structure:\n{_SEMANTIC_JUDGE_BATCH_EXAMPLE}"
            )
        else:
            output_section = (
                "[OUTPUT FORMAT]\n"
                "Return exactly ONE JSON object — nothing before or after it. Do not include markdown blocks like ```json.\n"
                'For EACH discrepancy field, provide a nested object containing brief mathematical/logical "reasoning" '
                '(max 15 words) and the final "match" verdict ("YES" or "NO").\n\n'
                f"Example structure:\n{_SEMANTIC_JUDGE_EXAMPLE}"
            )

        return f"""You are an analytical semantic equivalence judge and domain expert for the provided extraction task.
Task Domain: {task_name}

Adopt the appropriate technical terminology and standard conventions of this task domain to evaluate whether Predicted and Ground Truth values express the exact same underlying fact.

Schema Definition (Field Meanings & Constraints):
{schema_context}

--- JUDGE ROLE & SCOPE ---
You are a SEMANTIC EQUIVALENCE JUDGE. Your objective is to determine whether Predicted and Ground Truth values represent the exact same factual, quantitative, or conceptual reality within the context of the record.
CRITICAL SCHEMA COMPLIANCE RULE: If the Schema Definition specifies an explicit canonical format, unit, regex pattern, or closed set of allowed choices for a field, predictions violating those schema constraints MUST be judged "NO". Semantic equivalence applies ONLY when the schema does not mandate a specific surface form.
You DO NOT have access to the source document. Rely ONLY on the provided JSON record context and Schema Definition.

--- INSTRUCTIONS ---
PRE-PROCESSING RULE: If the schema splits a single concept across multiple coupled fields (e.g., `value` and `unit`), evaluate them jointly as a single unit-bearing quantity using the full record context.

Evaluate EACH discrepancy strictly using the following criteria:

[ANSWER "YES" (SEMANTICS PRESERVED) ONLY IF IT FITS ONE OF THESE:]
1. Quantitative Equivalence (Math/Ranges/Scaling): Values represent the exact same number, magnitude, interval, or relationship despite different scientific notation (e.g., 1.5e3 == 1500), equivalent unit conversions across coupled value/unit fields (e.g., 0.5 kg == 500 g), or equivalent precision/rounding within standard tolerance. Zero equals zero across compatible units.
2. Qualitative Equivalence (Synonyms/Ontology/Ordering): Terms are unambiguous domain synonyms, canonical vs. standard abbreviated forms, case variations (when case is not schema-constrained), or alternate orderings of identical sets/lists with the exact same elements.

[ANSWER "NO" (SEMANTICS VIOLATED) IF IT FITS ONE OF THESE:]
1. Factual Contradiction (Mutation): Both Ground Truth and Predicted contain values, but they convey different facts, magnitudes, entities, roles, or incomplete/extra list elements (e.g., GT lists 3 items, Pred lists 2).
2. Schema / Format Violation: Predicted violates explicit schema constraints (e.g., invalid enum choice, wrong pattern, prohibited formatting) or changes a coupled unit without converting the numeric magnitude.
3. Missing or Unsupported Value: Either Ground Truth or Predicted is null/empty while the other is populated, or Predicted introduces unverified information not present in Ground Truth.

{output_section}"""

    def _build_judge_user_prompt(
        self,
        gt_json: Dict[str, Any],
        pred_json: Dict[str, Any],
        discrepancies: List[str],
    ) -> str:
        """Build dynamic user prompt containing compared records and discrepancies.

        Args:
            gt_json: Ground truth record as dictionary.
            pred_json: Predicted record as dictionary.
            discrepancies: List of field names with mismatches.

        Returns:
            Formatted user prompt string.
        """
        discrepancy_lines = []
        for field_name in discrepancies:
            gt_val = gt_json.get(field_name)
            pred_val = pred_json.get(field_name)
            gt_str = "null" if self._is_na(gt_val) else str(gt_val)
            pred_str = "null" if self._is_na(pred_val) else str(pred_val)
            discrepancy_lines.append(f"- {field_name}: GT='{gt_str}', Pred='{pred_str}'")
        discrepancies_text = "\n".join(discrepancy_lines)

        return f"""--- CONTEXT (Full Records) ---
Ground Truth (Reference):
{json.dumps(gt_json, indent=2, default=str)}

Predicted (Extraction):
{json.dumps(pred_json, indent=2, default=str)}

--- DISCREPANCIES TO EVALUATE ---
The following fields did not match strictly. Evaluate ONLY these fields based on the context above:
{discrepancies_text}"""

    def _build_judge_batch_user_prompt(
        self,
        batch_items: List[Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, Any], List[str]]],
    ) -> str:
        """Build batched user prompt containing multiple aligned record pairs and discrepancies."""
        sections = []
        for idx, (_, gt_json, pred_json, discrepancies) in enumerate(batch_items):
            discrepancy_lines = []
            for field_name in discrepancies:
                gt_val = gt_json.get(field_name)
                pred_val = pred_json.get(field_name)
                gt_str = "null" if self._is_na(gt_val) else str(gt_val)
                pred_str = "null" if self._is_na(pred_val) else str(pred_val)
                discrepancy_lines.append(f"- {field_name}: GT='{gt_str}', Pred='{pred_str}'")
            discrepancies_text = "\n".join(discrepancy_lines)
            sections.append(
                f'=== PAIR "{idx}" ===\n'
                f"Ground Truth (Reference):\n{json.dumps(gt_json, indent=2, default=str)}\n\n"
                f"Predicted (Extraction):\n{json.dumps(pred_json, indent=2, default=str)}\n\n"
                f'Discrepancies to evaluate for pair "{idx}":\n{discrepancies_text}'
            )
        return "\n\n".join(sections)

    def _partition_discrepancies(
        self,
        gt_json: Dict[str, Any],
        pred_json: Dict[str, Any],
        discrepancies: List[str],
        reasons_out: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Split mismatched fields into (judge_eval_fields, deterministic_no_fields)."""
        has_compatible_unit_discrepancy = False
        for uf in discrepancies:
            specs = self.field_validators.get(uf, ())
            if not any(s.name == "unit" for s in specs):
                continue
            u_gt = gt_json.get(uf)
            u_pred = pred_json.get(uf)
            if (
                self._is_empty_or_na(u_gt)
                or self._is_empty_or_na(u_pred)
                or self._violates_field_constraints(uf, u_pred)
            ):
                continue
            u_match, _ = self._compare_via_validators(uf, u_pred, u_gt)
            if u_match is not False:
                has_compatible_unit_discrepancy = True
                break

        eval_fields: List[str] = []
        deterministic_no: List[str] = []
        for f in discrepancies:
            gt_val = gt_json.get(f)
            pred_val = pred_json.get(f)
            if (
                self._is_empty_or_na(gt_val)
                or self._is_empty_or_na(pred_val)
                or self._violates_field_constraints(f, pred_val)
            ):
                deterministic_no.append(f)
                continue

            if f in self.field_validators:
                match_val, reason = self._compare_via_validators(f, pred_val, gt_val)
                if match_val is False:
                    if (
                        has_compatible_unit_discrepancy
                        and reason in ("numeric_mismatch", "std_mismatch")
                        and any(s.name == "numeric_quantity" for s in self.field_validators[f])
                    ):
                        eval_fields.append(f)
                        continue
                    deterministic_no.append(f)
                    if reasons_out is not None:
                        reasons_out[f] = reason
                    continue

            eval_fields.append(f)
        return eval_fields, deterministic_no

    def _make_discrepancy_cache_key(
        self,
        task_name: str,
        gt_json: Dict[str, Any],
        pred_json: Dict[str, Any],
        eval_fields: List[str],
    ) -> Tuple[Any, ...]:
        """Build hashable cache key pinned to judge config, prompt, schema, and full record context."""
        resolved_task = task_name or "unknown"
        prompt_hash = hashlib.sha256(
            self._build_judge_system_prompt(resolved_task, is_batch=False).encode("utf-8")
        ).hexdigest()[:16]
        all_keys = sorted(set(self.fields) | set(gt_json.keys()) | set(pred_json.keys()))
        context_payload = {
            "gt": {
                k: (None if self._is_na(gt_json.get(k)) else gt_json.get(k))
                for k in all_keys
            },
            "pred": {
                k: (None if self._is_na(pred_json.get(k)) else pred_json.get(k))
                for k in all_keys
            },
        }
        context_hash = hashlib.sha256(
            json.dumps(context_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        return (
            self._get_judge_model_id(),
            self._schema_constraints_hash,
            resolved_task,
            self._get_judge_config_signature(),
            prompt_hash,
            context_hash,
            tuple(sorted(eval_fields)),
        )

    def _call_semantic_judge(
        self,
        task_name: str,
        gt_json: Dict[str, Any],
        pred_json: Dict[str, Any],
        discrepancies: List[str],
    ) -> Dict[str, str]:
        """Call semantic judge LLM and parse response.

        Args:
            task_name: Name of the task.
            gt_json: Ground truth record as dictionary.
            pred_json: Predicted record as dictionary.
            discrepancies: List of field names with mismatches.

        Returns:
            Dictionary mapping field names to "YES" or "NO".
            Empty dict if LLM call fails (fallback to strict).
        """
        if not self.enable_semantic_judge:
            logger.debug("[SemanticJudge] Disabled, skipping evaluation")
            return {}

        if self.student_llm is None:
            logger.warning("[SemanticJudge] student_llm not provided, skipping evaluation")
            return {}

        eval_fields, deterministic_no = self._partition_discrepancies(
            gt_json, pred_json, discrepancies
        )
        verdicts: Dict[str, str] = {f: "NO" for f in deterministic_no}
        if not eval_fields:
            return verdicts

        cache_key = self._make_discrepancy_cache_key(task_name, gt_json, pred_json, eval_fields)
        with self._cache_lock:
            cached = self._verdict_cache.get(cache_key)
        if cached is not None:
            for f in eval_fields:
                verdicts[f] = cached.get(f, "NO")
            return verdicts

        response_text: Any = ""
        try:
            system_prompt = self._build_judge_system_prompt(task_name, is_batch=False)
            user_prompt = self._build_judge_user_prompt(gt_json, pred_json, eval_fields)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            logger.info(
                f"[SemanticJudge] Evaluating fields {eval_fields} via "
                f"{getattr(self.student_llm, 'model', type(self.student_llm).__name__)}..."
            )
            response = self.student_llm(messages)
            response_text = response[0] if isinstance(response, list) else response
            cleaned = _clean_judge_response(response_text)
            parsed = _parse_judge_json(cleaned)
            if parsed is None:
                self._record_judge_counter("judge_parse_failures")
                logger.warning(
                    f"[SemanticJudge] No valid JSON found. Raw response: {str(response_text)[:200]}..."
                )
                return {}

            evaluated_verdicts = {
                f: _extract_field_verdict(parsed.get(f))
                for f in eval_fields
            }
            evaluated_reasonings = {
                f: _extract_field_reasoning(parsed.get(f))
                for f in eval_fields
            }
            with self._cache_lock:
                self._verdict_cache[cache_key] = dict(evaluated_verdicts)
                self._reasoning_cache[cache_key] = dict(evaluated_reasonings)
            verdicts.update(evaluated_verdicts)
            return verdicts

        except Exception as e:
            self._record_judge_counter("judge_api_errors")
            logger.warning(f"[SemanticJudge] Failed: {e}")
            return {}

    def _call_semantic_judge_batch(
        self,
        task_name: str,
        batch_items: List[Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, Any], List[str]]],
    ) -> Dict[Tuple[Any, ...], Dict[str, str]]:
        """Evaluate multiple uncached record pairs in batched LLM calls and populate cache."""
        batch_results: Dict[Tuple[Any, ...], Dict[str, str]] = {}
        if not self.enable_semantic_judge or self.student_llm is None or not batch_items:
            return batch_results

        for start in range(0, len(batch_items), self.max_judge_batch_size):
            chunk = batch_items[start : start + self.max_judge_batch_size]
            if len(chunk) == 1:
                cache_key, gt_json, pred_json, eval_fields = chunk[0]
                batch_results[cache_key] = self._call_semantic_judge(
                    task_name, gt_json, pred_json, eval_fields
                )
                continue

            response_text: Any = ""
            try:
                system_prompt = self._build_judge_system_prompt(task_name, is_batch=True)
                user_prompt = self._build_judge_batch_user_prompt(chunk)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                logger.info(
                    f"[SemanticJudge] Batch evaluating {len(chunk)} discrepancy pair(s) via "
                    f"{getattr(self.student_llm, 'model', type(self.student_llm).__name__)}..."
                )
                response = self.student_llm(messages)
                response_text = response[0] if isinstance(response, list) else response
                cleaned = _clean_judge_response(response_text)
                parsed = _parse_judge_json(cleaned)
                if parsed is None:
                    self._record_judge_counter("judge_parse_failures")
                    logger.warning(
                        f"[SemanticJudge] No valid batch JSON found. Raw response: {str(response_text)[:200]}..."
                    )
                    for cache_key, gt_json, pred_json, eval_fields in chunk:
                        batch_results[cache_key] = self._call_semantic_judge(
                            task_name, gt_json, pred_json, eval_fields
                        )
                    continue

                for idx, (cache_key, gt_json, pred_json, eval_fields) in enumerate(chunk):
                    pair_verdicts = parsed.get(str(idx))
                    if not isinstance(pair_verdicts, dict):
                        self._record_judge_counter("judge_parse_failures")
                        logger.warning(
                            f"[SemanticJudge] Missing or malformed batch entry for pair '{idx}'; "
                            "falling back to single-pair judge call."
                        )
                        batch_results[cache_key] = self._call_semantic_judge(
                            task_name, gt_json, pred_json, eval_fields
                        )
                        continue

                    evaluated = {
                        f: _extract_field_verdict(pair_verdicts.get(f))
                        for f in eval_fields
                    }
                    evaluated_reasonings = {
                        f: _extract_field_reasoning(pair_verdicts.get(f))
                        for f in eval_fields
                    }
                    with self._cache_lock:
                        self._verdict_cache[cache_key] = dict(evaluated)
                        self._reasoning_cache[cache_key] = dict(evaluated_reasonings)
                    batch_results[cache_key] = evaluated
            except Exception as e:
                self._record_judge_counter("judge_api_errors")
                logger.warning(f"[SemanticJudge] Batch call failed: {e}")
                for cache_key, gt_json, pred_json, eval_fields in chunk:
                    batch_results[cache_key] = self._call_semantic_judge(
                        task_name, gt_json, pred_json, eval_fields
                    )

        return batch_results

    def _log_comparison_table(
        self,
        pred: ExperimentEntity,
        gold: ExperimentEntity,
        strict_matches: List[str],
        discrepancies: List[str],
        verdicts: Dict[str, str],
    ) -> None:
        """Log detailed comparison table for a pair of records.

        Args:
            pred: Predicted record entity.
            gold: Ground truth record entity.
            strict_matches: List of field names with strict matches.
            discrepancies: List of field names with discrepancies.
            verdicts: Dictionary mapping field names to judge verdicts (YES/NO).
        """
        table_data = []

        for field in self.fields:
            val_pred = getattr(pred, field, None)
            val_gold = getattr(gold, field, None)

            # Skip if both NA
            if self._is_na(val_gold) and self._is_na(val_pred):
                continue

            pred_str = "null" if self._is_na(val_pred) else str(val_pred)
            gold_str = "null" if self._is_na(val_gold) else str(val_gold)

            # Determine strict match status
            strict_match = "YES" if field in strict_matches else "NO"

            # Determine judge decision
            if field in strict_matches:
                judge_decision = "—"
            else:
                judge_decision = verdicts.get(field, "NO")

            table_data.append([field, pred_str, gold_str, strict_match, judge_decision])

        if table_data:
            table = tabulate(
                table_data,
                headers=["Field", "Extracted", "Ground Truth", "Strict Match", "Judge"],
                tablefmt="fancy_grid",
            )
            logger.info(f"\n{table}")

    def _process_false_negative(
        self,
        gold: Any,
        field_correct: Dict[str, int],
        field_total: Dict[str, int],
    ) -> tuple:
        """Process false negative case (pred=None, gold≠None)."""
        fn = 0
        for f in self.fields:
            if not self._is_na(getattr(gold, f, None)):
                fn += 1
                field_total[f] += 1
        return fn, field_correct, field_total

    def _process_false_positive(
        self,
        pred: Any,
        field_correct: Dict[str, int],
        field_total: Dict[str, int],
    ) -> tuple:
        """Process false positive case (gold=None, pred≠None)."""
        fp = 0
        for f in self.fields:
            if not self._is_na(getattr(pred, f, None)):
                fp += 1
                field_total[f] += 1
        return fp, field_correct, field_total

    def _process_aligned_pair(
        self,
        pred: Any,
        gold: Any,
        field_correct: Dict[str, int],
        field_total: Dict[str, int],
        task_name: Optional[str],
        batch_verdicts: Optional[Dict[Tuple[Any, ...], Dict[str, str]]] = None,
        field_equivalence_fn: Optional[Callable[[str, Any, Any], bool]] = None,
        pair_detail: Optional[Dict[str, Any]] = None,
        strict_field_correct: Optional[Dict[str, int]] = None,
    ) -> tuple:
        """Process aligned pair (both pred and gold exist)."""
        tp, fp, fn = 0, 0, 0
        strict_tp, strict_fp, strict_fn = 0, 0, 0
        strict_matches: List[str] = []
        soft_matched_fields: List[str] = []
        soft_match_details: Dict[str, Dict[str, Any]] = {}
        judge_reasoning: Dict[str, str] = {}
        discrepancies: List[str] = []

        for f in self.fields:
            val_p = getattr(pred, f, None)
            val_g = getattr(gold, f, None)
            g_na = self._is_na(val_g)
            p_na = self._is_na(val_p)

            if g_na and p_na:
                continue  # True Negative (Ignore)

            field_total[f] += 1  # Field participates in evaluation

            if not g_na and p_na:
                discrepancies.append(f)  # Missing value (Pure FN)
            elif g_na and not p_na:
                discrepancies.append(f)  # Hallucinated value (Pure FP)
            elif self._is_match(val_p, val_g):
                strict_matches.append(f)
            elif f in self.field_validators and self._compare_via_validators(f, val_p, val_g)[0] is True:
                strict_matches.append(f)
                soft_matched_fields.append(f)
                method, soft_reason = self._classify_soft_validator_reason(f, val_p, val_g)
                soft_match_details[f] = {
                    "method": method,
                    "reason": soft_reason,
                    "pred": val_p,
                    "gold": val_g,
                }
            elif field_equivalence_fn is not None and self._is_field_match(
                f, val_p, val_g, field_equivalence_fn=field_equivalence_fn
            ):
                strict_matches.append(f)
                soft_matched_fields.append(f)
                soft_match_details[f] = {
                    "method": "equivalence_fn",
                    "reason": "custom_equivalence",
                    "pred": val_p,
                    "gold": val_g,
                }
            else:
                discrepancies.append(f)  # Value mismatch (FP + FN candidate)

        # Count deterministic matches as TP for both strict and semantic metrics
        tp += len(strict_matches)
        strict_tp += len(strict_matches)
        for f in strict_matches:
            field_correct[f] += 1
            if strict_field_correct is not None:
                strict_field_correct[f] += 1

        matched_fields = list(strict_matches)
        mismatched_fields: Dict[str, Dict[str, Any]] = {}

        # Strict penalties for all discrepancies (independent of semantic judge)
        for field_name in discrepancies:
            val_p = getattr(pred, field_name, None)
            val_g = getattr(gold, field_name, None)
            s_tp_add, s_fp_add, s_fn_add = self._apply_strict_penalty(val_p, val_g)
            strict_tp += s_tp_add
            strict_fp += s_fp_add
            strict_fn += s_fn_add

        # Handle discrepancies
        if discrepancies and self.enable_semantic_judge:
            gt_json = {f: getattr(gold, f, None) for f in self.fields}
            pred_json = {f: getattr(pred, f, None) for f in self.fields}
            validator_reasons: Dict[str, Optional[str]] = {}
            eval_fields, deterministic_no = self._partition_discrepancies(
                gt_json, pred_json, discrepancies, reasons_out=validator_reasons
            )

            verdicts: Dict[str, str] = {f: "NO" for f in deterministic_no}
            cached_reasonings: Dict[str, str] = {}
            if eval_fields:
                resolved_task = task_name or "unknown"
                cache_key = self._make_discrepancy_cache_key(
                    resolved_task, gt_json, pred_json, eval_fields
                )
                if batch_verdicts is not None and cache_key in batch_verdicts:
                    judge_verdicts = batch_verdicts[cache_key]
                else:
                    judge_verdicts = self._call_semantic_judge(
                        task_name=resolved_task,
                        gt_json=gt_json,
                        pred_json=pred_json,
                        discrepancies=eval_fields,
                    )
                verdicts.update(judge_verdicts)
                with self._cache_lock:
                    cached_reasonings = dict(self._reasoning_cache.get(cache_key, {}))

            # Log comparison table
            self._log_comparison_table(pred, gold, strict_matches, discrepancies, verdicts)

            # Apply verdicts
            for field_name in discrepancies:
                verdict = verdicts.get(field_name, "NO")
                tp_add, fp_add, fn_add = self._apply_semantic_verdict(
                    field_name, pred, gold, verdict
                )
                tp += tp_add
                fp += fp_add
                fn += fn_add

                val_p = getattr(pred, field_name, None)
                val_g = getattr(gold, field_name, None)
                r_text = cached_reasonings.get(field_name, "")
                if r_text:
                    judge_reasoning[field_name] = r_text

                if verdict == "YES":
                    field_correct[field_name] += 1
                    matched_fields.append(field_name)
                    soft_matched_fields.append(field_name)
                    soft_match_details[field_name] = {
                        "method": "semantic_judge",
                        "reason": r_text or "semantic_equivalence",
                        "pred": val_p,
                        "gold": val_g,
                    }
                elif pair_detail is not None:
                    reason = (
                        "missing_pred"
                        if (self._is_na(val_p) and not self._is_na(val_g))
                        else (
                            "spurious_pred"
                            if (not self._is_na(val_p) and self._is_na(val_g))
                            else "value_mismatch"
                        )
                    )
                    if reason == "value_mismatch" and validator_reasons.get(field_name) == "stereo_diff":
                        reason = "stereo_diff"
                    entry: Dict[str, Any] = {
                        "pred": val_p,
                        "gold": val_g,
                        "reason": reason,
                    }
                    if r_text:
                        entry["reasoning"] = r_text
                    mismatched_fields[field_name] = entry
        else:
            # No discrepancies or judge disabled - apply strict penalties
            for field_name in discrepancies:
                val_p = getattr(pred, field_name, None)
                val_g = getattr(gold, field_name, None)
                tp_add, fp_add, fn_add = self._apply_strict_penalty(
                    val_p, val_g
                )
                tp += tp_add
                fp += fp_add
                fn += fn_add
                if pair_detail is not None:
                    reason = (
                        "missing_pred"
                        if (self._is_na(val_p) and not self._is_na(val_g))
                        else (
                            "spurious_pred"
                            if (not self._is_na(val_p) and self._is_na(val_g))
                            else "value_mismatch"
                        )
                    )
                    if (
                        reason == "value_mismatch"
                        and field_name in self.field_validators
                        and not self._is_na(val_p)
                        and not self._is_na(val_g)
                    ):
                        _, v_reason = self._compare_via_validators(field_name, val_p, val_g)
                        if v_reason == "stereo_diff":
                            reason = "stereo_diff"
                    mismatched_fields[field_name] = {
                        "pred": val_p,
                        "gold": val_g,
                        "reason": reason,
                    }

        if pair_detail is not None:
            pair_detail["matched_fields"] = matched_fields
            pair_detail["strict_matched_fields"] = strict_matches
            pair_detail["soft_matched_fields"] = soft_matched_fields
            pair_detail["soft_match_details"] = soft_match_details
            pair_detail["mismatched_fields"] = mismatched_fields
            pair_detail["judge_reasoning"] = judge_reasoning
            pair_detail["strict_counts"] = (strict_tp, strict_fp, strict_fn)

        return tp, fp, fn, field_correct, field_total

    def _apply_semantic_verdict(
        self,
        field_name: str,
        pred: Any,
        gold: Any,
        verdict: str,
    ) -> tuple:
        """Apply semantic judge verdict to scoring."""
        if verdict == "YES":
            return 1, 0, 0
        val_p = getattr(pred, field_name, None)
        val_g = getattr(gold, field_name, None)
        return self._apply_strict_penalty(val_p, val_g)

    def _apply_strict_penalty(
        self,
        val_p: Any,
        val_g: Any,
    ) -> tuple:
        """Apply strict penalty for discrepancies without semantic judge."""
        p_na = self._is_na(val_p)
        g_na = self._is_na(val_g)
        if p_na and not g_na:
            return 0, 0, 1  # Pure Miss
        if not p_na and g_na:
            return 0, 1, 0  # Pure Hallucination
        return 0, 1, 1  # Value mismatch

    def _annotate_structural_errors(
        self,
        preds: List[ExperimentEntity],
        gts: List[ExperimentEntity],
        pair_results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Use POT unbalanced OT and RapidFuzz to tag duplicate_of_tp, row_merge, and row_split."""
        diagnostics: Dict[str, List[Dict[str, Any]]] = {
            "duplicate_of_tp": [],
            "row_merge": [],
            "row_split": [],
        }
        if not preds or not gts:
            return diagnostics

        fp_entries = [
            r for r in pair_results if r["status"] == "fp" and r.get("pred_index") is not None
        ]
        fn_entries = [
            r for r in pair_results if r["status"] == "fn" and r.get("gt_index") is not None
        ]
        if not fp_entries and not fn_entries:
            return diagnostics

        import numpy as np
        import ot
        from rapidfuzz import fuzz

        n_preds, n_gts = len(preds), len(gts)
        n_fields = len(self.fields)
        sim_matrix = np.zeros((n_preds, n_gts))

        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                sim_sum = 0.0
                for field in self.fields:
                    vp = getattr(p, field, None)
                    vg = getattr(g, field, None)
                    p_na, g_na = self._is_na(vp), self._is_na(vg)
                    if p_na and g_na:
                        sim_sum += 0.1
                    elif p_na or g_na:
                        sim_sum += 0.0
                    elif self._is_field_match(field, vp, vg):
                        sim_sum += 1.0
                    else:
                        sp, sg = str(vp).strip(), str(vg).strip()
                        fuzz_score = 0.5 * (
                            fuzz.token_set_ratio(sp, sg) + fuzz.ratio(sp, sg)
                        )
                        sim_sum += fuzz_score / 100.0
                sim_matrix[i, j] = sim_sum / n_fields if n_fields else 0.0

        cost_matrix = 1.0 - sim_matrix
        try:
            transport = ot.unbalanced.sinkhorn_knopp_unbalanced(
                np.ones(n_preds),
                np.ones(n_gts),
                cost_matrix,
                reg=0.05,
                reg_m=0.5,
            )
            if not np.all(np.isfinite(transport)):
                return diagnostics
        except Exception as exc:
            logger.debug(f"POT unbalanced OT skipped due to solver error: {exc}")
            return diagnostics

        row_sums = transport.sum(axis=1, keepdims=True)
        col_sums = transport.sum(axis=0, keepdims=True)
        gt_pair_by_idx = {
            r["gt_index"]: r for r in pair_results if r.get("gt_index") is not None
        }
        pred_pair_by_idx = {
            r["pred_index"]: r for r in pair_results if r.get("pred_index") is not None
        }
        fp_indices = {int(r["pred_index"]) for r in fp_entries}
        fn_indices = {int(r["gt_index"]) for r in fn_entries}
        tagged_dup_preds: set[int] = set()

        # 1. duplicate_of_tp: FP pred couples strongly to a GT already matched as TP
        for pi in sorted(fp_indices):
            j_best = int(np.argmax(transport[pi]))
            t_val = float(transport[pi, j_best])
            row_share = t_val / max(float(row_sums[pi, 0]), 1e-12)
            s_val = float(sim_matrix[pi, j_best])
            gt_entry = gt_pair_by_idx.get(j_best)
            if (
                gt_entry is not None
                and gt_entry.get("status") == "tp"
                and s_val >= 0.80
                and (t_val >= 0.15 or row_share >= 0.35)
            ):
                tagged_dup_preds.add(pi)
                pred_entry = pred_pair_by_idx[pi]
                pred_entry["diagnostic_tag"] = "duplicate_of_tp"
                pred_entry["diagnostic_related_gt_indices"] = [j_best]
                pred_entry["diagnostic_score"] = round(s_val, 4)
                diagnostics["duplicate_of_tp"].append(
                    {
                        "pred_index": pi,
                        "gt_index": j_best,
                        "matched_pred_index": gt_entry.get("pred_index"),
                        "similarity": round(s_val, 4),
                        "transport_mass": round(t_val, 4),
                    }
                )

        # 2. row_merge: single pred couples strongly to >= 2 GTs (at least one FN)
        for pi in range(n_preds):
            if pi in tagged_dup_preds:
                continue
            r_sum = max(float(row_sums[pi, 0]), 1e-12)
            coupled_gts = [
                j
                for j in range(n_gts)
                if sim_matrix[pi, j] >= 0.65
                and (transport[pi, j] >= 0.12 or (transport[pi, j] / r_sum) >= 0.20)
            ]
            merged_fns = [j for j in coupled_gts if j in fn_indices]
            if len(coupled_gts) >= 2 and merged_fns:
                for gj in merged_fns:
                    fn_entry = gt_pair_by_idx[gj]
                    if not fn_entry.get("diagnostic_tag"):
                        fn_entry["diagnostic_tag"] = "row_merge"
                        fn_entry["diagnostic_related_pred_indices"] = [pi]
                        fn_entry["diagnostic_score"] = round(float(sim_matrix[pi, gj]), 4)
                pred_entry = pred_pair_by_idx.get(pi)
                if pred_entry is not None and not pred_entry.get("diagnostic_tag"):
                    pred_entry["diagnostic_tag"] = "row_merge"
                    pred_entry["diagnostic_related_gt_indices"] = list(coupled_gts)
                diagnostics["row_merge"].append(
                    {
                        "pred_index": pi,
                        "gt_indices": coupled_gts,
                        "fn_gt_indices": merged_fns,
                        "similarities": {j: round(float(sim_matrix[pi, j]), 4) for j in coupled_gts},
                    }
                )

        # 3. row_split: >= 2 non-duplicate preds couple strongly to 1 GT (at least one FP)
        for gj in range(n_gts):
            gt_entry = gt_pair_by_idx.get(gj)
            if gt_entry is not None and gt_entry.get("status") == "tp":
                continue
            c_sum = max(float(col_sums[0, gj]), 1e-12)
            coupled_preds = [
                i
                for i in range(n_preds)
                if i not in tagged_dup_preds
                and sim_matrix[i, gj] >= 0.65
                and (transport[i, gj] >= 0.12 or (transport[i, gj] / c_sum) >= 0.20)
            ]
            split_fps = [i for i in coupled_preds if i in fp_indices]
            if len(coupled_preds) >= 2 and split_fps:
                for pi in split_fps:
                    fp_entry = pred_pair_by_idx[pi]
                    if not fp_entry.get("diagnostic_tag"):
                        fp_entry["diagnostic_tag"] = "row_split"
                        fp_entry["diagnostic_related_gt_indices"] = [gj]
                        fp_entry["diagnostic_score"] = round(float(sim_matrix[pi, gj]), 4)
                if gt_entry is not None and not gt_entry.get("diagnostic_tag"):
                    gt_entry["diagnostic_tag"] = "row_split"
                    gt_entry["diagnostic_related_pred_indices"] = list(coupled_preds)
                diagnostics["row_split"].append(
                    {
                        "gt_index": gj,
                        "pred_indices": coupled_preds,
                        "fp_pred_indices": split_fps,
                        "similarities": {i: round(float(sim_matrix[i, gj]), 4) for i in coupled_preds},
                    }
                )

        return diagnostics

    def _evaluate_pairs_core(
        self,
        pair_items: List[Tuple[Optional[int], Optional[int], Optional[Any], Optional[Any]]],
        task_name: Optional[str] = None,
        field_equivalence_fn: Optional[Callable[[str, Any, Any], bool]] = None,
        collect_pair_results: bool = False,
        preds: Optional[List[ExperimentEntity]] = None,
        gts: Optional[List[ExperimentEntity]] = None,
    ) -> Dict[str, Any]:
        """Shared evaluation engine for `_compute_stats`, `evaluate_pairs`, and `get_detailed_report`."""
        tp, fp, fn = 0, 0, 0
        strict_tp, strict_fp, strict_fn = 0, 0, 0
        field_correct = {f: 0 for f in self.fields}
        strict_field_correct = {f: 0 for f in self.fields}
        field_total = {f: 0 for f in self.fields}
        soft_matched_fields_count = 0
        judge_reasonings: List[Dict[str, Any]] = []

        prev_counters = getattr(self._judge_counter_local, "counters", None)
        active_counters = {"judge_parse_failures": 0, "judge_api_errors": 0}
        self._judge_counter_local.counters = active_counters
        try:
            batch_verdicts: Optional[Dict[Tuple[Any, ...], Dict[str, str]]] = None
            if (
                self.enable_semantic_judge
                and self.student_llm is not None
                and getattr(self._call_semantic_judge, "__func__", None)
                is RecordMatcher._call_semantic_judge
            ):
                resolved_task = task_name or "unknown"
                uncached_by_key: Dict[
                    Tuple[Any, ...], Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, Any], List[str]]
                ] = {}
                for _, _, pred, gold in pair_items:
                    if pred is None or gold is None:
                        continue
                    gt_json = {f: getattr(gold, f, None) for f in self.fields}
                    pred_json = {f: getattr(pred, f, None) for f in self.fields}
                    pair_discrepancies = [
                        f
                        for f in self.fields
                        if not (self._is_na(gt_json[f]) and self._is_na(pred_json[f]))
                        and not self._is_field_match(
                            f, pred_json[f], gt_json[f], field_equivalence_fn=field_equivalence_fn
                        )
                    ]
                    eval_fields, _ = self._partition_discrepancies(
                        gt_json, pred_json, pair_discrepancies
                    )
                    if not eval_fields:
                        continue
                    cache_key = self._make_discrepancy_cache_key(
                        resolved_task, gt_json, pred_json, eval_fields
                    )
                    with self._cache_lock:
                        is_cached = cache_key in self._verdict_cache
                    if not is_cached and cache_key not in uncached_by_key:
                        uncached_by_key[cache_key] = (cache_key, gt_json, pred_json, eval_fields)

                if len(uncached_by_key) > 1:
                    batch_verdicts = self._call_semantic_judge_batch(
                        resolved_task, list(uncached_by_key.values())
                    )

            pair_results: List[Dict[str, Any]] = []

            for pred_idx, gt_idx, pred, gold in pair_items:
                if pred is None and gold is not None:
                    fn_inc, field_correct, field_total = self._process_false_negative(
                        gold, field_correct, field_total
                    )
                    fn += fn_inc
                    strict_fn += fn_inc
                    if collect_pair_results:
                        pair_results.append(
                            {
                                "status": "fn",
                                "pred_index": None,
                                "gt_index": gt_idx,
                                "pred": None,
                                "gold": gold,
                                "matched_fields": [],
                                "soft_matched_fields": [],
                                "mismatched_fields": {},
                                "judge_reasoning": {},
                                "diagnostic_tag": None,
                            }
                        )
                    continue

                if gold is None and pred is not None:
                    fp_inc, field_correct, field_total = self._process_false_positive(
                        pred, field_correct, field_total
                    )
                    fp += fp_inc
                    strict_fp += fp_inc
                    if collect_pair_results:
                        pair_results.append(
                            {
                                "status": "fp",
                                "pred_index": pred_idx,
                                "gt_index": None,
                                "pred": pred,
                                "gold": None,
                                "matched_fields": [],
                                "soft_matched_fields": [],
                                "mismatched_fields": {},
                                "judge_reasoning": {},
                                "diagnostic_tag": None,
                            }
                        )
                    continue

                if pred is None and gold is None:
                    continue

                detail: Dict[str, Any] = {}
                tp_inc, fp_inc, fn_inc, field_correct, field_total = self._process_aligned_pair(
                    pred,
                    gold,
                    field_correct,
                    field_total,
                    task_name,
                    batch_verdicts=batch_verdicts,
                    field_equivalence_fn=field_equivalence_fn,
                    pair_detail=detail,
                    strict_field_correct=strict_field_correct,
                )
                tp += tp_inc
                fp += fp_inc
                fn += fn_inc
                s_tp_inc, s_fp_inc, s_fn_inc = detail.get("strict_counts", (tp_inc, fp_inc, fn_inc))
                strict_tp += s_tp_inc
                strict_fp += s_fp_inc
                strict_fn += s_fn_inc

                soft_fields = detail.get("soft_matched_fields", [])
                soft_matched_fields_count += len(soft_fields)
                pair_reasoning = detail.get("judge_reasoning", {})
                if pair_reasoning:
                    judge_reasonings.append(
                        {
                            "pred_index": pred_idx,
                            "gt_index": gt_idx,
                            "reasoning": pair_reasoning,
                        }
                    )

                if collect_pair_results:
                    mismatched = detail.get("mismatched_fields", {})
                    pair_results.append(
                        {
                            "status": "tp" if not mismatched else "field_mismatch",
                            "pred_index": pred_idx,
                            "gt_index": gt_idx,
                            "pred": pred,
                            "gold": gold,
                            "matched_fields": detail.get("matched_fields", []),
                            "strict_matched_fields": detail.get("strict_matched_fields", []),
                            "soft_matched_fields": soft_fields,
                            "soft_match_details": detail.get("soft_match_details", {}),
                            "mismatched_fields": mismatched,
                            "judge_reasoning": pair_reasoning,
                            "diagnostic_tag": None,
                        }
                    )
        finally:
            self._judge_counter_local.counters = prev_counters

        diagnostics: Dict[str, List[Dict[str, Any]]] = {
            "duplicate_of_tp": [],
            "row_merge": [],
            "row_split": [],
        }
        if collect_pair_results and preds is not None and gts is not None:
            diagnostics = self._annotate_structural_errors(preds, gts, pair_results)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        strict_precision = strict_tp / (strict_tp + strict_fp) if (strict_tp + strict_fp) > 0 else 0.0
        strict_recall = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) > 0 else 0.0
        strict_f1 = (
            2 * strict_precision * strict_recall / (strict_precision + strict_recall)
            if (strict_precision + strict_recall) > 0
            else 0.0
        )

        field_scores = {
            f: (field_correct[f] / field_total[f]) if field_total[f] > 0 else 1.0
            for f in self.fields
        }
        strict_field_scores = {
            f: (strict_field_correct[f] / field_total[f]) if field_total[f] > 0 else 1.0
            for f in self.fields
        }

        result: Dict[str, Any] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "field_scores": field_scores,
            "field_correct": dict(field_correct),
            "field_total": dict(field_total),
            "semantic_precision": precision,
            "semantic_recall": recall,
            "semantic_f1": f1,
            "semantic_field_scores": dict(field_scores),
            "strict_precision": strict_precision,
            "strict_recall": strict_recall,
            "strict_f1": strict_f1,
            "strict_field_scores": strict_field_scores,
            "strict_field_correct": dict(strict_field_correct),
            "strict_field_total": dict(field_total),
            "soft_matched_fields_count": soft_matched_fields_count,
            "judge_reasonings": judge_reasonings,
            "judge_parse_failures": active_counters["judge_parse_failures"],
            "judge_api_errors": active_counters["judge_api_errors"],
            "diagnostics": diagnostics,
        }
        if collect_pair_results:
            result["pair_results"] = pair_results
        return result

    def _compute_stats(
        self,
        pairs: List[Tuple[Optional[Any], Optional[Any]]],
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Micro-F1/Precision/Recall with semantic judge fallback.

        Args:
            pairs: List of aligned pairs (pred, gt).
            task_name: Optional task name for semantic judge context.

        Returns:
            Dict with precision, recall, f1 scores, strict metrics, and field_scores.
        """
        pair_items = [(None, None, pred, gold) for pred, gold in pairs]
        return self._evaluate_pairs_core(
            pair_items, task_name=task_name, collect_pair_results=False
        )

    def evaluate_pairs(
        self,
        preds: List[ExperimentEntity],
        gts: List[ExperimentEntity],
        task_name: Optional[str] = None,
        field_equivalence_fn: Optional[Callable[[str, Any, Any], bool]] = None,
    ) -> Dict[str, Any]:
        """Align and evaluate predictions vs GT, returning detailed metrics and per-pair diagnostics."""
        indexed = self.align_pairs_with_indices(preds, gts)
        pair_items = [
            (
                pi,
                gi,
                preds[pi] if pi is not None else None,
                gts[gi] if gi is not None else None,
            )
            for pi, gi in indexed
        ]
        result = self._evaluate_pairs_core(
            pair_items,
            task_name=task_name,
            field_equivalence_fn=field_equivalence_fn,
            collect_pair_results=True,
            preds=preds,
            gts=gts,
        )
        result["counts"] = {"preds": len(preds), "gts": len(gts)}
        return result

    def get_optimization_score(
        self,
        preds: List[ExperimentEntity],
        gts: List[ExperimentEntity],
        task_name: Optional[str] = None,
    ) -> float:
        """Get optimization score (F1) for use in teleprompter.

        Args:
            preds: List of predicted experiment entities.
            gts: List of ground truth experiment entities.
            task_name: Optional task name for semantic judge context.

        Returns:
            F1 score.
        """
        pairs = self.align_pairs(preds, gts)
        return self._compute_stats(pairs, task_name)["f1"]

    def get_detailed_report(
        self,
        preds: List[ExperimentEntity],
        gts: List[ExperimentEntity],
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get detailed evaluation report with strict/semantic metrics and diagnostics.

        Args:
            preds: List of predicted experiment entities.
            gts: List of ground truth experiment entities.
            task_name: Optional task name for semantic judge context.

        Returns:
            Dict with detailed evaluation metrics.
        """
        indexed = self.align_pairs_with_indices(preds, gts)
        pair_items = [
            (
                pi,
                gi,
                preds[pi] if pi is not None else None,
                gts[gi] if gi is not None else None,
            )
            for pi, gi in indexed
        ]
        stats = self._evaluate_pairs_core(
            pair_items,
            task_name=task_name,
            collect_pair_results=True,
            preds=preds,
            gts=gts,
        )

        return {
            "f1": stats["f1"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "fields": stats["field_scores"],
            "field_correct": stats["field_correct"],
            "field_total": stats["field_total"],
            "semantic_f1": stats["semantic_f1"],
            "semantic_precision": stats["semantic_precision"],
            "semantic_recall": stats["semantic_recall"],
            "semantic_fields": stats["semantic_field_scores"],
            "strict_f1": stats["strict_f1"],
            "strict_precision": stats["strict_precision"],
            "strict_recall": stats["strict_recall"],
            "strict_fields": stats["strict_field_scores"],
            "strict_field_correct": stats["strict_field_correct"],
            "strict_field_total": stats["strict_field_total"],
            "soft_matched_fields_count": stats["soft_matched_fields_count"],
            "judge_reasonings": stats["judge_reasonings"],
            "judge_parse_failures": stats["judge_parse_failures"],
            "judge_api_errors": stats["judge_api_errors"],
            "diagnostics": stats["diagnostics"],
            "pair_results": stats["pair_results"],
            "counts": {"preds": len(preds), "gts": len(gts)},
        }


# Backward-compatible alias
ExperimentMatcher = RecordMatcher
