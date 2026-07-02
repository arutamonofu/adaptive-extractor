"""Evaluation engine for comparing extracted chemical experiments against ground truth."""

import functools
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

from pydantic import BaseModel
from tabulate import tabulate  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_SEMANTIC_JUDGE_EXAMPLE = (
    '{\n'
    '  "km_value": {"reasoning": "0.091 mM is mathematically equivalent to 91 \\u03bcM.", "match": "YES"},\n'
    '  "activity": {"reasoning": "GT has 3 elements, Pred extracted only 2. Incomplete list.", "match": "NO"},\n'
    '  "surface": {"reasoning": "GT has a specific value, Pred is null. Extractor missed it.", "match": "NO"}\n'
    '}'
)

ExperimentEntity: TypeAlias = Union[BaseModel, Any]


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


_RE_STRICT_CLEAN = re.compile(r"\s+")
_DASH_MAP = str.maketrans({"−": "-", "–": "-", "—": "-"})


@functools.lru_cache(maxsize=4096)
def _normalize_text_cached(val_str: str) -> str:
    """Helper to normalize text with caching."""
    return _RE_STRICT_CLEAN.sub("", val_str.translate(_DASH_MAP))


class ExperimentMatcher:
    """Evaluation engine for comparing extracted chemical experiments against ground truth.

    - Strings: Normalized Exact Match (removes spaces, standardizes dashes).
    - Floats: Tolerance Interval (default ±5%).
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
    ):
        """Initialize the ExperimentMatcher.

        Args:
            fields_to_compare: List of field names to compare between entities.
            float_tolerance: Tolerance for float comparisons (0.0 to 1.0).
                            Kept for backward compatibility, not used in strict mode.
            student_llm: DSPy LLM object for semantic judgment (optional).
            field_descriptions: Dictionary of field descriptions (optional).
            enable_semantic_judge: Flag to enable/disable semantic judge (default: True).

        Raises:
            ValueError: If fields_to_compare is empty or float_tolerance is invalid.
        """
        if not fields_to_compare:
            raise ValueError("fields_to_compare cannot be empty")
        if not 0 <= float_tolerance <= 1:
            raise ValueError("float_tolerance must be between 0 and 1")

        self.fields = fields_to_compare
        self.tolerance = float_tolerance  # Kept but unused
        self.student_llm = student_llm
        self.field_descriptions = field_descriptions or {}
        self.enable_semantic_judge = enable_semantic_judge

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
        """Compare two float values with strict tolerance.

        Uses math.isclose with relative tolerance 1e-9.

        Args:
            val_pred: Predicted float value.
            val_gold: Ground truth float value.

        Returns:
            True if values are close, False otherwise.
        """
        return math.isclose(val_pred, val_gold, rel_tol=1e-9)

    def _is_match(self, pred: Any, gold: Any) -> bool:
        """Check if two values match according to strict rules.

        Args:
            pred: Predicted value.
            gold: Ground truth value.

        Returns:
            bool: True if values match, False otherwise.
        """
        # Handle None cases
        if gold is None:
            return pred is None
        if pred is None:
            return False

        # Numerical comparison
        if isinstance(gold, (int, float)):
            try:
                return self._compare_floats(float(pred), float(gold))
            except (ValueError, TypeError):
                # Fall back to string comparison if conversion fails
                pass

        # String comparison
        return self._normalize_text(pred) == self._normalize_text(gold)

    def align_pairs(
        self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]
    ) -> List[Tuple[Optional[ExperimentEntity], Optional[ExperimentEntity]]]:
        """Align prediction objects to ground truth objects to maximize total similarity
        using the Hungarian Algorithm.

        Args:
            preds: List of predicted experiment entities.
            gts: List of ground truth experiment entities.

        Returns:
            List of aligned pairs (pred, gt), with None for unaligned entities.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        # Handle edge cases
        if not preds and not gts:
            return []

        if not preds:
            return [(None, gt) for gt in gts]
        if not gts:
            return [(pred, None) for pred in preds]

        # Create cost matrix
        cost_matrix = np.zeros((len(preds), len(gts)))

        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                matches = sum(
                    1 for field in self.fields
                    if self._is_match(getattr(p, field, None), getattr(g, field, None))
                )

                # Normalize score to [0, 1] range
                score = matches / len(self.fields) if self.fields else 0
                cost_matrix[i, j] = 1 - score  # Convert to cost (minimization problem)

        # Solve assignment problem
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        # Create result pairs
        matched_pred_indices = set(row_inds)
        matched_gt_indices = set(col_inds)

        pairs: List[Tuple[Optional[ExperimentEntity], Optional[ExperimentEntity]]] = []

        # Add matched pairs
        pairs.extend((preds[r], gts[c]) for r, c in zip(row_inds, col_inds))

        # Add unmatched Predictions (False Positives)
        pairs.extend((pred, None) for i, pred in enumerate(preds) if i not in matched_pred_indices)

        # Add unmatched GTs (False Negatives)
        pairs.extend((None, gt) for j, gt in enumerate(gts) if j not in matched_gt_indices)

        return pairs

    def _build_judge_system_prompt(self, task_name: str) -> str:
        """Build static system prompt for semantic judge containing instructions and schema.

        Args:
            task_name: Name of the task (e.g., "nanozymes").

        Returns:
            Formatted system prompt string.
        """
        # Build schema context from field descriptions
        schema_lines = []
        for field_name, description in self.field_descriptions.items():
            schema_lines.append(f"- {field_name}: {description}")
        schema_context = "\n".join(schema_lines)

        return f"""You are an elite analytical judge and a senior domain expert specializing strictly in the scientific or professional field of the provided task.
Task Domain: {task_name}

Your primary directive is to dynamically adopt the required technical expertise, terminology, and standard practices of this specific domain to accurately evaluate semantic equivalence.

Schema Definition (Field Meanings):
{schema_context}

--- JUDGE ROLE & SCOPE ---
You are a SEMANTIC EQUIVALENCE JUDGE. Your strict objective is to evaluate whether the Predicted and Ground Truth values represent the exact same factual, physical, or logical reality. 
You DO NOT enforce text-matching or extraction policies. Calculations, unit conversions, strict filtering, structural reshaping, or literal-only rules applied by the Extractor are IRRELEVANT to your judgment, provided the underlying truth remains identical. 
You have NO access to the source document. Rely ONLY on the provided JSON context.

--- INSTRUCTIONS ---
PRE-PROCESSING RULE: If the schema splits a single physical concept into multiple fields (e.g., `value` and `unit`), evaluate them collectively as a single physical entity.

Evaluate EACH discrepancy strictly routing it through the following logic tree:

[ANSWER "YES" (SEMANTICS PRESERVED) IF IT FITS ONE OF THESE:]
1. Quantitative Equivalence (Math/Ranges): Values represent the exact same magnitude, interval, or physical state despite different scientific notation, unit scaling (e.g., 91 μM == 0.091 mM), or overlapping valid approximations (e.g., "~30" == "30±5"). Zero equals zero in any dimension.
2. Qualitative Equivalence (Text/Ontology): Terms are standard domain synonyms, formal nomenclatures vs. common names, case variations, or alternate orderings of fully identical lists/mixtures. The functional truth in the context is unchanged.
3. Deductive Equivalence (Implicit Nulls): Ground Truth is 'null' AND Predicted contains a value that is strictly, mathematically, or logically guaranteed by other populated fields in the Ground Truth context.

[ANSWER "NO" (SEMANTICS VIOLATED) IF IT FITS ONE OF THESE:]
1. Mutation (Factual Contradiction): Both GT and Predicted contain values, but they represent fundamentally different realities. This includes wrong orders of magnitude, swapped functional roles, or incomplete lists (e.g., GT has 3 elements, Pred extracted only 2).
2. Addition (Hallucination/Guessing): Ground Truth is 'null' AND Predicted contains a value that CANNOT be strictly deduced from the provided context (including guesses based on general domain knowledge).
3. Subtraction (Omission): Ground Truth contains a specific value, but Predicted is 'null' or empty. The extractor missed a required fact.

[OUTPUT FORMAT]
Return exactly ONE JSON object — nothing before or after it. Do not include markdown blocks like ```json.
For EACH discrepancy field, you must provide a nested object containing brief mathematical/logical "reasoning" (max 15 words) and the final "match" boolean ("YES" or "NO").

Example structure:
{_SEMANTIC_JUDGE_EXAMPLE}"""

    def _build_judge_user_prompt(
        self,
        gt_json: Dict[str, Any],
        pred_json: Dict[str, Any],
        discrepancies: List[str],
    ) -> str:
        """Build dynamic user prompt containing compared experiments and discrepancies.

        Args:
            gt_json: Ground truth experiment as dictionary.
            pred_json: Predicted experiment as dictionary.
            discrepancies: List of field names with mismatches.

        Returns:
            Formatted user prompt string.
        """
        # Build discrepancies list with values
        discrepancy_lines = []
        for field_name in discrepancies:
            gt_val = gt_json.get(field_name)
            pred_val = pred_json.get(field_name)
            gt_str = "null" if gt_val is None else str(gt_val)
            pred_str = "null" if pred_val is None else str(pred_val)
            discrepancy_lines.append(f"- {field_name}: GT='{gt_str}', Pred='{pred_str}'")
        discrepancies_text = "\n".join(discrepancy_lines)

        return f"""--- CONTEXT (Full Experiments) ---
Ground Truth (Reference):
{json.dumps(gt_json, indent=2, default=str)}

Predicted (Extraction):
{json.dumps(pred_json, indent=2, default=str)}

--- DISCREPANCIES TO EVALUATE ---
The following fields did not match strictly. Evaluate ONLY these fields based on the context above:
{discrepancies_text}"""

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
            gt_json: Ground truth experiment as dictionary.
            pred_json: Predicted experiment as dictionary.
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

        try:
            # Build prompts
            system_prompt = self._build_judge_system_prompt(task_name)
            user_prompt = self._build_judge_user_prompt(gt_json, pred_json, discrepancies)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call LLM via DSPy interface with prompt_caching enabled
            # DSPy LM returns list of strings, take first
            # Force reasoning/thinking enabled for semantic judge regardless of config
            response = self.student_llm(
                messages,
                reasoning={"enabled": True},  # OpenRouter API reasoning models
                enable_thinking=True,  # Transformers thinking-capable models
                prompt_caching=True,  # Ensure prompt caching is active for semantic judge
            )
            response_text = response[0] if isinstance(response, list) else response

            # Extract JSON from response (handle markdown wrappers, extra text, and thinking blocks)
            # Strip thinking/reasoning blocks first (e.g., <think>...</think>, <think>...</think>)
            cleaned = re.sub(r"<think>.*?</think>", "", str(response_text), flags=re.DOTALL)
            cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

            # Find the first JSON object by brace balancing
            json_str = _extract_first_json(cleaned)
            if json_str is None:
                logger.warning(f"[SemanticJudge] No JSON found. Raw response: {str(response_text)[:200]}...")
                return {}

            # Parse JSON response
            verdicts = json.loads(json_str)

            # Validate and filter verdicts
            # Supports new nested format: {"field": {"reasoning": "...", "match": "YES"}}
            # and legacy flat format: {"field": "YES"} for backward compatibility
            valid_verdicts = {}
            for field_name in discrepancies:
                raw = verdicts.get(field_name)
                if raw is None:
                    # Field not in response, treat as NO
                    valid_verdicts[field_name] = "NO"
                elif isinstance(raw, dict):
                    # New nested format: extract "match" key
                    match_val = str(raw.get("match", "NO")).strip().upper()
                    valid_verdicts[field_name] = "YES" if match_val == "YES" else "NO"
                else:
                    # Legacy flat format fallback: {"field": "YES"}
                    valid_verdicts[field_name] = (
                        "YES" if str(raw).strip().upper() == "YES" else "NO"
                    )

            return valid_verdicts

        except json.JSONDecodeError as e:
            logger.warning(f"[SemanticJudge] JSON parse error: {e}. Raw response: {str(response_text)[:200]}...")
            return {}
        except Exception as e:
            logger.warning(f"[SemanticJudge] Failed: {e}")
            return {}


    def _log_comparison_table(
        self,
        pred: ExperimentEntity,
        gold: ExperimentEntity,
        strict_matches: List[str],
        discrepancies: List[str],
        verdicts: Dict[str, str],
    ) -> None:
        """Log detailed comparison table for a pair of experiments.

        Args:
            pred: Predicted experiment entity.
            gold: Ground truth experiment entity.
            strict_matches: List of field names with strict matches.
            discrepancies: List of field names with discrepancies.
            verdicts: Dictionary mapping field names to judge verdicts (YES/NO).
        """
        table_data = []

        for field in self.fields:
            val_pred = getattr(pred, field, None)
            val_gold = getattr(gold, field, None)

            # Skip if both None
            if val_gold is None and val_pred is None:
                continue

            pred_str = "null" if val_pred is None else str(val_pred)
            gold_str = "null" if val_gold is None else str(val_gold)

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
            if getattr(gold, f, None) is not None:
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
            if getattr(pred, f, None) is not None:
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
    ) -> tuple:
        """Process aligned pair (both pred and gold exist)."""
        tp, fp, fn = 0, 0, 0
        strict_matches = []
        discrepancies = []

        for f in self.fields:
            val_p = getattr(pred, f, None)
            val_g = getattr(gold, f, None)

            if val_g is None and val_p is None:
                continue  # True Negative (Ignore)

            field_total[f] += 1  # Поле участвует в оценке

            if val_g is not None and val_p is None:
                discrepancies.append(f)  # Missing value (Pure FN candidate)
            elif val_g is None and val_p is not None:
                discrepancies.append(f)  # Hallucinated value (Pure FP candidate)
            else:
                # Both present, check strict equality
                if self._is_match(val_p, val_g):
                    strict_matches.append(f)
                else:
                    discrepancies.append(f)  # Mismatch (FP + FN candidate)

        # Count strict matches as TP
        tp += len(strict_matches)
        for f in strict_matches:
            field_correct[f] += 1

        # Handle discrepancies
        if discrepancies and self.enable_semantic_judge:
            # Convert to JSON for judge (ensure primitive types only)
            gt_json = {f: getattr(gold, f, None) for f in self.fields}
            pred_json = {f: getattr(pred, f, None) for f in self.fields}

            # Call semantic judge
            verdicts = self._call_semantic_judge(
                task_name=task_name or "unknown",
                gt_json=gt_json,
                pred_json=pred_json,
                discrepancies=discrepancies,
            )

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

                # Обновляем per-field score с учётом вердикта судьи
                if verdict == "YES":
                    field_correct[field_name] += 1
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

        return tp, fp, fn, field_correct, field_total

    def _apply_semantic_verdict(
        self,
        field_name: str,
        pred: Any,
        gold: Any,
        verdict: str,
    ) -> tuple:
        """Apply semantic judge verdict to scoring."""
        tp, fp, fn = 0, 0, 0
        if verdict == "YES":
            tp += 1  # Amnesty granted
        else:
            # Вердикт NO - возвращаемся к исходной природе ошибки
            val_p = getattr(pred, field_name, None)
            val_g = getattr(gold, field_name, None)

            if val_p is None and val_g is not None:
                fn += 1  # Pure Miss (модель промолчала)
            elif val_p is not None and val_g is None:
                fp += 1  # Pure Hallucination (модель придумала)
            else:
                # Mismatch (wrong value: модель ошиблась значением)
                fp += 1
                fn += 1
        return tp, fp, fn

    def _apply_strict_penalty(
        self,
        val_p: Any,
        val_g: Any,
    ) -> tuple:
        """Apply strict penalty for discrepancies without semantic judge."""
        tp, fp, fn = 0, 0, 0
        if val_p is None and val_g is not None:
            fn += 1  # Pure Miss
        elif val_p is not None and val_g is None:
            fp += 1  # Pure Hallucination
        else:
            # Mismatch
            fp += 1
            fn += 1
        return tp, fp, fn

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
            Dict with precision, recall, f1 scores and field_scores.
        """
        tp, fp, fn = 0, 0, 0

        # Словари для хранения статистики по каждому полю
        field_correct = {f: 0 for f in self.fields}
        field_total = {f: 0 for f in self.fields}

        for pred, gold in pairs:
            # Case 3: False Negative (Missing Experiment)
            if pred is None and gold is not None:
                fn_inc, field_correct, field_total = self._process_false_negative(
                    gold, field_correct, field_total
                )
                fn += fn_inc
                continue

            # Case 2: False Positive (Hallucinated Experiment)
            if gold is None and pred is not None:
                fp_inc, field_correct, field_total = self._process_false_positive(
                    pred, field_correct, field_total
                )
                fp += fp_inc
                continue

            # Case 1: Aligned Experiment - Check field-wise
            tp_inc, fp_inc, fn_inc, field_correct, field_total = self._process_aligned_pair(
                pred, gold, field_correct, field_total, task_name
            )
            tp += tp_inc
            fp += fp_inc
            fn += fn_inc

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Расчет per-field scores
        field_scores = {
            f: (field_correct[f] / field_total[f]) if field_total[f] > 0 else 1.0
            for f in self.fields
        }

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "field_scores": field_scores,
        }

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
        """Get detailed evaluation report.

        Args:
            preds: List of predicted experiment entities.
            gts: List of ground truth experiment entities.
            task_name: Optional task name for semantic judge context.

        Returns:
            Dict with detailed evaluation metrics.
        """
        pairs = self.align_pairs(preds, gts)
        stats = self._compute_stats(pairs, task_name)

        return {
            "f1": stats["f1"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "fields": stats["field_scores"],
            "counts": {"preds": len(preds), "gts": len(gts)}
        }
