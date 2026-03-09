"""Evaluation engine for comparing extracted chemical experiments against ground truth."""

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

ExperimentEntity: TypeAlias = Union[BaseModel, Any]


class ExperimentMatcher:
    """Evaluation engine for comparing extracted chemical experiments against ground truth.

    - Strings: Normalized Exact Match (removes spaces, standardizes dashes).
    - Floats: Tolerance Interval (default ±5%).
    """

    # Pre-compiled regex for performance
    _RE_STRICT_CLEAN = re.compile(r"\s+")

    # Dash normalization mapping
    _DASH_MAP = str.maketrans({"−": "-", "–": "-", "—": "-"})

    def __init__(
        self,
        fields_to_compare: List[str],
        float_tolerance: float,
        teacher_llm: Optional[Any] = None,
        field_descriptions: Optional[Dict[str, str]] = None,
        enable_semantic_judge: bool = True,
    ):
        """Initialize the ExperimentMatcher.

        Args:
            fields_to_compare: List of field names to compare between entities.
            float_tolerance: Tolerance for float comparisons (0.0 to 1.0).
                            Kept for backward compatibility, not used in strict mode.
            teacher_llm: DSPy LLM object for semantic judgment (optional).
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
        self.teacher_llm = teacher_llm
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
        return self._RE_STRICT_CLEAN.sub("", str(value).translate(self._DASH_MAP))

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

    def _build_judge_prompt(
        self,
        task_name: str,
        gt_json: Dict[str, Any],
        pred_json: Dict[str, Any],
        discrepancies: List[str],
    ) -> str:
        """Build prompt for semantic judge.

        Args:
            task_name: Name of the task (e.g., "nanozymes").
            gt_json: Ground truth experiment as dictionary (primitive types only).
            pred_json: Predicted experiment as dictionary (primitive types only).
            discrepancies: List of field names with mismatches.

        Returns:
            Formatted prompt string.
        """
        # Build schema context from field descriptions
        schema_lines = []
        for field_name, description in self.field_descriptions.items():
            schema_lines.append(f"- {field_name}: {description}")
        schema_context = "\n".join(schema_lines)

        # Build discrepancies list with values
        discrepancy_lines = []
        for field_name in discrepancies:
            gt_val = gt_json.get(field_name)
            pred_val = pred_json.get(field_name)
            gt_str = "null" if gt_val is None else str(gt_val)
            pred_str = "null" if pred_val is None else str(pred_val)
            discrepancy_lines.append(f"- {field_name}: GT='{gt_str}', Pred='{pred_str}'")
        discrepancies_text = "\n".join(discrepancy_lines)

        prompt = f"""You are an expert scientist evaluating an automated data extraction system.
Task: {task_name}

Schema Definition (Field Meanings):
{schema_context}

--- CONTEXT (Full Experiments) ---
Ground Truth (Reference):
{json.dumps(gt_json, indent=2, default=str)}

Predicted (Extraction):
{json.dumps(pred_json, indent=2, default=str)}

--- DISCREPANCIES TO EVALUATE ---
The following fields did not match strictly. Evaluate ONLY these fields based on the context above:
{discrepancies_text}

--- INSTRUCTIONS ---
For EACH discrepancy, determine if the Predicted value is SEMANTICALLY EQUIVALENT to Ground Truth.
Acceptable differences:
1. Implicit Values: Null in Predicted is logically implied by other fields in Context (e.g., width=null for cubic).
2. Physical Equivalence: Different units but same value (e.g., 50 uM == 0.05 mM).
3. Synonyms/Order: "H2O2 + TMB" == "TMB + H2O2" (unless roles differ).
4. Case Variations: "Cubic" == "cubic" for crystal systems.

Response Format: JSON ONLY where keys are field names and values are "YES" (Accept) or "NO" (Reject).
Example: {{"width": "YES", "reaction_type": "NO"}}

Respond with JSON only:"""

        return prompt

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

        if self.teacher_llm is None:
            logger.warning("[SemanticJudge] teacher_llm not provided, skipping evaluation")
            return {}

        try:
            # Build prompt
            prompt = self._build_judge_prompt(task_name, gt_json, pred_json, discrepancies)

            # Call LLM via DSPy interface
            # DSPy LM returns list of strings, take first
            response = self.teacher_llm(prompt)
            response_text = response[0] if isinstance(response, list) else response

            # Extract JSON from response (handle markdown wrappers and extra text)
            json_match = re.search(r"\{.*\}", str(response_text), re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            # Parse JSON response
            verdicts = json.loads(response_text)

            # Validate and filter verdicts
            valid_verdicts = {}
            for field_name in discrepancies:
                if field_name in verdicts:
                    # Only accept "YES", everything else is "NO"
                    valid_verdicts[field_name] = (
                        "YES" if str(verdicts[field_name]).strip().upper() == "YES" else "NO"
                    )
                else:
                    # Field not in response, treat as NO
                    valid_verdicts[field_name] = "NO"

            logger.info(
                f"[SemanticJudge] Checked {len(discrepancies)} fields -> "
                f"{{{' + '.join(f'{k}: {v}' for k, v in valid_verdicts.items())}}}"
            )

            return valid_verdicts

        except json.JSONDecodeError as e:
            logger.warning(f"[SemanticJudge] JSON parse error: {e}. Raw response: {response_text[:200]}...")
            return {}
        except Exception as e:
            logger.warning(f"[SemanticJudge] Failed: {e}")
            return {}

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
                for f in self.fields:
                    if getattr(gold, f, None) is not None:
                        fn += 1
                        field_total[f] += 1
                continue

            # Case 2: False Positive (Hallucinated Experiment)
            if gold is None and pred is not None:
                for f in self.fields:
                    if getattr(pred, f, None) is not None:
                        fp += 1
                        field_total[f] += 1
                continue

            # Case 1: Aligned Experiment - Check field-wise
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

                # Apply verdicts with correct penalty logic
                for field_name in discrepancies:
                    verdict = verdicts.get(field_name, "NO")

                    if verdict == "YES":
                        tp += 1  # Amnesty granted
                        field_correct[field_name] += 1
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
            else:
                # No discrepancies or judge disabled - apply strict penalties
                for field_name in discrepancies:
                    val_p = getattr(pred, field_name, None)
                    val_g = getattr(gold, field_name, None)

                    if val_p is None and val_g is not None:
                        fn += 1  # Pure Miss
                    elif val_p is not None and val_g is None:
                        fp += 1  # Pure Hallucination
                    else:
                        # Mismatch
                        fp += 1
                        fn += 1

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
