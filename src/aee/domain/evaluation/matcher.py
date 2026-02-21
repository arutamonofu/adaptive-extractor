"""Evaluation engine for comparing extracted chemical experiments against ground truth."""

import logging
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

    def __init__(self, fields_to_compare: List[str], float_tolerance: float):
        """Initialize the ExperimentMatcher.

        Args:
            fields_to_compare: List of field names to compare between entities.
            float_tolerance: Tolerance for float comparisons (0.0 to 1.0).

        Raises:
            ValueError: If fields_to_compare is empty or float_tolerance is invalid.
        """
        if not fields_to_compare:
            raise ValueError("fields_to_compare cannot be empty")
        if not 0 <= float_tolerance <= 1:
            raise ValueError("float_tolerance must be between 0 and 1")

        self.fields = fields_to_compare
        self.tolerance = float_tolerance

    def _normalize_text(self, value: Any) -> str:
        """Normalize input values to strings for comparison.

        Handles OCR dash artifacts and whitespace.

        Args:
            value: Input value to normalize.

        Returns:
            str: Normalized string value.
        """
        if value is None:
            return ""

        # Convert to string, lowercase, normalize dashes, and remove whitespace
        return self._RE_STRICT_CLEAN.sub("", str(value).lower().translate(self._DASH_MAP))

    def _compare_floats(self, val_pred: float, val_gold: float) -> bool:
        """Compare two float values with tolerance.

        Args:
            val_pred: Predicted float value.
            val_gold: Ground truth float value.

        Returns:
            bool: True if values are within tolerance, False otherwise.
        """
        # Handle exact zero case
        if val_gold == 0:
            return abs(val_pred - val_gold) < 1e-9

        # Handle relative tolerance
        return abs(val_pred - val_gold) / abs(val_gold) <= self.tolerance

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

    def _compute_stats(self, pairs: List[Tuple[Optional[Any], Optional[Any]]]) -> Dict[str, float]:
        """Calculate Micro-F1/Precision/Recall.

        Args:
            pairs: List of aligned pairs (pred, gt).

        Returns:
            Dict with precision, recall, and f1 scores.
        """
        tp, fp, fn = 0, 0, 0

        for pred, gold in pairs:
            # Case 3: False Negative (Missing Experiment)
            if pred is None and gold is not None:
                # Count all non-None fields in Gold as Missed
                fn += sum(1 for f in self.fields if getattr(gold, f, None) is not None)
                continue

            # Case 2: False Positive (Hallucinated Experiment)
            if gold is None and pred is not None:
                # Count all non-None fields in Pred as False Positives
                fp += sum(1 for f in self.fields if getattr(pred, f, None) is not None)
                continue

            # Case 1: Aligned Experiment (Check field-wise)
            for f in self.fields:
                val_p = getattr(pred, f, None)
                val_g = getattr(gold, f, None)

                if val_g is None and val_p is None:
                    continue  # True Negative (Ignore)

                if val_g is not None and val_p is None:
                    fn += 1  # Missing value
                elif val_g is None and val_p is not None:
                    fp += 1  # Hallucinated value
                else:
                    # Both present, check equality
                    if self._is_match(val_p, val_g):
                        tp += 1
                    else:
                        fp += 1  # Wrong value predicted
                        fn += 1  # Correct value missed

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def get_optimization_score(
        self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]
    ) -> float:
        """Get optimization score (F1) for use in teleprompter.

        Args:
            preds: List of predicted experiment entities.
            gts: List of ground truth experiment entities.

        Returns:
            float: F1 score.
        """
        pairs = self.align_pairs(preds, gts)
        return self._compute_stats(pairs)["f1"]

    def get_detailed_report(self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]) -> Dict[str, Any]:
        """Get detailed evaluation report.

        Args:
            preds: List of predicted experiment entities.
            gts: List of ground truth experiment entities.

        Returns:
            Dict with detailed evaluation metrics.
        """
        pairs = self.align_pairs(preds, gts)
        stats = self._compute_stats(pairs)

        # Calculate per-field scores
        field_scores = {}
        for field in self.fields:
            correct = 0
            total = 0
            for p, g in pairs:
                val_p = getattr(p, field, None) if p else None
                val_g = getattr(g, field, None) if g else None

                if val_g is not None:
                    total += 1
                    if self._is_match(val_p, val_g):
                        correct += 1
                elif val_p is not None:
                    total += 1  # False Positive field

            field_scores[field] = correct / total if total > 0 else 1.0

        return {
            "f1": stats["f1"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "fields": field_scores,
            "counts": {"preds": len(preds), "gts": len(gts)}
        }
