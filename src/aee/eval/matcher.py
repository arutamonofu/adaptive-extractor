# src/aee/eval/matcher.py

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
# Удален импорт Levenshtein
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

ExperimentEntity: TypeAlias = Union[BaseModel, Any]


class ExperimentMatcher:
    """
    Evaluation engine for comparing extracted chemical experiments against ground truth.
    - Strings: Normalized Exact Match (removes spaces, standardizes dashes).
    - Floats: Tolerance Interval (default ±5%).
    """

    # Pre-compiled regex for performance
    _RE_STRICT_CLEAN = re.compile(r"\s+")

    def __init__(self, fields_to_compare: List[str], float_tolerance: float = 0.05):
        self.fields = fields_to_compare
        self.tolerance = float_tolerance

    def _normalize_text(self, value: Any) -> str:
        """
        Normalizes input values to strings for comparison.
        Handles OCR dash artifacts and whitespace.
        """
        if value is None:
            return ""

        text = str(value).lower()

        # Normalize dashes (OCR often confuses hyphen, en-dash, minus sign)
        text = text.replace("−", "-").replace("–", "-").replace("—", "-")

        # Remove all whitespace
        return self._RE_STRICT_CLEAN.sub("", text)

    def _compare_floats(self, val_pred: float, val_gold: float) -> bool:
        """Compares two float values with tolerance."""
        if val_gold == 0:
            return abs(val_pred - val_gold) < 1e-6
        return abs(val_pred - val_gold) / abs(val_gold) <= self.tolerance

    def _is_match(self, pred: Any, gold: Any) -> bool:
        """
        Checks if two values match according to strict rules.
        """
        if gold is None:
            return pred is None
        if pred is None:
            return False

        # Numerical comparison
        if isinstance(gold, (float, int)):
            try:
                return self._compare_floats(float(pred), float(gold))
            except (ValueError, TypeError):
                return False

        # String comparison
        return self._normalize_text(pred) == self._normalize_text(gold)

    def align_pairs(
        self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]
    ) -> List[Tuple[Optional[ExperimentEntity], Optional[ExperimentEntity]]]:
        """
        Aligns prediction objects to ground truth objects to maximize total similarity
        using the Hungarian Algorithm.
        """
        if not preds and not gts:
            return []

        if not preds:
            return [(None, gt) for gt in gts]
        if not gts:
            return [(pred, None) for pred in preds]

        cost_matrix = np.zeros((len(preds), len(gts)))

        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                matches = 0
                for field in self.fields:
                    val_p = getattr(p, field, None)
                    val_g = getattr(g, field, None)
                    if self._is_match(val_p, val_g):
                        matches += 1
                
                # Normalize score to [0, 1] range
                score = matches / len(self.fields) if self.fields else 0
                cost_matrix[i, j] = -score  # Minimize negative score

        # Solve assignment problem
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        pairs = []
        # Add matched pairs
        for r, c in zip(row_inds, col_inds):
            pairs.append((preds[r], gts[c]))

        # Add unmatched Predictions (False Positives)
        for i in range(len(preds)):
            if i not in row_inds:
                pairs.append((preds[i], None))

        # Add unmatched GTs (False Negatives)
        for j in range(len(gts)):
            if j not in col_inds:
                pairs.append((None, gts[j]))

        return pairs

    def _compute_stats(self, pairs: List[Tuple[Any, Any]]) -> Dict[str, float]:
        """Calculates Micro-F1/Precision/Recall."""
        tp, fp, fn = 0, 0, 0
        
        for pred, gold in pairs:
            # Case 3: False Negative (Missing Experiment)
            if pred is None and gold is not None:
                # Count all non-None fields in Gold as Missed
                for f in self.fields:
                    if getattr(gold, f, None) is not None:
                        fn += 1
                continue
            
            # Case 2: False Positive (Hallucinated Experiment)
            if gold is None and pred is not None:
                # Count all non-None fields in Pred as False Positives
                for f in self.fields:
                    if getattr(pred, f, None) is not None:
                        fp += 1
                continue

            # Case 1: Aligned Experiment (Check field-wise)
            for f in self.fields:
                val_p = getattr(pred, f, None)
                val_g = getattr(gold, f, None)

                if val_g is None and val_p is None:
                    continue # True Negative (Ignore)

                if val_g is not None and val_p is None:
                    fn += 1 # Missing value
                elif val_g is None and val_p is not None:
                    fp += 1 # Hallucinated value
                else:
                    # Both present, check equality
                    if self._is_match(val_p, val_g):
                        tp += 1
                    else:
                        fp += 1 # Wrong value predicted
                        fn += 1 # Correct value missed

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def get_optimization_score(
        self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]
    ) -> float:
        pairs = self.align_pairs(preds, gts)
        return self._compute_stats(pairs)["f1"]