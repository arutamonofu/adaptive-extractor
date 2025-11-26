# src/aee/eval/matcher.py

import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel

class ExperimentMatcher:
    """
    Handles comparison of extracted entities against ground truth using the Hungarian Algorithm.
    Supports fuzzy matching for strings and tolerance intervals for floats.
    """

    def __init__(self, fields_to_compare: List[str], field_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            fields_to_compare: List of Pydantic model field names to evaluate.
            field_weights: Optional dictionary of weights for weighted similarity calculation.
                           Defaults to 1.0 for all fields if not provided.
        """
        self.fields = fields_to_compare
        self.weights = field_weights or {}

    def _normalize(self, s: Any) -> str:
        """Removes whitespace and lowers case for robust string comparison."""
        if s is None:
            return ""
        # Remove all whitespace (handles "Fe 3 O 4" vs "Fe3O4")
        return re.sub(r'\s+', '', str(s).lower())

    def _compare_values(self, pred: Any, gold: Any) -> bool:
        """Determines if two values match based on their type."""
        if gold is None:
            return pred is None
        if pred is None:
            return False

        # Float comparison with 5% tolerance
        if isinstance(gold, (float, int)):
            try:
                diff = abs(float(pred) - float(gold))
                if float(gold) == 0:
                    return diff < 1e-6
                return (diff / abs(float(gold))) <= 0.05
            except (ValueError, TypeError):
                return False

        # String comparison
        return self._normalize(pred) == self._normalize(gold)

    def _calculate_similarity(self, exp1: BaseModel, exp2: BaseModel) -> float:
        """Calculates weighted similarity score (0.0 - 1.0) between two experiment objects."""
        matches = 0.0
        total_weight = 0.0

        for field in self.fields:
            if not hasattr(exp1, field) or not hasattr(exp2, field):
                continue

            val1 = getattr(exp1, field)
            val2 = getattr(exp2, field)

            # Skip fields that are empty in Ground Truth (don't penalize optional fields)
            if val2 is None:
                continue

            weight = self.weights.get(field, 1.0)
            total_weight += weight

            if self._compare_values(val1, val2):
                matches += weight

        return matches / total_weight if total_weight > 0 else 0.0

    def match_experiments(self, preds: List[BaseModel], gts: List[BaseModel]) -> Tuple[List[Tuple[BaseModel, BaseModel]], List[int], List[int]]:
        """
        Aligns predictions with ground truth using the Hungarian algorithm.

        Returns:
            Tuple containing:
            1. List of matched pairs (prediction, ground_truth)
            2. List of indices of unmatched predictions (False Positives)
            3. List of indices of unmatched ground truths (False Negatives)
        """
        if not gts:
            return [], list(range(len(preds))), []
        if not preds:
            return [], [], list(range(len(gts)))

        # Build Cost Matrix (Cost = negative similarity)
        cost_matrix = np.zeros((len(preds), len(gts)))
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                cost_matrix[i, j] = -self._calculate_similarity(p, g)

        # Solve assignment
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_p = set(range(len(preds)))
        unmatched_g = set(range(len(gts)))

        for row, col in zip(row_inds, col_inds):
            # Threshold: similarity must be > 0.3 to be considered a "match"
            if -cost_matrix[row, col] > 0.3:
                matched.append((preds[row], gts[col]))
                unmatched_p.discard(row)
                unmatched_g.discard(col)

        return matched, list(unmatched_p), list(unmatched_g)

    def calculate_f1(self, preds: List[BaseModel], gts: List[BaseModel]) -> float:
        """Calculates F1 score for a single document (used by DSPy optimizer)."""
        metrics = self._compute_metrics([preds], [gts])
        return metrics["f1"]

    def evaluate_dataset(self, all_preds: List[List[BaseModel]], all_gts: List[List[BaseModel]]) -> Dict[str, float]:
        """Calculates global Precision, Recall, and F1 across the entire dataset."""
        return self._compute_metrics(all_preds, all_gts)

    def _compute_metrics(self, all_preds: List[List[BaseModel]], all_gts: List[List[BaseModel]]) -> Dict[str, Any]:
        """Internal helper to compute field-level metrics."""
        tp, fp, fn = 0, 0, 0

        for preds, gts in zip(all_preds, all_gts):
            matched, unmatched_p_idx, unmatched_g_idx = self.match_experiments(preds, gts)

            # 1. Analyze matched pairs (Check individual fields)
            for p, g in matched:
                for field in self.fields:
                    val_p = getattr(p, field, None)
                    val_g = getattr(g, field, None)

                    if val_g is not None:
                        if self._compare_values(val_p, val_g):
                            tp += 1
                        else:
                            fp += 1 # Value mismatch
                    elif val_p is not None:
                        fp += 1 # Hallucinated field

            # 2. Analyze unmatched predictions (False Positives)
            for idx in unmatched_p_idx:
                p = preds[idx]
                for field in self.fields:
                    if getattr(p, field, None) is not None:
                        fp += 1

            # 3. Analyze unmatched ground truths (False Negatives)
            for idx in unmatched_g_idx:
                g = gts[idx]
                for field in self.fields:
                    if getattr(g, field, None) is not None:
                        fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn
        }