# src/aee/eval/matcher.py

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
from Levenshtein import ratio as levenshtein_ratio
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

ExperimentEntity: TypeAlias = Union[BaseModel, Any]


class ExperimentMatcher:
    """
    Evaluation engine for comparing extracted chemical experiments against ground truth.

    Strategies:
    1. AEE Strict (Optimization Target):
       - Strings: Normalized Exact Match (removes spaces, standardizes dashes).
       - Floats: Tolerance Interval (±5%).
       - Goal: Precision.

    2. nanoMINER Legacy (Benchmarking Comparison):
       - Strings: Fuzzy Match (removes spaces, dashes, @) & Levenshtein Distance.
       - Floats: Exact Match.
       - Goal: Backward compatibility with SOTA paper results.
    """

    # Mapping from nanoMINER code for backward compatibility
    SYNGONY_MAP = {
        0: "amorphous",
        1: "triclinic",
        2: "monoclinic",
        3: "orthorhombic",
        4: "tetragonal",
        5: "trigonal",
        6: "hexagonal",
        7: "cubic",
    }

    # Pre-compiled regex for performance
    _RE_STRICT_CLEAN = re.compile(r"\s+")
    _RE_LEGACY_CLEAN = re.compile(r"[\s\-@]+")

    def __init__(self, fields_to_compare: List[str], float_tolerance: float = 0.05):
        """
        Args:
            fields_to_compare: List of field names to check in the Pydantic models.
            float_tolerance: Acceptable relative error for float comparison (default 5%).
        """
        self.fields = fields_to_compare
        self.tolerance = float_tolerance

    def _normalize_text(self, value: Any, strict_mode: bool = True) -> str:
        """
        Normalizes input values to strings for comparison.
        Handles OCR dash artifacts and whitespace.
        """
        if value is None:
            return ""

        # Handle legacy enum mapping
        if isinstance(value, (int, float)) and value in self.SYNGONY_MAP:
            value = self.SYNGONY_MAP[int(value)]

        text = str(value).lower()

        # Critical: Normalize dashes (OCR often confuses hyphen, en-dash, minus sign)
        # Replaces '−' (U+2212), '–' (U+2013), '—' (U+2014) with standard '-'
        text = text.replace("−", "-").replace("–", "-").replace("—", "-")

        if strict_mode:
            # Remove all whitespace
            return self._RE_STRICT_CLEAN.sub("", text)
        else:
            # Aggressive cleanup for legacy mode
            return self._RE_LEGACY_CLEAN.sub("", text)

    def _compare_floats(self, val_pred: float, val_gold: float, strict: bool) -> bool:
        """Compares two float values with tolerance (strict) or exact match (legacy)."""
        if strict:
            if val_gold == 0:
                return abs(val_pred - val_gold) < 1e-6
            return abs(val_pred - val_gold) / abs(val_gold) <= self.tolerance
        else:
            return val_pred == val_gold

    def _is_match_strict(self, pred: Any, gold: Any) -> bool:
        """
        AEE Strict Match:
        - Floats: Within tolerance.
        - Strings: Normalized exact match (dashes normalized).
        """
        if gold is None:
            return pred is None
        if pred is None:
            return False

        # Numerical comparison
        if isinstance(gold, (float, int)):
            try:
                return self._compare_floats(float(pred), float(gold), strict=True)
            except (ValueError, TypeError):
                return False

        # String comparison
        return self._normalize_text(pred, True) == self._normalize_text(gold, True)

    def _is_match_legacy(self, pred: Any, gold: Any) -> bool:
        """
        nanoMINER Legacy Match:
        - Floats: Exact equality.
        - Strings: Aggressive cleanup.
        """
        if gold is None:
            return pred is None
        if pred is None:
            return False

        if isinstance(gold, (float, int)):
            try:
                return self._compare_floats(float(pred), float(gold), strict=False)
            except (ValueError, TypeError):
                return False

        return self._normalize_text(pred, False) == self._normalize_text(gold, False)

    def _get_levenshtein_dist(self, pred: Any, gold: Any) -> float:
        """Calculates normalized distance (0.0 = Perfect match, 1.0 = Mismatch)."""
        if gold is None or pred is None:
            return 1.0

        # For numerics, fallback to strict binary check (0 or 1)
        if isinstance(gold, (float, int)):
            return 0.0 if self._is_match_strict(pred, gold) else 1.0

        s1 = self._normalize_text(pred, False)
        s2 = self._normalize_text(gold, False)

        if not s1 and not s2:
            return 0.0

        return 1.0 - levenshtein_ratio(s1, s2)

    def align_pairs(
        self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]
    ) -> List[Tuple[Optional[ExperimentEntity], Optional[ExperimentEntity]]]:
        """
        Aligns prediction objects to ground truth objects to maximize total similarity.
        Uses AEE Strict rules for alignment cost calculation.

        Returns:
            List of (Prediction, GroundTruth) tuples. Values can be None (FP/FN).
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
                    if self._is_match_strict(val_p, val_g):
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

    def _compute_aee_stats(
        self, pairs: List[Tuple[Any, Any]]
    ) -> Dict[str, float]:
        """Calculates Micro-F1/Precision/Recall based on Strict rules."""
        tp, fp, fn = 0, 0, 0

        for pred, gold in pairs:
            for field in self.fields:
                val_p = getattr(pred, field, None) if pred else None
                val_g = getattr(gold, field, None) if gold else None

                if val_g is not None:
                    if self._is_match_strict(val_p, val_g):
                        tp += 1
                    else:
                        # Value exists in GT but Pred is wrong/missing
                        fp += 1 
                elif val_p is not None:
                    # Value does not exist in GT but Pred found something (Hallucination)
                    fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
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
                    if self._is_match_strict(val_p, val_g):
                        tp += 1
                    else:
                        fp += 1 # Wrong value predicted
                        fn += 1 # Correct value missed

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_legacy_stats(
        self, pairs: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """Calculates Avg Levenshtein Distance and Column-wise stats (Legacy Mode)."""
        lev_scores = []
        col_stats = {f: {"correct": 0, "total_pred": 0, "total_gt": 0} for f in self.fields}

        for pred, gold in pairs:
            for f in self.fields:
                val_p = getattr(pred, f, None) if pred else None
                val_g = getattr(gold, f, None) if gold else None

                # Levenshtein Score Accumulation
                if val_p is not None and val_g is not None:
                    lev_scores.append(self._get_levenshtein_dist(val_p, val_g))
                elif val_g is not None:
                    lev_scores.append(1.0) # Maximum penalty for missing

                # Column Metrics
                is_match = self._is_match_legacy(val_p, val_g)
                if is_match and val_g is not None:
                    col_stats[f]["correct"] += 1
                
                if val_p is not None:
                    col_stats[f]["total_pred"] += 1
                if val_g is not None:
                    col_stats[f]["total_gt"] += 1

        # Aggregate Column Stats
        final_col_stats = {}
        for f, counts in col_stats.items():
            p = counts["correct"] / counts["total_pred"] if counts["total_pred"] else 0.0
            r = counts["correct"] / counts["total_gt"] if counts["total_gt"] else 0.0
            final_col_stats[f] = {"precision": p, "recall": r}

        avg_lev = sum(lev_scores) / len(lev_scores) if lev_scores else 0.0

        return {"avg_levenshtein": avg_lev, "column_metrics": final_col_stats}

    def get_optimization_score(
        self, preds: List[ExperimentEntity], gts: List[ExperimentEntity]
    ) -> float:
        """
        Fast path for DSPy Optimizer. Returns AEE Strict F1 only.
        """
        pairs = self.align_pairs(preds, gts)
        return self._compute_aee_stats(pairs)["f1"]

    def get_full_report(
        self,
        all_preds: List[List[ExperimentEntity]],
        all_gts: List[List[ExperimentEntity]],
    ) -> Dict[str, Any]:
        """
        Detailed report for benchmarking. Returns both AEE and Legacy metrics.
        Aggregates pairs from multiple documents.
        """
        all_pairs = []
        for p_list, g_list in zip(all_preds, all_gts):
            all_pairs.extend(self.align_pairs(p_list, g_list))

        return {
            "aee_strict": self._compute_aee_stats(all_pairs),
            "nanominer_legacy": self._compute_legacy_stats(all_pairs),
        }