# src/aee/eval/dspy.py

import dspy
from typing import Any, Dict
from aee.eval.matcher import ExperimentMatcher

class TaskMetric:
    """
    DSPy-compatible metric callable.
    Wraps ExperimentMatcher to calculate F1 score on a per-example basis for optimization.
    """

    def __init__(self, task_config: Dict[str, Any]):
        """
        Args:
            task_config: Configuration dict containing 'compare_fields' key.
        """
        self.matcher = ExperimentMatcher(fields_to_compare=task_config["compare_fields"])

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
        """
        Calculates field-level F1 score for the given prediction.

        Args:
            example: Ground truth example (must contain extracted_data.experiments).
            prediction: Model prediction (must contain extracted_data.experiments).
            trace: DSPy trace object (unused).

        Returns:
            F1 score between 0.0 and 1.0.
        """
        # Safely retrieve lists, defaulting to empty if missing
        ground_truth = getattr(example.extracted_data, "experiments", [])
        preds = getattr(prediction.extracted_data, "experiments", [])

        if preds is None:
            return 0.0

        # Delegate calculation to the core matcher logic
        return self.matcher.calculate_f1(preds, ground_truth)