# src/aee/eval/dspy.py

import logging
import dspy
from typing import Any, Dict
from aee.eval.matcher import ExperimentMatcher

logger = logging.getLogger(__name__)

class TaskMetric:
    """
    Metric wrapper for DSPy optimization (MIPROv2).
    
    Strategy:
    - Uses AEE Strict F1 score to force the agent to learn precise extraction.
    - penalizes malformed outputs (0.0) instead of crashing.
    - Rewards correct 'empty' predictions (True Negatives).
    """

    def __init__(self, task_config: Dict[str, Any]):
        self.matcher = ExperimentMatcher(fields_to_compare=task_config["compare_fields"])

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
        """
        Computes score for a single example. Returns 0.0 on failure.
        """
        try:
            gts = getattr(example.extracted_data, "experiments", [])

            extracted = getattr(prediction, "extracted_data", None)
            preds = getattr(extracted, "experiments", []) if extracted else []

            # Case 1: True Negative (Nothing to extract, nothing extracted) -> Perfect
            if not gts and not preds:
                return 1.0

            # Case 2: False Negative (Data exists, but nothing extracted)
            if not preds:
                return 0.0

            # Case 3: Calculate Strict F1 via Hungarian Algorithm
            return self.matcher.get_optimization_score(preds, gts)

        except Exception:
            # Fail gracefully on Pydantic/Logic errors to keep optimizer running
            return 0.0