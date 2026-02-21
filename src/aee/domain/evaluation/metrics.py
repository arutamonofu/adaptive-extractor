# src/aee/evaluation/metrics.py
"""Task-specific evaluation metrics for AutoEvoExtractor."""

import logging
from typing import Any, Dict, List, Union

import dspy
from pydantic import BaseModel

from aee.domain.evaluation.matcher import ExperimentEntity, ExperimentMatcher

logger = logging.getLogger(__name__)


class TaskMetric:
    """Task-specific evaluation metric for AutoEvoExtractor.
    
    Calculates F1 score and detailed metrics for extracted chemical experiments
    by comparing predictions against ground truth data.
    """
    
    def __init__(self, task_config: Dict[str, Any], float_tolerance: float) -> None:
        """Initialize the task metric.

        Args:
            task_config: Configuration dictionary for the task.
                         Must contain 'compare_fields' key with list of field names.
            float_tolerance: Float tolerance for comparisons (0.0 to 1.0).
        """

        self.matcher = ExperimentMatcher(
            fields_to_compare=task_config["compare_fields"],
            float_tolerance=float_tolerance
        )
        self.fields_to_compare = task_config["compare_fields"]

    def _extract_experiments(self, obj: Union[dspy.Example, dspy.Prediction]) -> List[ExperimentEntity]:
        """Extract experiments from a DSPy object.
        
        Args:
            obj: DSPy Example or Prediction object.
            
        Returns:
            List of experiment entities.
        """
        extracted_data = getattr(obj, "extracted_data", None)
        if extracted_data is None:
            return []
        return getattr(extracted_data, "experiments", [])

    def _format_field_details(self, fields: Dict[str, float]) -> str:
        """Format field-level metrics for logging.
        
        Args:
            fields: Dictionary of field names and their scores.
            
        Returns:
            Formatted string of field details.
        """
        return " | ".join([f"{field}: {score:.2f}" for field, score in fields.items()])

    def _log_metrics(self, report: Dict[str, Any]) -> None:
        """Log detailed metrics information.

        Args:
            report: Detailed evaluation report from ExperimentMatcher.
        """
        field_details = self._format_field_details(report["fields"])
        logger.info(
            f"F1: {report['f1']:.3f} | "
            f"Precision: {report['precision']:.3f} | "
            f"Recall: {report['recall']:.3f} | "
            f"Count: (P:{report['counts']['preds']} / G:{report['counts']['gts']}) | "
            f"Fields: {field_details}"
        )

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
        """Calculate the metric score for a prediction.
        
        Args:
            example: Ground truth example containing extracted_data.experiments.
            prediction: Predicted result containing extracted_data.experiments.
            trace: Optional trace information (unused).
            
        Returns:
            float: F1 score metric (0.0 to 1.0).
        """
        try:
            # Extract experiments from ground truth and prediction
            ground_truth_experiments = self._extract_experiments(example)
            predicted_experiments = self._extract_experiments(prediction)
            
            # Calculate detailed metrics using ExperimentMatcher
            report = self.matcher.get_detailed_report(predicted_experiments, ground_truth_experiments)
            score = report["f1"]
            
            # Log detailed metrics if logger is enabled for INFO level
            if logger.isEnabledFor(logging.INFO):
                self._log_metrics(report)
                
            return score
            
        except (AttributeError, KeyError, TypeError) as e:
            logger.error(f"Error in metric calculation: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error in metric calculation: {e}")
            return 0.0