# src/ae/evaluation/metrics.py
"""Task-specific evaluation metrics for Adaptive Extractor."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ae.core.evaluation.matcher import ExperimentEntity, ExperimentMatcher

if TYPE_CHECKING:
    import dspy

logger = logging.getLogger(__name__)


class TaskMetric:
    """Task-specific evaluation metric for Adaptive Extractor.

    Calculates F1 score and detailed metrics for extracted chemical experiments
    by comparing predictions against ground truth data.
    """

    def __init__(
        self,
        task_config: Dict[str, Any],
        float_tolerance: float,
        student_llm: Optional[Any] = None,
        field_descriptions: Optional[Dict[str, str]] = None,
        enable_semantic_judge: bool = True,
    ) -> None:
        """Initialize the task metric.

        Args:
            task_config: Configuration dictionary for the task.
                        Must contain 'compare_fields' key with list of field names.
            float_tolerance: Float tolerance for comparisons (0.0 to 1.0).
            student_llm: DSPy LLM object for semantic judgment.
            field_descriptions: Dictionary of field descriptions (optional).
            enable_semantic_judge: Flag to enable/disable semantic judge.
        """

        self.matcher = ExperimentMatcher(
            fields_to_compare=task_config["compare_fields"],
            float_tolerance=float_tolerance,
            student_llm=student_llm,
            field_descriptions=field_descriptions or {},
            enable_semantic_judge=enable_semantic_judge,
        )
        self.fields_to_compare = task_config["compare_fields"]
        self.task_name = task_config.get("name", "unknown")

    def _extract_experiments(self, obj: Union["dspy.Example", "dspy.Prediction"]) -> List[ExperimentEntity]:
        """Extract experiments from a DSPy object.

        Args:
            obj: DSPy Example or Prediction object.

        Returns:
            List of experiment entities.
        """
        extracted_data = getattr(obj, "extracted_data", None)
        if extracted_data is None:
            return []

        experiments = []
        if isinstance(extracted_data, str):
            import json
            parsed = None
            
            # 1. Try to parse directly as JSON first
            try:
                parsed = json.loads(extracted_data)
            except Exception:
                pass

            # 2. Extract JSON block using _extract_first_json (within extracted_data marker if present)
            if parsed is None:
                try:
                    from ae.core.evaluation.matcher import _extract_first_json
                    cleaned_data = extracted_data
                    if "[[ ## extracted_data ## ]]" in extracted_data:
                        cleaned_data = extracted_data.split("[[ ## extracted_data ## ]]")[-1].split("[[")[0]
                    
                    json_str = _extract_first_json(cleaned_data) or cleaned_data
                    parsed = json.loads(json_str)
                except Exception:
                    pass

            # 3. Fallback: Parse single-quoted Python dict representation using ast.literal_eval
            if parsed is None:
                try:
                    import ast
                    import re
                    cleaned_data = extracted_data
                    if "[[ ## extracted_data ## ]]" in extracted_data:
                        cleaned_data = extracted_data.split("[[ ## extracted_data ## ]]")[-1].split("[[")[0]
                    
                    # Normalize JS/JSON bare keywords to Python equivalents safely
                    cleaned_data_norm = re.sub(r'\bnull\b', 'None', cleaned_data)
                    cleaned_data_norm = re.sub(r'\btrue\b', 'True', cleaned_data_norm)
                    cleaned_data_norm = re.sub(r'\bfalse\b', 'False', cleaned_data_norm)
                    
                    parsed = ast.literal_eval(cleaned_data_norm.strip())
                except Exception as e:
                    logger.warning(f"Error parsing predicted experiments JSON/literal: {e}")
                    return []

            if isinstance(parsed, dict):
                experiments = parsed.get("experiments", []) or parsed.get("extracted_data", [])
            elif isinstance(parsed, list):
                experiments = parsed
            else:
                experiments = []
        elif isinstance(extracted_data, dict):
            experiments = extracted_data.get("experiments", [])
        elif isinstance(extracted_data, list):
            experiments = extracted_data
        elif hasattr(extracted_data, "experiments"):
            experiments = extracted_data.experiments
        else:
            try:
                if hasattr(extracted_data, "model_dump"):
                    experiments = extracted_data.model_dump().get("experiments", [])
                elif hasattr(extracted_data, "dict"):
                    experiments = extracted_data.dict().get("experiments", [])
            except Exception:
                pass

        from types import SimpleNamespace
        return [SimpleNamespace(**exp) if isinstance(exp, dict) else exp for exp in experiments]

    def _log_metrics(self, report: Dict[str, Any]) -> None:
        """Log evaluation metrics as formatted tables."""
        from tabulate import tabulate  # type: ignore[import-untyped]

        summary = [
            ["F1", f"{report['f1']:.3f}"],
            ["Precision", f"{report['precision']:.3f}"],
            ["Recall", f"{report['recall']:.3f}"],
            ["Count", f"P:{report['counts']['preds']} / G:{report['counts']['gts']}"],
        ]

        fields = [[f, f"{s:.2f}"] for f, s in sorted(report["fields"].items())]

        logger.info("\n" + tabulate(summary, headers=["Metric", "Value"], tablefmt="fancy_grid"))
        logger.info("\n" + tabulate(fields, headers=["Field", "Score"], tablefmt="fancy_grid"))

    def __call__(self, example: "dspy.Example", prediction: "dspy.Prediction", trace: Any = None) -> float:
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
            report = self.matcher.get_detailed_report(
                predicted_experiments,
                ground_truth_experiments,
                task_name=self.task_name,
            )
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
