# src/ae/evaluation/metrics.py
"""Task-specific evaluation metrics for Adaptive Extractor."""

import ast
import json
import logging
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from tabulate import tabulate  # type: ignore[import-untyped]

from ae.core.evaluation.matcher import RecordEntity, RecordMatcher, _extract_first_json

if TYPE_CHECKING:
    import dspy

logger = logging.getLogger(__name__)


class TaskMetric:
    """Task-specific evaluation metric for Adaptive Extractor.

    Calculates F1 score and detailed metrics for extracted structured records
    by comparing predictions against ground truth data.
    """

    def __init__(
        self,
        task_config: Dict[str, Any],
        float_tolerance: float,
        student_llm: Optional[Any] = None,
        field_descriptions: Optional[Dict[str, str]] = None,
        enable_semantic_judge: bool = True,
        field_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        field_validators: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """Initialize the task metric.

        Args:
            task_config: Configuration dictionary for the task.
                        Must contain 'compare_fields' key with list of field names.
            float_tolerance: Float tolerance for comparisons (0.0 to 1.0).
            student_llm: DSPy LLM object for semantic judgment.
            field_descriptions: Dictionary of field descriptions (optional).
            enable_semantic_judge: Flag to enable/disable semantic judge.
            field_constraints: Optional schema constraints per field (choices, pattern, min/max).
            field_validators: Optional declarative validators per field (used for canonical equivalence).
        """

        self.matcher = RecordMatcher(
            fields_to_compare=task_config["compare_fields"],
            float_tolerance=float_tolerance,
            student_llm=student_llm,
            field_descriptions=field_descriptions or {},
            enable_semantic_judge=enable_semantic_judge,
            field_constraints=field_constraints or task_config.get("field_constraints"),
            field_validators=field_validators or task_config.get("field_validators"),
        )
        self.fields_to_compare = task_config["compare_fields"]
        self.task_name = task_config.get("name", "unknown")

    def _extract_records(self, obj: Union["dspy.Example", "dspy.Prediction"]) -> List[RecordEntity]:
        """Extract records from a DSPy object.

        Args:
            obj: DSPy Example or Prediction object.

        Returns:
            List of record entities.
        """
        extracted_data = getattr(obj, "extracted_data", None)
        if extracted_data is None:
            return []

        records = []
        if isinstance(extracted_data, str):
            parsed = None
            
            # 1. Try to parse directly as JSON first
            try:
                parsed = json.loads(extracted_data)
            except Exception:
                pass

            # 2. Extract JSON block using _extract_first_json (within extracted_data marker if present)
            if parsed is None:
                try:
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
                    cleaned_data = extracted_data
                    if "[[ ## extracted_data ## ]]" in extracted_data:
                        cleaned_data = extracted_data.split("[[ ## extracted_data ## ]]")[-1].split("[[")[0]
                    
                    # Normalize JS/JSON bare keywords to Python equivalents safely
                    cleaned_data_norm = re.sub(r'\bnull\b', 'None', cleaned_data)
                    cleaned_data_norm = re.sub(r'\btrue\b', 'True', cleaned_data_norm)
                    cleaned_data_norm = re.sub(r'\bfalse\b', 'False', cleaned_data_norm)
                    
                    parsed = ast.literal_eval(cleaned_data_norm.strip())
                except Exception as e:
                    logger.warning(f"Error parsing predicted records JSON/literal: {e}")
                    return []

            if isinstance(parsed, dict):
                records = parsed.get("records", []) or parsed.get("experiments", []) or parsed.get("extracted_data", [])
            elif isinstance(parsed, list):
                records = parsed
            else:
                records = []
        elif isinstance(extracted_data, dict):
            records = extracted_data.get("records") or extracted_data.get("experiments", [])
        elif isinstance(extracted_data, list):
            records = extracted_data
        elif hasattr(extracted_data, "records"):
            records = extracted_data.records
        elif hasattr(extracted_data, "experiments"):
            records = extracted_data.experiments
        else:
            try:
                if hasattr(extracted_data, "model_dump"):
                    dump = extracted_data.model_dump()
                    records = dump.get("records") or dump.get("experiments", [])
                elif hasattr(extracted_data, "dict"):
                    dump = extracted_data.dict()
                    records = dump.get("records") or dump.get("experiments", [])
            except Exception:
                pass

        return [SimpleNamespace(**rec) if isinstance(rec, dict) else rec for rec in records]

    # Backward-compatible alias
    _extract_experiments = _extract_records

    def _log_metrics(self, report: Dict[str, Any]) -> None:
        """Log evaluation metrics as formatted tables."""

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
            example: Ground truth example containing extracted_data.records.
            prediction: Predicted result containing extracted_data.records.
            trace: Optional trace information (unused).

        Returns:
            float: F1 score metric (0.0 to 1.0).
        """
        try:
            # Extract records from ground truth and prediction
            ground_truth_records = self._extract_records(example)
            predicted_records = self._extract_records(prediction)

            # Calculate detailed metrics using RecordMatcher
            report = self.matcher.get_detailed_report(
                predicted_records,
                ground_truth_records,
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
