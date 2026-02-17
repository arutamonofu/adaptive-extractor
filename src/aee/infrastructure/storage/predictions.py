"""Predictions repository for loading and managing prediction results.

This module provides a clean interface for loading prediction results
from JSON files, with improved error handling and validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from aee.shared.exceptions import DataValidationError, InvalidDataFormatError, RepositoryError

logger = logging.getLogger(__name__)


class PredictionRepository:
    """Repository for managing prediction results.

    This repository handles loading predictions from JSON files, validating
    them against schemas, and organizing them by document.

    Example:
        ```python
        from aee.domain.tasks.nanozymes import NanozymeExperiment

        repo = PredictionRepository()
        predictions = repo.load(
            results_dir=Path("data/predictions"),
            experiment_model=NanozymeExperiment
        )
        # predictions: Dict[str, List[NanozymeExperiment]]
        ```
    """

    def __init__(self):
        """Initialize the predictions repository."""
        logger.debug("Initialized PredictionRepository")

    def load(
        self,
        results_dir: Path,
        experiment_model: Type[BaseModel],
        strict: bool = False,
    ) -> Dict[str, List[Any]]:
        """Load predictions from JSON files in a directory.

        Args:
            results_dir: Directory containing prediction JSON files.
            experiment_model: Pydantic model for validating experiments.
            strict: If True, raise error on invalid predictions. If False, skip them.

        Returns:
            Dictionary mapping document keys to lists of experiments.

        Raises:
            RepositoryError: If directory doesn't exist or other IO errors.
            DataValidationError: If strict=True and validation fails.
        """
        # Validate directory exists
        if not results_dir.exists():
            if strict:
                raise RepositoryError(
                    "PredictionRepository",
                    "load",
                    f"Results directory does not exist: {results_dir}"
                )
            else:
                logger.warning(f"Predictions directory does not exist: {results_dir}")
                return {}

        if not results_dir.is_dir():
            raise RepositoryError(
                "PredictionRepository",
                "load",
                f"Path is not a directory: {results_dir}"
            )

        # Validate experiment model
        if not issubclass(experiment_model, BaseModel):
            raise TypeError("experiment_model must be a Pydantic BaseModel subclass")

        # Load all prediction files
        predictions: Dict[str, List[Any]] = {}
        stats = {"total": 0, "success": 0, "errors": 0, "experiments": 0}

        for file_path in sorted(results_dir.glob("*.json")):
            stats["total"] += 1
            try:
                doc_predictions = self._load_prediction_file(
                    file_path, experiment_model, strict
                )
                if doc_predictions:
                    doc_key = self._extract_document_key(file_path, doc_predictions)
                    predictions[doc_key] = doc_predictions
                    stats["success"] += 1
                    stats["experiments"] += len(doc_predictions)
                    logger.debug(
                        f"Loaded {len(doc_predictions)} predictions from {file_path.name}"
                    )
            except Exception as e:
                stats["errors"] += 1
                if strict:
                    raise
                else:
                    logger.warning(f"Failed to load predictions from {file_path.name}: {e}")
                    # Add empty entry to track which documents failed
                    predictions[file_path.stem.lower()] = []

        # Log summary
        logger.info(
            f"Loaded predictions: {stats['success']}/{stats['total']} files, "
            f"{stats['experiments']} total experiments, {stats['errors']} errors"
        )

        return predictions

    def _load_prediction_file(
        self,
        file_path: Path,
        experiment_model: Type[BaseModel],
        strict: bool,
    ) -> List[Any]:
        """Load and validate a single prediction file.

        Args:
            file_path: Path to prediction JSON file.
            experiment_model: Pydantic model for validation.
            strict: Whether to raise errors or skip invalid experiments.

        Returns:
            List of validated experiments.

        Raises:
            InvalidDataFormatError: If JSON format is invalid.
            DataValidationError: If strict=True and validation fails.
        """
        # Load JSON
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidDataFormatError(
                str(file_path), f"Invalid JSON: {e}"
            ) from e
        except Exception as e:
            raise InvalidDataFormatError(
                str(file_path), f"Cannot read file: {e}"
            ) from e

        # Extract experiments from nested structure
        raw_experiments = self._extract_experiments(data, file_path)

        # Validate each experiment
        validated_experiments = []
        validation_errors = []

        for idx, exp_data in enumerate(raw_experiments):
            try:
                validated_exp = experiment_model(**exp_data)
                validated_experiments.append(validated_exp)
            except ValidationError as e:
                error_msg = f"Experiment {idx}: {e}"
                validation_errors.append(error_msg)
                logger.debug(f"Validation error in {file_path.name}: {error_msg}")
            except Exception as e:
                error_msg = f"Experiment {idx}: Unexpected error: {e}"
                validation_errors.append(error_msg)

        # Handle validation errors
        if validation_errors:
            if strict:
                raise DataValidationError(
                    f"Predictions in {file_path.name}",
                    validation_errors
                )
            else:
                logger.warning(
                    f"Skipped {len(validation_errors)}/{len(raw_experiments)} "
                    f"invalid experiments in {file_path.name}"
                )

        return validated_experiments

    def _extract_experiments(self, data: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Extract experiments from prediction data structure.

        Handles various prediction file formats.

        Args:
            data: Loaded JSON data.
            file_path: Path to file (for error messages).

        Returns:
            List of raw experiment dictionaries.
        """
        # Try multiple common structures
        experiments = None

        # Structure 1: {"extraction": {"experiments": [...]}}
        if "extraction" in data:
            extraction = data["extraction"]
            if isinstance(extraction, dict) and "experiments" in extraction:
                experiments = extraction["experiments"]

        # Structure 2: {"experiments": [...]}
        elif "experiments" in data:
            experiments = data["experiments"]

        # Structure 3: {"extracted_data": {"experiments": [...]}}
        elif "extracted_data" in data:
            extracted_data = data["extracted_data"]
            if isinstance(extracted_data, dict) and "experiments" in extracted_data:
                experiments = extracted_data["experiments"]

        # Structure 4: Direct list of experiments
        elif isinstance(data, list):
            experiments = data

        # No valid structure found
        if experiments is None:
            logger.warning(
                f"No experiments found in {file_path.name}. "
                f"Expected 'extraction.experiments', 'experiments', or 'extracted_data.experiments'"
            )
            return []

        # Validate it's a list
        if not isinstance(experiments, list):
            logger.warning(
                f"Experiments field in {file_path.name} is not a list: {type(experiments)}"
            )
            return []

        return experiments

    def _extract_document_key(self, file_path: Path, predictions: List[Any]) -> str:
        """Extract document key from file path or prediction metadata.

        Args:
            file_path: Path to prediction file.
            predictions: List of predictions (unused, for future metadata extraction).

        Returns:
            Normalized document key.
        """
        # Remove common suffixes from filename
        filename = file_path.stem
        for suffix in ["_result", "_prediction", "_predictions", "_pred"]:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break

        # Normalize: lowercase, no extension
        return filename.lower().strip()

    def save(
        self,
        predictions: List[BaseModel],
        output_path: Path,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save predictions to JSON file.

        Args:
            predictions: List of experiment predictions.
            output_path: Path to output JSON file.
            document_metadata: Optional metadata to include in output.

        Raises:
            RepositoryError: If save operation fails.
        """
        try:
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build output structure
            output_data = {
                "extraction": {
                    "experiments": [exp.model_dump() for exp in predictions]
                }
            }

            # Add metadata if provided
            if document_metadata:
                output_data["source_metadata"] = document_metadata

            # Write JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(predictions)} predictions to {output_path}")

        except Exception as e:
            raise RepositoryError(
                "PredictionRepository",
                "save",
                f"Failed to save predictions: {e}"
            ) from e

    def compare_counts(
        self,
        predictions: Dict[str, List[Any]],
        ground_truth: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Compare prediction and ground truth counts.

        Args:
            predictions: Predictions loaded from load().
            ground_truth: Ground truth loaded from GroundTruthRepository.

        Returns:
            Dictionary with comparison statistics.
        """
        pred_docs = set(predictions.keys())
        gt_docs = set(ground_truth.keys())

        matched_docs = pred_docs & gt_docs
        missing_preds = gt_docs - pred_docs
        extra_preds = pred_docs - gt_docs

        # Count experiments per document
        doc_comparisons = {}
        for doc in matched_docs:
            doc_comparisons[doc] = {
                "predicted": len(predictions[doc]),
                "ground_truth": len(ground_truth[doc]),
                "difference": len(predictions[doc]) - len(ground_truth[doc]),
            }

        stats = {
            "total_pred_documents": len(pred_docs),
            "total_gt_documents": len(gt_docs),
            "matched_documents": len(matched_docs),
            "missing_predictions": sorted(missing_preds),
            "extra_predictions": sorted(extra_preds),
            "document_comparisons": doc_comparisons,
            "total_predicted_experiments": sum(len(exps) for exps in predictions.values()),
            "total_gt_experiments": sum(len(exps) for exps in ground_truth.values()),
        }

        return stats
