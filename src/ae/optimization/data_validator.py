"""Data validation services for checking dataset consistency."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ae.core.storage import DataSplitRepository, GroundTruthRepository
from ae.core.utils import normalize_document_key

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.stats.update(other.stats)
        if not other.success:
            self.success = False
        return self


class DataValidator:
    """Service for validating data consistency."""

    def __init__(
        self,
        gt_repo: Optional[GroundTruthRepository] = None,
        split_repo: Optional[DataSplitRepository] = None,
    ):
        """Initialize the data validator."""
        self.gt_repo = gt_repo or GroundTruthRepository()
        self.split_repo = split_repo or DataSplitRepository()
        logger.debug("Initialized DataValidator")

    def _normalize_split_id(self, doc_id: str) -> str:
        """Normalize a document ID from splits to match ground truth keys."""
        return normalize_document_key(doc_id)

    def _check_splits_file_exists(
        self,
        split_path: Path,
        result: ValidationResult,
    ) -> bool:
        """Check if splits file exists."""
        if not split_path.exists():
            result.add_error(
                f"Data splits file not found: {split_path}\n"
                f"Please create {split_path.name} with train/val/test splits.\n"
                f"See docs/data_artifacts.md for details."
            )
            return False
        return True

    def _check_required_splits(
        self,
        splits: Dict[str, Set[str]],
        required_splits: List[str],
        split_path: Path,
        result: ValidationResult,
    ) -> bool:
        """Check if required splits are present."""
        for split_name in required_splits:
            if split_name not in splits:
                result.add_error(
                    f"Required split '{split_name}' not found in {split_path}"
                )
        return result.success

    def _validate_single_split(
        self,
        split_name: str,
        split_ids: Set[str],
        gt_doc_ids: Set[str],
        main_split_ids: Set[str],
        all_split_ids: Set[str],
        is_main_split: bool,
        result: ValidationResult,
    ) -> None:
        """Validate a single split."""
        normalized_split_ids = [self._normalize_split_id(doc_id) for doc_id in split_ids]
        split_ids_set = set(normalized_split_ids)

        if len(split_ids) != len(split_ids_set):
            result.add_warning(f"Split '{split_name}' contains duplicate IDs")

        if is_main_split:
            overlap = main_split_ids & split_ids_set
            if overlap:
                result.add_error(
                    f"Split '{split_name}' overlaps with previous splits: {overlap}"
                )
            main_split_ids.update(split_ids_set)

        all_split_ids.update(split_ids_set)

        missing_in_gt = split_ids_set - gt_doc_ids
        if missing_in_gt:
            result.add_error(
                f"Split '{split_name}' contains {len(missing_in_gt)} document(s) "
                f"not found in ground truth: {sorted(missing_in_gt)[:5]}..."
            )

        result.stats[f"{split_name}_size"] = len(split_ids)
        result.stats[f"{split_name}_with_gt"] = len(split_ids_set & gt_doc_ids)

    def _validate_split_sizes(
        self,
        result: ValidationResult,
    ) -> None:
        """Validate train and validation split sizes."""
        train_size = result.stats.get("train_size", 0)
        val_size = result.stats.get("val_size", 0)

        if train_size == 0:
            result.add_error("Training split is empty")
        if val_size == 0:
            result.add_error("Validation split is empty")
        elif val_size < 3:
            result.add_warning(
                f"Validation split is very small ({val_size} examples). "
                f"Recommend at least 3-5 examples for reliable evaluation."
            )

    def validate_splits(
        self,
        gt_path: Path,
        split_path: Path,
        task: Any,
        gt_data: Dict[str, Any],
        required_splits: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate data splits against ground truth."""
        result = ValidationResult(success=True)
        required_splits = required_splits or ["train", "val"]

        if not self._check_splits_file_exists(split_path, result):
            return result

        try:
            gt_doc_ids = set(gt_data.keys())
            result.stats["ground_truth_docs"] = len(gt_doc_ids)

            splits = self.split_repo.load_all_splits(split_path)
            result.stats["total_splits"] = len(splits)

            if not self._check_required_splits(splits, required_splits, split_path, result):
                return result

            main_splits = ["train", "val", "test"]
            all_split_ids: Set[str] = set()
            main_split_ids: Set[str] = set()

            for split_name, split_ids in splits.items():
                is_main_split = split_name in main_splits
                self._validate_single_split(
                    split_name=split_name,
                    split_ids=split_ids,
                    gt_doc_ids=gt_doc_ids,
                    main_split_ids=main_split_ids,
                    all_split_ids=all_split_ids,
                    is_main_split=is_main_split,
                    result=result,
                )

            unused_docs = gt_doc_ids - all_split_ids
            if unused_docs:
                result.add_warning(
                    f"{len(unused_docs)} document(s) in ground truth but not in any split: "
                    f"{sorted(unused_docs)[:5]}..."
                )

            self._validate_split_sizes(result)
            result.stats["total_docs_in_splits"] = len(all_split_ids)
            result.stats["unused_gt_docs"] = len(unused_docs)

        except Exception as e:
            result.add_error(f"Validation failed: {e}")
            logger.error(f"Data validation error: {e}", exc_info=True)

        return result

    def validate_ground_truth(
        self,
        gt_path: Path,
        task: Any,
        gt_data: Dict[str, Any],
        min_examples: int = 1,
    ) -> ValidationResult:
        """Validate ground truth data quality."""
        result = ValidationResult(success=True)

        try:
            total_docs = len(gt_data)
            result.stats["ground_truth_docs"] = total_docs

            if total_docs < min_examples:
                result.add_error(
                    f"Ground truth has {total_docs} examples, minimum required: {min_examples}"
                )

            empty_docs = 0
            total_experiments = 0
            for doc_id, experiments in gt_data.items():
                if not experiments:
                    empty_docs += 1
                    result.add_warning(f"Document '{doc_id}' has no experiments")
                total_experiments += len(experiments)

            if empty_docs > 0:
                result.stats["empty_documents"] = empty_docs

            result.stats["total_experiments"] = total_experiments
            result.stats["avg_experiments_per_doc"] = (
                total_experiments / total_docs if total_docs > 0 else 0
            )

        except Exception as e:
            result.add_error(f"Ground truth validation failed: {e}")
            logger.error(f"Ground truth validation error: {e}", exc_info=True)

        return result

    def validate_dataset(
        self,
        dataset: List[Any],
        split_name: str,
        min_examples: int = 1,
    ) -> ValidationResult:
        """Validate a DSPy dataset."""
        result = ValidationResult(success=True)

        dataset_size = len(dataset)
        result.stats[f"{split_name}_size"] = dataset_size

        if dataset_size < min_examples:
            result.add_error(
                f"Dataset '{split_name}' has {dataset_size} examples, "
                f"minimum required: {min_examples}"
            )

        empty_docs = 0
        for i, example in enumerate(dataset):
            if not hasattr(example, 'document_text') or not example.document_text:
                empty_docs += 1
                if empty_docs <= 3:
                    result.add_warning(f"Example {i} in '{split_name}' has empty document_text")

        if empty_docs > 0:
            result.stats[f"{split_name}_empty_docs"] = empty_docs

        missing_data = 0
        for i, example in enumerate(dataset):
            if not hasattr(example, 'extracted_data') or example.extracted_data is None:
                missing_data += 1
                if missing_data <= 3:
                    result.add_warning(
                        f"Example {i} in '{split_name}' has missing extracted_data"
                    )

        if missing_data > 0:
            result.stats[f"{split_name}_missing_data"] = missing_data

        return result

    def log_validation_result(
        self,
        result: ValidationResult,
        context: str = "Validation",
    ) -> None:
        """Log validation results."""
        logger.info(f"{'=' * 60}")
        logger.info(f"{context} - {'✓ PASSED' if result.success else '✗ FAILED'}")
        logger.info(f"{'=' * 60}")

        if result.stats:
            logger.info("Statistics:")
            for key, value in result.stats.items():
                logger.info(f"  {key}: {value}")

        if result.warnings:
            logger.warning(f"Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                logger.warning(f"  ⚠ {warning}")

        if result.errors:
            logger.error(f"Errors ({len(result.errors)}):")
            for error in result.errors:
                logger.error(f"  ✗ {error}")

        logger.info(f"{'=' * 60}")
