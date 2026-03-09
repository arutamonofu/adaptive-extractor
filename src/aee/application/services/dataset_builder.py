"""Dataset builder service for creating training/evaluation datasets.

This service handles the creation of DSPy Example datasets from processed
documents and ground truth data, with support for filtering and sampling.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy

from aee.infrastructure.storage import DocumentRepository, GroundTruthRepository, DataSplitRepository
from aee.shared.exceptions import DataValidationError, UseCaseExecutionError

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Service for building DSPy datasets from documents and ground truth.

    This service orchestrates the creation of training and evaluation datasets
    by combining processed documents with ground truth annotations.

    Example:
        ```python
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir),
            gt_repo=GroundTruthRepository(),
            split_repo=DataSplitRepository()
        )

        # Build training dataset
        train_dataset = builder.build_from_split(
            task=nanozyme_task,
            gt_path=Path("data/ground_truth/gt.csv"),
            split_path=Path("data/splits/nanozymes.json"),
            split_name="train",
            limit=50
        )

        # Build dataset from specific IDs
        dataset = builder.build_from_ids(
            task=nanozyme_task,
            document_ids=["doc1", "doc2"],
            gt_data=ground_truth
        )
        ```
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        gt_repo: Optional[GroundTruthRepository] = None,
        split_repo: Optional[DataSplitRepository] = None,
    ):
        """Initialize the dataset builder.

        Args:
            document_repo: Repository for loading documents.
            gt_repo: Optional repository for loading ground truth.
            split_repo: Optional repository for loading splits.
        """
        self.document_repo = document_repo
        self.gt_repo = gt_repo or GroundTruthRepository()
        self.split_repo = split_repo or DataSplitRepository()
        logger.debug("Initialized DatasetBuilder")

    def build_from_split(
        self,
        task: Dict[str, Any],
        gt_path: Path,
        split_path: Path,
        split_name: str,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> List[dspy.Example]:
        """Build dataset from a data split.

        Args:
            task: Task dict with config, row_converter, etc.
            gt_path: Path to ground truth CSV.
            split_path: Path to splits JSON file.
            split_name: Name of split to load (e.g., "train", "test").
            limit: Optional maximum number of examples.
            seed: Random seed for sampling when limiting.

        Returns:
            List of DSPy Examples.

        Raises:
            UseCaseExecutionError: If dataset building fails.
        """
        try:
            # Load ground truth
            gt_data = self.gt_repo.load(gt_path, task["row_converter"])

            # Load split
            allowed_ids = list(self.split_repo.load_split(
                split_path, split_name, normalize_keys=True
            ))

            # Build dataset
            return self.build_from_ids(
                task=task,
                document_ids=allowed_ids,
                gt_data=gt_data,
                limit=limit,
                seed=seed,
            )

        except Exception as e:
            raise UseCaseExecutionError(
                "DatasetBuilder.build_from_split",
                f"Failed to build dataset from split '{split_name}': {e}"
            ) from e

    def build_from_ids(
        self,
        task: Dict[str, Any],
        document_ids: List[str],
        gt_data: Dict[str, List[Any]],
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> List[dspy.Example]:
        """Build dataset from specific document IDs.

        Args:
            task: Task dict with config, output_model, etc.
            document_ids: List of document IDs to include.
            gt_data: Ground truth data mapping doc IDs to experiments.
            limit: Optional maximum number of examples.
            seed: Random seed for sampling when limiting.

        Returns:
            List of DSPy Examples.

        Raises:
            DataValidationError: If inputs are invalid.
            UseCaseExecutionError: If dataset building fails.
        """
        # Validate inputs
        self._validate_inputs(task, document_ids, gt_data, limit, seed)

        try:
            # Filter to documents that have ground truth
            candidates = [doc_id for doc_id in document_ids if doc_id in gt_data]

            if not candidates:
                logger.warning(
                    f"No documents with ground truth found. "
                    f"Requested: {len(document_ids)}, GT available: {len(gt_data)}"
                )
                return []

            # Apply limit with sampling
            if limit is not None and len(candidates) > limit:
                rng = random.Random(seed)
                rng.shuffle(candidates)
                candidates = candidates[:limit]

            logger.info(
                f"Building dataset: {len(candidates)} documents "
                f"(limit={limit}, total_requested={len(document_ids)})"
            )

            # Build examples
            dataset = self._build_examples(task, candidates, gt_data)

            # Validate built dataset
            if not dataset:
                raise UseCaseExecutionError(
                    "DatasetBuilder.build_from_ids",
                    "Built dataset is empty. Check that documents exist and have content."
                )

            logger.info(
                f"Successfully built dataset: {len(dataset)} examples "
                f"from {len(candidates)} candidates"
            )

            return dataset

        except Exception as e:
            raise UseCaseExecutionError(
                "DatasetBuilder.build_from_ids",
                f"Failed to build dataset: {e}"
            ) from e

    def _build_examples(
        self,
        task: Dict[str, Any],
        document_ids: List[str],
        gt_data: Dict[str, List[Any]],
    ) -> List[dspy.Example]:
        """Build DSPy examples from documents and ground truth.

        Args:
            task: Task dict with config, output_model, etc.
            document_ids: List of document IDs.
            gt_data: Ground truth data.

        Returns:
            List of DSPy Examples.
        """
        dataset: List[dspy.Example] = []
        stats = {"success": 0, "missing": 0, "empty": 0, "errors": 0}

        # Load all documents (batch)
        all_docs = self.document_repo.load_all()

        for doc_id in document_ids:
            try:
                # Get document text
                doc_text = all_docs.get(doc_id)
                if doc_text is None:
                    stats["missing"] += 1
                    logger.debug(f"Document not found: {doc_id}")
                    continue

                # Validate document has content
                if not doc_text or not doc_text.strip():
                    stats["empty"] += 1
                    logger.debug(f"Skipping empty document: {doc_id}")
                    continue

                # Create DSPy example
                example = dspy.Example(
                    document_text=doc_text,
                    extracted_data=task["output_model"](experiments=gt_data[doc_id])
                ).with_inputs("document_text")

                dataset.append(example)
                stats["success"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"Failed to create example for {doc_id}: {e}")
                continue

        # Log statistics
        if stats["missing"] > 0 or stats["empty"] > 0 or stats["errors"] > 0:
            logger.warning(
                f"Dataset building stats: success={stats['success']}, "
                f"missing={stats['missing']}, empty={stats['empty']}, "
                f"errors={stats['errors']}"
            )

        return dataset

    def _validate_inputs(
        self,
        task: Dict[str, Any],
        document_ids: List[str],
        gt_data: Dict[str, List[Any]],
        limit: Optional[int],
        seed: int,
    ) -> None:
        """Validate inputs for dataset building.

        Args:
            task: Task dict with config, output_model, etc.
            document_ids: List of document IDs.
            gt_data: Ground truth data.
            limit: Optional limit.
            seed: Random seed.

        Raises:
            DataValidationError: If validation fails.
        """
        errors = []

        if not isinstance(document_ids, list):
            errors.append("document_ids must be a list")

        if not document_ids:
            errors.append("document_ids cannot be empty")

        if not isinstance(gt_data, dict):
            errors.append("gt_data must be a dictionary")

        if not gt_data:
            errors.append("gt_data cannot be empty")

        if limit is not None and (not isinstance(limit, int) or limit < 1):
            errors.append("limit must be a positive integer or None")

        if not isinstance(seed, int):
            errors.append("seed must be an integer")

        if errors:
            raise DataValidationError("Dataset builder inputs", errors)

    def get_dataset_statistics(
        self, dataset: List[dspy.Example]
    ) -> Dict[str, Any]:
        """Get statistics about a dataset.

        Args:
            dataset: List of DSPy Examples.

        Returns:
            Dictionary with dataset statistics.
        """
        if not dataset:
            return {
                "total_examples": 0,
                "avg_text_length": 0,
                "avg_experiments_per_example": 0,
            }

        total_text_length = sum(len(ex.document_text) for ex in dataset)
        total_experiments = sum(
            len(ex.extracted_data.experiments) for ex in dataset
        )

        return {
            "total_examples": len(dataset),
            "avg_text_length": total_text_length / len(dataset),
            "avg_experiments_per_example": total_experiments / len(dataset),
            "total_experiments": total_experiments,
        }
