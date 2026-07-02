"""Dataset builder services for creating training/evaluation datasets."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
import pandas as pd

from ae.core.exceptions import DataValidationError, UseCaseExecutionError
from ae.core.storage import (
    DataSplitRepository,
    DocumentRepository,
    GroundTruthRepository,
)

logger = logging.getLogger(__name__)


def get_global_snapshot(df: pd.DataFrame, top_k: int = 10, tail_n: int = 5) -> Dict[str, Any]:
    """
    Generate a representative snapshot of the ground truth dataset to provide
    a 'Baseline Reality' context to the LLM and prevent false generalizations.
    
    Args:
        df: Ground Truth DataFrame.
        top_k: Number of most frequent categorical values to include.
        tail_n: Number of random rare categorical values to include.
        
    Returns:
        A dictionary profiling each column.
    """
    snapshot = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
            
        if pd.api.types.is_numeric_dtype(series):
            snapshot[col] = {
                "type": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median())
            }
        else:
            # Treat as categorical/string
            series_str = series.astype(str)
            counts = series_str.value_counts()
            top = counts.head(top_k).index.tolist()
            
            remaining = list(set(series_str.unique()) - set(top))
            # Sort remaining for reproducibility before sampling
            remaining.sort()
            tail = random.sample(remaining, min(tail_n, len(remaining))) if remaining else []
            
            snapshot[col] = {
                "type": "categorical",
                "values": top + tail
            }
            
    return snapshot


class DatasetBuilder:
    """Service for building DSPy datasets from documents and ground truth."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        gt_repo: Optional[GroundTruthRepository] = None,
        split_repo: Optional[DataSplitRepository] = None,
    ):
        """Initialize the dataset builder."""
        self.document_repo = document_repo
        self.gt_repo = gt_repo or GroundTruthRepository()
        self.split_repo = split_repo or DataSplitRepository()
        logger.debug("Initialized DatasetBuilder")

    def build_from_split(
        self,
        task: Any,
        gt_path: Path,
        split_path: Path,
        split_name: str,
        gt_data: Dict[str, Any],
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> List[dspy.Example]:
        """Build dataset from a data split."""
        try:
            allowed_ids = list(self.split_repo.load_split(
                split_path, split_name, normalize_keys=True
            ))

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
        task: Any,
        document_ids: List[str],
        gt_data: Dict[str, List[Any]],
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> List[dspy.Example]:
        """Build dataset from specific document IDs."""
        self._validate_inputs(task, document_ids, gt_data, limit, seed)

        try:
            candidates = [doc_id for doc_id in document_ids if doc_id in gt_data]

            if not candidates:
                logger.warning(
                    f"No documents with ground truth found. "
                    f"Requested: {len(document_ids)}, GT available: {len(gt_data)}"
                )
                return []

            if limit is not None and len(candidates) > limit:
                rng = random.Random(seed)
                rng.shuffle(candidates)
                candidates = candidates[:limit]

            logger.info(
                f"Building dataset: {len(candidates)} documents "
                f"(limit={limit}, total_requested={len(document_ids)})"
            )

            dataset = self._build_examples(task, candidates, gt_data)

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
        task: Any,
        document_ids: List[str],
        gt_data: Dict[str, List[Any]],
    ) -> List[dspy.Example]:
        """Build DSPy examples from documents and ground truth."""
        dataset: List[dspy.Example] = []
        stats = {"success": 0, "missing": 0, "empty": 0, "errors": 0}

        output_model = task.output_model if hasattr(task, "output_model") else task["output_model"]

        for doc_id in document_ids:
            try:
                doc_text = self.document_repo.get(doc_id)
                if doc_text is None:
                    stats["missing"] += 1
                    logger.debug(f"Document not found: {doc_id}")
                    continue

                if not doc_text or not doc_text.strip():
                    stats["empty"] += 1
                    logger.debug(f"Skipping empty document: {doc_id}")
                    continue

                example = dspy.Example(
                    document_text=doc_text,
                    extracted_data=output_model(experiments=gt_data[doc_id])
                ).with_inputs("document_text")

                dataset.append(example)
                stats["success"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"Failed to create example for {doc_id}: {e}")
                continue

        if stats["missing"] > 0 or stats["empty"] > 0 or stats["errors"] > 0:
            logger.warning(
                f"Dataset building stats: success={stats['success']}, "
                f"missing={stats['missing']}, empty={stats['empty']}, "
                f"errors={stats['errors']}"
            )

        return dataset

    def _validate_inputs(
        self,
        task: Any,
        document_ids: List[str],
        gt_data: Dict[str, List[Any]],
        limit: Optional[int],
        seed: int,
    ) -> None:
        """Validate inputs for dataset building."""
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
        """Get statistics about a dataset."""
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
