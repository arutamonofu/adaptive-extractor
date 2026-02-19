"""Functional API for ground truth data management.

This module provides simple functions for loading ground truth data,
replacing the class-based GroundTruthRepository for simpler use cases.

Example:
    ```python
    from aee.infrastructure.storage.ground_truth import load_ground_truth

    gt_data = load_ground_truth(
        csv_path=Path("data/ground_truth.csv"),
        row_converter=row_to_nanozyme,
    )
    # gt_data: Dict[str, List[NanozymeExperiment]]
    ```
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd

from aee.shared.exceptions import DataNotFoundError, DataValidationError, InvalidDataFormatError

logger = logging.getLogger(__name__)

# Valid ID column names (in priority order)
ID_COLUMNS = ["pdf", "filename", "source", "doi", "document"]


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV file into pandas DataFrame.

    Args:
        csv_path: Path to CSV file.

    Returns:
        pandas DataFrame.

    Raises:
        DataNotFoundError: If file not found.
        InvalidDataFormatError: If CSV cannot be parsed.
    """
    if not csv_path.exists():
        raise DataNotFoundError("Ground truth CSV", str(csv_path))

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise InvalidDataFormatError(
            "Ground truth CSV", f"Failed to parse CSV: {e}"
        ) from e


def _identify_id_column(df: pd.DataFrame, csv_path: Path) -> str:
    """Identify the ID column in the DataFrame.

    Args:
        df: pandas DataFrame.
        csv_path: Path to CSV file (for error messages).

    Returns:
        Name of the ID column.

    Raises:
        DataValidationError: If no valid ID column found.
    """
    for col in ID_COLUMNS:
        if col in df.columns:
            return col

    # Fallback: use first column
    if len(df.columns) > 0:
        logger.warning(
            f"No standard ID column found in {csv_path}. "
            f"Using first column: {df.columns[0]}"
        )
        return df.columns[0]

    raise DataValidationError(
        "Ground truth CSV",
        f"No ID column found. Expected one of: {ID_COLUMNS}"
    )


def _normalize_document_key(doc_id: str) -> str:
    """Normalize document ID by removing extensions.

    Args:
        doc_id: Document identifier.

    Returns:
        Normalized document key.
    """
    doc_id = str(doc_id).strip().lower()

    # Remove common extensions
    for ext in [".pdf", ".txt"]:
        if doc_id.endswith(ext):
            doc_id = doc_id[:-len(ext)]
            break

    return doc_id


def _group_and_convert(
    df: pd.DataFrame,
    id_column: str,
    row_converter: Callable[[pd.Series], Optional[Any]],
    csv_path: Path,
) -> Dict[str, List[Any]]:
    """Group DataFrame by ID and convert to experiments.

    Args:
        df: pandas DataFrame.
        id_column: Name of ID column.
        row_converter: Function to convert rows to experiments.
        csv_path: Path to CSV file (for error messages).

    Returns:
        Dictionary mapping document keys to lists of experiments.

    Raises:
        InvalidDataFormatError: If conversion fails.
    """
    gt_data: Dict[str, List[Any]] = {}

    for doc_id, group in df.groupby(id_column):
        doc_key = _normalize_document_key(doc_id)
        experiments = []

        for _, row in group.iterrows():
            try:
                exp = row_converter(row)
                if exp is not None:
                    experiments.append(exp)
            except Exception as e:
                logger.warning(
                    f"Failed to convert row in {csv_path}: {e}. Skipping."
                )

        if experiments:
            gt_data[doc_key] = experiments

    return gt_data


def load_ground_truth(
    csv_path: Path,
    row_converter: Callable[[pd.Series], Optional[Any]],
) -> Dict[str, List[Any]]:
    """Load ground truth from CSV file.

    Args:
        csv_path: Path to the ground truth CSV file.
        row_converter: Function to convert rows to experiment objects.

    Returns:
        Dictionary mapping document keys to lists of experiments.

    Raises:
        DataNotFoundError: If CSV file not found.
        InvalidDataFormatError: If CSV format is invalid.
        DataValidationError: If data validation fails.
    """
    df = _load_csv(csv_path)

    if df.empty:
        raise InvalidDataFormatError("Ground truth CSV", "CSV file is empty")

    id_column = _identify_id_column(df, csv_path)
    gt_data = _group_and_convert(df, id_column, row_converter, csv_path)

    logger.info(
        f"Loaded ground truth: {len(gt_data)} documents, "
        f"{sum(len(exps) for exps in gt_data.values())} experiments"
    )

    return gt_data


def validate_coverage(
    gt_data: Dict[str, List[Any]],
    available_docs: Set[str],
) -> Dict[str, Any]:
    """Validate ground truth coverage against available documents.

    Args:
        gt_data: Ground truth data.
        available_docs: Set of available document keys.

    Returns:
        Dictionary with coverage information.
    """
    gt_docs = set(gt_data.keys())
    available_normalized = {_normalize_document_key(d) for d in available_docs}

    covered = gt_docs.intersection(available_normalized)
    missing = available_normalized - gt_docs

    coverage_pct = (len(covered) / len(available_normalized) * 100) if available_normalized else 0.0

    return {
        "covered_documents": len(covered),
        "total_documents": len(available_normalized),
        "coverage_percentage": round(coverage_pct, 2),
        "missing_documents": list(missing),
    }


# Keep GroundTruthRepository for backward compatibility
class GroundTruthRepository:
    """Repository for managing ground truth experiment data.

    This repository handles loading ground truth from CSV files, grouping
    experiments by document, and converting rows to experiment objects.

    Note: This class is maintained for backward compatibility.
    New code should use the functional API (load_ground_truth, etc.).
    """

    def __init__(self):
        """Initialize the ground truth repository."""
        logger.debug("Initialized GroundTruthRepository")

    def load(
        self,
        csv_path: Path,
        row_converter: Callable[[pd.Series], Optional[Any]],
    ) -> Dict[str, List[Any]]:
        """Load ground truth from CSV file.

        Args:
            csv_path: Path to the ground truth CSV file.
            row_converter: Function to convert rows to experiment objects.

        Returns:
            Dictionary mapping document keys to lists of experiments.
        """
        return load_ground_truth(csv_path, row_converter)

    def validate_coverage(
        self,
        gt_data: Dict[str, List[Any]],
        available_docs: Set[str],
    ) -> Dict[str, Any]:
        """Validate ground truth coverage.

        Args:
            gt_data: Ground truth data.
            available_docs: Set of available document keys.

        Returns:
            Dictionary with coverage information.
        """
        return validate_coverage(gt_data, available_docs)
