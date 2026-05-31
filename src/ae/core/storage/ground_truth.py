"""Ground truth repository and functional API for managing ground truth data.

This module provides functions and repository classes for loading ground truth experiment
data from CSV files, grouping them by document, and validating dataset coverage.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

import pandas as pd

from ae.core.exceptions import DataNotFoundError, DataValidationError, InvalidDataFormatError

logger = logging.getLogger(__name__)

# Valid ID column names (in priority order)
ID_COLUMNS = ["pdf", "filename", "source", "doi", "document"]


def _normalize_document_key(doc_id: str) -> str:
    """Normalize document ID by removing extensions and standardizing.

    Args:
        doc_id: Document identifier.

    Returns:
        Normalized document key (lowercase, stripped, no extension).
    """
    doc_id = str(doc_id).strip().lower()

    # Remove common extensions
    for ext in [".pdf", ".txt", ".doc"]:
        if doc_id.endswith(ext):
            doc_id = doc_id[:-len(ext)]
            break

    return doc_id


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
    except pd.errors.EmptyDataError:
        raise InvalidDataFormatError(str(csv_path), "CSV (file is empty)")
    except Exception as e:
        raise InvalidDataFormatError(
            str(csv_path), f"CSV (read error: {e})"
        ) from e

    if df.empty:
        logger.warning(f"Ground truth CSV is empty: {csv_path}")
        return df

    # Normalize column names for consistent access
    df.columns = df.columns.str.lower().str.strip()
    return df


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

    raise DataValidationError(
        "Ground truth CSV",
        [
            f"No valid document ID column found in {csv_path}",
            f"Expected one of: {', '.join(ID_COLUMNS)}",
            f"Found columns: {', '.join(df.columns.tolist())}",
        ]
    )


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
    """
    gt_data: Dict[str, List[Any]] = {}
    conversion_errors = 0
    total_rows = 0

    for doc_id, group in df.groupby(id_column):
        doc_key = _normalize_document_key(str(doc_id))
        experiments = []

        for idx, row in group.iterrows():
            total_rows += 1
            try:
                exp = row_converter(row)
                if exp is not None:
                    experiments.append(exp)
                else:
                    logger.debug(
                        f"Row converter returned None for row {idx} in {doc_id}"
                    )
                    conversion_errors += 1
            except Exception as e:
                logger.warning(
                    f"Failed to convert row {idx} for {doc_id}: {e}"
                )
                conversion_errors += 1

        if experiments:
            gt_data[doc_key] = experiments

    if conversion_errors > 0:
        logger.warning(
            f"Ground truth conversion: {conversion_errors}/{total_rows} rows failed"
        )

    if not gt_data:
        logger.warning(f"No valid experiments found in {csv_path}")

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
    try:
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

    except (DataNotFoundError, InvalidDataFormatError, DataValidationError):
        raise
    except Exception as e:
        raise DataValidationError(
            "Ground truth",
            [f"Unexpected error loading ground truth: {e}"]
        ) from e


def validate_coverage(
    gt_data: Dict[str, List[Any]],
    available_docs: Union[Iterable[str], Set[str], List[str]],
) -> Dict[str, Any]:
    """Validate ground truth coverage against available documents.

    Args:
        gt_data: Ground truth data.
        available_docs: Iterable of available document keys.

    Returns:
        Dictionary with coverage information.
    """
    available_set = {_normalize_document_key(d) for d in available_docs}
    gt_set = set(gt_data.keys())

    covered = gt_set.intersection(available_set)
    missing = available_set - gt_set
    extra_gt = gt_set - available_set

    total_experiments = sum(len(exps) for exps in gt_data.values())

    stats = {
        "total_gt_documents": len(gt_set),
        "total_available_documents": len(available_set),
        "total_documents": len(available_set),
        "covered_documents": len(covered),
        "coverage_percentage": (len(covered) / len(available_set) * 100) if available_set else 0.0,
        "missing_documents": sorted(missing),
        "extra_gt_documents": sorted(extra_gt),
        "total_experiments": total_experiments,
    }

    if missing:
        logger.warning(
            f"Ground truth missing for {len(missing)} documents: {list(missing)[:5]}..."
        )
    if extra_gt:
        logger.warning(
            f"Ground truth exists for {len(extra_gt)} unavailable documents"
        )

    return stats


class GroundTruthRepository:
    """Repository for managing ground truth experiment data.

    Note: Maintained for backward compatibility.
    New code should use the functional API (load_ground_truth, etc.).
    """

    def __init__(self):
        """Initialize the ground truth repository."""
        logger.debug("Initialized GroundTruthRepository")

    def _normalize_document_key(self, document_id: str) -> str:
        """Normalize document ID. Added for test compatibility."""
        return _normalize_document_key(document_id)

    def load(
        self,
        csv_path: Path,
        row_converter: Callable[[pd.Series], Optional[Any]],
    ) -> Dict[str, List[Any]]:
        """Load ground truth from CSV file."""
        return load_ground_truth(csv_path, row_converter)

    def validate_coverage(
        self,
        gt_data: Dict[str, List[Any]],
        available_documents: Union[Iterable[str], Set[str], List[str]],
    ) -> Dict[str, Any]:
        """Validate ground truth coverage."""
        return validate_coverage(gt_data, available_documents)
