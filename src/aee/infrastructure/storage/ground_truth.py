"""Ground truth repository for loading and managing ground truth data.

This module provides a clean interface for loading ground truth experiments
from CSV files, with improved error handling and validation.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from aee.shared.exceptions import DataNotFoundError, DataValidationError, InvalidDataFormatError

logger = logging.getLogger(__name__)


class GroundTruthRepository:
    """Repository for managing ground truth experiment data.

    This repository handles loading ground truth from CSV files, grouping
    experiments by document, and converting rows to experiment objects.

    Example:
        ```python
        from aee.domain.tasks.nanozymes import row_to_nanozyme

        repo = GroundTruthRepository()
        gt_data = repo.load(
            csv_path=Path("data/ground_truth.csv"),
            row_converter=row_to_nanozyme
        )
        # gt_data: Dict[str, List[NanozymeExperiment]]
        ```
    """

    # Valid ID column names (in priority order)
    ID_COLUMNS = ["pdf", "filename", "source", "doi", "document"]

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

        Raises:
            DataNotFoundError: If CSV file not found.
            InvalidDataFormatError: If CSV format is invalid.
            DataValidationError: If data validation fails.
        """
        # Validate file exists
        if not csv_path.exists():
            raise DataNotFoundError("Ground truth CSV", str(csv_path))

        try:
            df = self._load_csv(csv_path)
            id_column = self._identify_id_column(df, csv_path)
            gt_data = self._group_and_convert(df, id_column, row_converter, csv_path)

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

    def _load_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load and validate CSV file.

        Args:
            csv_path: Path to CSV file.

        Returns:
            Loaded DataFrame.

        Raises:
            InvalidDataFormatError: If CSV cannot be parsed or is empty.
        """
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            raise InvalidDataFormatError(str(csv_path), "CSV (file is empty)")
        except pd.errors.ParserError as e:
            raise InvalidDataFormatError(
                str(csv_path), f"CSV (parse error: {e})"
            ) from e
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

    def _identify_id_column(self, df: pd.DataFrame, csv_path: Path) -> str:
        """Identify the document ID column in the DataFrame.

        Args:
            df: DataFrame to search.
            csv_path: Path to CSV file (for error messages).

        Returns:
            Name of the ID column.

        Raises:
            DataValidationError: If no valid ID column found.
        """
        if df.empty:
            return self.ID_COLUMNS[0]  # Return default if empty

        # Find first matching column from priority list
        for col_name in self.ID_COLUMNS:
            if col_name in df.columns:
                logger.debug(f"Using '{col_name}' as document ID column")
                return col_name

        # No valid ID column found
        raise DataValidationError(
            "Ground truth",
            [
                f"No valid document ID column found in {csv_path}",
                f"Expected one of: {', '.join(self.ID_COLUMNS)}",
                f"Found columns: {', '.join(df.columns.tolist())}",
            ]
        )

    def _group_and_convert(
        self,
        df: pd.DataFrame,
        id_column: str,
        row_converter: Callable,
        csv_path: Path,
    ) -> Dict[str, List[Any]]:
        """Group rows by document and convert to experiments.

        Args:
            df: DataFrame to process.
            id_column: Name of the document ID column.
            row_converter: Function to convert rows to experiments.
            csv_path: Path to CSV file (for error messages).

        Returns:
            Dictionary mapping document keys to experiment lists.
        """
        gt_data: Dict[str, List[Any]] = {}
        conversion_errors = 0
        total_rows = 0

        # Group by document ID
        for document_id, group in df.groupby(id_column):
            # Normalize document key (remove extension, lowercase)
            doc_key = self._normalize_document_key(str(document_id))

            # Convert each row to an experiment
            experiments = []
            for idx, row in group.iterrows():
                total_rows += 1
                try:
                    experiment = row_converter(row)
                    if experiment is not None:
                        experiments.append(experiment)
                    else:
                        logger.debug(
                            f"Row converter returned None for row {idx} in {document_id}"
                        )
                        conversion_errors += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to convert row {idx} for {document_id}: {e}"
                    )
                    conversion_errors += 1
                    continue

            # Only add documents with at least one valid experiment
            if experiments:
                gt_data[doc_key] = experiments
                logger.debug(
                    f"Loaded {len(experiments)} experiments for document '{doc_key}'"
                )

        # Log conversion statistics
        if conversion_errors > 0:
            logger.warning(
                f"Ground truth conversion: {conversion_errors}/{total_rows} rows failed"
            )

        if not gt_data:
            logger.warning(f"No valid experiments found in {csv_path}")

        return gt_data

    def _normalize_document_key(self, document_id: str) -> str:
        """Normalize document identifier to a consistent key.

        Args:
            document_id: Raw document identifier from CSV.

        Returns:
            Normalized key (lowercase, no extension).
        """
        # Remove common file extensions
        for ext in [".pdf", ".PDF", ".txt", ".TXT", ".doc", ".DOC"]:
            if document_id.endswith(ext):
                document_id = document_id[:-len(ext)]
                break

        # Lowercase and strip whitespace
        return document_id.lower().strip()

    def validate_coverage(
        self,
        gt_data: Dict[str, List[Any]],
        available_documents: List[str],
    ) -> Dict[str, Any]:
        """Validate ground truth coverage against available documents.

        Args:
            gt_data: Ground truth data from load().
            available_documents: List of available document keys.

        Returns:
            Dictionary with coverage statistics.
        """
        available_set = {self._normalize_document_key(doc) for doc in available_documents}
        gt_set = set(gt_data.keys())

        covered = gt_set & available_set
        missing_docs = available_set - gt_set
        extra_gt = gt_set - available_set

        total_experiments = sum(len(exps) for exps in gt_data.values())

        stats = {
            "total_gt_documents": len(gt_set),
            "total_available_documents": len(available_set),
            "covered_documents": len(covered),
            "coverage_percentage": (len(covered) / len(available_set) * 100) if available_set else 0,
            "missing_documents": sorted(missing_docs),
            "extra_gt_documents": sorted(extra_gt),
            "total_experiments": total_experiments,
        }

        if missing_docs:
            logger.warning(
                f"Ground truth missing for {len(missing_docs)} documents: {list(missing_docs)[:5]}..."
            )
        if extra_gt:
            logger.warning(
                f"Ground truth exists for {len(extra_gt)} unavailable documents"
            )

        return stats
