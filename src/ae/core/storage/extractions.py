"""Extractions repository for loading and managing extraction results.

This module provides a clean interface for loading extraction results
from JSON files, with improved error handling and validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from ae.core.exceptions import DataValidationError, InvalidDataFormatError, RepositoryError
from ae.core.utils import normalize_document_key

logger = logging.getLogger(__name__)


class ExtractionRepository:
    """Repository for managing extraction results.

    This repository handles loading extractions from JSON files, validating
    them against schemas, and organizing them by document.

    Example:
        ```python
        from ae.core.schema import load_schema_complete

        bundle = load_schema_complete("config/schema.yaml")

        repo = ExtractionRepository()
        extractions = repo.load(
            results_dir=Path("data/extractions"),
            record_model=bundle.record_model
        )
        ```
    """

    def __init__(self):
        """Initialize the extractions repository."""
        logger.debug("Initialized ExtractionRepository")

    def load(
        self,
        results_dir: Path,
        record_model: Optional[Type[BaseModel]] = None,
        strict: bool = False,
        experiment_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, List[Any]]:
        """Load extractions from JSON files in a directory.

        Args:
            results_dir: Directory containing extraction JSON files.
            record_model: Pydantic model for validating records.
            strict: If True, raise error on invalid extractions. If False, skip them.
            experiment_model: Backward-compatible alias for record_model.

        Returns:
            Dictionary mapping document keys to lists of records.

        Raises:
            RepositoryError: If directory doesn't exist or other IO errors.
            DataValidationError: If strict=True and validation fails.
        """
        # Validate directory exists
        if not results_dir.exists():
            if strict:
                raise RepositoryError(
                    "ExtractionRepository",
                    "load",
                    f"Results directory does not exist: {results_dir}"
                )
            else:
                logger.warning(f"Extractions directory does not exist: {results_dir}")
                return {}

        if not results_dir.is_dir():
            raise RepositoryError(
                "ExtractionRepository",
                "load",
                f"Path is not a directory: {results_dir}"
            )

        # Validate record model
        model = record_model or experiment_model
        if model is None:
            raise ValueError("record_model must be provided")

        if not issubclass(model, BaseModel):
            raise TypeError("record_model must be a Pydantic BaseModel subclass")

        # Load all extraction files
        extractions: Dict[str, List[Any]] = {}
        stats = {"total": 0, "success": 0, "errors": 0, "records": 0}

        for file_path in sorted(results_dir.glob("*.json")):
            stats["total"] += 1
            try:
                doc_extractions = self._load_extraction_file(
                    file_path, model, strict
                )
                if doc_extractions:
                    doc_key = self._extract_document_key(file_path, doc_extractions)
                    extractions[doc_key] = doc_extractions
                    stats["success"] += 1
                    stats["records"] += len(doc_extractions)
                    logger.debug(
                        f"Loaded {len(doc_extractions)} extractions from {file_path.name}"
                    )
            except Exception as e:
                stats["errors"] += 1
                if strict:
                    raise
                else:
                    logger.warning(f"Failed to load extractions from {file_path.name}: {e}")
                    # Add empty entry to track which documents failed
                    extractions[file_path.stem.lower()] = []

        # Log summary
        logger.info(
            f"Loaded extractions: {stats['success']}/{stats['total']} files, "
            f"{stats['records']} total records, {stats['errors']} errors"
        )

        return extractions

    def _load_extraction_file(
        self,
        file_path: Path,
        record_model: Type[BaseModel],
        strict: bool,
    ) -> List[Any]:
        """Load and validate a single extraction file.

        Args:
            file_path: Path to extraction JSON file.
            record_model: Pydantic model for validation.
            strict: Whether to raise errors or skip invalid records.

        Returns:
            List of validated records.

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

        # Extract records from nested structure
        raw_records = self._extract_records(data, file_path)

        # Validate each record
        validated_records = []
        validation_errors = []

        for idx, rec_data in enumerate(raw_records):
            try:
                validated_rec = record_model(**rec_data)
                validated_records.append(validated_rec)
            except ValidationError as e:
                error_msg = f"Record {idx}: {e}"
                validation_errors.append(error_msg)
                logger.debug(f"Validation error in {file_path.name}: {error_msg}")
            except Exception as e:
                error_msg = f"Record {idx}: Unexpected error: {e}"
                validation_errors.append(error_msg)

        # Handle validation errors
        if validation_errors:
            if strict:
                raise DataValidationError(
                    f"Extractions in {file_path.name}",
                    validation_errors
                )
            else:
                logger.warning(
                    f"Skipped {len(validation_errors)}/{len(raw_records)} "
                    f"invalid records in {file_path.name}"
                )

        return validated_records

    def _extract_records(self, data: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Extract records from extraction data structure.

        Handles various extraction file formats.

        Args:
            data: Loaded JSON data.
            file_path: Path to file (for error messages).

        Returns:
            List of raw record dictionaries.
        """
        records = None

        # Structure 1: {"extraction": {"records": [...]}} or {"extraction": {"experiments": [...]}}
        if "extraction" in data:
            extraction = data["extraction"]
            if isinstance(extraction, dict):
                records = extraction.get("records") or extraction.get("experiments")

        # Structure 2: {"records": [...]} or {"experiments": [...]}
        elif "records" in data:
            records = data["records"]
        elif "experiments" in data:
            records = data["experiments"]

        # Structure 3: {"extracted_data": {"records": [...]}} or {"extracted_data": {"experiments": [...]}}
        elif "extracted_data" in data:
            extracted_data = data["extracted_data"]
            if isinstance(extracted_data, dict):
                records = extracted_data.get("records") or extracted_data.get("experiments")

        # Structure 4: Direct list of records
        elif isinstance(data, list):
            records = data

        # No valid structure found
        if records is None:
            logger.warning(
                f"No records found in {file_path.name}. "
                f"Expected 'extraction.records', 'records', or 'extracted_data.records'"
            )
            return []

        # Validate it's a list
        if not isinstance(records, list):
            logger.warning(
                f"Records field in {file_path.name} is not a list: {type(records)}"
            )
            return []

        return records

    def _extract_experiments(self, data: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Backward-compatible alias for _extract_records."""
        return self._extract_records(data, file_path)

    def _extract_document_key(self, file_path: Path, extractions: List[Any]) -> str:
        """Extract document key from file path or extraction metadata.

        Args:
            file_path: Path to extraction file.
            extractions: List of extractions (unused, for future metadata extraction).

        Returns:
            Normalized document key.
        """
        return normalize_document_key(str(file_path))

    def save(
        self,
        extractions: List[BaseModel],
        output_path: Path,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save extractions to JSON file.

        Args:
            extractions: List of experiment extractions.
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
                    "records": [rec.model_dump() for rec in extractions]
                }
            }

            # Add metadata if provided
            if document_metadata:
                output_data["source_metadata"] = document_metadata

            # Write JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(extractions)} extractions to {output_path}")

        except Exception as e:
            raise RepositoryError(
                "ExtractionRepository",
                "save",
                f"Failed to save extractions: {e}"
            ) from e

    def compare_counts(
        self,
        extractions: Dict[str, List[Any]],
        ground_truth: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Compare extraction and ground truth counts.

        Args:
            extractions: Extractions loaded from load().
            ground_truth: Ground truth loaded from GroundTruthRepository.

        Returns:
            Dictionary with comparison statistics.
        """
        ext_docs = set(extractions.keys())
        gt_docs = set(ground_truth.keys())

        matched_docs = ext_docs & gt_docs
        missing_ext = gt_docs - ext_docs
        extra_ext = ext_docs - gt_docs

        # Count experiments per document
        doc_comparisons = {}
        for doc in matched_docs:
            doc_comparisons[doc] = {
                "extracted": len(extractions[doc]),
                "ground_truth": len(ground_truth[doc]),
                "difference": len(extractions[doc]) - len(ground_truth[doc]),
            }

        stats = {
            "total_ext_documents": len(ext_docs),
            "total_gt_documents": len(gt_docs),
            "matched_documents": len(matched_docs),
            "missing_extractions": sorted(missing_ext),
            "extra_extractions": sorted(extra_ext),
            "document_comparisons": doc_comparisons,
            "total_extracted_experiments": sum(len(exps) for exps in extractions.values()),
            "total_gt_experiments": sum(len(exps) for exps in ground_truth.values()),
        }

        return stats
