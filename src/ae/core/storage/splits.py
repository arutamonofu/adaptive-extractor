"""Data split repository and functional API for managing train/test splits.

This module provides functions and repository classes for loading, saving,
creating, and validating dataset splits.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from ae.core.exceptions import DataNotFoundError, InvalidDataFormatError, RepositoryError

logger = logging.getLogger(__name__)

VALID_SPLIT_NAMES = ["train", "test", "val", "validation", "dev"]


def _normalize_document_key(doc_id: str) -> str:
    """Normalize document ID by removing extensions.

    Args:
        doc_id: Document identifier.

    Returns:
        Normalized document key (lowercase, no extension).
    """
    doc_id = str(doc_id).strip().lower()

    # Remove common extensions
    for ext in [".pdf", ".txt", ".doc"]:
        if doc_id.endswith(ext):
            doc_id = doc_id[:-len(ext)]
            break

    return doc_id


def _load_json(json_path: Path) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        json_path: Path to JSON file.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        DataNotFoundError: If file not found.
        InvalidDataFormatError: If JSON cannot be parsed.
    """
    if not json_path.exists():
        raise DataNotFoundError("JSON file", str(json_path))

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidDataFormatError("JSON file", f"Invalid JSON: {e}") from e


def load_all_splits(
    split_path: Path,
    normalize_keys: bool = True,
) -> Dict[str, Set[str]]:
    """Load all data splits from JSON file.

    Args:
        split_path: Path to splits JSON file.
        normalize_keys: Whether to normalize filenames.

    Returns:
        Dictionary mapping split names to sets of document keys.
    """
    splits = _load_json(split_path)

    result = {}
    for split_name, docs in splits.items():
        if normalize_keys:
            result[split_name] = {_normalize_document_key(d) for d in docs}
        else:
            result[split_name] = set(docs)

    return result


def load_split(
    split_path: Path,
    split_name: str,
    normalize_keys: bool = True,
) -> Set[str]:
    """Load a specific data split.

    Args:
        split_path: Path to splits JSON file.
        split_name: Name of split to load (e.g., "train", "test").
        normalize_keys: Whether to normalize filenames.

    Returns:
        Set of document keys in the split.

    Raises:
        DataNotFoundError: If split file not found.
        InvalidDataFormatError: If JSON format is invalid.
    """
    splits = load_all_splits(split_path, normalize_keys=normalize_keys)

    if split_name not in splits:
        logger.warning(
            f"Split '{split_name}' not found in {split_path}. "
            f"Available: {', '.join(splits.keys())}"
        )
        return set()

    return splits[split_name]


def save_splits(
    splits: Dict[str, Union[List[str], Set[str]]],
    output_path: Path,
) -> Path:
    """Save data splits to JSON file.

    Args:
        splits: Dictionary mapping split names to lists/sets of documents.
        output_path: Path for output JSON file.

    Returns:
        Path to saved file.

    Raises:
        RepositoryError: If save operation fails.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_data = {}
        for split_name, docs in splits.items():
            json_data[split_name] = sorted(list(docs))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Saved splits to {output_path}")
        return output_path

    except Exception as e:
        raise RepositoryError(
            "save_splits", "save", f"Failed to save splits: {e}"
        ) from e


def create_random_split(
    documents: List[str],
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> Dict[str, Set[str]]:
    """Create random train/test split.

    Args:
        documents: List of document keys.
        train_ratio: Ratio of documents for training (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train' and 'test' splits.

    Raises:
        ValueError: If train_ratio is invalid.
    """
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    if seed is not None:
        random.seed(seed)

    docs = list(documents)
    random.shuffle(docs)

    split_idx = int(len(docs) * train_ratio)

    return {
        "train": set(docs[:split_idx]),
        "test": set(docs[split_idx:]),
    }


def validate_splits(
    splits: Dict[str, Set[str]],
    available_docs: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Validate splits against available documents.

    Args:
        splits: Dictionary of splits.
        available_docs: Iterable of available document keys.

    Returns:
        Dictionary with validation results for each split.
    """
    available_set = {_normalize_document_key(d) for d in available_docs}
    results = {}

    for split_name, docs in splits.items():
        missing = docs - available_set
        results[split_name] = {
            "count": len(docs),
            "missing": sorted(list(missing)),
            "valid": len(missing) == 0,
        }

    return results


class DataSplitRepository:
    """Repository for managing train/test data splits.

    Note: Maintained for backward compatibility.
    New code should use the functional API.
    """

    def __init__(self):
        """Initialize the data split repository."""
        logger.debug("Initialized DataSplitRepository")

    def load_split(
        self,
        split_path: Path,
        split_name: str,
        normalize_keys: bool = True,
    ) -> Set[str]:
        """Load a specific data split."""
        return load_split(split_path, split_name, normalize_keys=normalize_keys)

    def load_all_splits(
        self,
        split_path: Path,
        normalize_keys: bool = True,
    ) -> Dict[str, Set[str]]:
        """Load all data splits."""
        return load_all_splits(split_path, normalize_keys=normalize_keys)

    def save_splits(
        self,
        splits: Dict[str, Union[List[str], Set[str]]],
        output_path: Path,
    ) -> Path:
        """Save data splits to JSON file."""
        return save_splits(splits, output_path)

    def create_random_split(
        self,
        documents: List[str],
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ) -> Dict[str, Set[str]]:
        """Create random train/test split."""
        return create_random_split(documents, train_ratio=train_ratio, seed=seed)

    def validate_splits(
        self,
        splits: Dict[str, Set[str]],
        available_docs: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Validate splits against available documents."""
        return validate_splits(splits, available_docs)
