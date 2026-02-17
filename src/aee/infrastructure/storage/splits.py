"""Data split repository for managing train/test splits.

This module provides a clean interface for loading and managing
data splits with improved error handling.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from aee.shared.exceptions import DataNotFoundError, InvalidDataFormatError, RepositoryError

logger = logging.getLogger(__name__)


class DataSplitRepository:
    """Repository for managing train/test data splits.

    This repository handles loading and saving data split configurations
    from/to JSON files.

    Example:
        ```python
        repo = DataSplitRepository()

        # Load train split
        train_files = repo.load_split(
            split_path=Path("data/splits/nanozymes.json"),
            split_name="train"
        )

        # Load all splits
        all_splits = repo.load_all_splits(Path("data/splits/nanozymes.json"))

        # Save splits
        repo.save_splits(
            splits={"train": [...], "test": [...]},
            output_path=Path("data/splits/nanozymes.json")
        )
        ```
    """

    VALID_SPLIT_NAMES = ["train", "test", "val", "validation", "dev"]

    def __init__(self):
        """Initialize the data split repository."""
        logger.debug("Initialized DataSplitRepository")

    def load_split(
        self,
        split_path: Path,
        split_name: str,
        normalize_keys: bool = True,
    ) -> Set[str]:
        """Load a specific data split.

        Args:
            split_path: Path to splits JSON file.
            split_name: Name of split to load (e.g., "train", "test").
            normalize_keys: Whether to normalize filenames (lowercase, no extension).

        Returns:
            Set of document keys in the split.

        Raises:
            DataNotFoundError: If split file not found.
            InvalidDataFormatError: If JSON format is invalid.
        """
        if not split_path.exists():
            raise DataNotFoundError("Data split file", str(split_path))

        try:
            splits = self.load_all_splits(split_path)

            if split_name not in splits:
                logger.warning(
                    f"Split '{split_name}' not found in {split_path}. "
                    f"Available: {', '.join(splits.keys())}"
                )
                return set()

            files = splits[split_name]

            # Normalize keys if requested
            if normalize_keys:
                files = {self._normalize_key(f) for f in files}

            logger.debug(f"Loaded {len(files)} files from '{split_name}' split")
            return files

        except (DataNotFoundError, InvalidDataFormatError):
            raise
        except Exception as e:
            raise InvalidDataFormatError(
                str(split_path), f"Error loading split: {e}"
            ) from e

    def load_all_splits(self, split_path: Path) -> Dict[str, List[str]]:
        """Load all splits from a file.

        Args:
            split_path: Path to splits JSON file.

        Returns:
            Dictionary mapping split names to lists of document keys.

        Raises:
            DataNotFoundError: If split file not found.
            InvalidDataFormatError: If JSON format is invalid.
        """
        if not split_path.exists():
            raise DataNotFoundError("Data split file", str(split_path))

        try:
            with open(split_path, "r", encoding="utf-8") as f:
                splits = json.load(f)

            # Validate structure
            if not isinstance(splits, dict):
                raise InvalidDataFormatError(
                    str(split_path),
                    "Expected dict mapping split names to file lists"
                )

            # Validate each split
            for split_name, files in splits.items():
                if not isinstance(files, list):
                    raise InvalidDataFormatError(
                        str(split_path),
                        f"Split '{split_name}' must be a list, got {type(files)}"
                    )

            logger.info(
                f"Loaded {len(splits)} splits from {split_path}: "
                f"{', '.join(f'{k}({len(v)})' for k, v in splits.items())}"
            )

            return splits

        except json.JSONDecodeError as e:
            raise InvalidDataFormatError(
                str(split_path), f"Invalid JSON: {e}"
            ) from e
        except (DataNotFoundError, InvalidDataFormatError):
            raise
        except Exception as e:
            raise InvalidDataFormatError(
                str(split_path), f"Cannot load splits: {e}"
            ) from e

    def save_splits(
        self,
        splits: Dict[str, List[str]],
        output_path: Path,
    ) -> Path:
        """Save data splits to JSON file.

        Args:
            splits: Dictionary mapping split names to lists of document keys.
            output_path: Path to save splits to.

        Returns:
            Path to the saved file.

        Raises:
            RepositoryError: If save operation fails.
        """
        try:
            # Validate splits structure
            if not isinstance(splits, dict):
                raise ValueError("Splits must be a dictionary")

            for split_name, files in splits.items():
                if not isinstance(files, list):
                    raise ValueError(
                        f"Split '{split_name}' must be a list, got {type(files)}"
                    )

            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(splits, f, indent=2, ensure_ascii=False)

            total_files = sum(len(files) for files in splits.values())
            logger.info(
                f"Saved {len(splits)} splits ({total_files} total files) to {output_path}"
            )

            return output_path

        except Exception as e:
            raise RepositoryError(
                "DataSplitRepository",
                "save_splits",
                f"Failed to save splits: {e}"
            ) from e

    def _normalize_key(self, filename: str) -> str:
        """Normalize a filename to a consistent key.

        Args:
            filename: Raw filename from split.

        Returns:
            Normalized key (lowercase, no extension).
        """
        # Remove common extensions
        for ext in [".pdf", ".PDF", ".json", ".JSON", ".txt", ".TXT"]:
            if filename.endswith(ext):
                filename = filename[:-len(ext)]
                break

        # Remove common suffixes
        for suffix in ["_parsed", "_processed", "_result"]:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break

        return filename.lower().strip()

    def validate_splits(
        self,
        splits: Dict[str, List[str]],
        available_documents: List[str],
    ) -> Dict[str, Any]:
        """Validate splits against available documents.

        Args:
            splits: Loaded splits dictionary.
            available_documents: List of available document keys.

        Returns:
            Dictionary with validation statistics.
        """
        available_set = {self._normalize_key(doc) for doc in available_documents}
        validation_results = {}

        for split_name, split_files in splits.items():
            split_set = {self._normalize_key(f) for f in split_files}

            valid_files = split_set & available_set
            missing_files = split_set - available_set

            validation_results[split_name] = {
                "total": len(split_files),
                "valid": len(valid_files),
                "missing": sorted(missing_files),
                "coverage": (len(valid_files) / len(split_files) * 100) if split_files else 0,
            }

            if missing_files:
                logger.warning(
                    f"Split '{split_name}': {len(missing_files)}/{len(split_files)} "
                    f"files not found in available documents"
                )

        return validation_results

    def create_random_split(
        self,
        documents: List[str],
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Create a random train/test split.

        Args:
            documents: List of document keys to split.
            train_ratio: Ratio of documents for training (0-1).
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with "train" and "test" splits.

        Raises:
            ValueError: If train_ratio is invalid.
        """
        import random

        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        if seed is not None:
            random.seed(seed)

        # Shuffle documents
        shuffled = documents.copy()
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * train_ratio)
        train_docs = shuffled[:split_idx]
        test_docs = shuffled[split_idx:]

        splits = {
            "train": sorted(train_docs),
            "test": sorted(test_docs),
        }

        logger.info(
            f"Created random split: {len(train_docs)} train, "
            f"{len(test_docs)} test (ratio={train_ratio})"
        )

        return splits
