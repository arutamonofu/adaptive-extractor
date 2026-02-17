"""Document repository for managing parsed documents.

This module provides a clean interface for loading and saving parsed
documents with improved error handling and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from aee.domain.entities import ProcessedDocument
from aee.shared.exceptions import DataNotFoundError, InvalidDataFormatError, RepositoryError

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for managing parsed documents.

    This repository handles loading and saving ProcessedDocument instances
    from/to JSON files.

    Example:
        ```python
        repo = DocumentRepository(parsed_dir=Path("data/parsed"))

        # Load a single document
        doc = repo.load(Path("data/parsed/document.json"))

        # Load all documents
        all_docs = repo.load_all()

        # Save a document
        repo.save(document, Path("data/parsed/new_doc.json"))
        ```
    """

    def __init__(self, parsed_dir: Optional[Path] = None):
        """Initialize the document repository.

        Args:
            parsed_dir: Default directory for parsed documents.
        """
        self.parsed_dir = Path(parsed_dir) if parsed_dir else None
        if self.parsed_dir:
            self.parsed_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized DocumentRepository at {self.parsed_dir}")

    def load(self, file_path: Path) -> ProcessedDocument:
        """Load a single processed document.

        Args:
            file_path: Path to document JSON file.

        Returns:
            Loaded ProcessedDocument.

        Raises:
            DataNotFoundError: If file not found.
            InvalidDataFormatError: If JSON format is invalid.
        """
        if not file_path.exists():
            raise DataNotFoundError("Parsed document", str(file_path))

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate and create ProcessedDocument
            document = ProcessedDocument(**data)
            logger.debug(f"Loaded document from {file_path}")
            return document

        except json.JSONDecodeError as e:
            raise InvalidDataFormatError(
                str(file_path), f"Invalid JSON: {e}"
            ) from e
        except ValidationError as e:
            raise InvalidDataFormatError(
                str(file_path), f"Invalid document structure: {e}"
            ) from e
        except Exception as e:
            raise InvalidDataFormatError(
                str(file_path), f"Cannot load document: {e}"
            ) from e

    def load_all(
        self,
        directory: Optional[Path] = None,
        pattern: str = "*.json",
    ) -> Dict[str, ProcessedDocument]:
        """Load all documents from a directory.

        Args:
            directory: Directory to load from (uses default if None).
            pattern: Glob pattern for matching files.

        Returns:
            Dictionary mapping document keys to ProcessedDocument instances.

        Raises:
            RepositoryError: If directory doesn't exist or is invalid.
        """
        load_dir = Path(directory) if directory else self.parsed_dir

        if load_dir is None:
            raise RepositoryError(
                "DocumentRepository",
                "load_all",
                "No directory specified and no default directory set"
            )

        if not load_dir.exists():
            raise RepositoryError(
                "DocumentRepository",
                "load_all",
                f"Directory does not exist: {load_dir}"
            )

        documents: Dict[str, ProcessedDocument] = {}
        stats = {"total": 0, "success": 0, "errors": 0}

        for file_path in sorted(load_dir.glob(pattern)):
            stats["total"] += 1
            try:
                doc = self.load(file_path)
                doc_key = self._extract_document_key(file_path, doc)
                documents[doc_key] = doc
                stats["success"] += 1
            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"Failed to load document {file_path.name}: {e}")

        logger.info(
            f"Loaded {stats['success']}/{stats['total']} documents "
            f"({stats['errors']} errors)"
        )

        return documents

    def save(
        self,
        document: ProcessedDocument,
        file_path: Path,
    ) -> None:
        """Save a processed document to JSON.

        Args:
            document: ProcessedDocument to save.
            file_path: Path to save to.

        Raises:
            RepositoryError: If save operation fails.
        """
        try:
            # Create output directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize to JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    document.model_dump(),
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            logger.info(f"Saved document to {file_path}")

        except Exception as e:
            raise RepositoryError(
                "DocumentRepository",
                "save",
                f"Failed to save document: {e}"
            ) from e

    def _extract_document_key(
        self,
        file_path: Path,
        document: ProcessedDocument,
    ) -> str:
        """Extract document key from filename or metadata.

        Args:
            file_path: Path to document file.
            document: Loaded document.

        Returns:
            Normalized document key.
        """
        # Try to use filename from metadata first
        if hasattr(document.metadata, "filename"):
            filename = document.metadata.filename
        else:
            filename = file_path.name

        # Remove extension and normalize
        key = Path(filename).stem.lower().strip()

        # Remove common suffixes
        for suffix in ["_parsed", "_processed", "_result"]:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                break

        return key

    def list_document_keys(
        self,
        directory: Optional[Path] = None,
    ) -> List[str]:
        """List all document keys in a directory.

        Args:
            directory: Directory to scan (uses default if None).

        Returns:
            List of document keys.
        """
        load_dir = Path(directory) if directory else self.parsed_dir

        if load_dir is None or not load_dir.exists():
            return []

        keys = []
        for file_path in sorted(load_dir.glob("*.json")):
            try:
                doc = self.load(file_path)
                key = self._extract_document_key(file_path, doc)
                keys.append(key)
            except Exception as e:
                logger.debug(f"Skipped {file_path.name}: {e}")

        return keys

    def exists(self, document_key: str, directory: Optional[Path] = None) -> bool:
        """Check if a document exists.

        Args:
            document_key: Document key to check.
            directory: Directory to check (uses default if None).

        Returns:
            True if document exists, False otherwise.
        """
        load_dir = Path(directory) if directory else self.parsed_dir

        if load_dir is None:
            return False

        # Try exact match
        exact_path = load_dir / f"{document_key}.json"
        if exact_path.exists():
            return True

        # Try with common suffixes
        for suffix in ["_parsed", "_processed", "_result"]:
            path = load_dir / f"{document_key}{suffix}.json"
            if path.exists():
                return True

        return False
