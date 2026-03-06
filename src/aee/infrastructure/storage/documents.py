"""Document repository for managing parsed documents.

This module provides a clean interface for loading and saving parsed
documents as Markdown files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from aee.shared.exceptions import DataNotFoundError, RepositoryError

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for managing parsed documents as Markdown files.

    This repository handles loading and saving documents as plain
    Markdown (.md) files containing only the text content.

    Example:
        ```python
        repo = DocumentRepository(parsed_dir=Path("data/parsed"))

        # Load a single document
        text = repo.load(Path("data/parsed/document.md"))

        # Load all documents
        all_docs = repo.load_all()  # Dict[str, str]

        # Save a document
        repo.save(markdown_text, Path("data/parsed/new_doc.md"))
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

    def load(self, file_path: Path) -> str:
        """Load a single document text from a Markdown file.

        Args:
            file_path: Path to document .md file.

        Returns:
            Document text content.

        Raises:
            DataNotFoundError: If file not found.
        """
        if not file_path.exists():
            raise DataNotFoundError("Parsed document", str(file_path))

        try:
            text = file_path.read_text(encoding="utf-8")
            logger.debug(f"Loaded document from {file_path}")
            return text

        except Exception as e:
            raise RepositoryError(
                "DocumentRepository",
                "load",
                f"Failed to load document: {e}"
            ) from e

    def load_all(
        self,
        directory: Optional[Path] = None,
        pattern: str = "*.md",
    ) -> Dict[str, str]:
        """Load all documents from a directory.

        Args:
            directory: Directory to load from (uses default if None).
            pattern: Glob pattern for matching files.

        Returns:
            Dictionary mapping document keys to text content.

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

        documents: Dict[str, str] = {}
        stats = {"total": 0, "success": 0, "errors": 0}

        for file_path in sorted(load_dir.glob(pattern)):
            stats["total"] += 1
            try:
                text = self.load(file_path)
                doc_key = self._extract_document_key(file_path)
                documents[doc_key] = text
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
        text: str,
        file_path: Path,
    ) -> None:
        """Save a document text to a Markdown file.

        Args:
            text: Document text content to save.
            file_path: Path to save to.

        Raises:
            RepositoryError: If save operation fails.
        """
        try:
            # Create output directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write text to file
            file_path.write_text(text, encoding="utf-8")

            logger.info(f"Saved document to {file_path}")

        except Exception as e:
            raise RepositoryError(
                "DocumentRepository",
                "save",
                f"Failed to save document: {e}"
            ) from e

    def _extract_document_key(self, file_path: Path) -> str:
        """Extract document key from filename.

        Args:
            file_path: Path to document file.

        Returns:
            Normalized document key.
        """
        # Remove .md extension and normalize
        key = file_path.stem.lower().strip()

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
        for file_path in sorted(load_dir.glob("*.md")):
            try:
                key = self._extract_document_key(file_path)
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
        exact_path = load_dir / f"{document_key}.md"
        if exact_path.exists():
            return True

        # Try with common suffixes
        for suffix in ["_parsed", "_processed", "_result"]:
            path = load_dir / f"{document_key}{suffix}.md"
            if path.exists():
                return True

        return False
