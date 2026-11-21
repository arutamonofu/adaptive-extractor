"""Document repository for managing parsed documents.

This module provides a clean interface for loading and saving parsed
documents as Markdown files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ae.core.exceptions import DataNotFoundError, RepositoryError
from ae.core.utils import normalize_document_key

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for managing parsed documents as Markdown files.

    This repository handles loading and saving documents as plain
    Markdown (.md) files containing only the text content.

    Example:
        ```python
        repo = DocumentRepository(ingestion_dir=Path("data/interim/ingestion"))

        # Load a single document
        text = repo.load(Path("data/interim/ingestion/document.md"))

        # Load all documents
        all_docs = repo.load_all()  # Dict[str, str]

        # Save a document
        repo.save(markdown_text, Path("data/interim/ingestion/new_doc.md"))
        ```
    """

    def __init__(self, ingestion_dir: Optional[Path] = None):
        """Initialize the document repository.

        Args:
            ingestion_dir: Default directory for parsed documents.
        """
        self.ingestion_dir = Path(ingestion_dir) if ingestion_dir else None
        if self.ingestion_dir:
            self.ingestion_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized DocumentRepository at {self.ingestion_dir}")

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
        load_dir = Path(directory) if directory else self.ingestion_dir

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

        if pattern == "*.md":
            doc_keys = self.list_document_keys(directory=load_dir)
            for doc_key in doc_keys:
                stats["total"] += 1
                try:
                    text = self.get(doc_key, directory=load_dir)
                    if text is not None:
                        documents[doc_key] = text
                        stats["success"] += 1
                    else:
                        stats["errors"] += 1
                        logger.warning(f"Failed to load document content for {doc_key}")
                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(f"Failed to load document {doc_key}: {e}")
        else:
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
        return normalize_document_key(str(file_path))

    def _find_document_path(
        self,
        document_key: str,
        directory: Optional[Path] = None,
    ) -> Optional[Path]:
        """Find the filesystem path for a document key, supporting flat and nested layouts.

        Args:
            document_key: Document identifier or key.
            directory: Directory to search (uses default if None).

        Returns:
            Path to the document .md file if found, None otherwise.
        """
        load_dir = Path(directory) if directory else self.ingestion_dir
        if load_dir is None or not load_dir.exists():
            return None

        raw_key = document_key.strip()
        key_lower = raw_key.lower()
        norm_key = normalize_document_key(raw_key)

        suffixes = ["", "_parsed", "_processed", "_result"]

        # 1. Check flat paths: load_dir / f"{key}{suffix}.md"
        for k in (key_lower, raw_key, norm_key):
            for suffix in suffixes:
                candidate = load_dir / f"{k}{suffix}.md"
                if candidate.is_file():
                    return candidate

        # 2. Check nested per-document directory: load_dir / folder / f"{doc_name}.md"
        for folder_name in (key_lower, raw_key, norm_key):
            sub_dir = load_dir / folder_name
            if sub_dir.is_dir():
                for doc_name in (folder_name, key_lower, raw_key, norm_key):
                    for suffix in suffixes:
                        candidate = sub_dir / f"{doc_name}{suffix}.md"
                        if candidate.is_file():
                            return candidate
                for f in sub_dir.iterdir():
                    if f.is_file() and f.suffix == ".md":
                        if f.stem.lower() in (key_lower, norm_key, folder_name.lower()):
                            return f

        # 3. Fallback scan top-level for case-insensitive / normalized match
        for item in load_dir.iterdir():
            if item.is_file() and item.suffix == ".md":
                if (
                    item.stem.lower() in (key_lower, norm_key)
                    or normalize_document_key(item.stem) == norm_key
                ):
                    return item
            elif item.is_dir():
                sub_norm = normalize_document_key(item.name)
                if item.name.lower() in (key_lower, norm_key) or sub_norm == norm_key:
                    target_file = item / f"{item.name}.md"
                    if target_file.is_file():
                        return target_file
                    for f in item.iterdir():
                        if f.is_file() and f.suffix == ".md":
                            if f.stem.lower() in (key_lower, norm_key, item.name.lower()):
                                return f

        return None

    def list_document_keys(
        self,
        directory: Optional[Path] = None,
    ) -> List[str]:
        """List all document keys in a directory (both flat and nested layouts).

        Args:
            directory: Directory to scan (uses default if None).

        Returns:
            List of unique document keys.
        """
        load_dir = Path(directory) if directory else self.ingestion_dir

        if load_dir is None or not load_dir.exists():
            return []

        keys_set = set()
        keys_list = []

        def add_key(file_path: Path) -> None:
            try:
                key = self._extract_document_key(file_path)
                if key and key not in keys_set:
                    keys_set.add(key)
                    keys_list.append(key)
            except Exception as e:
                logger.debug(f"Skipped {file_path.name}: {e}")

        # 1. Scan flat files
        for file_path in sorted(load_dir.glob("*.md")):
            add_key(file_path)

        # 2. Scan subdirectories (nested layout from multi-stage ingestion)
        for item in sorted(load_dir.iterdir()):
            if item.is_dir():
                item_norm = normalize_document_key(item.name)
                primary_file = item / f"{item.name}.md"
                if primary_file.is_file():
                    add_key(primary_file)
                else:
                    for sub_file in sorted(item.iterdir()):
                        if sub_file.is_file() and sub_file.suffix == ".md":
                            if (
                                sub_file.stem.lower() == item.name.lower()
                                or normalize_document_key(sub_file.stem) == item_norm
                            ):
                                add_key(sub_file)
                                break

        return sorted(keys_list)

    def get(self, document_key: str, directory: Optional[Path] = None) -> Optional[str]:
        """Load a document by key, using case-insensitive lookup across flat and nested layouts.

        Args:
            document_key: Document key (e.g. ``ANGE.201904751``).
            directory: Directory to search (uses default if None).

        Returns:
            Document text content, or ``None`` if the document is not found.
        """
        path = self._find_document_path(document_key, directory)
        if path is None:
            return None
        try:
            return self.load(path)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None

    def get_canonical_manifest_text(
        self, document_key: str, directory: Optional[Path] = None
    ) -> Optional[str]:
        """Load the canonical manifest text if available, falling back to the main document .md.

        Prefers ``02_manifest_visuals/manifest_enriched_text.md`` in the document directory
        when present and non-empty, since visual unpacking may modify the main ``.md`` in place.
        """
        doc_path = self._find_document_path(document_key, directory)
        if doc_path is None:
            return None

        manifest_md = doc_path.parent / "02_manifest_visuals" / "manifest_enriched_text.md"
        if manifest_md.is_file() and manifest_md.stat().st_size > 0:
            try:
                return self.load(manifest_md)
            except Exception as e:
                logger.warning(f"Failed to read canonical manifest {manifest_md}: {e}")

        try:
            return self.load(doc_path)
        except Exception as e:
            logger.warning(f"Failed to read {doc_path}: {e}")
            return None

    def exists(self, document_key: str, directory: Optional[Path] = None) -> bool:
        """Check if a document exists across flat and nested layouts.

        Args:
            document_key: Document key to check.
            directory: Directory to check (uses default if None).

        Returns:
            True if document exists, False otherwise.
        """
        return self._find_document_path(document_key, directory) is not None
