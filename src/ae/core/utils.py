"""Utility functions for Adaptive Extractor."""

from pathlib import Path

def normalize_document_key(doc_id: str) -> str:
    """Normalize a document key/ID by stripping, converting to lowercase,
    and removing common document extensions and suffixes.

    Args:
        doc_id: The document identifier or file path string.

    Returns:
        The normalized document key.
    """
    if not doc_id:
        return ""

    # If it is a path or has path-like separators, get the stem/name
    path = Path(doc_id)
    name = path.stem

    # Just in case, lowercase and strip
    name = name.strip().lower()

    # Remove common extensions (just in case they are still in the stem)
    for ext in [".pdf", ".txt", ".doc", ".json", ".md"]:
        if name.endswith(ext):
            name = name[:-len(ext)]

    # Remove common suffixes
    for suffix in [
        "_parsed",
        "_processed",
        "_result",
        "_extraction",
        "_extractions",
        "_ext",
    ]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    return name.lower().strip()
