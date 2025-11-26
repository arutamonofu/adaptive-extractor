# src/aee/core/types.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """
    Metadata associated with a source document.

    Attributes:
        source_path: Absolute system path to the source file.
        filename: Name of the file with extension.
        page_count: Total number of pages in the document.
        extra: Additional metadata extracted by parsers (DOI, authors, etc.).
    """
    source_path: str
    filename: str
    page_count: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class ProcessedDocument(BaseModel):
    """
    Unified representation of a processed document ready for AI ingestion.

    Attributes:
        text_content: Full text content in Markdown format.
        tables: List of extracted tables (Markdown or HTML representation).
        images: Paths or descriptions of extracted images.
        metadata: Document metadata object.
    """
    text_content: str = Field(..., description="Full text in Markdown format")
    tables: List[str] = Field(default_factory=list, description="Extracted tables")
    images: List[str] = Field(default_factory=list, description="Extracted image paths/descriptions")
    metadata: DocumentMetadata