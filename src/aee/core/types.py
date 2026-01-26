# src/aee/core/types.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class DocumentMetadata(BaseModel):
    """
    Immutable metadata associated with a source document.
    """
    model_config = ConfigDict(frozen=True)

    source_path: str
    filename: str
    page_count: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class ProcessedDocument(BaseModel):
    """
    Unified representation of an ingested document.
    
    Attributes:
        text_content: Hybrid content (Markdown text + HTML tables).
        tables: List of raw extracted tables (optional, parser-dependent).
        images: List of extracted image paths or descriptions.
    """
    model_config = ConfigDict(frozen=False)

    text_content: str
    metadata: DocumentMetadata