"""Ingestion module for Adaptive Extractor.

This module handles document parsing (e.g. PDF to Markdown conversion)
using the Gemini parser pipeline (Ingestion v2).
"""

from .parsers import BaseParser, get_parser
from .use_case import (
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    IngestDocumentsUseCase,
)

__all__ = [
    "BaseParser",
    "get_parser",
    "IngestDocumentsRequest",
    "IngestDocumentsResponse",
    "IngestDocumentsUseCase",
]
