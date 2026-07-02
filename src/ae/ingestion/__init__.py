"""Ingestion module for Adaptive Extractor.

This module handles document parsing (e.g. PDF to Markdown conversion)
using various parsing strategies (Gemini and visual enricher).
"""

from .parsers import BaseParser, get_parser
from .use_case import (
    ParseDocumentsRequest,
    ParseDocumentsResponse,
    ParseDocumentsUseCase,
)

__all__ = [
    "BaseParser",
    "get_parser",
    "ParseDocumentsRequest",
    "ParseDocumentsResponse",
    "ParseDocumentsUseCase",
]
