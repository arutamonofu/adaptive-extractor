"""Ingestion module for Adaptive Extractor.

This module handles document parsing (e.g. PDF to Markdown conversion)
using various parsing strategies (Gemini and visual enricher).
"""

from .base_parser import BaseParser
from .parsers import get_parser
from .pipeline import (
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
