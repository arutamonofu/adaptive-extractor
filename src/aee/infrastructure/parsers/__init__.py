"""Document parsers for AutoEvoExtractor.

This module provides parsers for extracting text and tables from PDF
documents using various parsing backends (Docling, Marker, etc.).
"""

from .base import BaseParser
from .cleaning import TextCleaner
from .parsers import DoclingParser, MarkerParser, get_parser

# Alias for backward compatibility
DocumentParser = BaseParser

__all__ = [
    "BaseParser",
    "DocumentParser",
    "TextCleaner",
    "DoclingParser",
    "MarkerParser",
    "get_parser",
]
