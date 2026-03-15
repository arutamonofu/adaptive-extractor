"""Document parsers for AutoEvoExtractor.

This module provides parsers for extracting text and tables from PDF
documents using the Marker parsing backend or Google Gemini API.
"""

from .base import BaseParser
from .parsers import MarkerParser, GeminiParser, get_parser

# Alias for backward compatibility
DocumentParser = BaseParser

__all__ = [
    "BaseParser",
    "DocumentParser",
    "MarkerParser",
    "GeminiParser",
    "get_parser",
]
