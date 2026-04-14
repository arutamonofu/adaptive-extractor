"""Document parsers for AutoEvoExtractor.

This module provides parsers for extracting text and tables from PDF
documents using the Marker parsing backend or Google Gemini API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseParser
from .parsers import GeminiParser, get_parser

# Alias for backward compatibility
if TYPE_CHECKING:
    # DocumentParser is an alias for BaseParser (defined via __getattr__)
    from .base import BaseParser as DocumentParser  # type: ignore[assignment]
    from .parsers import MarkerParser
else:

    def __getattr__(name: str):
        if name == "MarkerParser":
            from .parsers import MarkerParser

            return MarkerParser
        if name == "DocumentParser":
            from .base import BaseParser

            return BaseParser
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseParser",
    "DocumentParser",
    "MarkerParser",
    "GeminiParser",
    "get_parser",
]
