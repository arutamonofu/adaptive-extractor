# src/aee/ingestion/__init__.py
"""Document ingestion module for AutoEvoExtractor."""

from aee.ingestion.base import BaseParser
from aee.ingestion.cleaning import TextCleaner
from aee.ingestion.parsers import (
    DoclingParser,
    MarkerParser
)

__all__ = [
    "BaseParser",
    "DoclingParser",
    "MarkerParser",
    "TextCleaner"
]