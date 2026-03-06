# src/aee/ingestion/base.py
"""Base classes for document parsing strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class BaseParser(ABC):
    """Interface for document parsing strategies."""

    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a source file into markdown text.

        Args:
            file_path: Path to the input file.

        Returns:
            str: Markdown text content.
        """
        pass
