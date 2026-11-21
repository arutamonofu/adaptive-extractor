"""Base classes for document parsing strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class BaseParser(ABC):
    """Interface for document parsing strategies."""

    def __init__(self, config: Any = None) -> None:
        """Initialize parser with optional configuration."""
        self.config = config

    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a source file into markdown text.

        Args:
            file_path: Path to the input file.

        Returns:
            str: Markdown text content.
        """
        pass
