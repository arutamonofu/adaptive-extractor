# src/aee/ingestion/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from aee.core.types import ProcessedDocument

class BaseParser(ABC):
    """Abstract base class defining the interface for all document parsers."""

    @abstractmethod
    def parse(self, file_path: str | Path) -> ProcessedDocument:
        """
        Parses a source file into a structured ProcessedDocument.

        Args:
            file_path: Path to the input file (e.g., PDF).

        Returns:
            A populated ProcessedDocument instance containing text and metadata.

        Raises:
            FileNotFoundError: If the source file does not exist.
        """
        pass