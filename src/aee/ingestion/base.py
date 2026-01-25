# src/aee/ingestion/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from aee.core.types import ProcessedDocument

class BaseParser(ABC):
    """Interface for document parsing strategies."""

    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """
        Parses a source file into a structured ProcessedDocument.

        Args:
            file_path: Path to the input file.

        Returns:
            ProcessedDocument containing text, tables, and metadata.
        """
        pass