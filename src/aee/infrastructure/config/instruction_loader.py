"""Instruction loader for managing initial instructions.

This module provides utilities for loading initial instructions from files
for use in DSPy signature optimization.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from aee.shared.exceptions import MissingConfigError

logger = logging.getLogger(__name__)


class InstructionLoader:
    """Loader for initial instruction files.

    This class handles loading instruction text from files located in the
    config directory, with validation and error handling.

    Example:
        ```python
        loader = InstructionLoader(config_dir=Path("config"))
        instruction = loader.load("initial_instructions/nanozymes_sota.txt")
        ```
    """

    def __init__(self, config_dir: Path):
        """Initialize the instruction loader.

        Args:
            config_dir: Base directory containing instruction files.
        """
        self.config_dir = Path(config_dir)
        logger.debug(f"Initialized InstructionLoader with config_dir={self.config_dir}")

    def load(self, instruction_file: str) -> str:
        """Load an instruction from a file.

        Args:
            instruction_file: Relative path to the instruction file from config_dir.

        Returns:
            The instruction text content.

        Raises:
            MissingConfigError: If the instruction file does not exist.
        """
        instruction_path = self.config_dir / instruction_file

        if not instruction_path.exists():
            raise MissingConfigError(
                f"Initial instruction file not found: {instruction_path}. "
                f"Check your configuration."
            )

        if not instruction_path.is_file():
            raise MissingConfigError(
                f"Initial instruction path is not a file: {instruction_path}"
            )

        content = instruction_path.read_text(encoding="utf-8")

        if not content or not content.strip():
            raise MissingConfigError(
                f"Initial instruction file is empty: {instruction_path}"
            )

        logger.info(f"Loaded initial instruction from {instruction_path} ({len(content)} chars)")
        return content

    @staticmethod
    def compute_hash(instruction: str, algorithm: str = "sha256") -> str:
        """Compute a hash of the instruction content.

        Args:
            instruction: The instruction text to hash.
            algorithm: Hash algorithm to use (default: sha256).

        Returns:
            Hexadecimal hash string (first 12 characters).
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(instruction.encode("utf-8"))
        return hash_obj.hexdigest()[:12]

    def load_with_metadata(self, instruction_file: str) -> dict:
        """Load an instruction with metadata.

        Args:
            instruction_file: Relative path to the instruction file from config_dir.

        Returns:
            Dictionary with instruction text and metadata.
        """
        instruction = self.load(instruction_file)
        instruction_path = self.config_dir / instruction_file

        return {
            "instruction": instruction,
            "instruction_file": str(instruction_file),
            "instruction_path": str(instruction_path.absolute()),
            "instruction_length": len(instruction),
            "instruction_hash": self.compute_hash(instruction),
        }
