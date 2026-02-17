"""Base agent abstraction for the domain layer.

This module defines the abstract interface for extraction agents,
allowing the application layer to depend on abstractions rather
than concrete infrastructure implementations.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for extraction agents.

    This interface allows the application layer to work with agents
    without depending on specific DSPy implementations.

    Example:
        ```python
        class UniversalExtractor(BaseAgent):
            def __init__(self, signature_class):
                self.prog = dspy.ChainOfThought(signature_class)

            def forward(self, document_text: str) -> Any:
                return self.prog(document_text=document_text)

            def save(self, path: Path) -> None:
                # Save implementation
        ```
    """

    @abstractmethod
    def forward(self, document_text: str) -> Any:
        """Execute the extraction pipeline.

        Args:
            document_text: The input text to process.

        Returns:
            Extraction result (DSPy Prediction or similar).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent to a file.

        Args:
            path: Path to save the agent.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent from a file.

        Args:
            path: Path to load the agent from.
        """
        pass
