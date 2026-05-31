"""Extraction agents and lifecycle management.

This module defines base agent interfaces, concrete extractor implementations,
and protocols for serialization.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Type, runtime_checkable

import dspy

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for extraction agents.

    Allows the application layer to work with agents without depending on
    specific DSPy implementations.
    """

    @abstractmethod
    def forward(self, document_text: str) -> Any:
        """Execute the extraction pipeline.

        Args:
            document_text: The input text to process.

        Returns:
            Extraction result.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent to a file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent from a file."""
        pass


class UniversalExtractorMeta(type(BaseAgent), type(dspy.Module)):  # type: ignore[misc]
    """Metaclass for UniversalExtractor to resolve metaclass conflict."""
    pass


class UniversalExtractor(BaseAgent, dspy.Module, metaclass=UniversalExtractorMeta):
    """Task-agnostic extraction agent.

    Wraps a specific task signature with Chain-of-Thought reasoning.
    """

    def __init__(self, signature_class: Type[dspy.Signature]):
        """Initialize the UniversalExtractor."""
        BaseAgent.__init__(self)
        dspy.Module.__init__(self)
        self.prog = dspy.ChainOfThought(signature_class)

    def forward(self, document_text: str) -> dspy.Prediction:
        """Execute the extraction pipeline."""
        return self.prog(document_text=document_text)

    def save(self, path: str) -> None:
        """Save the agent to a file."""
        dspy.Module.save(self, path)

    def load(self, path: str) -> None:
        """Load the agent from a file."""
        dspy.Module.load(self, path)


@runtime_checkable
class SerializableAgent(Protocol):
    """Protocol for agents that can be serialized."""

    def dump_state(self) -> Dict[str, Any]:
        """Dump agent state to a dictionary."""
        ...


@runtime_checkable
class SaveableAgent(Protocol):
    """Protocol for agents that can be saved to a file."""

    def save(self, path: str) -> None:
        """Save agent to a file."""
        ...
