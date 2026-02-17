"""Task-agnostic extraction agent for AutoEvoExtractor."""

import json
from pathlib import Path
from typing import Any, Type

import dspy

from aee.domain.agents.base import BaseAgent


class UniversalExtractorMeta(type(BaseAgent), type(dspy.Module)):
    """Metaclass for UniversalExtractor to resolve metaclass conflict."""
    pass


class UniversalExtractor(BaseAgent, dspy.Module, metaclass=UniversalExtractorMeta):
    """Task-agnostic extraction agent.

    Wraps a specific task signature (e.g., Nanozymes) with Chain-of-Thought reasoning.
    """

    def __init__(self, signature_class: Type[dspy.Signature]):
        """Initialize the UniversalExtractor.

        Args:
            signature_class: The DSPy signature defining input/output fields and instructions.
        """
        BaseAgent.__init__(self)
        dspy.Module.__init__(self)
        self.prog = dspy.ChainOfThought(signature_class)

    def forward(self, document_text: str) -> dspy.Prediction:
        """Execute the extraction pipeline.

        Args:
            document_text: The full content of the document (Markdown/HTML hybrid).

        Returns:
            dspy.Prediction: Contains 'reasoning' (str) and 'extracted_data' (Pydantic model).
        """
        return self.prog(document_text=document_text)

    def save(self, path: str) -> None:
        """Save the agent to a file.

        Args:
            path: Path to save the agent.
        """
        dspy.Module.save(self, path)

    def load(self, path: str) -> None:
        """Load the agent from a file.

        Args:
            path: Path to load the agent from.
        """
        dspy.Module.load(self, path)
