"""Task-agnostic extraction agent for AutoEvoExtractor."""

from typing import Type

import dspy

from aee.domain.agents.base import BaseAgent


class UniversalExtractorMeta(type(BaseAgent), type(dspy.Module)):  # type: ignore[misc]
    """Metaclass for UniversalExtractor to resolve metaclass conflict."""
    pass


class UniversalExtractor(BaseAgent, dspy.Module, metaclass=UniversalExtractorMeta):
    """Task-agnostic extraction agent.

    Wraps a specific task signature (e.g., Nanozymes) with Chain-of-Thought reasoning.

    Performance note:
        ChainOfThought generates reasoning tokens BEFORE the final answer, which
        can add 500–1500 tokens of output. With local Transformers inference this
        means 1.5–4 minutes of extra generation time per document.

        To disable CoT and use direct prediction (faster, but potentially lower quality):
            Replace dspy.ChainOfThought(signature_class) with dspy.Predict(signature_class)

        The max_new_tokens config limits total output; ensure it is high enough
        to accommodate both reasoning and the answer (e.g., 1024+ with CoT,
        512+ without).
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
