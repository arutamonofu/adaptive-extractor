# src/aee/agents/extractor.py

import dspy
from typing import Type

class UniversalExtractor(dspy.Module):
    """
    A task-agnostic extraction agent powered by DSPy.
    
    It wraps a specific task signature (e.g., Nanozymes) in a Chain-of-Thought 
    reasoning module to improve extraction accuracy and handle complex logic.
    """

    def __init__(self, signature_class: Type[dspy.Signature]):
        """
        Args:
            signature_class: The DSPy signature defining input/output fields.
        """
        super().__init__()
        self.prog = dspy.ChainOfThought(signature_class)

    def forward(self, document_text: str) -> dspy.Prediction:
        """
        Executes the extraction pipeline.

        Args:
            document_text: The full text content of the document.

        Returns:
            A DSPy Prediction object containing:
            - reasoning: The model's chain of thought.
            - extracted_data: The structured output (Pydantic model).
        """
        return self.prog(document_text=document_text)