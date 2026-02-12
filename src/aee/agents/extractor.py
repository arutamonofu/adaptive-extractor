# src/aee/agents/extractor.py
"""Task-agnostic extraction agent for AutoEvoExtractor."""

import dspy
from typing import Type

class UniversalExtractor(dspy.Module):
    """Task-agnostic extraction agent.
    
    Wraps a specific task signature (e.g., Nanozymes) with Chain-of-Thought reasoning.
    """

    def __init__(self, signature_class: Type[dspy.Signature]):
        """Initialize the UniversalExtractor.
        
        Args:
            signature_class: The DSPy signature defining input/output fields and instructions.
        """
        super().__init__()
        self.prog = dspy.ChainOfThought(signature_class)

    def forward(self, document_text: str) -> dspy.Prediction:
        """Execute the extraction pipeline.
        
        Args:
            document_text: The full content of the document (Markdown/HTML hybrid).
            
        Returns:
            dspy.Prediction: Contains 'reasoning' (str) and 'extracted_data' (Pydantic model).
        """
        return self.prog(document_text=document_text)