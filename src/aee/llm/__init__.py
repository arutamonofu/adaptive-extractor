# src/aee/llm/__init__.py
"""LLM provider module for AutoEvoExtractor."""

from aee.llm.provider import setup_student, setup_teacher, create_lm

__all__ = ["setup_student", "setup_teacher", "create_lm"]