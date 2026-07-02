# src/ae/reverse_engineering/__init__.py
"""Reverse Engineering (RE) data annotation rules module.

This module contains the pipeline for automatically discovering, consolidatating,
and generalizing extraction guidelines/rules from existing ground truth annotations
using a teacher LLM.
"""

from .use_case import (
    ReverseEngineeringUseCase,
    ReverseEngineeringRequest,
    ReverseEngineeringResponse,
)

__all__ = [
    "ReverseEngineeringUseCase",
    "ReverseEngineeringRequest",
    "ReverseEngineeringResponse",
]
