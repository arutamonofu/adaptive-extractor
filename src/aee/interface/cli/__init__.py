"""CLI commands for AutoEvoExtractor.

This module provides command-line interfaces for all major operations.
"""

from .optimize import optimize_command
from .parse import parse_command
from .predict import predict_command

__all__ = [
    "optimize_command",
    "parse_command",
    "predict_command",
]
