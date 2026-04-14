"""CLI commands for AutoEvoExtractor.

This module provides command-line interfaces for all major operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .extract import extract_command
    from .optimize import optimize_command
    from .parse import parse_command


def __getattr__(name: str):
    """Lazy loading — CLI modules pull in the full application stack."""
    if name == "extract_command":
        from .extract import extract_command

        return extract_command
    if name == "optimize_command":
        from .optimize import optimize_command

        return optimize_command
    if name == "parse_command":
        from .parse import parse_command

        return parse_command
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    "extract_command",
    "optimize_command",
    "parse_command",
]
