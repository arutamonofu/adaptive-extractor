"""Domain layer agents module.

This module provides abstract interfaces for agents in the domain layer,
enabling dependency inversion for the application layer.
"""

from aee.domain.agents.base import BaseAgent

__all__ = ["BaseAgent"]
