"""Document parser module exposing the registry factory and base interface."""

from .base import BaseParser
from .registry import PARSER_REGISTRY, get_parser

__all__ = ["BaseParser", "get_parser", "PARSER_REGISTRY"]
