"""Parser factory for Adaptive Extractor."""

from typing import Any

from ae.core.config.settings import (
    AEVisualParserConfig,
    GeminiParserConfig,
)
from ae.ingestion.base_parser import BaseParser
from ae.ingestion.gemini_parser import GeminiParser
from ae.ingestion.visual_parser import AEVisualParser


def get_parser(parser_name: str, config: Any = None) -> BaseParser:
    """Factory function to get a parser instance by name.

    Args:
        parser_name: Name of the parser ("marker", "gemini", or "gemini_visual").
        config: Configuration for the parser (MarkerConfig, GeminiParserConfig, or AEVisualParserConfig).

    Returns:
        Parser instance.

    Raises:
        ValueError: If parser_name is not recognized or config is invalid.
    """
    parser_name = parser_name.lower()

    if parser_name == "gemini":
        if config is None or not isinstance(config, GeminiParserConfig):
            raise ValueError(
                f"GeminiParser requires GeminiParserConfig, got {type(config).__name__}"
            )
        return GeminiParser(config)

    elif parser_name == "gemini_visual":
        if config is None or not isinstance(config, AEVisualParserConfig):
            raise ValueError(
                f"AEVisualParser requires AEVisualParserConfig, got {type(config).__name__}"
            )
        return AEVisualParser(config)

    else:
        raise ValueError(
            f"Unknown parser: {parser_name}. Available parsers: 'gemini', 'gemini_visual'"
        )
