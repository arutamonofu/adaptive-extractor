"""Parser factory for Adaptive Extractor."""

from typing import Any, Dict, Tuple, Type

from ae.core.config.settings import IngestionConfig
from ae.ingestion.parsers.base import BaseParser
from ae.ingestion.parsers.mineru.parser import MinerUParser


class ParserRegistry:
    """Registry for document parsers (Open-Closed Principle compliant)."""

    def __init__(self) -> None:
        self._parsers: Dict[str, Tuple[Type[BaseParser], Type]] = {}

    def register(self, name: str, parser_class: Type[BaseParser], config_class: Type) -> None:
        """Register a parser class with its config class."""
        self._parsers[name.lower()] = (parser_class, config_class)

    def get_parser(self, name: str, config: Any = None) -> BaseParser:
        """Get an instance of the registered parser."""
        name_lower = name.lower()
        if name_lower not in self._parsers:
            raise ValueError(
                f"Unknown parser: {name}. Available parsers: {sorted(list(self._parsers.keys()))}"
            )

        parser_class, config_class = self._parsers[name_lower]
        if config is None or not isinstance(config, config_class):
            raise ValueError(
                f"{parser_class.__name__} requires {config_class.__name__}, got {type(config).__name__}"
            )

        return parser_class(config)


# Global registry singleton
PARSER_REGISTRY = ParserRegistry()

# Register default parsers
PARSER_REGISTRY.register("mineru", MinerUParser, IngestionConfig)


def get_parser(parser_name: str, config: Any = None) -> BaseParser:
    """Factory function to get a parser instance by name.

    Args:
        parser_name: Name of the parser (e.g., "mineru").
        config: Configuration for the parser (e.g., IngestionConfig).

    Returns:
        Parser instance.

    Raises:
        ValueError: If parser_name is not recognized or config is invalid.
    """
    return PARSER_REGISTRY.get_parser(parser_name, config)
