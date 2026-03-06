# src/aee/ingestion/parsers.py
"""Document parsers for AutoEvoExtractor."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

# Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import (
    TableItem, TextItem, SectionHeaderItem, ListItem
)

# Marker
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Project
from aee.infrastructure.parsers.base import BaseParser
from aee.infrastructure.parsers.cleaning import TextCleaner
from aee.infrastructure.config.settings import MarkerConfig

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Parser using Docling library."""

    def __init__(self, config: Any):
        """Initialize the Docling parser.

        Args:
            config: Configuration for the parser. Required.
        """
        if config is None:
            raise ValueError("Configuration object is required for DoclingParser")
        self.cfg = config
        self.converter = self._create_converter()

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the Docling document converter.

        Returns:
            DocumentConverter: Configured converter instance.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.cfg.do_ocr
        pipeline_options.do_table_structure = self.cfg.do_table_structure
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        # Configure OCR engine (RapidOCR with specified backend)
        if self.cfg.do_ocr:
            ocr_options = RapidOcrOptions(
                backend=self.cfg.ocr_backend,
            )
            pipeline_options.ocr_options = ocr_options
            logger.info(f"RapidOCR configured with backend: {self.cfg.ocr_backend}")

        # Map device string to accelerator device enum
        device_mapping = {
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS
        }
        device_type = device_mapping.get(self.cfg.device, AcceleratorDevice.CPU)

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=self.cfg.num_threads,
            device=device_type
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _build_hybrid_content(self, doc) -> str:
        """Build hybrid content from document items.

        Args:
            doc: Document object.

        Returns:
            str: Hybrid content string.
        """
        output = []
        for item, _ in doc.iterate_items():
            content = self._extract_item_content(item, doc)
            if content:
                output.append(content)
        return "\n\n".join(output)

    def _extract_item_content(self, item, doc) -> Optional[str]:
        """Extract content from a document item based on its type.

        Args:
            item: Document item to extract content from.
            doc: Parent document object.

        Returns:
            str: Extracted content or None if no content.
        """
        if isinstance(item, TableItem):
            return self._extract_table_content(item, doc)
        elif isinstance(item, SectionHeaderItem):
            return self._extract_header_content(item)
        elif isinstance(item, ListItem):
            return self._extract_list_item_content(item)
        elif isinstance(item, TextItem):
            return self._extract_text_content(item)
        return None

    def _extract_table_content(self, item: TableItem, doc) -> Optional[str]:
        """Extract table content as HTML.

        Args:
            item: Table item to extract.
            doc: Parent document object.

        Returns:
            str: HTML representation of the table or None.
        """
        if html := item.export_to_html(doc=doc):
            return f"\n{html}\n"
        return None

    def _extract_header_content(self, item: SectionHeaderItem) -> Optional[str]:
        """Extract section header content as markdown.

        Args:
            item: Section header item to extract.

        Returns:
            str: Markdown header or None.
        """
        if text := TextCleaner.clean_docling_markdown(item.text):
            prefix = "#" * getattr(item, "level", 1)
            return f"\n{prefix} {text}\n"
        return None

    def _extract_list_item_content(self, item: ListItem) -> Optional[str]:
        """Extract list item content as markdown.

        Args:
            item: List item to extract.

        Returns:
            str: Markdown list item or None.
        """
        if text := TextCleaner.clean_docling_markdown(item.text):
            marker = "1." if item.enumerated else "-"
            return f"{marker} {text}"
        return None

    def _extract_text_content(self, item: TextItem) -> Optional[str]:
        """Extract text content, filtering out headers/footers.

        Args:
            item: Text item to extract.

        Returns:
            str: Cleaned text content or None.
        """
        # Skip headers and footers
        if item.label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}:
            return None

        if text := TextCleaner.clean_docling_markdown(item.text):
            return text
        return None

    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a PDF file using Docling.

        Args:
            file_path: Path to the PDF file.

        Returns:
            str: Hybrid markdown text content.

        Raises:
            Exception: If parsing fails.
        """
        path = Path(file_path)
        logger.info(f"Docling processing: {path.name} (device: {self.cfg.device})")

        try:
            result = self.converter.convert(path)
            hybrid_text = self._build_hybrid_content(result.document)
            return hybrid_text

        except Exception as e:
            logger.error(f"Docling parsing failed for {path.name}: {str(e)}")
            raise


class MarkerParser(BaseParser):
    """Parser using Marker library."""

    def __init__(self, config: MarkerConfig):
        """Initialize the Marker parser.

        Args:
            config: Configuration for the parser. Required.

        Raises:
            ValueError: If config is None.
        """
        if config is None:
            raise ValueError("Configuration object is required for MarkerParser")
        self.cfg = config
        logger.info(f"Initializing Marker on {self.cfg.device}...")
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(device=self.cfg.device)
        )

    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a PDF file using Marker.

        Args:
            file_path: Path to the PDF file.

        Returns:
            str: Markdown text content.

        Raises:
            Exception: If parsing fails.
        """
        path = Path(file_path)
        logger.info(f"Marker processing: {path.name}")

        try:
            rendered = self.converter(str(path))

            # Extract text content with fallback chain
            text = (
                getattr(rendered, "markdown", None) or
                getattr(rendered, "text", None) or
                str(rendered)
            )

            return text

        except Exception as e:
            logger.error(f"Marker parsing failed for {path.name}: {str(e)}")
            raise


def get_parser(parser_name: str, config: Any = None) -> BaseParser:
    """Factory function to get a parser instance by name.

    Args:
        parser_name: Name of the parser ("docling" or "marker").
        config: Optional configuration for the parser.

    Returns:
        Parser instance.

    Raises:
        ValueError: If parser_name is not recognized.
    """
    parser_name = parser_name.lower()

    if parser_name == "docling":
        return DoclingParser(config)
    elif parser_name == "marker":
        return MarkerParser(config)
    else:
        raise ValueError(
            f"Unknown parser: {parser_name}. "
            f"Available parsers: 'docling', 'marker'"
        )
