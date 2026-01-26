# src/aee/ingestion/parsers.py

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import (
    TableItem, TextItem, SectionHeaderItem, ListItem
)

# Marker
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Project
from aee.ingestion.base import BaseParser
from aee.ingestion.cleaning import TextCleaner
from aee.core.types import ProcessedDocument, DocumentMetadata
from aee.core.config import settings

logger = logging.getLogger(__name__)

class DoclingParser(BaseParser):
    def __init__(self, config: Optional[Any] = None):
        self.cfg = config or settings.parsing.docling
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.cfg.do_ocr
        pipeline_options.do_table_structure = self.cfg.do_table_structure
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True 
        )
        
        device_type = AcceleratorDevice.CPU
        if self.cfg.device == "cuda":
            device_type = AcceleratorDevice.CUDA
        elif self.cfg.device == "mps":
            device_type = AcceleratorDevice.MPS

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=self.cfg.num_threads,
            device=device_type
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _build_hybrid_content(self, doc) -> str:
        output = []
        for item, _ in doc.iterate_items():
            if isinstance(item, TableItem):
                if html := item.export_to_html(doc=doc):
                    output.append(f"\n{html}\n")
            elif isinstance(item, SectionHeaderItem):
                if text := TextCleaner.clean_docling_markdown(item.text):
                    prefix = "#" * getattr(item, "level", 1)
                    output.append(f"\n{prefix} {text}\n")
            elif isinstance(item, ListItem):
                if text := TextCleaner.clean_docling_markdown(item.text):
                    marker = "1." if item.enumerated else "-"
                    output.append(f"{marker} {text}")
            elif isinstance(item, TextItem):
                if item.label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}:
                    continue
                if text := TextCleaner.clean_docling_markdown(item.text):
                    output.append(text)
        return "\n\n".join(output)

    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        path = Path(file_path)
        logger.info(f"Docling processing: {path.name} (device: {self.cfg.device})")
        
        result = self.converter.convert(path)
        hybrid_text = self._build_hybrid_content(result.document)
        
        return ProcessedDocument(
            text_content=hybrid_text,
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=len(result.document.pages) if hasattr(result.document, "pages") else None,
                extra={"parser": "Docling", "device": self.cfg.device}
            )
        )

class MarkerParser(BaseParser):
    def __init__(self, config: Optional[Any] = None):
        self.cfg = config or settings.parsing.marker
        logger.info(f"Initializing Marker on {self.cfg.device}...")
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(device=self.cfg.device)
        )

    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        path = Path(file_path)
        logger.info(f"Marker processing: {path.name}")
        rendered = self.converter(str(path))
        
        text = getattr(rendered, "markdown", None) or getattr(rendered, "text", str(rendered))
        meta = getattr(rendered, "metadata", {})

        return ProcessedDocument(
            text_content=text,
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=meta.get("page_count"),
                extra={"parser": "Marker", **meta}
            )
        )