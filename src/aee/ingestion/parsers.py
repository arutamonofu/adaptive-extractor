# src/aee/ingestion/parsers.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Union

# Third-party
import fitz  # PyMuPDF
import pdfplumber

# Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc.document import (
    TableItem, TextItem, SectionHeaderItem, ListItem, GroupItem
)
from docling_core.types.doc.labels import DocItemLabel

# Marker
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Project
from aee.ingestion.base import BaseParser
from aee.ingestion.cleaning import TextCleaner
from aee.core.types import ProcessedDocument, DocumentMetadata

logger = logging.getLogger(__name__)

# --- 1. Docling Parser (Hybrid: HTML Tables + Clean Markdown Text) ---

class DoclingParser(BaseParser):
    """
    State-of-the-art parser using IBM Docling.
    Strategy: Manual Hybrid Export.
    - Tables -> HTML (Preserves rowspans/colspans critical for chemistry).
    - Text -> Markdown (Cleaned via TextCleaner).
    """

    def __init__(self, num_threads: int = 4):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True 
        )
        
        # Force CPU to avoid OOM and ensure stability on Free Tier/Standard VMs
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=AcceleratorDevice.CPU
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _build_hybrid_content(self, doc) -> str:
        """Constructs document content: HTML for tables, Cleaned MD for text."""
        output = []
        
        for item, _ in doc.iterate_items():
            
            # 1. Tables: Export as HTML to preserve structure
            if isinstance(item, TableItem):
                # Pass 'doc' to resolve references if necessary
                if html := item.export_to_html(doc=doc):
                    output.append(f"\n{html}\n")
            
            # 2. Headers: Convert to MD & Clean
            elif isinstance(item, SectionHeaderItem):
                if text := TextCleaner.clean_docling_markdown(item.text):
                    prefix = "#" * getattr(item, "level", 1)
                    output.append(f"\n{prefix} {text}\n")
            
            # 3. Lists: Convert to MD & Clean
            elif isinstance(item, ListItem):
                if text := TextCleaner.clean_docling_markdown(item.text):
                    marker = "1." if item.enumerated else "-"
                    output.append(f"{marker} {text}")
            
            # 4. Text: Filter furniture, Clean & Append
            elif isinstance(item, TextItem):
                if item.label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}:
                    continue
                if text := TextCleaner.clean_docling_markdown(item.text):
                    output.append(text)

        return "\n\n".join(output)

    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Docling processing: {path.name} (Strategy: Hybrid)")
        
        result = self.converter.convert(path)
        hybrid_text = self._build_hybrid_content(result.document)
        
        return ProcessedDocument(
            text_content=hybrid_text,
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=len(result.document.pages) if hasattr(result.document, "pages") else None,
                extra={"parser": "Docling", "strategy": "hybrid_manual"}
            )
        )

# --- 2. Marker Parser ---

class MarkerParser(BaseParser):
    """GPU-optimized parser using Marker."""

    def __init__(self, device: str = "cpu"):
        logger.info(f"Initializing Marker on {device}...")
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(device=device)
        )

    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Marker processing: {path.name}")
        rendered = self.converter(str(path))
        
        # Handle API variations
        text = getattr(rendered, "markdown", None) or getattr(rendered, "text", str(rendered))
        meta = getattr(rendered, "metadata", {})

        return ProcessedDocument(
            text_content=text,
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=meta.get("page_count"),
                extra=meta
            )
        )

# --- 3. PyMuPDF Parser ---

class PyMuPDFParser(BaseParser):
    """Fast, layout-preserving heuristic parser."""

    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"PyMuPDF processing: {path.name}")
        doc = fitz.open(path)
        full_text = []

        for i, page in enumerate(doc):
            # Extract text blocks, filter out noise
            blocks = page.get_text("blocks")
            page_text = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
            
            full_text.append(f"## Page {i + 1}\n\n" + "\n\n".join(page_text))

        return ProcessedDocument(
            text_content="\n\n".join(full_text),
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=doc.page_count,
                extra={"producer": doc.metadata.get("producer", "")}
            )
        )

# --- 4. PDFPlumber Parser ---

class PlumberParser(BaseParser):
    """Table-focused parser."""

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Simple list-of-lists to Markdown conversion."""
        if not table: return ""
        # Filter None
        rows = [[str(c) if c is not None else "" for c in r] for r in table]
        if not rows: return ""

        headers = rows[0]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        lines.extend(["| " + " | ".join(r) + " |" for r in rows[1:]])
        return "\n".join(lines)

    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Plumber processing: {path.name}")
        content_parts = []
        
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(layout=False) or ""
                tables = [self._table_to_markdown(t) for t in page.extract_tables()]
                tables = [t for t in tables if t]

                page_content = f"## Page {i + 1}\n\n{text}\n"
                if tables:
                    page_content += "\n### Tables:\n" + "\n\n".join(tables)
                
                content_parts.append(page_content)

        return ProcessedDocument(
            text_content="\n\n".join(content_parts),
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=len(pdf.pages)
            )
        )
    
# --- 5. NanoMiner Legacy Parser ---

class NanoPlumberParser(BaseParser):
    """
    Legacy parser from the nanoMINER project. 
    Uses specific character alignment logic for scientific texts based on pdfplumber.
    """

    def _align_chars(self, text: str, chars: List[Dict[str, Any]]) -> str:
        """Aligns raw text layout with character objects to preserve flow."""
        # Normalize text: collapse spaces, keep line breaks
        lines = [re.sub(" +", " ", line.strip()) for line in text.split("\n")]
        clean_text = "\n".join(l for l in lines if l) # Skip empty lines

        # Allowed alphabet for alignment
        alphabet = "zxcvbasdfgqwertnmhjklyuiop1234567890-+=/*©()[]"
        
        result = []
        t_idx, c_idx = 0, 0
        
        while t_idx < len(clean_text) and c_idx < len(chars):
            t_char = clean_text[t_idx]
            c_char_obj = chars[c_idx]
            c_text = c_char_obj.get("text", "")

            if t_char in (" ", "\n"):
                result.append(t_char)
                t_idx += 1
            elif c_text.lower() not in alphabet:
                c_idx += 1
            elif t_char.lower() not in alphabet:
                result.append(t_char)
                t_idx += 1
            elif t_char.lower() != c_text.lower():
                # Mismatch: prefer text stream
                result.append(t_char)
                t_idx += 1
            else:
                # Match: use char object
                result.append(c_text)
                t_idx += 1
                c_idx += 1
        
        return "".join(result) + "\n"

    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        path = Path(file_path)
        logger.info(f"NanoMiner processing: {path.name}")
        
        extracted_text = ""
        page_count = 0
        
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text(
                    x_tolerance=2, y_tolerance=3, layout=True, use_text_flow=True
                )
                if text:
                    extracted_text += self._align_chars(text, page.chars)

        # Truncate References section to save LLM context window
        extracted_text = re.sub(r"References.*", "References", extracted_text, flags=re.DOTALL)

        return ProcessedDocument(
            text_content=extracted_text,
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=page_count
            )
        )