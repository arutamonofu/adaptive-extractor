# src/aee/ingestion/parsers.py

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

# Third-party imports (assuming installed)
import fitz  # PyMuPDF
import pdfplumber
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# Marker imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from aee.ingestion.base import BaseParser
from aee.core.types import ProcessedDocument, DocumentMetadata

logger = logging.getLogger(__name__)

# --- 1. Docling Parser ---

class DoclingParser(BaseParser):
    """Parser based on IBM Docling (optimized for CPU)."""

    def __init__(self, num_threads: int = 4):
        options = PdfPipelineOptions()
        options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=AcceleratorDevice.CPU
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def parse(self, file_path: str | Path) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Docling processing: {path.name}")
        result = self.converter.convert(path)
        
        return ProcessedDocument(
            text_content=result.document.export_to_markdown(),
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=len(result.document.pages) if hasattr(result.document, "pages") else None
            )
        )


# --- 2. Marker Parser ---

class MarkerParser(BaseParser):
    """Parser based on Marker (requires GPU for reasonable speed)."""

    def __init__(self, device: str = "cpu"):
        logger.info(f"Initializing Marker on {device}...")
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(device=device)
        )

    def parse(self, file_path: str | Path) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Marker processing: {path.name}")
        rendered = self.converter(str(path))
        
        # Handle different Marker API versions
        text = getattr(rendered, "markdown", None) or getattr(rendered, "text", str(rendered))
        meta_raw = getattr(rendered, "metadata", {})

        return ProcessedDocument(
            text_content=text,
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=meta_raw.get("page_count"),
                extra=meta_raw
            )
        )


# --- 3. PyMuPDF Parser ---

class PyMuPDFParser(BaseParser):
    """Fast parser using PyMuPDF (fitz). Preserves layout via heuristics."""

    def parse(self, file_path: str | Path) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"PyMuPDF processing: {path.name}")
        doc = fitz.open(path)
        full_text = []

        for i, page in enumerate(doc):
            # Extract blocks: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            # Filter text blocks (type 0) and strip content
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
    """Parser using pdfplumber. Good for table extraction."""

    def _table_to_markdown(self, table: List[List[Union[str, None]]]) -> str:
        """Converts a raw list-of-lists table to a Markdown string."""
        if not table:
            return ""
        # Clean None values
        clean_rows = [[str(cell) if cell is not None else "" for cell in row] for row in table]
        if not clean_rows:
            return ""

        headers = clean_rows[0]
        # Construct MD table
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        lines.extend(["| " + " | ".join(row) + " |" for row in clean_rows[1:]])
        return "\n".join(lines)

    def parse(self, file_path: str | Path) -> ProcessedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Plumber processing: {path.name}")
        text_parts = []
        md_tables = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract tables
                for table in page.extract_tables():
                    if md_table := self._table_to_markdown(table):
                        md_tables.append(md_table)

                # Extract text
                text = page.extract_text(layout=False) or ""
                
                # Compose page content
                page_content = f"## Page {i + 1}\n\n{text}\n\n"
                if md_tables:
                    # Append tables found on this page (simple heuristic)
                    # Note: Ideally tables should be inserted where they occur, 
                    # but plumber separates these streams.
                    page_content += "### Extracted Tables:\n\n" + "\n\n".join(md_tables[-len(page.extract_tables()):])
                
                text_parts.append(page_content)

            return ProcessedDocument(
                text_content="\n\n".join(text_parts),
                tables=md_tables,
                metadata=DocumentMetadata(
                    source_path=str(path.absolute()),
                    filename=path.name,
                    page_count=len(pdf.pages),
                    extra=dict(pdf.metadata)
                )
            )


# --- 5. NanoMiner Legacy Parser ---

class NanoPlumberParser(BaseParser):
    """
    Legacy parser from nanoMINER project. 
    Uses specific character alignment logic for scientific texts.
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
                # Match: use char object (potentially richer info, though here just text)
                result.append(c_text)
                t_idx += 1
                c_idx += 1
        
        return "".join(result) + "\n"

    def parse(self, file_path: str | Path) -> ProcessedDocument:
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

        # Truncate References to save context window
        extracted_text = re.sub(r"References.*", "References", extracted_text, flags=re.DOTALL)

        return ProcessedDocument(
            text_content=extracted_text,
            metadata=DocumentMetadata(
                source_path=str(path.absolute()),
                filename=path.name,
                page_count=page_count
            )
        )