# src/aee/ingestion/parsers.py
"""Document parsers for AutoEvoExtractor."""

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from aee.infrastructure.config.settings import GeminiParserConfig, MarkerConfig

# Project
from aee.infrastructure.parsers.base import BaseParser
from aee.infrastructure.parsers.marker_config import (
    get_custom_processors,
    get_marker_config_dict,
    get_torch_device,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Gemini PDF-to-Markdown conversion prompt for chemistry papers
GEMINI_PDF_TO_MD_PROMPT = """You are a PDF-to-Markdown converter for chemistry papers.

Convert the PDF content into Markdown by copying the text as faithfully as possible.
The conversion should be mechanical. Avoid summarizing, interpreting, or rewriting the content.
Return only the converted Markdown.
If the document is long, continue converting sequentially until the entire document is processed.
Prioritize completeness of extracted text.

---
# Document coverage
Convert the document from beginning to end.
Conversion starts at the first page of the PDF and continues until the final page of the PDF.

Include:
• main text
• appendices
• supplementary information
• supporting information
• notes appearing after the references section

Content may sometimes look repetitive. Keep it in the output.
The references section is not the end of the document.
Maintain the original reading order.

For two-column layouts:
read the left column first, then the right column.
Ignore page numbers, headers, footers, and page break markers.

---
# Scientific characters
Preserve scientific characters exactly as they appear.

Examples include:
• Greek letters (α, β, γ, Δ, λ, μ, π, Ω)
• mathematical symbols (±, ×, ≤, ≥, ∑, ∫, √, ∞)
• chemical radicals (•)
• minus sign (−)
• degree symbol (°)
• micro symbol (μ)

Keep these characters in their original form rather than converting them to ASCII alternatives.

---
# Chemical and mathematical notation
Subscripts and superscripts follow these rules.

Outside LaTeX:
use HTML tags `<sub>` and `<sup>`

Inside LaTeX expressions:
use `_{} and ^{}`

Chemical formulas and reaction notation should remain exactly as written.

---
# Mathematical expressions
Inline math:  $...$
Block math:  $$...$$
Keep formulas intact rather than splitting them into multiple blocks.
Units remain unchanged (μM, mM, °C, etc).

---
# Citations
Keep citation markers as written: [1] [2] [3]
They remain within the paragraph where they appear.
If a bibliography section appears at the end, it may be omitted.
Content that appears after the references heading (appendix or supplementary sections) should still be included.

---
# Tables
Tables are mandatory structural elements.
Every table must be converted into HTML inside Markdown:

<table>
...
</table>

Preserve exactly:
• column order
• row order
• headers
• colspan / rowspan

Table captions and notes should always be kept, even if the same data appears in the main text.

---

# Figures and visual anchors

Images themselves can be skipped.
Figure captions, numbers, descriptive text, and notes should always be included as text.
Even if the same information appears elsewhere in the main text, keep the figure captions in their original location.
Treat each figure and its caption as a distinct block that must appear in the output.

For every actual numbered visual object caption, insert a stable HTML comment anchor immediately before the caption.

Use this format:

<!-- AEE_VISUAL_ANCHOR: <normalized_visual_id> -->

Then preserve the caption text exactly as written.

Examples:
• Fig. 1 → <!-- AEE_VISUAL_ANCHOR: main_fig_1 -->
• Figure 2 → <!-- AEE_VISUAL_ANCHOR: main_fig_2 -->
• Scheme 1 → <!-- AEE_VISUAL_ANCHOR: main_scheme_1 -->
• Fig. S1 → <!-- AEE_VISUAL_ANCHOR: si_fig_s1 -->
• Figure S2 → <!-- AEE_VISUAL_ANCHOR: si_fig_s2 -->
• Scheme S3 → <!-- AEE_VISUAL_ANCHOR: si_scheme_s3 -->
• Supplementary Figure S4 → <!-- AEE_VISUAL_ANCHOR: si_fig_s4 -->

Normalization rules:
• use lowercase
• use underscores
• remove punctuation
• preserve the figure number or supplementary label
• use prefix `main_` for main article figures
• use prefix `si_` for Supporting Information / Supplementary figures
• use `fig`, `scheme` according to the label

Do not create anchors for ordinary textual references such as:
• “as shown in Fig. 1”
• “see Fig. S2”
• “Figures 3 and 4 demonstrate...”

Create anchors only for actual figure caption blocks or actual visual-object locations.

If a figure has multiple panels, create only one anchor for the whole figure, not one anchor per panel.

If a visual object has no caption but has a clearly visible label such as “Fig. 3” or “Scheme 2” at the visual location, insert the anchor at that visual location and preserve any visible label text if present.

If a visual object has no clear caption and no clear label, do not invent an anchor.

If you are unsure whether something is an actual caption or only a textual reference, preserve the text but do not add an anchor.

Do not insert chart-to-table conversions.
Do not insert extracted numerical data from figures.
Do not insert AEE_CHART_TABLE blocks. These will be added later by another pipeline stage.

---
# Output format
Return the converted document as Markdown containing headings, paragraphs, formulas, tables, figure captions, and AEE visual anchor comments.

AEE visual anchor comments are part of the converted Markdown and should be included exactly in this form:

<!-- AEE_VISUAL_ANCHOR: <normalized_visual_id> -->

Do not include explanations or additional commentary.
Do not wrap the output in code fences."""


class MarkerParser(BaseParser):
    """Parser using Marker library with detailed configuration.

    This parser uses the configuration from marker_config.py which contains
    ~70 parameters optimized for scientific chemistry PDF extraction.

    The MarkerConfig from settings is still accepted for backward compatibility,
    but the detailed settings come from the marker_config module.
    """

    def __init__(self, config: MarkerConfig):
        """Initialize the Marker parser.

        Args:
            config: Configuration for the parser. Required.
                Note: Only used for backward compatibility. Detailed settings
                are loaded from marker_config.py.

        Raises:
            ValueError: If config is None.
        """
        # Lazy import — marker is only needed when MarkerParser is instantiated
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        if config is None:
            raise ValueError("Configuration object is required for MarkerParser")
        self.cfg = config

        # Get device and config from marker_config module
        torch_device = get_torch_device()
        config_dict = get_marker_config_dict()

        logger.info(f"Initializing Marker on {torch_device}...")
        logger.info(f"LLM enabled: {config_dict.get('use_llm', False)}")
        logger.info(
            f"OCR settings: force_ocr={config_dict.get('force_ocr')}, "
            f"strip_existing_ocr={config_dict.get('strip_existing_ocr')}"
        )

        # Create config parser and converter
        config_parser = ConfigParser(config_dict)

        self.converter = PdfConverter(
            artifact_dict=create_model_dict(device=torch_device),
            config=config_parser.generate_config_dict(),
            processor_list=get_custom_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
            if config_dict.get("use_llm")
            else None,
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
                getattr(rendered, "markdown", None)
                or getattr(rendered, "text", None)
                or str(rendered)
            )

            return text

        except Exception as e:
            logger.error(f"Marker parsing failed for {path.name}: {str(e)}")
            raise


class GeminiParser(BaseParser):
    """Parser using Google Gemini API for PDF-to-Markdown conversion."""

    def __init__(self, config: GeminiParserConfig):
        """Initialize the Gemini parser.

        Args:
            config: Configuration for the Gemini parser. Required.

        Raises:
            ValueError: If config is None or API key is not set.
        """
        if config is None:
            raise ValueError("Configuration object is required for GeminiParser")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable must be set in .env file. "
                "Add 'GEMINI_API_KEY=your_key' to your .env file."
            )

        self.cfg = config
        self.api_key = api_key

        # Import here to avoid dependency when not using Gemini
        from google import genai
        from google.genai import types

        self.client = genai.Client()
        self.types = types

        logger.info(f"Initializing Gemini parser with model: {self.cfg.model_name}")

    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a PDF file using Gemini API with retry logic for network errors.

        Args:
            file_path: Path to the PDF file.

        Returns:
            str: Markdown text content.

        Raises:
            Exception: If parsing fails after all retry attempts.
        """
        path = Path(file_path)

        # Retry loop for network errors
        for attempt in range(self.cfg.max_retries):
            try:
                logger.info(f"Gemini processing: {path.name} (attempt {attempt + 1}/{self.cfg.max_retries})")
                return self._do_parse(path)
            except Exception as e:
                error_msg = str(e)
                # Check if this is a retryable network error
                retryable_errors = ["disconnected", "connection", "timeout", "network", "503", "504"]
                is_retryable = any(err in error_msg.lower() for err in retryable_errors)

                if attempt < self.cfg.max_retries - 1 and is_retryable:
                    delay = 10.0 * (attempt + 1)  # Progressive delay: 10s, 20s, 30s
                    logger.warning(
                        f"Network error for {path.name}: {error_msg}. "
                        f"Retrying in {delay}s... ({attempt + 1}/{self.cfg.max_retries})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini parsing failed for {path.name} after {attempt + 1} attempts: {error_msg}")
                    raise

        raise RuntimeError(
            f"Gemini parsing failed for {path.name} after exhausting all retry attempts"
        )

    def _do_parse(self, path: Path) -> str:
        """Internal method to perform actual Gemini parsing.

        Args:
            path: Path to the PDF file.

        Returns:
            str: Markdown text content.
        """
        from google.genai import types

        uploaded_file = None

        try:
            # Upload file to Google servers
            logger.info("Uploading file to Google server...")
            uploaded_file = self.client.files.upload(file=str(path))

            # Wait for file processing
            logger.info("Waiting for file to be ready...")
            while (
                uploaded_file.state is not None
                and uploaded_file.state.name == "PROCESSING"
            ):
                logger.info(".")
                time.sleep(3)
                if uploaded_file.name is None:
                    raise RuntimeError("Uploaded file has no name")
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state is not None and uploaded_file.state.name == "FAILED":
                raise RuntimeError(
                    f"Failed to process file {path.name} on Google server"
                )

            # Generate Markdown content using streaming
            logger.info("Generating Markdown (streaming mode)...")

            # Build safety settings
            safety_settings = []
            if self.cfg.safety_settings:
                safety_settings = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]

            response_stream = self.client.models.generate_content_stream(
                model=self.cfg.model_name,
                contents=[uploaded_file, GEMINI_PDF_TO_MD_PROMPT],  # type: ignore[arg-type]
                config=types.GenerateContentConfig(
                    safety_settings=safety_settings,
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )

            # Stream response to string
            markdown_content = []
            for chunk in response_stream:
                if chunk.text:
                    markdown_content.append(chunk.text)

            result = "".join(markdown_content)

            if not result:
                logger.warning(f"Gemini returned empty response for {path.name}")

            return result

        finally:
            # Clean up uploaded file
            if uploaded_file and uploaded_file.name:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info("Temporary file deleted from server")
                except Exception:
                    pass  # Ignore cleanup errors


def get_parser(parser_name: str, config: Any = None) -> BaseParser:
    """Factory function to get a parser instance by name.

    Args:
        parser_name: Name of the parser ("marker" or "gemini").
        config: Configuration for the parser (MarkerConfig or GeminiParserConfig).

    Returns:
        Parser instance.

    Raises:
        ValueError: If parser_name is not recognized or config is invalid.
    """
    from aee.infrastructure.config.settings import GeminiParserConfig, MarkerConfig

    parser_name = parser_name.lower()

    if parser_name == "marker":
        if config is None or not isinstance(config, MarkerConfig):
            raise ValueError(
                f"MarkerParser requires MarkerConfig, got {type(config).__name__}"
            )
        return MarkerParser(config)

    elif parser_name == "gemini":
        if config is None or not isinstance(config, GeminiParserConfig):
            raise ValueError(
                f"GeminiParser requires GeminiParserConfig, got {type(config).__name__}"
            )
        return GeminiParser(config)

    else:
        raise ValueError(
            f"Unknown parser: {parser_name}. Available parsers: 'marker', 'gemini'"
        )
