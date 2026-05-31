"""Gemini PDF-to-Markdown parser for Adaptive Extractor."""

import logging
import os
import time
from pathlib import Path
from typing import Union

from ae.core.config.settings import GeminiParserConfig
from ae.ingestion.base_parser import BaseParser

logger = logging.getLogger(__name__)

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

<!-- AE_VISUAL_ANCHOR: <normalized_visual_id> -->

Then preserve the caption text exactly as written.

Examples:
• Fig. 1 → <!-- AE_VISUAL_ANCHOR: main_fig_1 -->
• Figure 2 → <!-- AE_VISUAL_ANCHOR: main_fig_2 -->
• Scheme 1 → <!-- AE_VISUAL_ANCHOR: main_scheme_1 -->
• Fig. S1 → <!-- AE_VISUAL_ANCHOR: si_fig_s1 -->
• Figure S2 → <!-- AE_VISUAL_ANCHOR: si_fig_s2 -->
• Scheme S3 → <!-- AE_VISUAL_ANCHOR: si_scheme_s3 -->
• Supplementary Figure S4 → <!-- AE_VISUAL_ANCHOR: si_fig_s4 -->

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

If a visual object has no caption but has a clearly visible label such as “Fig. 3” or “Scheme 2”
at the visual location, insert the anchor at that visual location and preserve any visible label text if present.

If a visual object has no clear caption and no clear label, do not invent an anchor.

If you are unsure whether something is an actual caption or only a textual reference,
preserve the text but do not add an anchor.

Do not insert chart-to-table conversions.
Do not insert extracted numerical data from figures.
Do not insert AE_CHART_TABLE blocks. These will be added later by another pipeline stage.

---
# Output format
Return the converted document as Markdown containing headings, paragraphs, formulas, tables,
figure captions, and AE visual anchor comments.

AE visual anchor comments are part of the converted Markdown and should be included exactly in this form:

<!-- AE_VISUAL_ANCHOR: <normalized_visual_id> -->

Do not include explanations or additional commentary.
Do not wrap the output in code fences."""


class GeminiParser(BaseParser):
    """Parser using Google Gemini API for PDF-to-Markdown conversion."""

    def __init__(self, config: GeminiParserConfig):
        """Initialize the Gemini parser.

        Args:
            config: Configuration for the Gemini parser. Required.
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
        """Parse a PDF file using Gemini API with retry logic for network errors."""
        path = Path(file_path)

        for attempt in range(self.cfg.max_retries):
            try:
                logger.info(f"Gemini processing: {path.name} (attempt {attempt + 1}/{self.cfg.max_retries})")
                return self._do_parse(path)
            except Exception as e:
                error_msg = str(e)
                retryable_errors = ["disconnected", "connection", "timeout", "network", "503", "504"]
                is_retryable = any(err in error_msg.lower() for err in retryable_errors)

                if attempt < self.cfg.max_retries - 1 and is_retryable:
                    delay = 10.0 * (attempt + 1)
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
        """Internal method to perform actual Gemini parsing."""
        from google.genai import types

        uploaded_file = None

        try:
            logger.info("Uploading file to Google server...")
            uploaded_file = self.client.files.upload(file=str(path))

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

            logger.info("Generating Markdown (streaming mode)...")

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
                    thinking_config=types.ThinkingConfig(thinking_level="low"),  # type: ignore[arg-type]
                ),
            )

            markdown_content = []
            for chunk in response_stream:
                if chunk.text:
                    markdown_content.append(chunk.text)

            result = "".join(markdown_content)

            if not result:
                logger.warning(f"Gemini returned empty response for {path.name}")

            return result

        finally:
            if uploaded_file and uploaded_file.name:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info("Temporary file deleted from server")
                except Exception:
                    pass
