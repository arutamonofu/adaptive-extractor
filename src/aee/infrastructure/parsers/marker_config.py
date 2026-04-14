# src/aee/infrastructure/parsers/marker_config.py
"""Marker PDF converter configuration.

This module contains all configuration parameters for the Marker PDF converter,
extracted from research/notebooks/test_marker_settings.ipynb.

These settings are optimized for data extraction from scientific chemistry PDFs
using Qwen2.5-VL as the LLM backend.

Note: Configuration is defined in code (not YAML) for type safety and version control.
To modify these settings, edit this file directly.
"""

from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# CORE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_FORMAT = "markdown"
"""Output format: 'markdown' is the only supported format."""

PAGE_RANGE = None
"""Range of pages to process. None = all pages."""

PAGINATE_OUTPUT = False
"""Whether to paginate the output."""

DEBUG = False
"""Enable debug mode."""

DEBUG_PRINT = False
"""Enable debug printing."""

DISABLE_TQDM = False
"""Disable tqdm progress bars."""


# ═══════════════════════════════════════════════════════════════════════════════
# OCR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

FORCE_OCR = True
"""Force OCR on all pages, even if text is detected."""

STRIP_EXISTING_OCR = True
"""Strip existing OCR data and re-run OCR."""

DISABLE_OCR = False
"""Disable OCR entirely."""

KEEP_CHARS = False
"""Keep character-level bounding boxes."""


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DISABLE_IMAGE_EXTRACTION = True
"""Disable image extraction from PDFs."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM SETTINGS (Global)
# ═══════════════════════════════════════════════════════════════════════════════

USE_LLM = True
"""Enable LLM-based processing for equations, tables, and math blocks."""

REDO_INLINE_MATH = True
"""Re-process inline math expressions with LLM."""


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT BUILDER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DocumentBuilder_lowres_image_dpi = 256
"""DPI for low-resolution image extraction."""

DocumentBuilder_highres_image_dpi = 600
"""DPI for high-resolution image extraction."""


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT BUILDER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LayoutBuilder_layout_batch_size = None
"""Batch size for layout detection."""

LayoutBuilder_force_layout_block = None
"""Force specific layout block type."""

LayoutBuilder_max_expand_frac = 0.04
"""Maximum fraction to expand layout blocks."""

LayoutBuilder_detection_batch_size = None
"""Batch size for layout detection model."""


# ═══════════════════════════════════════════════════════════════════════════════
# LINE BUILDER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LineBuilder_layout_coverage_min_lines = 1
"""Minimum number of lines for layout coverage."""

LineBuilder_layout_coverage_threshold = 0.05
"""Threshold for layout coverage detection."""

LineBuilder_min_document_ocr_threshold = 0.7
"""Minimum OCR threshold for document-level OCR."""

LineBuilder_detection_line_min_confidence = 0.7
"""Minimum confidence for line detection."""

LineBuilder_recognition_batch_size = None
"""Batch size for text recognition."""


# ═══════════════════════════════════════════════════════════════════════════════
# OCR BUILDER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

OcrBuilder_ocr_task_name = "ocr_with_boxes"
"""OCR task name for Surya OCR."""

OcrBuilder_disable_ocr_math = False
"""Disable OCR for math expressions."""

OcrBuilder_drop_repeated_text = True
"""Drop repeated text detected by OCR."""

OcrBuilder_block_mode_intersection_thresh = 0.45
"""Intersection threshold for block mode OCR."""

OcrBuilder_block_mode_max_lines = 8
"""Maximum lines per block in block mode."""

OcrBuilder_block_mode_max_height_frac = 0.3
"""Maximum height fraction for block mode."""


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE BUILDER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

StructureBuilder_gap_threshold = 0.04
"""Threshold for detecting gaps between blocks."""

StructureBuilder_list_gap_threshold = 0.08
"""Threshold for detecting list item gaps."""


# ═══════════════════════════════════════════════════════════════════════════════
# EQUATION PROCESSOR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

EquationProcessor_model_max_length = 2048
"""Maximum sequence length for equation model."""

EquationProcessor_equation_batch_size = None
"""Batch size for equation processing."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM EQUATION PROCESSOR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LLMEquationProcessor_max_concurrency = 1
"""Maximum concurrent LLM requests for equations."""

LLMEquationProcessor_image_expansion_ratio = 0.06
"""Ratio to expand equation images."""

LLMEquationProcessor_min_equation_height = 0.025
"""Minimum height for equation detection."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM MATH BLOCK PROCESSOR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LLMMathBlockProcessor_max_concurrency = 1
"""Maximum concurrent LLM requests for math blocks."""

LLMMathBlockProcessor_image_expansion_ratio = 0.04
"""Ratio to expand math block images."""

LLMMathBlockProcessor_inlinemath_min_ratio = 0.35
"""Minimum ratio for inline math detection."""


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE PROCESSOR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

TableProcessor_row_split_threshold = 0.55
"""Threshold for splitting table rows."""

TableProcessor_pdftext_workers = 1
"""Number of workers for pdftext processing."""

TableProcessor_drop_repeated_table_text = True
"""Drop repeated text in tables."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM TABLE PROCESSOR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LLMTableProcessor_max_concurrency = 1
"""Maximum concurrent LLM requests for tables."""

LLMTableProcessor_image_expansion_ratio = 0.05
"""Ratio to expand table images."""

LLMTableProcessor_max_rows_per_batch = 70
"""Maximum rows per batch for LLM table processing."""

LLMTableProcessor_max_table_rows = 200
"""Maximum number of rows in a table."""

LLMTableProcessor_table_image_expansion_ratio = 0.0
"""Additional ratio for table image expansion."""

LLMTableProcessor_rotation_max_wh_ratio = 0.6
"""Maximum width/height ratio for rotation detection."""

LLMTableProcessor_max_table_iterations = 2
"""Maximum iterations for table refinement."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM TABLE MERGE PROCESSOR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LLMTableMergeProcessor_max_concurrency = 1
"""Maximum concurrent LLM requests for table merging."""

LLMTableMergeProcessor_image_expansion_ratio = 0.05
"""Ratio to expand table merge images."""

LLMTableMergeProcessor_table_height_threshold = 0.65
"""Height threshold for table merging."""

LLMTableMergeProcessor_table_start_threshold = 0.2
"""Start position threshold for table merging."""

LLMTableMergeProcessor_vertical_table_height_threshold = 0.3
"""Height threshold for vertical table detection."""

LLMTableMergeProcessor_vertical_table_distance_threshold = 18
"""Distance threshold for vertical table merging."""

LLMTableMergeProcessor_horizontal_table_width_threshold = 0.3
"""Width threshold for horizontal table detection."""

LLMTableMergeProcessor_horizontal_table_distance_threshold = 12
"""Distance threshold for horizontal table merging."""

LLMTableMergeProcessor_column_gap_threshold = 45
"""Gap threshold between columns."""

LLMTableMergeProcessor_no_merge_tables_across_pages = False
"""Prevent merging tables across page boundaries."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM PAGE CORRECTION PROCESSOR SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LLMPageCorrectionProcessor_max_concurrency = 1
"""Maximum concurrent LLM requests for page correction."""

LLMPageCorrectionProcessor_image_expansion_ratio = 0.03
"""Ratio to expand page correction images."""

LLMPageCorrectionProcessor_block_correction_prompt = """\
You are a strict data-extraction formatter. Your ONLY job is to fix OCR errors in scientific text.
CRITICAL RULES:
1) Precisely fix scientific notation (e.g., convert '105' to '$10^5$' if context implies math).
2) Precisely format chemical formulas and parameters with proper LaTeX subscripts/superscripts \
(e.g., '$H_2O_2$', '$K^{app}_m$').
3) Fix words broken by hyphens across lines.
4) DO NOT output conversational filler like 'Here is the corrected text'.
5) DO NOT add, infer, or summarize any scientific data. Output only the corrected raw text.
"""
"""Prompt for block correction with LLM."""


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROVIDER & RENDERER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DISABLE_LINKS = True
"""Disable hyperlink extraction."""

MarkdownRenderer_keep_pageheader_in_output = False
"""Keep page headers in output."""

MarkdownRenderer_keep_pagefooter_in_output = False
"""Keep page footers in output."""

MarkdownRenderer_add_block_ids = False
"""Add block IDs to output."""

MarkdownRenderer_page_separator = "-" * 48
"""Separator between pages in output."""

MarkdownRenderer_html_tables_in_markdown = True
"""Render tables as HTML within Markdown."""


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE & GLOBAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

LLM_SERVICE = "ollama"
"""LLM service backend: 'ollama' or other supported services."""

OllamaService_timeout = 600
"""Timeout for Ollama API requests (seconds)."""

OllamaService_max_retries = 3
"""Maximum number of retries for Ollama requests."""

OllamaService_retry_wait_time = 30
"""Wait time between retries (seconds)."""

OllamaService_max_output_tokens = None
"""Maximum output tokens for Ollama (None = use model default)."""

OllamaService_ollama_base_url = "https://aicltr.itmo.ru/ollama"
"""Base URL for Ollama API."""

OllamaService_ollama_model = "qwen2.5vl:72b"
"""Ollama model name for LLM processing."""


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch

    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_DEVICE = "cpu"
"""PyTorch device for model inference: 'cuda' or 'cpu'."""


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSOR LIST
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_PROCESSORS: List[str] = [
    "marker.processors.order.OrderProcessor",
    "marker.processors.block_relabel.BlockRelabelProcessor",
    "marker.processors.line_merge.LineMergeProcessor",
    "marker.processors.blockquote.BlockquoteProcessor",
    "marker.processors.code.CodeProcessor",
    "marker.processors.document_toc.DocumentTOCProcessor",
    "marker.processors.equation.EquationProcessor",
    "marker.processors.footnote.FootnoteProcessor",
    "marker.processors.ignoretext.IgnoreTextProcessor",
    "marker.processors.line_numbers.LineNumbersProcessor",
    "marker.processors.list.ListProcessor",
    "marker.processors.page_header.PageHeaderProcessor",
    "marker.processors.sectionheader.SectionHeaderProcessor",
    "marker.processors.table.TableProcessor",
    # LLM processors
    "marker.processors.llm.llm_table.LLMTableProcessor",
    "marker.processors.llm.llm_table_merge.LLMTableMergeProcessor",
    "marker.processors.llm.llm_equation.LLMEquationProcessor",
    "marker.processors.llm.llm_mathblock.LLMMathBlockProcessor",
    # LLMPageCorrectionProcessor is commented out by default
    # "marker.processors.llm.llm_page_correction.LLMPageCorrectionProcessor",
    "marker.processors.text.TextProcessor",
    "marker.processors.reference.ReferenceProcessor",
    "marker.processors.blank_page.BlankPageProcessor",
    "marker.processors.debug.DebugProcessor",
]
"""Custom processor stack for Marker PDF converter."""


def get_marker_config_dict(
    output_dir: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the complete Marker configuration dictionary.

    This function assembles all configuration parameters into a dictionary
    compatible with Marker's ConfigParser.

    Args:
        output_dir: Optional output directory override.
        ollama_base_url: Optional Ollama base URL override.
        ollama_model: Optional Ollama model override.

    Returns:
        Dictionary containing all Marker configuration parameters.
    """
    config_dict: Dict[str, Any] = {
        # Core settings
        "output_format": OUTPUT_FORMAT,
        "page_range": PAGE_RANGE,
        "paginate_output": PAGINATE_OUTPUT,
        "debug": DEBUG,
        "debug_print": DEBUG_PRINT,
        "disable_tqdm": DISABLE_TQDM,
        # OCR settings
        "force_ocr": FORCE_OCR,
        "strip_existing_ocr": STRIP_EXISTING_OCR,
        "disable_ocr": DISABLE_OCR,
        "keep_chars": KEEP_CHARS,
        # LLM (global)
        "use_llm": USE_LLM,
        # DocumentBuilder
        "lowres_image_dpi": DocumentBuilder_lowres_image_dpi,
        "highres_image_dpi": DocumentBuilder_highres_image_dpi,
        # LayoutBuilder
        "layout_batch_size": LayoutBuilder_layout_batch_size,
        "force_layout_block": LayoutBuilder_force_layout_block,
        "max_expand_frac": LayoutBuilder_max_expand_frac,
        "layout_detection_batch_size": LayoutBuilder_detection_batch_size,
        # LineBuilder
        "layout_coverage_min_lines": LineBuilder_layout_coverage_min_lines,
        "layout_coverage_threshold": LineBuilder_layout_coverage_threshold,
        "min_document_ocr_threshold": LineBuilder_min_document_ocr_threshold,
        "detection_line_min_confidence": LineBuilder_detection_line_min_confidence,
        "line_recognition_batch_size": LineBuilder_recognition_batch_size,
        # OcrBuilder
        "ocr_task_name": OcrBuilder_ocr_task_name,
        "disable_ocr_math": OcrBuilder_disable_ocr_math,
        "drop_repeated_text": OcrBuilder_drop_repeated_text,
        "block_mode_intersection_thresh": OcrBuilder_block_mode_intersection_thresh,
        "block_mode_max_lines": OcrBuilder_block_mode_max_lines,
        "block_mode_max_height_frac": OcrBuilder_block_mode_max_height_frac,
        # StructureBuilder
        "gap_threshold": StructureBuilder_gap_threshold,
        "list_gap_threshold": StructureBuilder_list_gap_threshold,
        # EquationProcessor
        "equation_model_max_length": EquationProcessor_model_max_length,
        "equation_batch_size": EquationProcessor_equation_batch_size,
        # LLMEquationProcessor
        "LLMEquationProcessor_max_concurrency": LLMEquationProcessor_max_concurrency,
        "LLMEquationProcessor_image_expansion_ratio": LLMEquationProcessor_image_expansion_ratio,
        "LLMEquationProcessor_min_equation_height": LLMEquationProcessor_min_equation_height,
        "LLMEquationProcessor_redo_inline_math": REDO_INLINE_MATH,
        "LLMEquationProcessor_use_llm": USE_LLM,
        # LLMMathBlockProcessor
        "LLMMathBlockProcessor_max_concurrency": LLMMathBlockProcessor_max_concurrency,
        "LLMMathBlockProcessor_image_expansion_ratio": LLMMathBlockProcessor_image_expansion_ratio,
        "LLMMathBlockProcessor_redo_inline_math": REDO_INLINE_MATH,
        "LLMMathBlockProcessor_inlinemath_min_ratio": LLMMathBlockProcessor_inlinemath_min_ratio,
        "LLMMathBlockProcessor_use_llm": USE_LLM,
        # LLMTableProcessor
        "LLMTableProcessor_max_concurrency": LLMTableProcessor_max_concurrency,
        "LLMTableProcessor_image_expansion_ratio": LLMTableProcessor_image_expansion_ratio,
        "LLMTableProcessor_max_rows_per_batch": LLMTableProcessor_max_rows_per_batch,
        "LLMTableProcessor_max_table_rows": LLMTableProcessor_max_table_rows,
        "LLMTableProcessor_table_image_expansion_ratio": LLMTableProcessor_table_image_expansion_ratio,
        "LLMTableProcessor_rotation_max_wh_ratio": LLMTableProcessor_rotation_max_wh_ratio,
        "LLMTableProcessor_max_table_iterations": LLMTableProcessor_max_table_iterations,
        "LLMTableProcessor_use_llm": USE_LLM,
        # LLMTableMergeProcessor
        "LLMTableMergeProcessor_max_concurrency": LLMTableMergeProcessor_max_concurrency,
        "LLMTableMergeProcessor_image_expansion_ratio": LLMTableMergeProcessor_image_expansion_ratio,
        "LLMTableMergeProcessor_table_height_threshold": LLMTableMergeProcessor_table_height_threshold,
        "LLMTableMergeProcessor_table_start_threshold": LLMTableMergeProcessor_table_start_threshold,
        "LLMTableMergeProcessor_vertical_table_height_threshold": LLMTableMergeProcessor_vertical_table_height_threshold,
        "LLMTableMergeProcessor_vertical_table_distance_threshold": (
            LLMTableMergeProcessor_vertical_table_distance_threshold
        ),
        "LLMTableMergeProcessor_horizontal_table_width_threshold": (
            LLMTableMergeProcessor_horizontal_table_width_threshold
        ),
        "LLMTableMergeProcessor_horizontal_table_distance_threshold": (
            LLMTableMergeProcessor_horizontal_table_distance_threshold
        ),
        "LLMTableMergeProcessor_column_gap_threshold": LLMTableMergeProcessor_column_gap_threshold,
        "LLMTableMergeProcessor_no_merge_tables_across_pages": (
            LLMTableMergeProcessor_no_merge_tables_across_pages
        ),
        "LLMTableMergeProcessor_use_llm": USE_LLM,
        # LLMPageCorrectionProcessor (commented out by default)
        # "LLMPageCorrectionProcessor_max_concurrency": LLMPageCorrectionProcessor_max_concurrency,
        # "LLMPageCorrectionProcessor_image_expansion_ratio": LLMPageCorrectionProcessor_image_expansion_ratio,
        # "LLMPageCorrectionProcessor_block_correction_prompt": LLMPageCorrectionProcessor_block_correction_prompt,
        # "LLMPageCorrectionProcessor_use_llm": True,
        # TableProcessor
        "table_row_split_threshold": TableProcessor_row_split_threshold,
        "table_pdftext_workers": TableProcessor_pdftext_workers,
        "drop_repeated_table_text": TableProcessor_drop_repeated_table_text,
        # DocumentProvider
        "disable_links": DISABLE_LINKS,
        # MarkdownRenderer
        "extract_images": not DISABLE_IMAGE_EXTRACTION,
        "page_separator": MarkdownRenderer_page_separator,
        "html_tables_in_markdown": MarkdownRenderer_html_tables_in_markdown,
        "keep_pageheader_in_output": MarkdownRenderer_keep_pageheader_in_output,
        "keep_pagefooter_in_output": MarkdownRenderer_keep_pagefooter_in_output,
        "add_block_ids": MarkdownRenderer_add_block_ids,
    }

    # Inject Ollama service parameters
    if LLM_SERVICE == "ollama":
        config_dict["ollama_base_url"] = (
            ollama_base_url or OllamaService_ollama_base_url
        )
        config_dict["ollama_model"] = ollama_model or OllamaService_ollama_model
        config_dict["llm_service"] = "marker.services.ollama.OllamaService"

    # Add output directory if provided
    if output_dir:
        config_dict["output_dir"] = output_dir

    return config_dict


def get_custom_processors() -> List[str]:
    """Get the custom processor list for Marker.

    Returns:
        List of processor class paths.
    """
    return CUSTOM_PROCESSORS.copy()


def get_torch_device() -> str:
    """Get the PyTorch device setting.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    return TORCH_DEVICE
