# src/ae/ingestion/visual_parser.py
"""Visual-anchored document parser using Gemini and the ae_visual pipeline."""

import logging
import tempfile
from pathlib import Path
from typing import Union

from ae.core.config.settings import AEVisualParserConfig
from ae.ingestion.base_parser import BaseParser
from ae.ingestion.gemini_parser import GeminiParser
from ae.ingestion.visual_pipeline import run_visual_pipeline

logger = logging.getLogger(__name__)


class AEVisualParser(BaseParser):
    """Parser that first extracts markdown text using GeminiParser (with visual anchors),
    then runs the ae_visual pipeline to extract visual charts and tables and insert them.
    """

    def __init__(self, config: AEVisualParserConfig):
        """Initialize the AEVisualParser.

        Args:
            config: Configuration settings for the visual parser.
        """
        if config is None:
            raise ValueError("AEVisualParserConfig is required")

        self.cfg = config
        # Initialize the base parser using the nested Gemini configuration
        self.base_parser = GeminiParser(config.gemini)
        logger.info("Initialized AEVisualParser with nested Gemini parser configuration")

    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a PDF file into Markdown text with embedded charts/tables.

        Args:
            file_path: Path to the input PDF file.

        Returns:
            str: Enriched Markdown text containing inserted tables/charts.
        """
        pdf_path = Path(file_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"[AEVisualParser] Starting parsing pipeline for: {pdf_path.name}")

        # Stage 1: Base text extraction with GeminiParser
        logger.info("[AEVisualParser] Step 1/6: Running base text parser with Gemini to extract anchors...")
        initial_md = self.base_parser.parse(pdf_path)
        if not initial_md:
            logger.warning("[AEVisualParser] Gemini parser returned empty markdown content")
            return ""

        logger.info(
            f"[AEVisualParser] Step 2/6: Text parsing completed (Markdown size: {len(initial_md)} characters)"
        )

        logger.info("[AEVisualParser] Step 3/6: Resolving configuration paths and initializing temporary directory...")
        # Use temporary directory for the pipeline's inputs and outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir).resolve()
            temp_out_dir = temp_dir_path / "ae_visual_run"
            temp_out_dir.mkdir(parents=True, exist_ok=True)

            # Locate configuration paths
            pipeline_config = self.cfg.pipeline
            task_config = Path(self.cfg.task_config_path).resolve()

            # Retrieve active LM from DSPy settings (setup during app run)
            import dspy
            lm = getattr(dspy.settings, "lm", None)

            logger.info(f"[AEVisualParser] Step 4/6: Executing run_visual_pipeline in-process...")

            enriched_md = run_visual_pipeline(
                initial_md=initial_md,
                pdf_path=pdf_path,
                out_dir=temp_out_dir,
                pipeline_config=pipeline_config,
                task_config_path=task_config,
                lm=lm,
                force=True,  # Always force inside a fresh temp dir
            )

            logger.info("[AEVisualParser] Step 5/6: Local visual pipeline finished successfully")
            logger.info("[AEVisualParser] Step 6/6: Cleaned up temporary directory")
            return enriched_md
