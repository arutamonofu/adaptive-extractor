"""GeminiParser: orchestrator for Ingestion v2 pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from ae.core.config.settings import IngestionConfig
from ae.core.exceptions import GeminiEmptyResponseError
from ae.ingestion.parsers.base import BaseParser
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.stages import (
    run_manifest_visuals,
    run_parse_text,
    run_parse_visuals,
    run_render,
    run_render_manifest,
    run_render_pages_with_visuals,
    run_segment_and_crop,
)

logger = logging.getLogger(__name__)


class GeminiParser(BaseParser):
    """End-to-end PDF parser using Google Gemini API (Ingestion v2 Lazy Agentic Pipeline).

    Orchestrates the 4-stage ingestion pipeline:
    1. parse_text: PDF → Markdown + <visual id="..."> placeholders (with 1-based page)
    2. render_pages: PyMuPDF: pages from placeholders → PNG (scaled to <= 768 px, 1 tile)
    3. classify: Placeholders + targeted page previews → manifest (blocks, types, pages)
    4. render_manifest: Lightweight manifest stubs <visual id="{id}" type="{type}"> into document.md
    """

    def __init__(self, config: IngestionConfig) -> None:
        if config is None:
            raise ValueError("IngestionConfig is required for GeminiParser")

        self.cfg = config
        self._gemini_cfg = config.gemini
        if self._gemini_cfg is None:
            raise ValueError("GeminiParser requires a 'gemini' section in IngestionConfig")

        self._api_key = self._resolve_api_key()
        self._client: Optional[GeminiClient] = None
        logger.info("Initialised GeminiParser (Ingestion v2).")

    @property
    def client(self) -> GeminiClient:
        """Lazy initialization of GeminiClient."""
        if self._client is None:
            self._client = GeminiClient(self._gemini_cfg, self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key() -> str:
        """Retrieve Gemini API key from environment or Settings."""
        import os

        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key

        try:
            from ae.core.config.settings import Settings

            settings = Settings.load(load_env_file=True)
            if settings.gemini_api_key:
                return settings.gemini_api_key.get_secret_value()
        except Exception:
            pass

        raise EnvironmentError(
            "GEMINI_API_KEY not found. Set environment variable or configure in .env",
        )

    # ------------------------------------------------------------------
    # Parse entrypoint
    # ------------------------------------------------------------------

    def parse(self, file_path: Union[str, Path]) -> str:
        """Execute the full Ingestion v2 pipeline on a single PDF."""
        pdf_path = Path(file_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_name = pdf_path.stem
        artifacts = ArtifactManager(doc_name, base_dir=self.cfg.ingestion_dir)

        # Handle stage invalidation
        from_stage = getattr(self.cfg, "from_stage", None)
        overwrite = getattr(self.cfg, "overwrite", False)

        if overwrite:
            artifacts.invalidate_from_stage("parse_text")
        elif from_stage:
            artifacts.invalidate_from_stage(from_stage)

        # Stage 1: parse_text (parses 2-page chunks in parallel with page breaks)
        clean_md, placeholders = run_parse_text(
            self.client,
            None,
            artifacts,
            self.cfg,
            pdf_path=pdf_path,
        )

        # Stage 2: manifest_visuals (renders page previews and builds visual manifest)
        run_render_pages_with_visuals(
            pdf_path,
            placeholders,
            artifacts,
            self.cfg,
        )

        manifest = run_manifest_visuals(
            self.client,
            placeholders,
            artifacts,
            self.cfg,
        )

        # Ensure any pages introduced by manifest objects are rendered for localization
        run_render_pages_with_visuals(
            pdf_path,
            manifest,
            artifacts,
            self.cfg,
        )

        # Stage 4: render_manifest (embeds lightweight manifest stubs)
        manifest_md = run_render_manifest(
            clean_md,
            manifest,
            artifacts,
            self.client,
        )

        if not manifest_md or not manifest_md.strip():
            raise GeminiEmptyResponseError(
                stage="render_manifest",
                details=f"Final rendered Markdown for document '{doc_name}' is empty.",
            )

        return manifest_md
