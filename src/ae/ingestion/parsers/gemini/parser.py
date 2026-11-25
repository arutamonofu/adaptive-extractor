"""GeminiParser: orchestrator for Ingestion v2 pipeline."""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

import pymupdf as fitz

from ae.core.config.settings import IngestionConfig
from ae.core.exceptions import GeminiEmptyResponseError, GeminiMaxTokensError
from ae.ingestion.parsers.base import BaseParser
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import BatchRequestSpec, GeminiClient
from ae.ingestion.parsers.gemini.schemas import Manifest
from ae.ingestion.parsers.gemini.stages import (
    run_manifest_visuals,
    run_parse_text,
    run_render_manifest,
    run_render_pages_with_visuals,
)
from ae.ingestion.parsers.gemini.stages.manifest_visuals import (
    build_manifest_visuals_batch_spec,
    validate_and_retry_manifest,
)
from ae.ingestion.parsers.gemini.stages.parse_text import (
    _extract_pdf_pages_text,
    _extract_pdf_pages_visual_flags,
    _generate_chunk_markdown,
    _is_silent_error_guard_enabled,
    _read_prompt_template,
    assemble_parse_text_chunks,
    atomic_write_text,
    build_parse_text_batch_specs,
    is_visual_tag_silent_error,
    strip_outer_code_fences,
    validate_and_retry_chunk_markdown,
)
from ae.ingestion.parsers.gemini.stages.silent_error_guard import (
    detect_parse_text_silent_error,
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

        with fitz.open(pdf_path) as pdf_doc:
            total_pages = len(pdf_doc)

        scope_cm = (
            self.client.doc_scope(doc_name)
            if hasattr(self.client, "doc_scope")
            else contextlib.nullcontext()
        )
        with scope_cm:
            # Stage 1: parse_text (parses 1-page chunks in parallel via inline PDF)
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
                total_pages=total_pages,
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

    def parse_batch(self, file_paths: list[Union[str, Path]]) -> dict[Path, str]:
        """Execute Stage 01 (parse_text) and Stage 02 (classify) via Gemini Batch API."""
        resolved_paths = [Path(p).resolve() for p in file_paths]
        if not resolved_paths:
            return {}

        from_stage = getattr(self.cfg, "from_stage", None)
        overwrite = getattr(self.cfg, "overwrite", False)
        use_guard = _is_silent_error_guard_enabled(self.cfg)
        stage1_cfg = self.cfg.gemini.get_stage_config("parse_text")
        static_sys_prompt = _read_prompt_template()

        doc_states: dict[Path, dict] = {}
        all_chunk_specs: list[BatchRequestSpec] = []
        tmp_dirs: list[Path] = []

        # 1. Stage invalidation + prepare Stage 1 batch specs
        for pdf_path in resolved_paths:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            doc_name = pdf_path.stem
            artifacts = ArtifactManager(doc_name, base_dir=self.cfg.ingestion_dir)
            if overwrite:
                artifacts.invalidate_from_stage("parse_text")
            elif from_stage:
                artifacts.invalidate_from_stage(from_stage)

            with fitz.open(pdf_path) as pdf_doc:
                total_pages = len(pdf_doc)

            if artifacts.has_parse_text_artifact():
                clean_md, placeholders = run_parse_text(
                    self.client, None, artifacts, self.cfg, pdf_path=pdf_path
                )
                doc_states[pdf_path] = {
                    "artifacts": artifacts,
                    "clean_md": clean_md,
                    "placeholders": placeholders,
                    "chunks": [],
                    "total_pages": total_pages,
                }
                continue

            chunks, specs, tmp_dir = build_parse_text_batch_specs(
                self.client, pdf_path, artifacts, self.cfg
            )
            tmp_dirs.append(tmp_dir)
            all_chunk_specs.extend(specs)
            has_uncached = len(specs) > 0
            doc_states[pdf_path] = {
                "artifacts": artifacts,
                "chunks": chunks,
                "total_pages": total_pages,
                "page_texts": _extract_pdf_pages_text(pdf_path) if (use_guard and has_uncached) else {},
                "page_visual_flags": (
                    _extract_pdf_pages_visual_flags(pdf_path) if (use_guard and has_uncached) else {}
                ),
            }

        try:
            batch_chunk_results = (
                self.client.run_batch_requests(all_chunk_specs, display_name="ae-parse-text")
                if all_chunk_specs
                else {}
            )

            # Finalize Stage 1 per document (with Silent-Error Guard & sync fallback on any missing chunk)
            for pdf_path, state in doc_states.items():
                if "clean_md" in state:
                    continue
                artifacts = state["artifacts"]
                chunks = state["chunks"]
                page_texts = state["page_texts"]
                page_visual_flags = state.get("page_visual_flags", {})
                chunk_texts: list[Optional[str]] = []

                with self.client.doc_scope(artifacts.doc_name):
                    for start_page, end_page, chunk_pdf_path in chunks:
                        cache_path = artifacts.chunks_dir / f"chunk_{start_page}_{end_page}.md"
                        if cache_path.exists() and cache_path.stat().st_size > 0:
                            chunk_texts.append(cache_path.read_text(encoding="utf-8"))
                            continue

                        key = f"{artifacts.doc_name}__chunk_{start_page}_{end_page}"
                        item_res = batch_chunk_results.get(key)
                        chunk_md = item_res.text if (item_res and not item_res.error) else None
                        initial_max_tokens_exc: Optional[GeminiMaxTokensError] = None

                        if (
                            item_res is not None
                            and getattr(item_res, "error_type", None) == "GeminiMaxTokensError"
                        ):
                            exc = GeminiMaxTokensError(
                                stage="parse_text",
                                details=item_res.error or f"Batch item '{key}' truncated at MAX_TOKENS.",
                            )
                            if not use_guard:
                                raise exc
                            initial_max_tokens_exc = exc
                        elif not chunk_md or not chunk_md.strip():
                            logger.warning(
                                f"Batch item '{key}' failed/empty ({getattr(item_res, 'error', None)}); "
                                f"falling back to sync generate_text."
                            )
                            try:
                                chunk_md = _generate_chunk_markdown(
                                    self.client,
                                    chunk_pdf_path,
                                    start_page,
                                    end_page,
                                    static_sys_prompt,
                                    stage1_cfg,
                                    self.cfg,
                                )
                            except GeminiMaxTokensError as exc:
                                if not use_guard:
                                    raise
                                initial_max_tokens_exc = exc
                            except GeminiEmptyResponseError:
                                if not use_guard:
                                    raise
                                chunk_md = ""

                        raw_txt = ""
                        has_visual_content = False
                        if use_guard:
                            raw_txt = "\n".join(
                                page_texts.get(p, "") for p in range(start_page, end_page + 1)
                            )
                            has_visual_content = any(
                                page_visual_flags.get(p, False)
                                for p in range(start_page, end_page + 1)
                            )
                            chunk_md = validate_and_retry_chunk_markdown(
                                client=self.client,
                                chunk_md=chunk_md if chunk_md is not None else "",
                                chunk_pdf_path=chunk_pdf_path,
                                start_page=start_page,
                                end_page=end_page,
                                static_system_prompt=static_sys_prompt,
                                stage_cfg=stage1_cfg,
                                config=self.cfg,
                                raw_pdf_txt=raw_txt,
                                doc_name=artifacts.doc_name,
                                initial_max_tokens_exc=initial_max_tokens_exc,
                                has_visual_content=has_visual_content,
                            )
                        else:
                            chunk_md = strip_outer_code_fences(chunk_md or "")

                        if not chunk_md or not chunk_md.strip():
                            raise GeminiEmptyResponseError(
                                stage="parse_text",
                                details=(
                                    f"Empty chunk response for '{key}' of '{artifacts.doc_name}'."
                                ),
                            )

                        should_cache = True
                        if use_guard:
                            final_err = detect_parse_text_silent_error(
                                chunk_md,
                                pdf_page_text=raw_txt,
                                has_visual_content=has_visual_content,
                            )
                            if is_visual_tag_silent_error(final_err):
                                logger.warning(
                                    f"Stage 1 [parse_text]: Skipping cache write for '{key}' "
                                    f"due to unresolved visual tag error ({final_err})."
                                )
                                should_cache = False

                        if should_cache:
                            atomic_write_text(cache_path, chunk_md)
                        chunk_texts.append(chunk_md)

                    clean_md, placeholders = assemble_parse_text_chunks(
                        chunks, chunk_texts, artifacts
                    )
                    state["clean_md"] = clean_md
                    state["placeholders"] = placeholders
        finally:
            for spec in all_chunk_specs:
                for handle in getattr(spec, "uploaded_handles", []) or []:
                    self.client.delete_file(handle)
            for td in tmp_dirs:
                if td.exists():
                    shutil.rmtree(td, ignore_errors=True)

        # 2. Render page previews and prepare Stage 2 (classify) batch specs
        classify_specs: list[BatchRequestSpec] = []
        for pdf_path, state in doc_states.items():
            artifacts = state["artifacts"]
            placeholders = state["placeholders"]
            run_render_pages_with_visuals(pdf_path, placeholders, artifacts, self.cfg)
            spec = build_manifest_visuals_batch_spec(placeholders, artifacts, self.cfg)
            if spec is not None:
                classify_specs.append(spec)

        classify_results = (
            self.client.run_batch_requests(classify_specs, display_name="ae-classify")
            if classify_specs
            else {}
        )

        # 3. Finalize Stage 2 + Stage 4 render_manifest per document
        outputs: dict[Path, str] = {}
        for pdf_path, state in doc_states.items():
            artifacts = state["artifacts"]
            placeholders = state["placeholders"]
            clean_md = state["clean_md"]
            total_pages = state.get("total_pages")
            with self.client.doc_scope(artifacts.doc_name):
                if artifacts.has_manifest_artifact():
                    manifest = run_manifest_visuals(
                        self.client, placeholders, artifacts, self.cfg, total_pages=total_pages
                    )
                else:
                    c_key = f"{artifacts.doc_name}__classify"
                    c_res = classify_results.get(c_key)
                    if c_res and not c_res.error and isinstance(c_res.parsed, Manifest):
                        manifest = validate_and_retry_manifest(
                            self.client,
                            c_res.parsed,
                            placeholders,
                            artifacts,
                            self.cfg,
                            total_pages=total_pages,
                        )
                        artifacts.save_json(artifacts.manifest_json_path, manifest)
                    elif (
                        c_res is not None
                        and getattr(c_res, "error_type", None) == "GeminiMaxTokensError"
                    ):
                        exc = GeminiMaxTokensError(
                            stage="classify",
                            details=c_res.error or f"Batch item '{c_key}' truncated at MAX_TOKENS.",
                        )
                        if not use_guard:
                            raise exc
                        manifest = validate_and_retry_manifest(
                            self.client,
                            None,
                            placeholders,
                            artifacts,
                            self.cfg,
                            total_pages=total_pages,
                            initial_max_tokens_exc=exc,
                        )
                        artifacts.save_json(artifacts.manifest_json_path, manifest)
                    else:
                        logger.warning(
                            f"Batch classify failed for '{artifacts.doc_name}' "
                            f"({getattr(c_res, 'error', None)}); falling back to sync."
                        )
                        manifest = run_manifest_visuals(
                            self.client, placeholders, artifacts, self.cfg, total_pages=total_pages
                        )

                run_render_pages_with_visuals(pdf_path, manifest, artifacts, self.cfg)
                manifest_md = run_render_manifest(clean_md, manifest, artifacts, self.client)
                outputs[pdf_path] = manifest_md

        return outputs
