"""Stage 1: Parse PDF text to Markdown with visual block placeholders."""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Final, Optional

import pymupdf as fitz

from ae.core.exceptions import GeminiEmptyResponseError
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import (
    INLINE_PDF_MAX_BYTES,
    BatchRequestSpec,
    GeminiClient,
)
from ae.ingestion.parsers.gemini.schemas import PlaceholderItem
from ae.ingestion.parsers.gemini.stages.silent_error_guard import (
    detect_parse_text_silent_error,
)

logger = logging.getLogger(__name__)

CHUNK_PAGES: Final[int] = 1

_VISUAL_BLOCK_RE = re.compile(
    r"<visual\s+id=[\"'](?P<id>[\w.\-]+)[\"'](?:\s*/>|\s*>(?P<caption>.*?)</visual>)",
    re.DOTALL,
)
_PAGE_BREAK_RE = re.compile(r"<!--\s*page_break\s*-->")


def _read_prompt_template() -> str:
    prompt_file = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "01_parse_text"
        / "parse_text.txt"
    )
    return prompt_file.read_text(encoding="utf-8")


def format_parse_text_page_context(
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> str:
    """Format dynamic per-chunk page context for user contents (preserving static system prompt)."""
    if start_page is None:
        return ""

    if end_page is None or end_page == start_page:
        page_info = (
            f"Document Page Context: You are transcribing physical page {start_page} of the PDF document."
        )
        example_id = f"Unlabeled_p{start_page}_1"
    else:
        page_info = (
            f"Document Page Context: You are transcribing physical pages {start_page} to {end_page} "
            "of the PDF document."
        )
        example_id = f"Unlabeled_p{start_page}_1"

    return (
        f"### Chunk Context\n"
        f"- {page_info}\n"
        f"- When assigning IDs to uncaptioned visuals on this page, use the 1-based document page "
        f"number (e.g., `{example_id}`)."
    )


def compose_parse_text_prompt(
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> str:
    """Read base prompt template for text parsing and inject chunk page context."""
    template = _read_prompt_template()
    chunk_ctx = format_parse_text_page_context(start_page, end_page)
    if not chunk_ctx:
        return template
    return f"{template}\n\n{chunk_ctx}"


def split_pdf_into_chunks(
    pdf_path: Path,
    temp_dir: Path,
    chunk_pages: int = CHUNK_PAGES,
) -> list[tuple[int, int, Path]]:
    """Split a PDF into smaller chunk PDFs.

    Args:
        pdf_path: Source PDF path.
        temp_dir: Directory where chunk PDFs will be temporarily stored.
        chunk_pages: Number of pages per chunk (defaults to CHUNK_PAGES = 1).

    Returns:
        List of tuples: (start_page, end_page, chunk_pdf_path) with 1-based page indices.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[int, int, Path]] = []
    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        for start_idx in range(0, total_pages, chunk_pages):
            end_idx = min(start_idx + chunk_pages, total_pages)
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx - 1)
            chunk_path = temp_dir / f"chunk_{start_idx + 1}_{end_idx}.pdf"
            chunk_doc.save(chunk_path)
            chunk_doc.close()
            chunks.append((start_idx + 1, end_idx, chunk_path))
    finally:
        doc.close()
    return chunks


def parse_placeholders_from_markdown(
    raw_md: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    id_counts: Optional[dict[str, int]] = None,
    unlabeled_counts: Optional[dict[int, int]] = None,
) -> tuple[str, list[PlaceholderItem]]:
    """Parse visual tags from raw markdown, tracking page offsets and disambiguating IDs."""
    if id_counts is None:
        id_counts = {}
    if unlabeled_counts is None:
        unlabeled_counts = {}

    placeholders: list[PlaceholderItem] = []
    matches = list(_VISUAL_BLOCK_RE.finditer(raw_md))
    if not matches:
        return raw_md, []

    # Map character offsets to page numbers based on <!-- page_break -->
    page_breaks = [m.start() for m in _PAGE_BREAK_RE.finditer(raw_md)]

    def _offset_to_page(char_offset: int) -> int:
        if not page_breaks:
            return start_page
        idx = 0
        for pb_pos in page_breaks:
            if char_offset > pb_pos:
                idx += 1
            else:
                break
        resolved = start_page + idx
        if end_page is not None and resolved > end_page:
            resolved = end_page
        return resolved

    new_parts: list[str] = []
    last_idx = 0

    for m in matches:
        new_parts.append(raw_md[last_idx:m.start()])
        raw_id = m.group("id")
        raw_caption = m.group("caption") or ""
        caption = raw_caption.strip()

        visual_page = _offset_to_page(m.start())

        if raw_id.startswith("Unlabeled"):
            k = unlabeled_counts.get(visual_page, 0) + 1
            unlabeled_counts[visual_page] = k
            unique_id = f"Unlabeled_p{visual_page}_{k}"
        else:
            # Disambiguate duplicate IDs
            count = id_counts.get(raw_id, 0) + 1
            id_counts[raw_id] = count
            unique_id = raw_id if count == 1 else f"{raw_id}_dup{count}"

        item = PlaceholderItem(
            id=unique_id,
            caption=caption,
            page=visual_page,
            original_tag=m.group(0),
        )
        placeholders.append(item)

        if caption:
            replacement = (
                f'<visual id="{unique_id}">\n'
                f"{caption}\n"
                f"</visual>"
            )
        else:
            replacement = f'<visual id="{unique_id}" />'

        new_parts.append(replacement)
        last_idx = m.end()

    new_parts.append(raw_md[last_idx:])
    updated_md = "".join(new_parts)
    return updated_md, placeholders


def _strip_chunk_edge_page_breaks(text: str) -> str:
    """Strip accidental leading and trailing page break tags from a chunk response."""
    text = text.strip()
    text = re.sub(r"^(?:<!--\s*page_break\s*-->\s*)+", "", text)
    text = re.sub(r"(?:\s*<!--\s*page_break\s*-->)+$", "", text)
    return text.strip()


def _extract_pdf_pages_text(pdf_path: Path) -> dict[int, str]:
    """Extract raw PyMuPDF text per 1-based page for silent-error checks."""
    texts: dict[int, str] = {}
    try:
        doc = fitz.open(pdf_path)
        try:
            for idx in range(len(doc)):
                texts[idx + 1] = doc[idx].get_text() or ""
        finally:
            doc.close()
    except Exception as exc:
        logger.debug(f"Could not extract raw PDF text for silent-error check: {exc}")
    return texts


def _should_use_inline_pdf(config: Any, chunk_pdf_path: Path) -> bool:
    use_inline = getattr(getattr(config, "execution", None), "use_inline_pdf", None)
    if use_inline is None:
        use_inline = getattr(getattr(config, "gemini", None), "use_inline_pdf", True)
    if not bool(use_inline):
        return False
    try:
        return chunk_pdf_path.stat().st_size <= INLINE_PDF_MAX_BYTES
    except OSError:
        return False


def _is_silent_error_guard_enabled(config: Any) -> bool:
    guard = getattr(getattr(config, "execution", None), "silent_error_guard", None)
    if guard is None:
        guard = getattr(getattr(config, "gemini", None), "silent_error_guard", True)
    return bool(guard)


def assemble_parse_text_chunks(
    chunks: list[tuple[int, int, Path]],
    chunk_results: list[Optional[str]],
    artifacts: ArtifactManager,
) -> tuple[str, list[PlaceholderItem]]:
    """Assemble chronological chunk outputs into text.md and placeholders.json."""
    id_counts: dict[str, int] = {}
    unlabeled_counts: dict[int, int] = {}
    clean_chunk_texts: list[str] = []
    all_placeholders: list[PlaceholderItem] = []

    for (start_page, end_page, _), raw_chunk_md in zip(chunks, chunk_results):
        if not raw_chunk_md or not raw_chunk_md.strip():
            raise GeminiEmptyResponseError(
                stage="parse_text",
                details=f"Chunk covering pages {start_page}-{end_page} is empty.",
            )
        clean_chunk_md, chunk_phs = parse_placeholders_from_markdown(
            raw_chunk_md,
            start_page=start_page,
            end_page=end_page,
            id_counts=id_counts,
            unlabeled_counts=unlabeled_counts,
        )
        clean_chunk_texts.append(clean_chunk_md)
        all_placeholders.extend(chunk_phs)

    clean_md_parts: list[str] = []
    for chunk_text in clean_chunk_texts:
        cleaned = _strip_chunk_edge_page_breaks(chunk_text)
        cleaned = _PAGE_BREAK_RE.sub("", cleaned).strip()
        if cleaned:
            clean_md_parts.append(cleaned)

    clean_md = "\n\n<!-- page_break -->\n\n".join(clean_md_parts)

    if not clean_md or not clean_md.strip():
        raise GeminiEmptyResponseError(
            stage="parse_text",
            details=(
                f"Parsed text markdown for document '{artifacts.doc_name}' "
                f"is empty after combining chunks."
            ),
        )

    artifacts.save_text(artifacts.text_md_path, clean_md)
    artifacts.save_json(
        artifacts.placeholders_json_path,
        [p.model_dump() for p in all_placeholders],
    )
    return clean_md, all_placeholders


def _generate_chunk_markdown(
    client: GeminiClient,
    chunk_pdf_path: Path,
    start_page: int,
    end_page: int,
    static_system_prompt: str,
    stage_cfg: Any,
    config: Any,
    *,
    temperature_override: Optional[float] = None,
) -> str:
    """Invoke Gemini for a single PDF chunk using inline Part.from_bytes (or Files API fallback)."""
    page_ctx = format_parse_text_page_context(start_page, end_page)
    temp = stage_cfg.temperature if temperature_override is None else temperature_override

    if _should_use_inline_pdf(config, chunk_pdf_path):
        pdf_part = client.make_pdf_part(chunk_pdf_path)
        contents = [pdf_part, page_ctx] if page_ctx else [pdf_part]
        return client.generate_text(
            contents,
            system_instruction=static_system_prompt,
            stage_name="parse_text",
            model=stage_cfg.model,
            temperature=temp,
            max_output_tokens=stage_cfg.max_output_tokens,
            media_resolution=stage_cfg.media_resolution,
            thinking_level=stage_cfg.thinking_level,
            thinking_budget=stage_cfg.thinking_budget,
        )

    chunk_file_handle = None
    try:
        chunk_file_handle = client.upload_file(chunk_pdf_path)
        contents = [chunk_file_handle, page_ctx] if page_ctx else [chunk_file_handle]
        return client.generate_text(
            contents,
            system_instruction=static_system_prompt,
            stage_name="parse_text",
            model=stage_cfg.model,
            temperature=temp,
            max_output_tokens=stage_cfg.max_output_tokens,
            media_resolution=stage_cfg.media_resolution,
            thinking_level=stage_cfg.thinking_level,
            thinking_budget=stage_cfg.thinking_budget,
        )
    finally:
        if chunk_file_handle is not None:
            client.delete_file(chunk_file_handle)


def run_chunked_parse_text(
    client: GeminiClient,
    pdf_path: Path,
    artifacts: ArtifactManager,
    config: Any,
    chunk_pages: int = CHUNK_PAGES,
) -> tuple[str, list[PlaceholderItem]]:
    """Execute Stage 1 by processing document in 1-page chunks in parallel."""
    stage_cfg = config.gemini.get_stage_config("parse_text")
    temp_dir = artifacts.artifact_dir / "tmp_chunks"
    static_system_prompt = _read_prompt_template()
    use_guard = _is_silent_error_guard_enabled(config)
    page_texts = _extract_pdf_pages_text(pdf_path) if use_guard else {}

    logger.info(
        f"Stage 1 [parse_text]: Chunked text parsing enabled for '{artifacts.doc_name}' "
        f"({chunk_pages} page(s) per chunk).",
    )

    chunks = split_pdf_into_chunks(pdf_path, temp_dir=temp_dir, chunk_pages=chunk_pages)
    logger.info(
        f"Stage 1 [parse_text]: Split document into {len(chunks)} chunk(s).",
    )

    max_workers = getattr(config.gemini, "max_concurrent_requests", 4)
    chunk_results: list[Optional[str]] = [None] * len(chunks)

    def _process_chunk(idx: int, start_page: int, end_page: int, chunk_pdf_path: Path) -> str:
        chunk_cache_path = artifacts.chunks_dir / f"chunk_{start_page}_{end_page}.md"

        if chunk_cache_path.exists() and chunk_cache_path.stat().st_size > 0:
            logger.info(
                f"Stage 1 [parse_text]: Reusing cached chunk {idx}/{len(chunks)} "
                f"(pages {start_page}-{end_page}).",
            )
            return chunk_cache_path.read_text(encoding="utf-8")

        logger.info(
            f"Stage 1 [parse_text]: Parsing chunk {idx}/{len(chunks)} "
            f"(pages {start_page}-{end_page}) via Gemini…",
        )
        chunk_md = _generate_chunk_markdown(
            client=client,
            chunk_pdf_path=chunk_pdf_path,
            start_page=start_page,
            end_page=end_page,
            static_system_prompt=static_system_prompt,
            stage_cfg=stage_cfg,
            config=config,
        )

        if not chunk_md or not chunk_md.strip():
            raise GeminiEmptyResponseError(
                stage="parse_text",
                details=(
                    f"Gemini returned empty response for chunk {idx} "
                    f"(pages {start_page}-{end_page}) of '{artifacts.doc_name}'."
                ),
            )

        if use_guard:
            raw_pdf_txt = "\n".join(
                page_texts.get(p, "") for p in range(start_page, end_page + 1)
            )
            err_reason = detect_parse_text_silent_error(chunk_md, pdf_page_text=raw_pdf_txt)
            if err_reason:
                logger.warning(
                    f"Silent-Error Guard triggered on '{artifacts.doc_name}' pages {start_page}-{end_page} "
                    f"({err_reason}); retrying at temperature=0.5..."
                )
                if hasattr(client, "record_event"):
                    client.record_event("silent_error_retries")
                retry_md = _generate_chunk_markdown(
                    client=client,
                    chunk_pdf_path=chunk_pdf_path,
                    start_page=start_page,
                    end_page=end_page,
                    static_system_prompt=static_system_prompt,
                    stage_cfg=stage_cfg,
                    config=config,
                    temperature_override=0.5,
                )
                if retry_md and retry_md.strip():
                    retry_err = detect_parse_text_silent_error(retry_md, pdf_page_text=raw_pdf_txt)
                    if not retry_err or len(retry_md.strip()) > len(chunk_md.strip()):
                        chunk_md = retry_md

        chunk_cache_path.write_text(chunk_md, encoding="utf-8")
        return chunk_md

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, (start_page, end_page, chunk_pdf_path) in enumerate(chunks, 1):
                ctx = contextvars.copy_context()
                future = executor.submit(
                    ctx.run, _process_chunk, idx, start_page, end_page, chunk_pdf_path
                )
                future_to_idx[future] = idx - 1
            for future in concurrent.futures.as_completed(future_to_idx):
                i = future_to_idx[future]
                chunk_results[i] = future.result()

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    clean_md, all_placeholders = assemble_parse_text_chunks(chunks, chunk_results, artifacts)
    logger.info(
        f"Stage 1 [parse_text]: Finished chunked text parsing. "
        f"Found {len(all_placeholders)} visual placeholders across {len(chunks)} chunk(s).",
    )
    return clean_md, all_placeholders


def build_parse_text_batch_specs(
    client: GeminiClient,
    pdf_path: Path,
    artifacts: ArtifactManager,
    config: Any,
    chunk_pages: int = CHUNK_PAGES,
) -> tuple[list[tuple[int, int, Path]], list[BatchRequestSpec], Path]:
    """Prepare BatchRequestSpec objects for all uncached pages of a document."""
    stage_cfg = config.gemini.get_stage_config("parse_text")
    temp_dir = artifacts.artifact_dir / "tmp_chunks"
    static_system_prompt = _read_prompt_template()
    chunks = split_pdf_into_chunks(pdf_path, temp_dir=temp_dir, chunk_pages=chunk_pages)
    specs: list[BatchRequestSpec] = []

    for start_page, end_page, chunk_pdf_path in chunks:
        chunk_cache_path = artifacts.chunks_dir / f"chunk_{start_page}_{end_page}.md"
        if chunk_cache_path.exists() and chunk_cache_path.stat().st_size > 0:
            continue
        page_ctx = format_parse_text_page_context(start_page, end_page)
        pdf_part = client.make_pdf_part(chunk_pdf_path)
        contents = [pdf_part, page_ctx] if page_ctx else [pdf_part]
        specs.append(
            BatchRequestSpec(
                key=f"{artifacts.doc_name}__chunk_{start_page}_{end_page}",
                contents=contents,
                stage_name="parse_text",
                model=stage_cfg.model,
                system_instruction=static_system_prompt,
                max_output_tokens=stage_cfg.max_output_tokens,
                temperature=stage_cfg.temperature,
                media_resolution=stage_cfg.media_resolution,
                thinking_level=stage_cfg.thinking_level,
                thinking_budget=stage_cfg.thinking_budget,
                doc_scope=artifacts.doc_name,
            )
        )

    return chunks, specs, temp_dir


def run_parse_text(
    client: GeminiClient,
    file_handle: Any = None,
    artifacts: ArtifactManager = None,
    config: Any = None,
    pdf_path: Optional[Path] = None,
) -> tuple[str, list[PlaceholderItem]]:
    """Execute Stage 1: parse full text to Markdown and extract placeholders."""
    if artifacts.has_parse_text_artifact():
        logger.info(f"Stage 1 [parse_text]: Reusing cached artifacts for '{artifacts.doc_name}'")
        text_md = artifacts.load_text(artifacts.text_md_path)
        items_raw = artifacts.load_json(artifacts.placeholders_json_path)
        placeholders = [PlaceholderItem.model_validate(p) for p in items_raw]
        return text_md, placeholders

    if pdf_path and pdf_path.exists():
        return run_chunked_parse_text(
            client=client,
            pdf_path=pdf_path,
            artifacts=artifacts,
            config=config,
            chunk_pages=CHUNK_PAGES,
        )

    # Fallback for mock/handle-only invocations (e.g., tests without PDF files)
    stage_cfg = config.gemini.get_stage_config("parse_text")
    logger.info(f"Stage 1 [parse_text]: Parsing document '{artifacts.doc_name}' text via Gemini…")
    prompt = _read_prompt_template()

    local_handle = file_handle
    handle_created = False
    try:
        if local_handle is None and pdf_path and pdf_path.exists():
            local_handle = client.upload_file(pdf_path)
            handle_created = True

        contents = [local_handle] if local_handle else []
        raw_md = client.generate_text(
            contents,
            system_instruction=prompt,
            stage_name="parse_text",
            model=stage_cfg.model,
            temperature=stage_cfg.temperature,
            max_output_tokens=stage_cfg.max_output_tokens,
            media_resolution=stage_cfg.media_resolution,
            thinking_level=stage_cfg.thinking_level,
            thinking_budget=stage_cfg.thinking_budget,
        )
    finally:
        if handle_created and local_handle is not None:
            client.delete_file(local_handle)

    clean_md, placeholders = parse_placeholders_from_markdown(raw_md, start_page=1)
    if not clean_md or not clean_md.strip():
        raise GeminiEmptyResponseError(
            stage="parse_text",
            details=f"Parsed markdown for document '{artifacts.doc_name}' is empty.",
        )

    artifacts.save_text(artifacts.text_md_path, clean_md)
    artifacts.save_json(
        artifacts.placeholders_json_path,
        [p.model_dump() for p in placeholders],
    )
    logger.info(
        f"Stage 1 [parse_text]: Finished. Found {len(placeholders)} visual placeholders.",
    )
    return clean_md, placeholders
