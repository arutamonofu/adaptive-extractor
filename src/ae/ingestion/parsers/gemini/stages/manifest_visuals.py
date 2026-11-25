"""Stage 02: Manifest Visuals (02_manifest_visuals).

Renders page previews with visual candidates and classifies placeholders into
the document visual manifest.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import pymupdf as fitz

from ae.core.exceptions import GeminiMaxTokensError
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import BatchRequestSpec, GeminiClient
from ae.ingestion.parsers.gemini.schemas import (
    Manifest,
    ManifestVisual,
    PlaceholderItem,
)
from ae.ingestion.parsers.gemini.stages.silent_error_guard import (
    detect_manifest_silent_error,
)

logger = logging.getLogger(__name__)

_DUP_ID_RE = re.compile(r"_dup\d+$", re.IGNORECASE)
_CONT_ID_RE = re.compile(
    r"(?:^|[_\-.])cont(?:inued)?(?:[_\-.\d]|$)|(?:_|\d)cont(?:inued)?\d*$",
    re.IGNORECASE,
)


def _read_prompt_template() -> str:
    prompt_file = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "02_manifest_visuals"
        / "manifest_visuals.txt"
    )
    return prompt_file.read_text(encoding="utf-8")


def _split_manifest_prompt_template() -> tuple[str, str]:
    """Split manifest_visuals.txt into static system_instruction and dynamic user prompt."""
    template = _read_prompt_template()
    marker = "Placeholders:"
    if marker in template:
        idx = template.index(marker)
        return template[:idx].strip(), template[idx:].strip()
    return template.replace("{placeholders_list}", "").strip(), "Placeholders:\n{placeholders_list}"


def get_required_pages(
    source: Union[list[PlaceholderItem], Manifest],
    total_pages: Optional[int] = None,
) -> set[int]:
    """Get the union of 1-based page numbers required by placeholders or manifest visuals.

    For Unlabeled_* placeholders on page N, includes both N - 1 and N + 1 (bounded by
    1..total_pages) so both table continuations from previous pages and scheme/figure captions
    on the following page have visual context during classification.
    For *_dupN and *cont*/*continued* placeholders on page N (N > 1), includes N - 1.
    """
    pages: set[int] = set()
    if isinstance(source, Manifest):
        for v in source.visuals:
            for p in v.pages:
                if isinstance(p, int) and p >= 1 and (total_pages is None or p <= total_pages):
                    pages.add(p)
    else:
        for p in source:
            if hasattr(p, "page") and p.page and p.page >= 1:
                if total_pages is None or p.page <= total_pages:
                    pages.add(p.page)
                pid = getattr(p, "id", "") or ""
                pid_lower = pid.lower()
                if pid_lower.startswith("unlabeled"):
                    if p.page > 1 and (total_pages is None or p.page - 1 <= total_pages):
                        pages.add(p.page - 1)
                    if total_pages is None or p.page + 1 <= total_pages:
                        pages.add(p.page + 1)
                elif (_DUP_ID_RE.search(pid) or _CONT_ID_RE.search(pid)) and p.page > 1:
                    if total_pages is None or p.page - 1 <= total_pages:
                        pages.add(p.page - 1)
    return pages


def _get_render_page_max_size(config: Any) -> int:
    """Resolve Stage 2 page preview max dimension (px) from config."""
    try:
        stages = getattr(config, "stages", None) or getattr(getattr(config, "gemini", None), "stages", None)
        classify_stage = getattr(stages, "classify", None)
        if classify_stage is not None:
            val = getattr(classify_stage, "render_page_max_size", None)
            if val is not None:
                return int(val)
    except Exception:
        pass
    return int(getattr(getattr(config, "gemini", None), "render_page_max_size", 768) or 768)


def run_render_pages_with_visuals(
    pdf_path: Path,
    source: Union[list[PlaceholderItem], Manifest],
    artifacts: ArtifactManager,
    config: Any,
) -> set[int]:
    """Render unique pages containing visuals to PNG scaled to <= 768 px."""
    max_size = _get_render_page_max_size(config)

    doc = fitz.open(pdf_path)
    rendered_count = 0
    skipped_count = 0

    try:
        total_pdf_pages = len(doc)
        pages = get_required_pages(source, total_pages=total_pdf_pages)
        if not pages:
            logger.info("Stage 02 [manifest_visuals]: No pages required. Skipping page rendering.")
            return set()

        valid_pages = [p for p in sorted(pages) if 1 <= p <= total_pdf_pages]

        for p_num in valid_pages:
            img_path = artifacts.pages_dir / f"p{p_num}.png"
            if img_path.exists() and img_path.stat().st_size > 0:
                skipped_count += 1
                continue

            page = doc.load_page(p_num - 1)
            rect = page.rect
            w = rect.width
            h = rect.height

            # Scale to fit within max_size x max_size box preserving aspect ratio
            scale = min(1.0, max_size / max(w, h))
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(img_path))
            rendered_count += 1

    finally:
        doc.close()

    logger.info(
        f"Stage 02 [manifest_visuals]: Rendered {rendered_count} page(s) (<= {max_size} px), "
        f"reused {skipped_count} cached page(s) across {len(pages)} target page(s).",
    )
    return pages


# Alias for backward compatibility
run_render_pages = run_render_pages_with_visuals


def _normalize_visual_id(vis_id: str) -> str:
    """Normalize visual ID for fuzzy fallback matching (lowercase alphanumeric only)."""
    return re.sub(r"[^a-z0-9]+", "", vis_id.lower())


def _clean_visual_pages(
    pages: list[int],
    *,
    exp_page: Optional[int] = None,
    total_pages: Optional[int] = None,
) -> list[int]:
    """Validate, union expected placeholder page, sort, and deduplicate page numbers."""
    cleaned: set[int] = set()
    for p in pages:
        if not isinstance(p, int) or p < 1:
            continue
        if total_pages is not None and p > total_pages:
            continue
        cleaned.add(p)

    if exp_page and exp_page >= 1 and (total_pages is None or exp_page <= total_pages):
        cleaned.add(exp_page)

    return sorted(cleaned)


def _is_silent_error_guard_enabled(cfg: Any) -> bool:
    guard = getattr(getattr(cfg, "execution", None), "silent_error_guard", None)
    if guard is None:
        guard = getattr(getattr(cfg, "gemini", None), "silent_error_guard", True)
    return bool(guard)


def _build_manifest_contents(
    placeholders: list[PlaceholderItem],
    artifacts: ArtifactManager,
    user_template: Optional[str] = None,
    *,
    total_pages: Optional[int] = None,
) -> list[Any]:
    """Build multimodal prompt contents for Stage 02 manifest classification."""
    if user_template is None:
        _, user_template = _split_manifest_prompt_template()
    placeholders_list_str = json.dumps(
        [{"id": p.id, "caption": p.caption, "page": p.page} for p in placeholders],
        indent=2,
        ensure_ascii=False,
    )
    prompt = user_template.replace("{placeholders_list}", placeholders_list_str)

    target_pages = get_required_pages(placeholders, total_pages=total_pages)
    contents: list[Any] = []
    for p in sorted(target_pages):
        page_img = artifacts.pages_dir / f"p{p}.png"
        if page_img.exists() and page_img.stat().st_size > 0:
            contents.append(f"=== Page {p} ===")
            contents.append(page_img)
    contents.append(prompt)
    return contents


def reconcile_manifest(
    model_manifest: Manifest,
    expected_placeholders: list[PlaceholderItem],
    total_pages: Optional[int] = None,
) -> Manifest:
    """Reconcile model output against expected placeholders:

    - Keeps only valid visuals matching expected placeholder IDs (with normalized fuzzy fallback).
    - Deduplicates visual IDs inside visuals.
    - Resolves merged_from continuation placeholder IDs without putting them in discarded_ids.
    - Adds missing placeholders to discarded_ids.
    - Validates 1 <= page <= total_pages (when known) and sorts/deduplicates pages.
    """
    expected_ids = {p.id for p in expected_placeholders}
    ph_page_map = {p.id: p.page for p in expected_placeholders}

    norm_map: dict[str, list[str]] = {}
    for eid in expected_ids:
        norm_map.setdefault(_normalize_visual_id(eid), []).append(eid)

    def _resolve_expected_id(raw_id: str) -> Optional[str]:
        if raw_id in expected_ids:
            return raw_id
        cands = norm_map.get(_normalize_visual_id(raw_id), [])
        if len(cands) == 1:
            return cands[0]
        return None

    valid_visuals: list[ManifestVisual] = []
    valid_by_id: dict[str, ManifestVisual] = {}
    seen_ids: set[str] = set()

    for v in model_manifest.visuals:
        resolved_id = _resolve_expected_id(v.id)
        if resolved_id is None:
            logger.warning(
                f"Model returned unexpected visual ID '{v.id}'. Discarding.",
            )
            continue

        if resolved_id != v.id:
            logger.warning(
                f"Fuzzy-matched visual ID '{v.id}' to expected placeholder '{resolved_id}'.",
            )
            v = v.model_copy(update={"id": resolved_id})

        exp_page = ph_page_map.get(resolved_id)
        cleaned_pages = _clean_visual_pages(
            list(v.pages),
            exp_page=exp_page,
            total_pages=total_pages,
        )

        cleaned_merged: list[str] = []
        for raw_m in getattr(v, "merged_from", None) or []:
            resolved_m = _resolve_expected_id(raw_m)
            if resolved_m is not None and resolved_m != resolved_id and resolved_m not in cleaned_merged:
                cleaned_merged.append(resolved_m)

        if resolved_id in valid_by_id:
            logger.warning(
                f"Model returned duplicate visual ID '{resolved_id}'. Merging pages into first entry.",
            )
            existing = valid_by_id[resolved_id]
            existing.pages = _clean_visual_pages(
                list(existing.pages) + cleaned_pages,
                exp_page=exp_page,
                total_pages=total_pages,
            )
            merged_combined = list(existing.merged_from)
            for m_id in cleaned_merged:
                if m_id not in merged_combined:
                    merged_combined.append(m_id)
            existing.merged_from = merged_combined
            continue

        v = v.model_copy(update={"pages": cleaned_pages, "merged_from": cleaned_merged})
        valid_visuals.append(v)
        valid_by_id[resolved_id] = v
        seen_ids.add(resolved_id)

    all_merged_ids: set[str] = set()
    for v in valid_visuals:
        v.merged_from = [m for m in v.merged_from if m not in seen_ids]
        merged_pages = [ph_page_map[m] for m in v.merged_from if m in ph_page_map]
        if merged_pages:
            v.pages = _clean_visual_pages(
                list(v.pages) + merged_pages,
                total_pages=total_pages,
            )
        all_merged_ids.update(v.merged_from)

    discarded: set[str] = set()
    for d in model_manifest.discarded_ids:
        resolved_d = _resolve_expected_id(d)
        if resolved_d is not None:
            discarded.add(resolved_d)

    force_discarded: list[str] = []
    for p in expected_placeholders:
        if p.id not in seen_ids and p.id not in all_merged_ids:
            if p.id not in discarded:
                force_discarded.append(p.id)
            discarded.add(p.id)

    if force_discarded:
        logger.warning(
            f"Stage 02 [manifest_visuals]: Force-discarded {len(force_discarded)} unaccounted "
            f"placeholder(s) omitted by model: {force_discarded}"
        )

    discarded = {d for d in discarded if d not in seen_ids and d not in all_merged_ids}

    return Manifest(
        visuals=valid_visuals,
        discarded_ids=sorted(list(discarded)),
    )


def _manifest_error_metrics(
    raw_manifest: Manifest,
    reconciled_manifest: Manifest,
    expected_placeholders: list[PlaceholderItem],
) -> tuple[int, int]:
    """Return (unaccounted_count, rejected_raw_count) for comparing manifest quality."""
    expected_ids = {p.id for p in expected_placeholders}
    norm_map: dict[str, list[str]] = {}
    for eid in expected_ids:
        norm_map.setdefault(_normalize_visual_id(eid), []).append(eid)

    def _resolve_id(raw_id: str) -> Optional[str]:
        if raw_id in expected_ids:
            return raw_id
        cands = norm_map.get(_normalize_visual_id(raw_id), [])
        return cands[0] if len(cands) == 1 else None

    raw_visuals = getattr(raw_manifest, "visuals", []) or []
    raw_discarded = getattr(raw_manifest, "discarded_ids", []) or []
    reconciled_visuals = getattr(reconciled_manifest, "visuals", []) or []
    reconciled_vis_ids = {v.id for v in reconciled_visuals}
    reconciled_merged_ids = {
        m for v in reconciled_visuals for m in (getattr(v, "merged_from", None) or [])
    }
    raw_discarded_matched = {
        resolved
        for d in raw_discarded
        if (resolved := _resolve_id(d)) is not None
    }
    unaccounted_count = len(
        expected_ids - (reconciled_vis_ids | reconciled_merged_ids | raw_discarded_matched)
    )
    rejected_raw_count = sum(1 for v in raw_visuals if _resolve_id(v.id) is None)
    return unaccounted_count, rejected_raw_count


def validate_and_retry_manifest(
    client: GeminiClient,
    raw_manifest: Optional[Manifest],
    placeholders: list[PlaceholderItem],
    artifacts: ArtifactManager,
    cfg: Any,
    *,
    total_pages: Optional[int] = None,
    initial_max_tokens_exc: Optional[GeminiMaxTokensError] = None,
) -> Manifest:
    """Reconcile manifest and retry once at temperature=0.5 if Stage 2 silent/truncation error occurs."""
    manifest: Optional[Manifest] = None
    if initial_max_tokens_exc is not None:
        err_reason: Optional[str] = f"max_tokens_truncation: {initial_max_tokens_exc}"
    else:
        assert raw_manifest is not None
        manifest = reconcile_manifest(raw_manifest, placeholders, total_pages=total_pages)
        if not _is_silent_error_guard_enabled(cfg):
            return manifest
        err_reason = detect_manifest_silent_error(raw_manifest, manifest, placeholders)
        if not err_reason:
            return manifest

    logger.warning(
        f"Silent-Error Guard triggered on Stage 02 manifest for '{artifacts.doc_name}' "
        f"({err_reason}); retrying at temperature=0.5..."
    )
    if hasattr(client, "record_event"):
        client.record_event("silent_error_retries")

    stage_cfg = cfg.gemini.get_stage_config("classify")
    static_sys_prompt, user_tpl = _split_manifest_prompt_template()
    contents = _build_manifest_contents(
        placeholders, artifacts, user_template=user_tpl, total_pages=total_pages
    )
    try:
        retry_raw = client.generate_structured(
            contents,
            response_schema=Manifest,
            stage_name="classify",
            model=stage_cfg.model,
            system_instruction=static_sys_prompt,
            temperature=0.5,
            max_output_tokens=stage_cfg.max_output_tokens,
            media_resolution=stage_cfg.media_resolution,
            thinking_level=stage_cfg.thinking_level,
            thinking_budget=stage_cfg.thinking_budget,
        )
    except GeminiMaxTokensError:
        if initial_max_tokens_exc is not None:
            raise
        assert manifest is not None
        return manifest
    except Exception as exc:
        logger.warning(
            f"Silent-Error Guard Stage 02 retry failed for '{artifacts.doc_name}': {exc}"
        )
        if initial_max_tokens_exc is not None:
            raise initial_max_tokens_exc from exc
        assert manifest is not None
        return manifest

    retry_manifest = reconcile_manifest(retry_raw, placeholders, total_pages=total_pages)
    retry_err = detect_manifest_silent_error(retry_raw, retry_manifest, placeholders)
    if initial_max_tokens_exc is not None:
        if retry_err is not None:
            logger.error(
                f"Silent-Error Guard Stage 02 retry after MAX_TOKENS for '{artifacts.doc_name}' "
                f"failed validation ({retry_err}); refusing to return broken manifest."
            )
            raise initial_max_tokens_exc
        return retry_manifest
    assert raw_manifest is not None and manifest is not None
    if retry_err is None:
        return retry_manifest

    orig_unaccounted, orig_rejected = _manifest_error_metrics(raw_manifest, manifest, placeholders)
    retry_unaccounted, retry_rejected = _manifest_error_metrics(
        retry_raw, retry_manifest, placeholders
    )
    if (retry_unaccounted + retry_rejected) < (orig_unaccounted + orig_rejected) and (
        retry_rejected <= orig_rejected
    ):
        return retry_manifest
    return manifest


def run_manifest_visuals(
    client: GeminiClient,
    placeholders_or_handle: Any,
    artifacts_or_placeholders: Any,
    config_or_artifacts: Any,
    config: Any = None,
    file_handle: Any = None,
    *,
    total_pages: Optional[int] = None,
) -> Manifest:
    """Execute Stage 02: manifest visuals from placeholders and page previews."""
    if isinstance(placeholders_or_handle, list):
        placeholders: list[PlaceholderItem] = placeholders_or_handle
        artifacts: ArtifactManager = artifacts_or_placeholders
        cfg: Any = config_or_artifacts
    else:
        placeholders = artifacts_or_placeholders
        artifacts = config_or_artifacts
        cfg = config

    if artifacts.has_manifest_artifact():
        logger.info(f"Stage 02 [manifest_visuals]: Reusing cached manifest for '{artifacts.doc_name}'")
        raw = artifacts.load_json(artifacts.manifest_json_path)
        return Manifest.model_validate(raw)

    if not placeholders:
        logger.info("Stage 02 [manifest_visuals]: No placeholders found. Creating empty manifest.")
        empty_manifest = Manifest(visuals=[], discarded_ids=[])
        artifacts.save_json(artifacts.manifest_json_path, empty_manifest)
        return empty_manifest

    logger.info(
        f"Stage 02 [manifest_visuals]: Classifying {len(placeholders)} placeholders via Gemini…",
    )
    stage_cfg = cfg.gemini.get_stage_config("classify")
    static_sys_prompt, user_tpl = _split_manifest_prompt_template()
    contents = _build_manifest_contents(
        placeholders, artifacts, user_template=user_tpl, total_pages=total_pages
    )

    initial_max_tokens_exc: Optional[GeminiMaxTokensError] = None
    raw_manifest: Optional[Manifest] = None
    try:
        raw_manifest = client.generate_structured(
            contents,
            response_schema=Manifest,
            stage_name="classify",
            model=stage_cfg.model,
            system_instruction=static_sys_prompt,
            temperature=stage_cfg.temperature,
            max_output_tokens=stage_cfg.max_output_tokens,
            media_resolution=stage_cfg.media_resolution,
            thinking_level=stage_cfg.thinking_level,
            thinking_budget=stage_cfg.thinking_budget,
        )
    except GeminiMaxTokensError as exc:
        if not _is_silent_error_guard_enabled(cfg):
            raise
        initial_max_tokens_exc = exc

    manifest = validate_and_retry_manifest(
        client,
        raw_manifest,
        placeholders,
        artifacts,
        cfg,
        total_pages=total_pages,
        initial_max_tokens_exc=initial_max_tokens_exc,
    )
    artifacts.save_json(artifacts.manifest_json_path, manifest)

    total_visuals = len(manifest.visuals)
    logger.info(
        f"Stage 02 [manifest_visuals]: Manifest created with {total_visuals} visual(s), "
        f"{len(manifest.discarded_ids)} discarded visual(s).",
    )
    return manifest


# Alias for backward compatibility
run_classify = run_manifest_visuals


def build_manifest_visuals_batch_spec(
    placeholders: list[PlaceholderItem],
    artifacts: ArtifactManager,
    cfg: Any,
    *,
    total_pages: Optional[int] = None,
) -> Optional[BatchRequestSpec]:
    """Prepare a BatchRequestSpec for Stage 02 manifest classification if not already cached."""
    if artifacts.has_manifest_artifact():
        return None
    if not placeholders:
        empty_manifest = Manifest(visuals=[], discarded_ids=[])
        artifacts.save_json(artifacts.manifest_json_path, empty_manifest)
        return None

    stage_cfg = cfg.gemini.get_stage_config("classify")
    static_sys_prompt, user_tpl = _split_manifest_prompt_template()
    contents = _build_manifest_contents(
        placeholders, artifacts, user_template=user_tpl, total_pages=total_pages
    )

    return BatchRequestSpec(
        key=f"{artifacts.doc_name}__classify",
        contents=contents,
        stage_name="classify",
        model=stage_cfg.model,
        system_instruction=static_sys_prompt,
        max_output_tokens=stage_cfg.max_output_tokens,
        temperature=stage_cfg.temperature,
        media_resolution=stage_cfg.media_resolution,
        thinking_level=stage_cfg.thinking_level,
        thinking_budget=stage_cfg.thinking_budget,
        response_schema=Manifest,
        doc_scope=artifacts.doc_name,
    )
