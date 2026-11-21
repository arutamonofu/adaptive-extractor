"""Stage 02: Manifest Visuals (02_manifest_visuals).

Renders page previews with visual candidates and classifies placeholders into
the document visual manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import pymupdf as fitz

from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.schemas import (
    Manifest,
    ManifestVisual,
    PlaceholderItem,
)

logger = logging.getLogger(__name__)


def _read_prompt_template() -> str:
    prompt_file = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "02_manifest_visuals"
        / "manifest_visuals.txt"
    )
    return prompt_file.read_text(encoding="utf-8")


def get_required_pages(source: Union[list[PlaceholderItem], Manifest]) -> set[int]:
    """Get the union of 1-based page numbers required by placeholders or manifest visuals.

    For Unlabeled_* placeholders on page N (N > 1), also includes page N - 1 so multi-page
    tables have visual context of previous table headers during classification.
    """
    pages: set[int] = set()
    if isinstance(source, Manifest):
        for v in source.visuals:
            pages.update(v.pages)
    else:
        for p in source:
            if hasattr(p, "page") and p.page:
                pages.add(p.page)
                if getattr(p, "id", "").startswith("Unlabeled") and p.page > 1:
                    pages.add(p.page - 1)
    return pages


def run_render_pages_with_visuals(
    pdf_path: Path,
    source: Union[list[PlaceholderItem], Manifest],
    artifacts: ArtifactManager,
    config: Any,
) -> set[int]:
    """Render unique pages containing visuals to PNG scaled to <= 768 px."""
    pages = get_required_pages(source)
    if not pages:
        logger.info("Stage 02 [manifest_visuals]: No pages required. Skipping page rendering.")
        return set()

    max_size = getattr(config.gemini, "render_page_max_size", 768)

    doc = fitz.open(pdf_path)
    rendered_count = 0
    skipped_count = 0

    try:
        total_pdf_pages = len(doc)
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


def reconcile_manifest(
    model_manifest: Manifest,
    expected_placeholders: list[PlaceholderItem],
) -> Manifest:
    """Reconcile model output against expected placeholders:

    - Keeps only valid visuals matching expected placeholder IDs.
    - Adds missing placeholders to discarded_ids.
    - Ensures visuals preserve page invariants.
    """
    expected_ids = {p.id for p in expected_placeholders}
    ph_page_map = {p.id: p.page for p in expected_placeholders}

    valid_visuals: list[ManifestVisual] = []
    seen_ids: set[str] = set()

    for v in model_manifest.visuals:
        if v.id not in expected_ids:
            logger.warning(
                f"Model returned unexpected visual ID '{v.id}'. Discarding.",
            )
            continue

        exp_page = ph_page_map.get(v.id)
        if exp_page and exp_page not in v.pages:
            v.pages = sorted(set(v.pages) | {exp_page})

        valid_visuals.append(v)
        seen_ids.add(v.id)

    discarded = set(model_manifest.discarded_ids)
    for p in expected_placeholders:
        if p.id not in seen_ids:
            discarded.add(p.id)

    discarded = {d for d in discarded if d not in seen_ids}

    return Manifest(
        visuals=valid_visuals,
        discarded_ids=sorted(list(discarded)),
    )


def run_manifest_visuals(
    client: GeminiClient,
    placeholders_or_handle: Any,
    artifacts_or_placeholders: Any,
    config_or_artifacts: Any,
    config: Any = None,
    file_handle: Any = None,
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
    template = _read_prompt_template()

    placeholders_list_str = json.dumps(
        [{"id": p.id, "caption": p.caption, "page": p.page} for p in placeholders],
        indent=2,
        ensure_ascii=False,
    )
    prompt = template.replace("{placeholders_list}", placeholders_list_str)

    # Gather page previews for all placeholder pages
    target_pages: set[int] = set()
    for ph in placeholders:
        if ph.page:
            target_pages.add(ph.page)
        if ph.id.startswith("Unlabeled") and ph.page > 1:
            target_pages.add(ph.page - 1)

    contents: list[Any] = []
    for p in sorted(target_pages):
        page_img = artifacts.pages_dir / f"p{p}.png"
        if page_img.exists() and page_img.stat().st_size > 0:
            contents.append(f"=== Page {p} ===")
            contents.append(page_img)

    contents.append(prompt)

    raw_manifest = client.generate_structured(
        contents,
        response_schema=Manifest,
        stage_name="classify",
        model=stage_cfg.model,
        temperature=stage_cfg.temperature,
        max_output_tokens=stage_cfg.max_output_tokens,
        media_resolution=stage_cfg.media_resolution,
        thinking_level=stage_cfg.thinking_level,
        thinking_budget=stage_cfg.thinking_budget,
    )

    manifest = reconcile_manifest(raw_manifest, placeholders)
    artifacts.save_json(artifacts.manifest_json_path, manifest)

    total_visuals = len(manifest.visuals)
    logger.info(
        f"Stage 02 [manifest_visuals]: Manifest created with {total_visuals} visual(s), "
        f"{len(manifest.discarded_ids)} discarded visual(s).",
    )
    return manifest


# Alias for backward compatibility
run_classify = run_manifest_visuals
