"""Stage 03: Segment and Crop Visuals and Elements (03_segment_and_crop_visuals_elements).

Provides:
- Visuals level: full visual bbox localization on pages, vector PDF cropping
- Elements level: element bbox localization on visual crops, element vector PDF cropping
- Visual decomposition: single vs composite with composition_template
- Element filtering: selective sub-element selection by target hint
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pymupdf as fitz
from PIL import Image, ImageDraw

from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.schemas import (
    ElementLocalizationResult,
    ElementManifestItem,
    ElementSelectionResult,
    Manifest,
    PageLocalization,
    VisualBox,
    VisualDecomposition,
)

logger = logging.getLogger(__name__)

# Fallback DPI settings if configuration is missing
_DEFAULT_DPI_BOUNDS = {
    "molecular_structure": (400, 1200),
    "table": (400, 900),
    "chart": (300, 600),
    "mixed": (400, 1000),
}


# ==============================================================================
# Prompts
# ==============================================================================


def _read_visuals_bboxes_prompt() -> str:
    prompt_file = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "03_segment_and_crop_visuals_elements"
        / "visuals_bboxes.txt"
    )
    if not prompt_file.exists():
        prompt_file = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "03_segment_and_crop_visuals_atoms"
            / "visuals_bboxes.txt"
        )
    return prompt_file.read_text(encoding="utf-8")


def _read_visuals_segmentations_prompt() -> str:
    path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "03_segment_and_crop_visuals_elements"
        / "visuals_segmentations.txt"
    )
    if not path.exists():
        path = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "03_segment_and_crop_visuals_atoms"
            / "visuals_segmentations.txt"
        )
    return path.read_text(encoding="utf-8")


def _read_elements_bboxes_prompt() -> str:
    path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "03_segment_and_crop_visuals_elements"
        / "elements_bboxes.txt"
    )
    if not path.exists():
        path = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "03_segment_and_crop_visuals_atoms"
            / "atoms_bboxes.txt"
        )
    return path.read_text(encoding="utf-8")


_read_atoms_bboxes_prompt = _read_elements_bboxes_prompt


# ==============================================================================
# Visuals: Localization & Bounding Boxes
# ==============================================================================


def validate_box(box: VisualBox) -> VisualBox:
    """Validate normalized [ymin, xmin, ymax, xmax] coordinates.

    If invalid, sets found=False.
    """
    if not box.found:
        return box

    coords = box.box_2d
    if len(coords) != 4:
        box.found = False
        return box

    ymin, xmin, ymax, xmax = coords
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        box.found = False

    return box


def localize_page(
    client: GeminiClient,
    page_img_path: Path,
    page_num: int,
    manifest: Manifest,
    config: Any,
) -> PageLocalization:
    """Run PageLocalization via Gemini structured output for target page visuals."""
    visuals_for_page = [v for v in manifest.visuals if page_num in v.pages]
    if not visuals_for_page:
        logger.info(f"Page {page_num}: No visuals manifested. Returning empty.")
        return PageLocalization(page=page_num, boxes=[])

    visuals_list_text = "\n".join(
        f"- ID: {v.id}, Type: {v.type}, Description: {v.description}"
        for v in visuals_for_page
    )

    template = _read_visuals_bboxes_prompt()
    prompt = template.replace("{page_number}", str(page_num)).replace(
        "{visuals_list}", visuals_list_text
    )

    stage_cfg = config.gemini.get_stage_config("localize")
    contents = [page_img_path, prompt]

    logger.info(
        f"Localizing {len(visuals_for_page)} visuals on page {page_num} using Gemini ({stage_cfg.model})..."
    )
    result: PageLocalization = client.generate_structured(
        contents,
        response_schema=PageLocalization,
        stage_name="localize_visuals",
        model=stage_cfg.model,
        temperature=stage_cfg.temperature,
        max_output_tokens=stage_cfg.max_output_tokens,
        media_resolution=stage_cfg.media_resolution,
        thinking_level=stage_cfg.thinking_level,
        thinking_budget=stage_cfg.thinking_budget,
    )

    result.page = page_num
    manifest_ids = {v.id for v in visuals_for_page}
    boxes_by_id: dict[str, VisualBox] = {}

    for b in result.boxes:
        if b.id in manifest_ids:
            boxes_by_id[b.id] = validate_box(b)

    for v in visuals_for_page:
        if v.id not in boxes_by_id:
            boxes_by_id[v.id] = VisualBox(id=v.id, box_2d=[0, 0, 0, 0], found=False)

    result.boxes = list(boxes_by_id.values())
    found_count = sum(1 for b in result.boxes if b.found)
    logger.info(
        f"Page {page_num}: Localized {found_count}/{len(visuals_for_page)} visuals successfully."
    )
    return result


def run_localize_visuals(
    client: GeminiClient,
    manifest: Manifest,
    artifacts: ArtifactManager,
    config: Any,
) -> dict[int, PageLocalization]:
    """Stage 03: Localize bounding boxes for each page containing visuals."""
    if not manifest.visuals:
        logger.info("Manifest has zero visuals. Skipping localization.")
        artifacts.mark_visual_bboxes_empty()
        return {}

    needed_pages = sorted({p for v in manifest.visuals for p in v.pages})
    existing = artifacts.load_all_page_bboxes()
    missing_pages = [p for p in needed_pages if p not in existing]

    if not missing_pages:
        logger.info(
            f"All {len(needed_pages)} required pages have existing localization bboxes. Reusing."
        )
        return {p: existing[p] for p in needed_pages if p in existing}

    logger.info(
        f"Localization needed for pages: {missing_pages} (already cached: {list(existing.keys())})"
    )

    results = dict(existing)
    for p_num in missing_pages:
        img_path = artifacts.pages_dir / f"p{p_num}.png"
        if not img_path.exists():
            logger.warning(
                f"Page image {img_path} not found. Returning empty localization for p{p_num}."
            )
            empty_loc = PageLocalization(page=p_num, boxes=[])
            artifacts.save_page_bboxes(p_num, empty_loc)
            results[p_num] = empty_loc
            continue

        loc = localize_page(client, img_path, p_num, manifest, config)
        artifacts.save_page_bboxes(p_num, loc)
        results[p_num] = loc

    return results


run_localize = run_localize_visuals


# ==============================================================================
# Visuals: Vector PDF Cropping & Adaptive DPI
# ==============================================================================


def _compute_target_dpi(
    clip_rect: fitz.Rect,
    crop_cfg: Any,
    default_base_dpi: int = 400,
    default_min_dpi: int = 200,
    default_max_dpi: int = 1200,
) -> int:
    """Compute adaptive DPI for vector PDF rendering.

    Small physical crops are scaled up to guarantee clear text and fine bonds,
    while large full-page crops stay near base DPI to prevent token explosion.
    """
    base_dpi = getattr(crop_cfg, "base_dpi", default_base_dpi) if crop_cfg else default_base_dpi
    upscale_cfg = getattr(crop_cfg, "upscale", None) if crop_cfg else None
    min_long_side = 2000
    max_dpi = default_max_dpi
    if upscale_cfg:
        min_long_side = getattr(upscale_cfg, "min_long_side_px", 2000)
        max_dpi = getattr(upscale_cfg, "max_dpi", default_max_dpi)

    w = max(clip_rect.width, 1.0)
    h = max(clip_rect.height, 1.0)
    long_side_pt = max(w, h)

    ideal_dpi = int((min_long_side * 72.0) / long_side_pt)
    computed = max(base_dpi, ideal_dpi)
    return int(max(default_min_dpi, min(computed, max_dpi)))


def compute_crop_rect_and_dpi(
    arg1: Any,
    arg2: Any,
    crop_cfg: Any = None,
    *args: Any,
) -> tuple[fitz.Rect, int]:
    """Calculate the vector PDF clip rectangle with padding and dynamic DPI."""
    if isinstance(arg1, fitz.Page) or (isinstance(arg1, fitz.Rect) and not isinstance(arg2, fitz.Rect)):
        page_or_rect = arg1
        box = arg2
        if args and crop_cfg is not None and not hasattr(crop_cfg, "base_dpi") and not hasattr(crop_cfg, "padding_ratio"):
            crop_cfg = args[0]
    else:
        box = arg1
        page_or_rect = arg2

    box_2d = box.box_2d if hasattr(box, "box_2d") else box
    page_rect = (
        page_or_rect.rect
        if hasattr(page_or_rect, "rect") and not isinstance(page_or_rect, fitz.Rect)
        else page_or_rect
    )
    ymin, xmin, ymax, xmax = box_2d
    w = page_rect.width
    h = page_rect.height

    x0 = (xmin / 1000.0) * w
    y0 = (ymin / 1000.0) * h
    x1 = (xmax / 1000.0) * w
    y1 = (ymax / 1000.0) * h

    padding_ratio = getattr(crop_cfg, "padding_ratio", 0.02) if crop_cfg else 0.02
    pad_x = padding_ratio * w
    pad_y = padding_ratio * h

    clip_rect = fitz.Rect(
        max(page_rect.x0, x0 - pad_x),
        max(page_rect.y0, y0 - pad_y),
        min(page_rect.x1, x1 + pad_x),
        min(page_rect.y1, y1 + pad_y),
    )

    base_dpi = getattr(crop_cfg, "base_dpi", 300) if crop_cfg else 300
    upscale_cfg = getattr(crop_cfg, "upscale", None) if crop_cfg else None

    if upscale_cfg and getattr(upscale_cfg, "enabled", True):
        long_side_pt = max(clip_rect.width, clip_rect.height)
        min_long_side_px = getattr(upscale_cfg, "min_long_side_px", 2000)
        max_dpi = getattr(upscale_cfg, "max_dpi", 1200)
        ideal_dpi = int((min_long_side_px * 72.0) / max(long_side_pt, 1.0))
        dpi = int(min(max(base_dpi, ideal_dpi), max_dpi))
    else:
        dpi = int(base_dpi)

    return clip_rect, dpi


def generate_debug_overlay(
    page_img_path: Path,
    boxes: list[tuple[str, list[int], str]],
    output_path: Path,
    padding_ratio: float = 0.0,
) -> None:
    """Draw bounding boxes and labels onto page preview image for inspection."""
    if not page_img_path.exists():
        return

    type_colors = {
        "molecular_structure": "red",
        "table": "blue",
        "chart": "green",
        "mixed": "purple",
    }

    try:
        with Image.open(page_img_path) as im:
            draw_img = im.convert("RGB")
            draw = ImageDraw.Draw(draw_img)
            w, h = draw_img.size

            for b_id, coords, b_type in boxes:
                if len(coords) != 4:
                    continue
                ymin, xmin, ymax, xmax = coords
                x0 = (xmin / 1000.0) * w
                y0 = (ymin / 1000.0) * h
                x1 = (xmax / 1000.0) * w
                y1 = (ymax / 1000.0) * h

                if padding_ratio > 0.0:
                    px = padding_ratio * (x1 - x0)
                    py = padding_ratio * (y1 - y0)
                    x0 = max(0, x0 - px)
                    y0 = max(0, y0 - py)
                    x1 = min(w, x1 + px)
                    y1 = min(h, y1 + py)

                color = type_colors.get(b_type, "orange")
                draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
                draw.text((x0 + 4, max(0, y0 - 15)), f"{b_id} ({b_type})", fill=color)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            draw_img.save(str(output_path), "PNG")
    except Exception as exc:
        logger.warning(f"Could not generate debug overlay: {exc}")


def run_crop_visuals(
    pdf_path: Path,
    manifest: Manifest,
    localizations: dict[int, PageLocalization],
    artifacts: ArtifactManager,
    config: Any,
) -> dict[str, list[Path]]:
    """Stage 03: Crop all localized visuals directly from vector PDF pages."""
    crops_by_visual: dict[str, list[Path]] = {v.id: [] for v in manifest.visuals}
    crop_cfg = config.gemini.crop if config and hasattr(config, "gemini") else None

    visual_map = {v.id: v for v in manifest.visuals}
    doc = fitz.open(str(pdf_path))

    try:
        for page_num, loc in localizations.items():
            if page_num < 1 or page_num > len(doc):
                continue
            page = doc[page_num - 1]
            debug_boxes: list[tuple[str, list[int], str]] = []

            for box in loc.boxes:
                if not box.found or box.id not in visual_map:
                    continue

                v = visual_map[box.id]
                debug_boxes.append((box.id, box.box_2d, v.type))

                part_suffix = f"_p{page_num}" if len(v.pages) > 1 else ""
                crop_filename = f"{box.id}{part_suffix}.png"
                crop_path = artifacts.crops_dir / crop_filename

                if not crop_path.exists():
                    clip_rect, dpi = compute_crop_rect_and_dpi(page, box, v.type, crop_cfg)
                    pix = page.get_pixmap(dpi=dpi, clip=clip_rect)
                    pix.save(str(crop_path))
                    logger.debug(
                        f"Cropped {box.id} on p{page_num} at {dpi} DPI -> {crop_filename}"
                    )

                crops_by_visual[box.id].append(crop_path)

            if debug_boxes:
                page_img = artifacts.pages_dir / f"p{page_num}.png"
                debug_path = artifacts.debug_dir / f"p{page_num}_localized_bboxes.png"
                padding = getattr(crop_cfg, "padding_ratio", 0.04) if crop_cfg else 0.04
                generate_debug_overlay(page_img, debug_boxes, debug_path, padding_ratio=padding)

    finally:
        doc.close()

    logger.info(
        f"Completed cropping {sum(len(c) for c in crops_by_visual.values())} crop file(s) "
        f"across {len(crops_by_visual)} visual(s).",
    )
    return crops_by_visual


run_crop = run_crop_visuals


def run_segment_and_crop(
    pdf_path: Path,
    manifest: Manifest,
    localizations: dict[int, PageLocalization],
    artifacts: ArtifactManager,
    config: Any,
) -> dict[str, list[Path]]:
    """Stage 03 orchestration helper: crops visuals from PDF."""
    return run_crop_visuals(pdf_path, manifest, localizations, artifacts, config)


# ==============================================================================
# Visuals: Segmentation & Decomposition
# ==============================================================================


def decompose_visual(
    client: GeminiClient,
    visual_id: str,
    crop_paths: list[Path],
    caption: str,
    visual_type: str,
    artifacts: ArtifactManager,
    config: Any,
    description: str = "",
    local_context: str = "",
) -> VisualDecomposition:
    """Analyze and segment/decompose a cropped visual into single or composite with elements."""
    cached_path = artifacts.decompositions_dir / f"{visual_id}.json"
    if cached_path.exists():
        logger.debug(f"Loading cached segmentation for '{visual_id}' from {cached_path}")
        data = artifacts.load_json(cached_path)
        return VisualDecomposition.model_validate(data)

    template = _read_visuals_segmentations_prompt()
    prompt = (
        template.replace("{visual_id}", visual_id)
        .replace("{block_id}", visual_id)
        .replace("{caption}", caption or "(none)")
        .replace("{visual_type}", visual_type or "mixed")
        .replace("{block_type}", visual_type or "mixed")
        .replace("{description}", description or "(none)")
        .replace("{local_context}", local_context or "(none)")
    )

    stage_cfg = config.gemini.get_stage_config("decompose")
    contents = list(crop_paths) + [prompt]

    logger.info(f"Segmenting visual '{visual_id}' using Gemini ({stage_cfg.model})...")
    decomposition: VisualDecomposition = client.generate_structured(
        contents,
        response_schema=VisualDecomposition,
        stage_name="decompose",
        model=stage_cfg.model,
        temperature=stage_cfg.temperature,
        max_output_tokens=stage_cfg.max_output_tokens,
        media_resolution=stage_cfg.media_resolution,
        thinking_level=stage_cfg.thinking_level,
        thinking_budget=stage_cfg.thinking_budget,
    )

    artifacts.save_json(cached_path, decomposition)
    elements_count = len(decomposition.elements)
    logger.info(
        f"Visual '{visual_id}': mode='{decomposition.mode}', elements={elements_count}"
    )
    return decomposition


segment_visual = decompose_visual
decompose_block = decompose_visual


def select_elements_by_hint(
    client: GeminiClient,
    visual_id: str,
    elements: list[ElementManifestItem],
    target_hint: str,
    config: Any,
) -> list[str]:
    """Select relevant element IDs matching target_hint using a fast structured LLM call."""
    if not elements or not target_hint or not target_hint.strip():
        return [e.element_id for e in elements]

    elem_id_set = {e.element_id for e in elements}
    hint_lower = target_hint.lower()
    explicit_matches = [
        e.element_id
        for e in elements
        if (e.label and e.label.lower() in hint_lower) or e.element_id.lower() in hint_lower
    ]
    if explicit_matches:
        logger.info(
            f"Heuristic matched {len(explicit_matches)} element(s) for hint '{target_hint}': {explicit_matches}"
        )
        return explicit_matches

    elements_list_text = "\n".join(
        f"- ID: '{e.element_id}', Type: '{e.type}', Label: '{e.label}': {e.description}"
        for e in elements
    )
    prompt = (
        f"You are selecting relevant visual sub-elements of a figure/scheme for extraction.\n\n"
        f"Figure ID: {visual_id}\n\n"
        f"Available elements in this figure:\n"
        f"{elements_list_text}\n\n"
        f"Extraction Target Hint:\n"
        f"\"{target_hint}\"\n\n"
        f"Task:\n"
        f"Identify which element IDs are relevant to the extraction target hint.\n"
        f"- If the target asks for specific compounds, products, reactants, or chart panels, select ONLY the matching elements.\n"
        f"- If unsure or if the hint is broad/ambiguous, select all potentially relevant elements.\n"
        f"- Do NOT invent IDs; choose strictly from the IDs listed above."
    )

    stage_cfg = (
        config.gemini.get_stage_config("decompose")
        if config and getattr(config, "gemini", None)
        else None
    )
    model = getattr(stage_cfg, "model", None) or "gemini-2.5-flash"

    try:
        res: ElementSelectionResult = client.generate_structured(
            [prompt],
            response_schema=ElementSelectionResult,
            stage_name="select_elements_by_hint",
            model=model,
            temperature=0.0,
            max_output_tokens=500,
        )
        valid_selected = [eid for eid in res.selected_element_ids if eid in elem_id_set]
        if valid_selected:
            logger.info(
                f"Selected {len(valid_selected)}/{len(elements)} element(s) for '{visual_id}' based on hint '{target_hint}': {valid_selected}"
            )
            return valid_selected
    except Exception as exc:
        logger.warning(
            f"Failed to filter elements by hint '{target_hint}' for '{visual_id}': {exc}. Falling back to all elements."
        )

    return [e.element_id for e in elements]


select_atoms_by_hint = select_elements_by_hint


# ==============================================================================
# Elements: Localization & Bounding Boxes
# ==============================================================================


def localize_elements(
    client: GeminiClient,
    visual_id: str,
    crop_paths: list[Path],
    elements: list[ElementManifestItem],
    artifacts: ArtifactManager,
    config: Any,
) -> ElementLocalizationResult:
    """Detect bounding boxes of elements on a composite visual crop."""
    cached_path = artifacts.element_bboxes_dir / f"{visual_id}.json"
    if cached_path.exists():
        logger.debug(f"Loading cached element bboxes for '{visual_id}' from {cached_path}")
        data = artifacts.load_json(cached_path)
        return ElementLocalizationResult.model_validate(data)

    elements_list_text = "\n".join(
        f"- id='{e.element_id}', type='{e.type}', label='{e.label}': {e.description}"
        for e in elements
    )
    template = _read_elements_bboxes_prompt()
    prompt = template.replace("{visual_id}", visual_id).replace("{elements_list}", elements_list_text)

    stage_cfg = config.gemini.get_stage_config("localize")
    contents = list(crop_paths) + [prompt]

    logger.info(f"Localizing {len(elements)} elements for visual '{visual_id}'...")
    loc_result: ElementLocalizationResult = client.generate_structured(
        contents,
        response_schema=ElementLocalizationResult,
        stage_name="localize_elements",
        model=stage_cfg.model,
        temperature=stage_cfg.temperature,
        max_output_tokens=stage_cfg.max_output_tokens,
        media_resolution=stage_cfg.media_resolution,
        thinking_level=stage_cfg.thinking_level,
        thinking_budget=stage_cfg.thinking_budget,
    )

    for box in loc_result.boxes:
        coords = box.box_2d
        if len(coords) == 4:
            ymin, xmin, ymax, xmax = coords
            if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
                box.found = False
        else:
            box.found = False

        valid_extras: list[list[int]] = []
        for extra in getattr(box, "extra_boxes_2d", []):
            if len(extra) == 4:
                eymin, exmin, eymax, exmax = extra
                if 0 <= eymin < eymax <= 1000 and 0 <= exmin < exmax <= 1000:
                    valid_extras.append(extra)
        box.extra_boxes_2d = valid_extras

    artifacts.save_json(cached_path, loc_result)

    # Render debug overlay for element bounding boxes on first visual crop
    if crop_paths and crop_paths[0].exists():
        element_type_map = {e.element_id: e.type for e in elements}
        debug_boxes: list[tuple[str, list[int], str]] = []
        for box in loc_result.boxes:
            if not box.found:
                continue
            e_type = element_type_map.get(box.id, "molecular_structure")
            debug_boxes.append((box.id, box.box_2d, e_type))
            for idx, extra in enumerate(box.extra_boxes_2d, start=1):
                debug_boxes.append((f"{box.id}[+{idx}]", extra, e_type))
        if debug_boxes:
            debug_img_path = artifacts.debug_dir / f"{visual_id}_elements_bboxes.png"
            generate_debug_overlay(crop_paths[0], debug_boxes, debug_img_path, padding_ratio=0.04)

    return loc_result


localize_atoms = localize_elements


# ==============================================================================
# Elements: Vector PDF Cropping
# ==============================================================================


def crop_elements(
    pdf_path: Path,
    visual_id: str,
    page_num: int,
    visual_clip_rect: fitz.Rect,
    element_boxes: list[VisualBox],
    artifacts: ArtifactManager,
    config: Any,
) -> dict[str, list[Path]]:
    """Vector-crop elementary units directly from the PDF page."""
    element_crops: dict[str, list[Path]] = {}
    if not element_boxes:
        return element_crops

    element_crops_dir = artifacts.element_crops_dir
    element_crops_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_num - 1]
        width_pt = visual_clip_rect.width
        height_pt = visual_clip_rect.height

        crop_cfg = config.gemini.crop if config and hasattr(config, "gemini") else None

        for box in element_boxes:
            if not box.found:
                continue

            all_box_coords = [box.box_2d]
            if hasattr(box, "extra_boxes_2d") and box.extra_boxes_2d:
                all_box_coords.extend(box.extra_boxes_2d)

            crops_for_elem: list[Path] = []
            for part_idx, coords in enumerate(all_box_coords):
                ymin, xmin, ymax, xmax = coords
                part_x0 = visual_clip_rect.x0 + (xmin / 1000.0) * width_pt
                part_y0 = visual_clip_rect.y0 + (ymin / 1000.0) * height_pt
                part_x1 = visual_clip_rect.x0 + (xmax / 1000.0) * width_pt
                part_y1 = visual_clip_rect.y0 + (ymax / 1000.0) * height_pt

                pad_x = 0.04 * (part_x1 - part_x0)
                pad_y = 0.04 * (part_y1 - part_y0)
                clip_rect = fitz.Rect(
                    max(page.rect.x0, part_x0 - pad_x),
                    max(page.rect.y0, part_y0 - pad_y),
                    min(page.rect.x1, part_x1 + pad_x),
                    min(page.rect.y1, part_y1 + pad_y),
                )

                part_suffix = f"_part{part_idx}" if len(all_box_coords) > 1 else ""
                crop_file = element_crops_dir / f"{visual_id}__{box.id}{part_suffix}.png"
                if not crop_file.exists():
                    target_dpi = _compute_target_dpi(clip_rect, crop_cfg, default_base_dpi=400)
                    pix = page.get_pixmap(dpi=target_dpi, clip=clip_rect)
                    pix.save(str(crop_file))

                crops_for_elem.append(crop_file)

            if crops_for_elem:
                element_crops[box.id] = crops_for_elem

    finally:
        doc.close()

    return element_crops


crop_atoms = crop_elements
