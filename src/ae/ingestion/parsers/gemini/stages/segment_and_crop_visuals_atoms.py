"""Legacy proxy module for Stage 03 (segment_and_crop_visuals_atoms).

Maintained for backward compatibility. All implementations have been renamed
and migrated to `segment_and_crop_visuals_elements.py`.
"""

from __future__ import annotations

from ae.ingestion.parsers.gemini.stages.segment_and_crop_visuals_elements import (
    _compute_target_dpi,
    _DEFAULT_DPI_BOUNDS,
    _read_atoms_bboxes_prompt,
    _read_elements_bboxes_prompt,
    _read_visuals_bboxes_prompt,
    _read_visuals_segmentations_prompt,
    compute_crop_rect_and_dpi,
    crop_atoms,
    crop_elements,
    decompose_block,
    decompose_visual,
    generate_debug_overlay,
    localize_atoms,
    localize_elements,
    localize_page,
    run_crop,
    run_crop_visuals,
    run_localize,
    run_localize_visuals,
    run_segment_and_crop,
    segment_visual,
    select_atoms_by_hint,
    select_elements_by_hint,
    validate_box,
)

__all__ = [
    "_compute_target_dpi",
    "_DEFAULT_DPI_BOUNDS",
    "_read_atoms_bboxes_prompt",
    "_read_elements_bboxes_prompt",
    "_read_visuals_bboxes_prompt",
    "_read_visuals_segmentations_prompt",
    "compute_crop_rect_and_dpi",
    "crop_atoms",
    "crop_elements",
    "decompose_block",
    "decompose_visual",
    "generate_debug_overlay",
    "localize_atoms",
    "localize_elements",
    "localize_page",
    "run_crop",
    "run_crop_visuals",
    "run_localize_visuals",
    "run_segment_and_crop",
    "segment_visual",
    "select_atoms_by_hint",
    "select_elements_by_hint",
    "validate_box",
]
