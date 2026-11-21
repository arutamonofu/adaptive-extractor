"""Pipeline stages for the Ingestion v2 Gemini parser.

Canonically maps to:
- Stage 01: parse_text
- Stage 02: manifest_visuals
- Stage 03: segment_and_crop_visuals_elements
- Stage 04: parse_visuals
"""

from ae.ingestion.parsers.gemini.stages.manifest_visuals import (
    reconcile_manifest,
    run_classify,
    run_manifest_visuals,
    run_render_pages,
    run_render_pages_with_visuals,
)
from ae.ingestion.parsers.gemini.stages.parse_text import (
    compose_parse_text_prompt,
    parse_placeholders_from_markdown,
    run_chunked_parse_text,
    run_parse_text,
    split_pdf_into_chunks,
)
from ae.ingestion.parsers.gemini.stages.parse_visuals import (
    compose_parse_prompt,
    parse_single_visual,
    run_parse,
    run_parse_visuals,
)
from ae.ingestion.parsers.gemini.stages.render_markdown import (
    assemble_enriched_markdown,
    assemble_manifest_markdown,
    extract_unparsed_visual_ids,
    has_unparsed_visuals,
    render_composite_block,
    render_manifest_stub,
    render_parse_body,
    render_placeholder_block,
    replace_manifest_with_enrichment,
    run_render,
    run_render_manifest,
)
from ae.ingestion.parsers.gemini.stages.segment_and_crop_visuals_elements import (
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
    # Canonical Stage 01
    "run_parse_text",
    "compose_parse_text_prompt",
    "split_pdf_into_chunks",
    "parse_placeholders_from_markdown",
    "run_chunked_parse_text",
    # Canonical Stage 02
    "run_manifest_visuals",
    "run_render_pages_with_visuals",
    "reconcile_manifest",
    "run_classify",
    "run_render_pages",
    # Canonical Stage 03
    "run_localize_visuals",
    "run_crop_visuals",
    "decompose_visual",
    "segment_visual",
    "decompose_block",
    "localize_elements",
    "crop_elements",
    "select_elements_by_hint",
    "localize_atoms",
    "crop_atoms",
    "select_atoms_by_hint",
    "run_localize",
    "run_crop",
    "localize_page",
    "validate_box",
    "compute_crop_rect_and_dpi",
    "generate_debug_overlay",
    # Canonical Stage 04
    "run_parse_visuals",
    "run_parse",
    "parse_single_visual",
    "compose_parse_prompt",
    # Render utilities
    "run_render",
    "run_render_manifest",
    "render_manifest_stub",
    "replace_manifest_with_enrichment",
    "render_parse_body",
    "render_composite_block",
    "render_placeholder_block",
    "assemble_enriched_markdown",
    "assemble_manifest_markdown",
    "extract_unparsed_visual_ids",
    "has_unparsed_visuals",
]
