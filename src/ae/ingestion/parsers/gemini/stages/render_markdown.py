"""Render Manifest (Stage 4) and Markdown assembling utilities."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.schemas import (
    ChartParseResult,
    Manifest,
    ManifestVisual,
    MolecularParseResult,
    PipelineSummary,
    TableParseResult,
    VisualDecomposition,
    VisualParseRecord,
)

logger = logging.getLogger(__name__)

_VISUAL_BLOCK_RE = re.compile(
    r"<visual\s+id=[\"'](?P<id>[\w.\-]+)[\"'](?:\s*/>|\s*>(?P<caption>.*?)</visual>)",
    re.DOTALL,
)

_SINGLE_VISUAL_BLOCK_RE = re.compile(
    r"<visual\s+id=[\"'](?P<id>[^\"']+)[\"'][^>]*>(?P<body>.*?)</visual>",
    re.DOTALL,
)

_UNPARSED_ELEMENT_RE = re.compile(
    r"<(?:element|atom)\s+id=[\"'](?P<id>[^\"']+)[\"'][^>]*>\[Unparsed[^<]*\]</(?:element|atom)>",
    re.IGNORECASE,
)
_UNPARSED_ATOM_RE = _UNPARSED_ELEMENT_RE


def extract_unparsed_visual_ids(doc_text: str) -> list[str]:
    """Return list of visual and element IDs that are still unparsed stubs in doc_text."""
    unparsed: list[str] = []
    for match in _SINGLE_VISUAL_BLOCK_RE.finditer(doc_text):
        vis_id = match.group("id").strip()
        body = match.group("body").strip()
        if (
            "| ---" in body
            or "|:---" in body
            or "<table" in body.lower()
            or "Chemical Entities" in body
            or "*[Parsing failed]*" in body
            or f"[{vis_id}: parsing failed]" in body
        ):
            continue
        if re.search(r"^\s*>\s+", body, re.MULTILINE):
            unparsed.append(vis_id)

    # Detect unparsed sub-elements within partially enriched visuals
    for match in _UNPARSED_ELEMENT_RE.finditer(doc_text):
        elem_id = match.group("id").strip()
        if elem_id and elem_id not in unparsed:
            unparsed.append(elem_id)

    return unparsed


def has_unparsed_visuals(doc_text: str) -> bool:
    """Check whether doc_text contains at least one unparsed visual manifest stub."""
    return bool(extract_unparsed_visual_ids(doc_text))


def render_manifest_stub(visual: ManifestVisual, caption: str) -> str:
    """Render deterministic visual manifest placeholder stub for unextracted visual."""
    lines = [f'<visual id="{visual.id}" type="{visual.type}">']
    if caption.strip():
        lines.append(caption.strip())
    if visual.description.strip():
        lines.append(f"> {visual.description.strip()}")
    lines.append("</visual>")
    return "\n".join(lines)


def replace_manifest_with_enrichment(
    doc_text: str, visual_id: str, enrichment_text: str
) -> str:
    """Replace manifest stub or existing visual block for visual_id."""
    esc_id = re.escape(visual_id)
    vis_pat = re.compile(
        rf"<visual\s+id=[\"']{esc_id}[\"'][^>]*>.*?</visual>",
        re.DOTALL,
    )
    if vis_pat.search(doc_text):
        cleaned = enrichment_text.strip()
        return vis_pat.sub(lambda _: cleaned, doc_text)

    # If nowhere found, append to the end
    return doc_text.rstrip() + f"\n\n{enrichment_text.strip()}\n"


def _render_entities_table(entities: list[Any]) -> list[str]:
    """Render a Markdown table for chemical entities, including attachment points when present."""
    has_attachments = any(getattr(ent, "attachment_points", None) for ent in entities)
    if has_attachments:
        lines = [
            "| Label | SMILES | Attachment Points |",
            "| :--- | :--- | :--- |",
        ]
        for ent in entities:
            aps = getattr(ent, "attachment_points", None) or []
            ap_str = "; ".join(f"`{ap.placeholder}` = {ap.variable}" for ap in aps) if aps else "-"
            lines.append(f"| **{ent.label}** | `{ent.smiles}` | {ap_str} |")
    else:
        lines = [
            "| Label | SMILES |",
            "| :--- | :--- |",
        ]
        for ent in entities:
            lines.append(f"| **{ent.label}** | `{ent.smiles}` |")

    notes: list[str] = []
    for ent in entities:
        status = getattr(ent, "provenance_status", None)
        oracle_notes = (getattr(ent, "oracle_notes", None) or "").strip()
        is_override = "overrid" in oracle_notes.lower()
        if status == "mismatch_flagged" or is_override:
            tag = status or "oracle_note"
            detail = oracle_notes.replace("--", "-")
            entry = f"{ent.label}: {tag}" + (f" ({detail})" if detail else "")
            notes.append(entry)
    if notes:
        lines.append(f"<!-- oracle_notes: {'; '.join(notes)} -->")
    return lines


def _render_molecules_body(result: MolecularParseResult) -> list[str]:
    lines = []
    if getattr(result, "status", "success") != "success":
        lines.append("*[Parsing failed]*")
        return lines

    if result.entities:
        lines.append("#### Chemical Entities")
        lines.extend(_render_entities_table(result.entities))
        lines.append("")

    return lines


def _render_chart_body(result: ChartParseResult) -> list[str]:
    lines = []
    if getattr(result, "status", "success") != "success":
        lines.append("*[Parsing failed]*")
        return lines

    if result.content:
        lines.append(result.content.strip())
        lines.append("")

    return lines


def _render_table_body(result: TableParseResult) -> list[str]:
    lines = []
    if getattr(result, "status", "success") != "success":
        lines.append("*[Parsing failed]*")
        return lines

    if result.content:
        lines.append(result.content.strip())
        lines.append("")

    return lines


def render_parse_body(parse_result: Any) -> str:
    """Render the inner markdown body for any visual parse result object."""
    if isinstance(parse_result, MolecularParseResult):
        lines = _render_molecules_body(parse_result)
    elif isinstance(parse_result, ChartParseResult):
        lines = _render_chart_body(parse_result)
    elif isinstance(parse_result, TableParseResult):
        lines = _render_table_body(parse_result)
    elif isinstance(parse_result, VisualParseRecord):
        lines = [parse_result.rendered_markdown]
    else:
        lines = ["*[Parsing failed]*"]
    return "\n".join(lines).strip()


def render_composite_block(
    decomp: VisualDecomposition,
    element_results: Optional[dict[str, Any]] = None,
    visual_id: str = "",
    **kwargs: Any,
) -> str:
    """Assemble composite visual enrichment using composition_template and unified entity registry."""
    if element_results is None:
        element_results = kwargs.get("atom_results", {})
    template = decomp.composition_template or ""
    elements = getattr(decomp, "elements", None) or getattr(decomp, "atoms", [])
    element_map = {getattr(e, "element_id", getattr(e, "atom_id", "")): e for e in elements}
    all_entities = []
    seen_entity_keys: set[tuple[str, str]] = set()

    def _add_entities(entities: list[Any]) -> None:
        for ent in entities:
            key = (ent.label, ent.smiles)
            if key not in seen_entity_keys:
                seen_entity_keys.add(key)
                all_entities.append(ent)

    def _replace_element(match: re.Match) -> str:
        elem_id = match.group(1)
        elem_meta = element_map.get(elem_id)
        res = element_results.get(elem_id)
        label = elem_meta.label if elem_meta and elem_meta.label else elem_id

        if res is None:
            qualified_id = f"{visual_id}/{elem_id}" if visual_id else elem_id
            desc = elem_meta.description if elem_meta and elem_meta.description else ""
            e_type = elem_meta.type if elem_meta and elem_meta.type else "molecular_structure"
            desc_part = f": {desc}" if desc else ""
            return f'<element id="{qualified_id}" label="{label}" type="{e_type}">[Unparsed{desc_part}]</element>'

        if isinstance(res, MolecularParseResult):
            if res.entities:
                _add_entities(res.entities)
            return f"**{label}**"
        if isinstance(res, (ChartParseResult, TableParseResult)):
            body = render_parse_body(res)
            if body and body != "*[Parsing failed]*":
                header = f"**{label}**\n\n" if elem_meta and elem_meta.label else ""
                return f"\n\n{header}{body}\n\n"
        return f"**{label}**"

    rendered_narrative = re.sub(
        r"<(?:element|atom)\s+id=[\"']([^\"']+)[\"']\s*/>", _replace_element, template
    ).strip()

    # Collect any entities from element_results not in template
    for e_id, res in element_results.items():
        if isinstance(res, MolecularParseResult) and res.entities:
            _add_entities(res.entities)

    if not all_entities:
        return rendered_narrative

    # Build Chemical Entities Registry table
    lines = [rendered_narrative, "", "**Chemical Entities Registry:**"]
    lines.extend(_render_entities_table(all_entities))

    return "\n".join(lines).strip()


def render_placeholder_block(
    placeholder_id: str,
    caption: str,
    visuals: list[ManifestVisual],
    parsed_visuals: dict[str, Any],
) -> str:
    """Format a single visual placeholder with its caption and parsed objects."""
    if not visuals:
        return caption.strip()

    v_type = visuals[0].type if visuals else "table"
    lines = [
        f'<visual id="{placeholder_id}" type="{v_type}">',
    ]
    if caption.strip():
        lines.append(caption.strip())
        lines.append("")

    for vis in visuals:
        parsed_res = parsed_visuals.get(vis.id)
        if parsed_res is None:
            continue

        if isinstance(parsed_res, MolecularParseResult):
            lines.extend(_render_molecules_body(parsed_res))
        elif isinstance(parsed_res, ChartParseResult):
            lines.extend(_render_chart_body(parsed_res))
        elif isinstance(parsed_res, TableParseResult):
            lines.extend(_render_table_body(parsed_res))
        elif isinstance(parsed_res, VisualParseRecord):
            lines.append(parsed_res.rendered_markdown)
        else:
            lines.append("*[Parsing failed]*")
            lines.append("")

    lines.append("</visual>")
    return "\n".join(lines)


def assemble_enriched_markdown(
    raw_md: str,
    manifest: Manifest,
    parsed_visuals: dict[str, Any],
    caption_map: dict[str, str],
) -> str:
    """Format and substitute all visual placeholders in Markdown with parsed content."""
    vis_by_id: dict[str, ManifestVisual] = {v.id: v for v in manifest.visuals}

    def _replace_visual(match: re.Match) -> str:
        ph_id = match.group("id")
        caption = caption_map.get(ph_id)
        if caption is None:
            raw_c = match.group("caption")
            caption = raw_c.strip() if raw_c else ""
        vis = vis_by_id.get(ph_id)
        visuals_list = [vis] if vis else []
        return render_placeholder_block(ph_id, caption, visuals_list, parsed_visuals)

    return _VISUAL_BLOCK_RE.sub(_replace_visual, raw_md)


def assemble_manifest_markdown(
    raw_md: str,
    manifest: Manifest,
    caption_map: dict[str, str],
    artifacts: ArtifactManager,
) -> str:
    """Build markdown with document_id header and visual manifest stubs."""
    doc_id = artifacts.doc_name
    visuals_by_id = {v.id: v for v in manifest.visuals}
    discarded = set(manifest.discarded_ids)

    def _replace_visual(match: re.Match) -> str:
        ph_id = match.group("id")
        caption = caption_map.get(ph_id)
        if caption is None:
            raw_c = match.group("caption")
            caption = raw_c.strip() if raw_c else ""

        # Check if already parsed
        parsed_file = artifacts.parsed_visuals_dir / f"{ph_id}.json"
        if parsed_file.exists():
            try:
                raw_data = artifacts.load_json(parsed_file)
                rendered = raw_data.get("rendered_markdown", "")
                if rendered:
                    v_type = visuals_by_id[ph_id].type if ph_id in visuals_by_id else "table"
                    inner = f"{caption.strip()}\n\n{rendered}" if caption.strip() else rendered
                    return (
                        f'<visual id="{ph_id}" type="{v_type}">\n'
                        f"{inner}\n"
                        f"</visual>"
                    )
            except Exception:
                pass

        if ph_id in discarded:
            return caption.strip()

        if ph_id in visuals_by_id:
            return render_manifest_stub(visuals_by_id[ph_id], caption)

        # If not in visuals or discarded, keep caption
        return caption.strip()

    body = _VISUAL_BLOCK_RE.sub(_replace_visual, raw_md)
    doc_header = f"<!-- document_id: {doc_id} -->\n\n"
    if not body.startswith(f"<!-- document_id: {doc_id} -->"):
        return doc_header + body
    return body


def run_render_manifest(
    raw_md: str,
    manifest: Manifest,
    artifacts: ArtifactManager,
    client: Any = None,
) -> str:
    """Execute Stage 4 of lightweight Ingestion: render manifest stubs into doc.md."""
    caption_map: dict[str, str] = {}
    if artifacts.placeholders_json_path.exists():
        raw_ph = artifacts.load_json(artifacts.placeholders_json_path)
        caption_map = {item["id"]: item.get("caption", "") for item in raw_ph}

    manifest_md = assemble_manifest_markdown(raw_md, manifest, caption_map, artifacts)

    # Save snapshot to manifest stage and write to document_md_path
    artifacts.save_text(artifacts.manifest_enriched_text_path, manifest_md)
    artifacts.save_text(artifacts.document_md_path, manifest_md)

    # Summary
    summary = PipelineSummary()
    summary.total_placeholders = len(caption_map)
    summary.total_manifest_visuals = len(manifest.visuals)
    summary.discarded_placeholder_ids = list(manifest.discarded_ids)

    for v in manifest.visuals:
        summary.counts_by_type[v.type] = summary.counts_by_type.get(v.type, 0) + 1

    if client is not None and hasattr(client, "get_all_stage_usage"):
        usage_map = client.get_all_stage_usage()
        if isinstance(usage_map, dict):
            summary.stages_usage = usage_map
    if client is not None and hasattr(client, "get_scope_counters"):
        counters = client.get_scope_counters()
        if isinstance(counters, dict):
            summary.t4_skipped_calls = int(counters.get("t4_skipped_calls", 0) or 0)
            summary.t4_fallback_calls = int(counters.get("t4_fallback_calls", 0) or 0)
            summary.silent_error_retries = int(counters.get("silent_error_retries", 0) or 0)

    artifacts.save_json(artifacts.summary_json_path, summary)
    logger.info(
        f"Stage 4 [render_manifest]: Saved {len(manifest.visuals)} stubs to "
        f"{artifacts.document_md_path}"
    )
    return manifest_md


def run_render(
    raw_md: str,
    manifest: Manifest,
    parsed_visuals: dict[str, Any],
    artifacts: ArtifactManager,
    client: Any = None,
) -> tuple[str, PipelineSummary]:
    """Execute rendering (compatible with full visual parses or lightweight manifests)."""
    if not parsed_visuals and manifest.visuals:
        md = run_render_manifest(raw_md, manifest, artifacts, client)
        sum_data = artifacts.load_json(artifacts.summary_json_path)
        return md, PipelineSummary.model_validate(sum_data)

    caption_map: dict[str, str] = {}
    if artifacts.placeholders_json_path.exists():
        raw_ph = artifacts.load_json(artifacts.placeholders_json_path)
        caption_map = {item["id"]: item.get("caption", "") for item in raw_ph}

    vis_by_id: dict[str, ManifestVisual] = {v.id: v for v in manifest.visuals}

    def _replace_visual(match: re.Match) -> str:
        ph_id = match.group("id")
        caption = caption_map.get(ph_id)
        if caption is None:
            raw_c = match.group("caption")
            caption = raw_c.strip() if raw_c else ""
        vis = vis_by_id.get(ph_id)
        visuals_list = [vis] if vis else []
        return render_placeholder_block(ph_id, caption, visuals_list, parsed_visuals)

    enriched_md = _VISUAL_BLOCK_RE.sub(_replace_visual, raw_md)
    doc_header = f"<!-- document_id: {artifacts.doc_name} -->\n\n"
    if not enriched_md.startswith(f"<!-- document_id: {artifacts.doc_name} -->"):
        enriched_md = doc_header + enriched_md

    artifacts.save_text(artifacts.document_md_path, enriched_md)

    summary = PipelineSummary()
    summary.total_placeholders = len(caption_map)
    summary.total_manifest_visuals = len(manifest.visuals)
    if client is not None and hasattr(client, "get_all_stage_usage"):
        usage_map = client.get_all_stage_usage()
        if isinstance(usage_map, dict):
            summary.stages_usage = usage_map
    if client is not None and hasattr(client, "get_scope_counters"):
        counters = client.get_scope_counters()
        if isinstance(counters, dict):
            summary.t4_skipped_calls = int(counters.get("t4_skipped_calls", 0) or 0)
            summary.t4_fallback_calls = int(counters.get("t4_fallback_calls", 0) or 0)
            summary.silent_error_retries = int(counters.get("silent_error_retries", 0) or 0)
    artifacts.save_json(artifacts.summary_json_path, summary)

    return enriched_md, summary
