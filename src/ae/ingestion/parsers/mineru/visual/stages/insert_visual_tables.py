"""Stage for inserting extracted tables from charts into markdown by replacing image tags."""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _clean_cell(value: Any) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text.replace("|", r"\|")


def _column_title(column: Any) -> str:
    if isinstance(column, dict):
        name = _clean_cell(column.get("name", ""))
        unit = _clean_cell(column.get("unit", ""))
        return f"{name} ({unit})" if unit else name
    return _clean_cell(column)


def _render_table(table: dict[str, Any], result: dict[str, Any], warnings: list[str]) -> str:
    target_id = result.get("target_id", "result")
    columns = table.get("columns", [])
    rows = table.get("rows", [])
    if not isinstance(columns, list) or not columns:
        warnings.append(f"{target_id}:table_missing_columns")
        return ""
    if not isinstance(rows, list):
        warnings.append(f"{target_id}:table_rows_not_list")
        rows = []
    
    panel = table.get("panel")
    series_name = table.get("series_name")
    chart_type = table.get("chart_type")
    text_overlays = table.get("text_overlays", [])

    output_parts = []

    meta_info = []
    if panel:
        meta_info.append(f"Panel {panel}")
    if series_name:
        meta_info.append(f"Series: {series_name}")
    if chart_type:
        meta_info.append(f"Type: {chart_type}")

    if meta_info:
        output_parts.append(f"**{', '.join(meta_info)}**")

    headers = [_column_title(column) for column in columns]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row_idx, row in enumerate(rows, start=1):
        values = row if isinstance(row, list) else [row]
        if len(values) != len(headers):
            warnings.append(f"{target_id}:row_{row_idx}_column_count_mismatch")
        padded = (values + [""] * len(headers))[: len(headers)]
        table_lines.append("| " + " | ".join(_clean_cell(value) for value in padded) + " |")
    
    output_parts.append("\n".join(table_lines))

    if text_overlays:
        if isinstance(text_overlays, str):
            text_overlays = [text_overlays]
        if isinstance(text_overlays, list):
            overlay_lines = ["**Overlaid Annotations / Parameters:**"]
            for o in text_overlays:
                cleaned = _clean_cell(o)
                if cleaned:
                    overlay_lines.append(f"- {cleaned}")
            if len(overlay_lines) > 1:
                output_parts.append("\n".join(overlay_lines))

    return "\n\n".join(output_parts)


def _render_result(
    result: dict[str, Any],
    warnings: list[str],
    caption: str = "",
    img_path: str = ""
) -> str:
    tables = result.get("tables", [])
    
    rendered_tables: list[str] = []
    
    # Render global / header info first
    header_parts = []
    if caption and caption.strip():
        header_parts.append(f"**Figure Caption:** {caption.strip()}")
        
    if header_parts:
        rendered_tables.append("\n".join(header_parts))

    # Keep validation logic to populate warnings list, but do not append caution blocks
    if isinstance(tables, list) and tables:
        for table in tables:
            if not isinstance(table, dict):
                warnings.append("table_not_object")
                continue
            rendered = _render_table(table, result, warnings)
            if rendered:
                rendered_tables.append(rendered)
            
    return "\n\n".join(rendered_tables)


def replace_image_tags(
    markdown: str,
    results_by_img_path: Dict[str, Dict[str, Any]],
    warnings: List[str]
) -> str:
    """Find Markdown image tags and replace them with rendered tables or remove them."""
    # Pattern to match Markdown image tags like: ![caption](images/name.jpg)
    pattern = re.compile(r'!\[(.*?)\]\(((?:images|images_v2)\/[a-zA-Z0-9_\-\.\/]+)\)')
    
    def replacer(match: re.Match) -> str:
        img_path = match.group(2)
        caption = match.group(1)
        
        # Check if we have extraction results for this image path
        result = results_by_img_path.get(img_path)
        if not result:
            # Try to match by filename only if exact path doesn't match
            filename = Path(img_path).name
            result = next((r for path, r in results_by_img_path.items() if Path(path).name == filename), None)
            
        if not result:
            logger.debug(f"No visual extraction results found for image: {img_path}")
            return ""  # Delete from the markdown

        # Check if the image was evaluated as irrelevant
        if result.get("status") == "irrelevant":
            caption_text = result.get("caption") or caption or ""
            caption_text = caption_text.strip()
            if caption_text:
                return f"\n\n[Изображение удалено как нерелевантное: {caption_text}]\n\n"
            return f"\n\n[Изображение удалено как нерелевантное]\n\n"
            
        result.setdefault("target_id", Path(img_path).name)
            
        if result.get("status") == "failed":
            warnings.append(f"Failed extraction for {img_path}")
            return ""  # Delete from the markdown
            
        # Fallback to result caption if caption in markdown tag is empty
        final_caption = caption or result.get("caption") or ""
        rendered = _render_result(result, warnings, caption=final_caption, img_path=img_path)
        if not rendered:
            return ""  # Delete from the markdown
            
        return f"\n\n{rendered}\n\n"

    replaced = pattern.sub(replacer, markdown)
    
    # Remove trailing whitespaces on each line, and collapse consecutive empty lines
    lines = replaced.splitlines()
    cleaned_lines = [line.rstrip() for line in lines]
    reconstructed = "\n".join(cleaned_lines)
    
    cleaned_md = re.sub(r'\n{3,}', '\n\n', reconstructed).strip()
    if cleaned_md:
        return cleaned_md + '\n'
    return ""
