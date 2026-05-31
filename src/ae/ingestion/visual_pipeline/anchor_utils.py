from __future__ import annotations

import re
from typing import Any


def canonical_visual_ref(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_figure_label(value: Any) -> str:
    text = " ".join(str(value or "").strip().split()).lower()
    if not text:
        return ""
    text = text.rstrip(".:")
    is_supplementary = bool(re.search(r"\b(supplementary|supplemental|supp)\b", text))
    text = re.sub(r"\b(supplementary|supplemental|supp)\b\.?", "", text)
    text = re.sub(r"\bfigures?\b\.?", "fig", text)
    text = re.sub(r"\bextended\s+data\s+fig\b\.?", "extended_data_fig", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bfig\s+([0-9]+)\s*\(?[a-z]\)?\b", r"fig \1", text)
    text = re.sub(r"\bfig\s+(s[0-9]+)\s*\(?[a-z]\)?\b", r"fig \1", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if is_supplementary and text.startswith("fig_") and not text.startswith("supp_"):
        text = f"supp_{text}"
    return text


# Schema version and kinds
SCHEMA_VERSION = "aee.md_anchor_index.v1"
BEGIN_KIND = "BEGIN"
END_KIND = "END"

BLOCK_MARKER_RE = re.compile(r"<!--\s*(?:AE|AEE)_TABLES_(BEGIN|END):\s*(.*?)\s*-->", flags=re.S)
SINGLE_ANCHOR_RE = re.compile(r"<!--\s*AE_VISUAL_ANCHOR:\s*(.*?)\s*-->", flags=re.S)


def parse_md_anchors(markdown: str) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    invalid_blocks: list[dict[str, Any]] = []

    # Process single anchors (e.g. <!-- AE_VISUAL_ANCHOR: main_fig_1 -->)
    for match in SINGLE_ANCHOR_RE.finditer(markdown):
        raw_label = " ".join(match.group(1).split())
        visual_ref = canonical_visual_ref(match.group(1))
        if not visual_ref:
            invalid_blocks.append({
                "reason": "empty_anchor_label",
                "raw_label": raw_label,
                "visual_ref": visual_ref,
                "start_offset": match.start(),
                "end_offset": match.end(),
            })
            continue

        anchors.append({
            "raw_label": raw_label,
            "visual_ref": visual_ref,
            "begin_marker": match.group(0),
            "end_marker": "",
            "start_offset": match.start(),
            "content_start_offset": match.end(),
            "content_end_offset": match.end(),
            "end_offset": match.end(),
        })

    # Process block markers
    block_markers = [
        {
            "kind": match.group(1),
            "raw_label": " ".join(match.group(2).split()),
            "visual_ref": canonical_visual_ref(match.group(2)),
            "start_offset": match.start(),
            "end_offset": match.end(),
            "marker": match.group(0),
        }
        for match in BLOCK_MARKER_RE.finditer(markdown)
    ]

    i = 0
    while i < len(block_markers):
        marker = block_markers[i]
        if marker["kind"] == END_KIND:
            invalid_blocks.append({
                "reason": "unmatched_end",
                "raw_label": marker["raw_label"],
                "visual_ref": marker["visual_ref"],
                "start_offset": marker["start_offset"],
                "end_offset": marker["end_offset"],
            })
            i += 1
            continue
        if not marker["visual_ref"]:
            invalid_blocks.append({
                "reason": "empty_begin_label",
                "raw_label": marker["raw_label"],
                "visual_ref": marker["visual_ref"],
                "start_offset": marker["start_offset"],
                "end_offset": marker["end_offset"],
            })
            i += 1
            continue

        next_marker = block_markers[i + 1] if i + 1 < len(block_markers) else None
        if next_marker is None:
            invalid_blocks.append({
                "reason": "missing_end",
                "raw_label": marker["raw_label"],
                "visual_ref": marker["visual_ref"],
                "start_offset": marker["start_offset"],
                "end_offset": marker["end_offset"],
            })
            i += 1
            continue
        if next_marker["kind"] == BEGIN_KIND:
            invalid_blocks.append({
                "reason": "nested_or_missing_end",
                "raw_label": marker["raw_label"],
                "visual_ref": marker["visual_ref"],
                "start_offset": marker["start_offset"],
                "end_offset": marker["end_offset"],
            })
            i += 1
            continue
        if marker["visual_ref"] != next_marker["visual_ref"]:
            invalid_blocks.append({
                "reason": "begin_end_label_mismatch",
                "begin_raw_label": marker["raw_label"],
                "end_raw_label": next_marker["raw_label"],
                "begin_visual_ref": marker["visual_ref"],
                "end_visual_ref": next_marker["visual_ref"],
                "start_offset": marker["start_offset"],
                "end_offset": next_marker["end_offset"],
            })
            i += 2
            continue
        anchors.append({
            "raw_label": marker["raw_label"],
            "visual_ref": marker["visual_ref"],
            "begin_marker": marker["marker"],
            "end_marker": next_marker["marker"],
            "start_offset": marker["start_offset"],
            "content_start_offset": marker["end_offset"],
            "content_end_offset": next_marker["start_offset"],
            "end_offset": next_marker["end_offset"],
        })
        i += 2

    counts: dict[str, int] = {}
    for anchor in anchors:
        counts[anchor["visual_ref"]] = counts.get(anchor["visual_ref"], 0) + 1
    duplicates = [anchor for anchor in anchors if counts.get(anchor["visual_ref"], 0) > 1]
    index = {anchor["visual_ref"]: anchor for anchor in anchors if counts.get(anchor["visual_ref"], 0) == 1}
    return {
        "schema_version": SCHEMA_VERSION,
        "anchors": anchors,
        "index": index,
        "duplicates": duplicates,
        "invalid_blocks": invalid_blocks,
    }
