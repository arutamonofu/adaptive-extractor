"""Silent-error detection guards for Gemini ingestion stages (G-Ing-4)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

_VALID_VISUAL_BLOCK_RE = re.compile(
    r"<visual\s+id=[\"'][\w.\-]+[\"'](?:\s*/>|\s*>.*?</visual>)",
    re.DOTALL,
)
_VALID_VISUAL_OPEN_OR_SELF_RE = re.compile(r"^<visual\s+id=[\"'][\w.\-]+[\"']\s*/?>$")
_ANY_VISUAL_PREFIX_RE = re.compile(r"<visual\b", re.IGNORECASE)
_ANY_VISUAL_OPEN_RE = re.compile(r"<visual\b[^>]*>", re.IGNORECASE)
_VISUAL_TOKEN_RE = re.compile(r"<visual\b[^>]*>|</visual\s*>", re.IGNORECASE)
_OMITTED_RE = re.compile(r"^\s*<!--\s*omitted\s*-->\s*$", re.IGNORECASE)
_TABLE_TAG_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
_TABLE_TAG_CLOSE_RE = re.compile(r"</table>", re.IGNORECASE)
_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.DOTALL | re.IGNORECASE)


_BOILERPLATE_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+)?(?:\d+(?:\.\d+)*\.?\s+)?(?:"
    r"references|bibliography|acknowledg?ments?|author\s+contributions|"
    r"competing\s+interests?|conflicts?\s+of\s+interest|notes\s+and\s+references"
    r")\s*[:.]?\s*$",
    re.IGNORECASE,
)


def _is_likely_reference_or_boilerplate_page(raw_pdf_text: str) -> bool:
    lines = [ln.strip() for ln in raw_pdf_text.splitlines() if ln.strip()]
    if lines:
        if any(_BOILERPLATE_HEADING_RE.match(ln) for ln in lines[:3]):
            return True
    if len(lines) >= 6:
        ref_lines = sum(
            1 for ln in lines if re.match(r"^(?:\[\d+\]|\d+\.\s+[A-Z]|\([a-z]\)\s+[A-Z])", ln)
        )
        if ref_lines / len(lines) >= 0.50:
            return True
    return False


def detect_parse_text_silent_error(
    chunk_md: str,
    *,
    pdf_page_text: str = "",
    has_visual_content: bool = False,
) -> Optional[str]:
    """Detect silent transcription failures in a Stage 1 chunk response.

    Returns a short diagnostic reason string if a silent failure is detected, else None.
    """
    if not chunk_md or not chunk_md.strip():
        return "empty_chunk_response"

    stripped = chunk_md.strip()
    raw_pdf_text = (pdf_page_text or "").strip()

    # 1. Model output <!-- omitted --> or near-empty text without a valid visual placeholder
    has_valid_visual = bool(_VALID_VISUAL_BLOCK_RE.search(stripped))
    if _OMITTED_RE.match(stripped) or (len(stripped) < 30 and not has_valid_visual):
        if (
            len(raw_pdf_text) >= 150
            and not _is_likely_reference_or_boilerplate_page(raw_pdf_text)
        ) or (has_visual_content and len(raw_pdf_text) < 150):
            return (
                f"silent_omission: PDF page has {len(raw_pdf_text)} chars of extractable text "
                f"but transcription returned {len(stripped)} chars"
            )

    # 2. Malformed, nested, or unbalanced <visual ...> tags
    prefix_matches = _ANY_VISUAL_PREFIX_RE.findall(stripped)
    open_tags = _ANY_VISUAL_OPEN_RE.findall(stripped)
    if len(prefix_matches) != len(open_tags):
        return "malformed_visual_tag: unclosed '<visual' angle bracket"

    visual_stack: list[str] = []
    for token_match in _VISUAL_TOKEN_RE.finditer(stripped):
        token = token_match.group(0)
        if token.lower().startswith("</visual"):
            if not visual_stack:
                return f"stray_visual_close_tag: unmatched {token!r}"
            visual_stack.pop()
            continue

        if not _VALID_VISUAL_OPEN_OR_SELF_RE.match(token):
            return f"malformed_visual_tag: {token[:60]!r}"
        if visual_stack:
            return "malformed_visual_tag: nested <visual> tag"
        if not token.rstrip().endswith("/>"):
            visual_stack.append(token)

    if visual_stack:
        return f"unclosed_visual_tag: {len(visual_stack)} unclosed <visual> tag(s)"

    # 3. Soft completeness check for dense text pages truncated silently
    if (
        not has_valid_visual
        and len(raw_pdf_text) >= 500
        and not _is_likely_reference_or_boilerplate_page(raw_pdf_text)
        and (len(stripped) / len(raw_pdf_text)) < 0.15
    ):
        return (
            f"silent_omission: PDF page has {len(raw_pdf_text)} chars of extractable text "
            f"but transcription returned only {len(stripped)} chars "
            f"({len(stripped) / len(raw_pdf_text):.1%})"
        )

    # 4. Severe line repetition / decoding collapse
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip() and len(ln.strip()) >= 8]
    if len(lines) >= 8:
        top_line, top_count = Counter(lines).most_common(1)[0]
        if top_count / len(lines) >= 0.60 and top_count >= 5:
            return f"repetition_collapse: line {top_line[:40]!r} repeated {top_count}/{len(lines)} times"

    return None


def _norm_id(vis_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", vis_id.lower())


def detect_manifest_silent_error(
    raw_manifest: Any,
    reconciled_manifest: Any,
    expected_placeholders: list[Any],
) -> Optional[str]:
    """Detect silent failures in a Stage 2 visual manifest response."""
    if not expected_placeholders:
        return None

    expected_ids = {p.id for p in expected_placeholders}
    norm_map: dict[str, list[str]] = {}
    for eid in expected_ids:
        norm_map.setdefault(_norm_id(eid), []).append(eid)

    def _resolve_id(raw_id: str) -> Optional[str]:
        if raw_id in expected_ids:
            return raw_id
        cands = norm_map.get(_norm_id(raw_id), [])
        return cands[0] if len(cands) == 1 else None

    raw_visuals = getattr(raw_manifest, "visuals", []) or []
    raw_discarded = getattr(raw_manifest, "discarded_ids", []) or []
    reconciled_visuals = getattr(reconciled_manifest, "visuals", []) or []

    if not raw_visuals and not raw_discarded:
        return f"empty_manifest: 0/{len(expected_ids)} placeholders classified"

    reconciled_vis_ids = {v.id for v in reconciled_visuals}
    raw_discarded_matched = {
        resolved
        for d in raw_discarded
        if (resolved := _resolve_id(d)) is not None
    }
    accounted_ids = reconciled_vis_ids | raw_discarded_matched
    unaccounted_count = len(expected_ids - accounted_ids)
    rejected_raw_count = sum(1 for v in raw_visuals if _resolve_id(v.id) is None)

    if not reconciled_visuals:
        if not expected_ids.issubset(raw_discarded_matched):
            return f"empty_reconciled_visuals: 0/{len(expected_ids)} placeholders manifested"

    if (unaccounted_count >= 2 and (unaccounted_count / len(expected_ids)) >= 0.3) or (
        unaccounted_count >= 1 and (unaccounted_count / len(expected_ids)) >= 0.5
    ):
        return f"high_unexpected_drop: {unaccounted_count}/{len(expected_ids)} placeholders unaccounted"

    if rejected_raw_count > 0 and (rejected_raw_count / max(1, len(raw_visuals))) >= 0.5:
        return f"high_unmatched_ids: {rejected_raw_count}/{len(raw_visuals)} returned visual IDs unmatched"

    return None


def detect_visual_parse_silent_error(
    result: Any,
    visual_type: str,
) -> Optional[str]:
    """Detect silent parsing failures in a Stage 4 visual parse response.

    Returns a short diagnostic reason string if a silent failure is detected, else None.
    """
    if result is None:
        return "null_parse_result"

    status = getattr(result, "status", "success")
    if status in ("failed", "no_extractable_table", "no_extractable_chart", "no_extractable_structure"):
        return None

    if visual_type == "table":
        content = (getattr(result, "content", "") or "").strip()
        if not content:
            return "empty_table_content"
        if _TABLE_TAG_OPEN_RE.search(content) and not _TABLE_TAG_CLOSE_RE.search(content):
            return "truncated_html_table"

        # Check degenerate row repetition
        if "<tr" in content.lower():
            rows = [re.sub(r"\s+", " ", r.strip()) for r in _TR_RE.findall(content)]
        else:
            rows = [
                ln.strip()
                for ln in content.splitlines()
                if "|" in ln and not re.match(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$", ln)
            ]
        if len(rows) >= 6:
            top_row, top_count = Counter(rows).most_common(1)[0]
            if top_count / len(rows) >= 0.75 and top_count >= 5:
                return f"table_row_collapse: row repeated {top_count}/{len(rows)} times ({top_row[:40]!r})"

    elif visual_type == "chart":
        content = (getattr(result, "content", "") or "").strip()
        if not content:
            return "empty_chart_content"

    elif visual_type == "molecular_structure":
        entities = getattr(result, "entities", None)
        if not entities:
            return "empty_molecular_entities"

    return None
