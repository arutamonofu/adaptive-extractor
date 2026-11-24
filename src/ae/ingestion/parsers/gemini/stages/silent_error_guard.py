"""Silent-error detection guards for Gemini ingestion stages (G-Ing-4)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

_VISUAL_OPEN_RE = re.compile(r"<visual\s+id=[\"']([^\"']+)[\"'](?:\s*/>|\s*>)")
_VISUAL_CLOSE_RE = re.compile(r"</visual>")
_OMITTED_RE = re.compile(r"^\s*<!--\s*omitted\s*-->\s*$", re.IGNORECASE)
_TABLE_TAG_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
_TABLE_TAG_CLOSE_RE = re.compile(r"</table>", re.IGNORECASE)
_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.DOTALL | re.IGNORECASE)


_BOILERPLATE_PAGE_RE = re.compile(
    r"(?:^|\n)\s*(?:references|bibliography|acknowledg?ments?|author\s+contributions|"
    r"competing\s+interests?|conflicts?\s+of\s+interest|notes\s+and\s+references)\b",
    re.IGNORECASE,
)


def _is_likely_reference_or_boilerplate_page(raw_pdf_text: str) -> bool:
    if _BOILERPLATE_PAGE_RE.search(raw_pdf_text):
        return True
    lines = [ln.strip() for ln in raw_pdf_text.splitlines() if ln.strip()]
    if len(lines) >= 6:
        ref_lines = sum(
            1 for ln in lines if re.match(r"^(?:\[\d+\]|\d+\.\s+[A-Z]|\([a-z]\)\s+[A-Z])", ln)
        )
        if ref_lines / len(lines) >= 0.35:
            return True
    return False


def detect_parse_text_silent_error(
    chunk_md: str,
    *,
    pdf_page_text: str = "",
) -> Optional[str]:
    """Detect silent transcription failures in a Stage 1 chunk response.

    Returns a short diagnostic reason string if a silent failure is detected, else None.
    """
    if not chunk_md or not chunk_md.strip():
        return "empty_chunk_response"

    stripped = chunk_md.strip()
    raw_pdf_text = (pdf_page_text or "").strip()

    # 1. Model output <!-- omitted --> or near-empty text when PDF page clearly has body prose
    if _OMITTED_RE.match(stripped) or len(stripped) < 30:
        if len(raw_pdf_text) >= 150 and not _is_likely_reference_or_boilerplate_page(raw_pdf_text):
            return (
                f"silent_omission: PDF page has {len(raw_pdf_text)} chars of extractable text "
                f"but transcription returned {len(stripped)} chars"
            )

    # 2. Unclosed <visual id="..."> tag
    open_non_self = len(re.findall(r"<visual\s+id=[\"'][^\"']+[\"']\s*>", stripped))
    close_tags = len(_VISUAL_CLOSE_RE.findall(stripped))
    if open_non_self > close_tags:
        return f"unclosed_visual_tag: {open_non_self} opening vs {close_tags} closing tags"

    # 3. Severe line repetition / decoding collapse
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip() and len(ln.strip()) >= 8]
    if len(lines) >= 8:
        top_line, top_count = Counter(lines).most_common(1)[0]
        if top_count / len(lines) >= 0.60 and top_count >= 5:
            return f"repetition_collapse: line {top_line[:40]!r} repeated {top_count}/{len(lines)} times"

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
