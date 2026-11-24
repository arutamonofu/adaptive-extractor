"""Utilities for extracting local document and intra-visual context."""

from __future__ import annotations

import re
from typing import Optional

_ANY_VISUAL_RE = re.compile(
    r"<visual\s+id=[\"'](?P<id>[^\"']+)[\"'][^>]*>(?P<body>.*?)</visual>"
    r"|<visual\s+id=[\"'](?P<self_id>[^\"']+)[\"'][^>]*/>",
    re.DOTALL,
)
_COMMENT_RE = re.compile(r"<!--\s*(?:page_break|document_id:[^>]*)\s*-->")
_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.DOTALL | re.IGNORECASE)
_PIPE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?\s*$")


def _collapse_visual_blocks(
    text: str,
    caption_map: Optional[dict[str, str]] = None,
) -> str:
    """Replace <visual> blocks with compact one-line markers and strip HTML page comments."""

    def _repl(match: re.Match) -> str:
        vid = (match.group("id") or match.group("self_id") or "").strip()
        cap = (caption_map or {}).get(vid, "").strip()
        if not cap and match.group("body"):
            for line in match.group("body").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith(("|", "<", ">", "#", "*")):
                    cap = stripped
                    break
        if cap:
            first_line = cap.splitlines()[0].strip()[:140]
            return f"[Visual {vid}: {first_line}]"
        return f"[Visual {vid}]"

    collapsed = _ANY_VISUAL_RE.sub(_repl, text)
    collapsed = _COMMENT_RE.sub("", collapsed)
    return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


def _build_visual_mention_pattern(visual_id: str) -> Optional[re.Pattern[str]]:
    """Build a case-insensitive regex matching human-readable prose mentions of visual_id."""
    clean_id = re.sub(r"_dup\d+$", "", visual_id.strip())
    if not clean_id or clean_id.lower().startswith("unlabeled"):
        return None

    parts = [p for p in re.split(r"[_\s]+", clean_id) if p]
    if not parts:
        return None

    first = parts[0].lower().rstrip(".")
    if first in ("fig", "figure"):
        head = r"(?:Figure|Fig\.?)"
        rest = parts[1:]
    else:
        head = re.escape(parts[0])
        rest = parts[1:]

    if rest:
        tail = r"[\s._-]+".join(re.escape(p) for p in rest)
        pattern = rf"\b{head}[\s._-]*{tail}\b"
    else:
        pattern = rf"\b{head}\b"
    return re.compile(pattern, re.IGNORECASE)


def _trim_before(text: str, max_chars: int) -> tuple[str, bool]:
    """Take up to max_chars from the end of text, snapping to a paragraph or line boundary."""
    if len(text) <= max_chars:
        return text.strip(), False
    tail = text[-max_chars:]
    para_idx = tail.find("\n\n")
    if 0 <= para_idx < len(tail) // 2:
        return tail[para_idx + 2 :].strip(), True
    line_idx = tail.find("\n")
    if 0 <= line_idx < len(tail) // 3:
        return tail[line_idx + 1 :].strip(), True
    space_idx = tail.find(" ")
    if 0 <= space_idx < 80:
        return tail[space_idx + 1 :].strip(), True
    return tail.strip(), True


def _trim_after(text: str, max_chars: int) -> str:
    """Take up to max_chars from the start of text, snapping to a paragraph or line boundary."""
    if len(text) <= max_chars:
        return text.strip()
    head = text[:max_chars]
    para_idx = head.rfind("\n\n")
    if para_idx > len(head) // 2:
        return head[:para_idx].strip()
    line_idx = head.rfind("\n")
    if line_idx > len(head) // 2:
        return head[:line_idx].strip()
    space_idx = head.rfind(" ")
    if space_idx > len(head) - 80:
        return head[:space_idx].strip()
    return head.strip()


_PAGE_BREAK_SPLIT_RE = re.compile(r"<!--\s*page_break\s*-->")
_COMPOUND_RANGE_RE = re.compile(r"\b(\d{1,3})\s*[–\-—]\s*(\d{1,3})\b")
_COMPOUND_TOKEN_RE = re.compile(r"\b(\d{1,3}[a-z]?)\b", re.IGNORECASE)
_CHEM_CALLOUT_HINT_RE = re.compile(
    r"(?:HRMS|ESI-?MS|Anal\.?\s*Calcd|benzimidazol|pyridin|phenyl|chloro|fluoro|bromo|"
    r"methoxy|coumarin|oxadiazole|pyrazole|isobutyl|MIC|IC50|IC_?50|carbonyl|methanone)",
    re.IGNORECASE,
)


def _extract_target_compound_tokens(
    caption: str,
    description: str = "",
) -> list[str]:
    """Extract compound labels/ranges (e.g., '37-39' -> ['37','38','39']) from caption/description."""
    source = f"{caption}\n{description}".strip()
    if not source:
        return []
    tokens: list[str] = []
    for m in _COMPOUND_RANGE_RE.finditer(source):
        start_num, end_num = int(m.group(1)), int(m.group(2))
        if 1 <= start_num < end_num and (end_num - start_num) <= 20:
            for n in range(start_num, end_num + 1):
                s = str(n)
                if s not in tokens:
                    tokens.append(s)
    for m in _COMPOUND_TOKEN_RE.finditer(source):
        tok = m.group(1)
        if tok.isdigit() and int(tok) > 300:
            continue
        if len(tok) == 1 and tok.isdigit():
            continue
        if tok not in tokens:
            tokens.append(tok)
    return tokens[:24]


def extract_local_text_context(
    doc_md: str,
    visual_id: str,
    caption_map: Optional[dict[str, str]] = None,
    max_before_chars: int = 1800,
    max_after_chars: int = 900,
    max_ref_chars: int = 1000,
    max_callout_chars: int = 2000,
    description: str = "",
) -> str:
    """Extract N±1 page-window prose, cross-referenced paragraphs, and in-text callouts.

    Collapses neighboring <visual> blocks so large parsed tables or SMILES registries do
    not crowd out nearby prose, restricts immediate windowing to pages N-1..N+1 around the
    visual, and searches the rest of the document for explicit visual mentions (e.g. 'Table 11')
    and in-text chemical/compound callouts (IUPAC names, HRMS, substituent notes).
    """
    if not doc_md or not doc_md.strip():
        return ""

    esc_id = re.escape(visual_id)
    target_re = re.compile(
        rf"<visual\s+id=[\"']{esc_id}[\"'][^>]*>.*?</visual>"
        rf"|<visual\s+id=[\"']{esc_id}[\"'][^>]*/>",
        re.DOTALL,
    )

    # Split into physical pages to enforce the N±1 window around the target visual
    pages = _PAGE_BREAK_SPLIT_RE.split(doc_md)
    window_doc = doc_md
    target_page_idx: Optional[int] = None
    for idx, page_txt in enumerate(pages):
        if target_re.search(page_txt):
            target_page_idx = idx
            break
    if target_page_idx is not None and len(pages) > 1:
        p_start = max(0, target_page_idx - 1)
        p_end = min(len(pages), target_page_idx + 2)
        window_doc = "\n\n<!-- page_break -->\n\n".join(pages[p_start:p_end])

    match = target_re.search(window_doc)

    sections: list[str] = []
    window_text = ""

    if match:
        before_raw = window_doc[: match.start()]
        after_raw = window_doc[match.end() :]

        before_collapsed = _collapse_visual_blocks(before_raw, caption_map)
        after_collapsed = _collapse_visual_blocks(after_raw, caption_map)

        before_snippet, was_trimmed = _trim_before(before_collapsed, max_before_chars)
        if was_trimmed:
            headings = list(_HEADING_RE.finditer(before_collapsed))
            if headings:
                last_heading = headings[-1].group(0).strip()
                if last_heading not in before_snippet:
                    before_snippet = f"{last_heading}\n\n{before_snippet}"

        after_snippet = _trim_after(after_collapsed, max_after_chars)

        if before_snippet:
            sections.append(f"Preceding text:\n{before_snippet}")
        if after_snippet:
            sections.append(f"Following text:\n{after_snippet}")
        window_text = f"{before_snippet}\n{after_snippet}"

    prose_only = _ANY_VISUAL_RE.sub("", doc_md)
    prose_only = _COMMENT_RE.sub("", prose_only)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", prose_only) if p.strip()]

    # Find explicit visual cross-references outside the immediate window
    mention_re = _build_visual_mention_pattern(visual_id)
    ref_paras: list[str] = []
    if mention_re is not None:
        used_chars = 0
        for p_clean in paragraphs:
            if p_clean.startswith("#"):
                continue
            if p_clean in window_text:
                continue
            if mention_re.search(p_clean):
                if used_chars + len(p_clean) > max_ref_chars and ref_paras:
                    break
                ref_paras.append(p_clean)
                used_chars += len(p_clean) + 2
        if ref_paras:
            sections.append("Cross-referenced paragraphs:\n" + "\n\n".join(ref_paras))

    # Extract in-text callouts for compound IDs mentioned in caption/description (e.g. experimental IUPAC/HRMS)
    caption = (caption_map or {}).get(visual_id, "")
    compound_tokens = _extract_target_compound_tokens(caption, description)
    if compound_tokens and not visual_id.lower().startswith("unlabeled"):
        tok_pat = re.compile(
            r"(?:Compound|compd\.?|hybrid)?\s*\b(?:"
            + "|".join(re.escape(t) for t in compound_tokens)
            + r")\b|\((?:"
            + "|".join(re.escape(t) for t in compound_tokens)
            + r")\)",
            re.IGNORECASE,
        )
        callout_paras: list[str] = []
        used_callout_chars = 0
        already_included = window_text + "\n" + "\n".join(ref_paras)
        for idx, p_clean in enumerate(paragraphs):
            if p_clean in already_included:
                continue
            if tok_pat.search(p_clean) and _CHEM_CALLOUT_HINT_RE.search(p_clean):
                if used_callout_chars + len(p_clean) > max_callout_chars and callout_paras:
                    break
                callout_paras.append(p_clean)
                used_callout_chars += len(p_clean) + 2
            elif p_clean.startswith("#") and tok_pat.search(p_clean) and idx + 1 < len(paragraphs):
                # Include experimental heading + immediately following characterization paragraph
                combined = f"{p_clean}\n{paragraphs[idx + 1]}"
                if combined in already_included:
                    continue
                if used_callout_chars + len(combined) > max_callout_chars and callout_paras:
                    break
                callout_paras.append(combined)
                used_callout_chars += len(combined) + 2
        if callout_paras:
            sections.append("In-text callouts:\n" + "\n\n".join(callout_paras))

    return "\n\n".join(sections).strip()


def extract_element_template_context(
    composition_template: str,
    element_id: str,
    max_chars: int = 1500,
) -> str:
    """Extract the relevant table headers/row or scheme context for an element_id."""
    if not composition_template or not composition_template.strip():
        return ""

    elem_pattern = re.compile(
        rf"<(?:element|atom)\s+id=[\"']{re.escape(element_id)}[\"']\s*/>"
    )
    match = elem_pattern.search(composition_template)

    # Case 1: HTML <table>
    if "<table" in composition_template.lower():
        rows = _TR_RE.findall(composition_template)
        if rows:
            header_rows: list[str] = []
            target_row: Optional[str] = None
            seen_data_row = False
            for row in rows:
                is_target = bool(elem_pattern.search(row))
                if is_target and target_row is None:
                    target_row = row.strip()
                has_th = bool(re.search(r"<th\b", row, re.IGNORECASE))
                if has_th and not seen_data_row:
                    header_rows.append(row.strip())
                elif not has_th:
                    seen_data_row = True

            if target_row:
                parts = header_rows + [target_row]
                snippet = "<table>\n" + "\n".join(parts) + "\n</table>"
                return snippet[:max_chars]

    # Case 2: Markdown pipe table
    lines = composition_template.splitlines()
    for idx, line in enumerate(lines):
        if elem_pattern.search(line) and "|" in line:
            # Find contiguous pipe-table header above idx
            sep_idx = None
            for j in range(idx - 1, -1, -1):
                if _PIPE_SEP_RE.match(lines[j]):
                    sep_idx = j
                    break
                if "|" not in lines[j]:
                    break
            if sep_idx is not None:
                header_start = sep_idx
                while header_start > 0 and "|" in lines[header_start - 1]:
                    header_start -= 1
                header_block = lines[header_start : sep_idx + 1]
                return "\n".join(header_block + [line.strip()])[:max_chars]

    # Case 3: Scheme / narrative template
    if len(composition_template) <= max_chars or not match:
        return composition_template.strip()[:max_chars]

    half = max_chars // 2
    start = max(0, match.start() - half)
    end = min(len(composition_template), match.end() + half)
    return composition_template[start:end].strip()
