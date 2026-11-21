"""LangChain tool for lazy on-demand visual parsing."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool, tool

from ae.core.config.settings import Settings
from ae.ingestion.service import VisualParsingService

logger = logging.getLogger(__name__)


def _format_visual_results(results: dict[str, Any]) -> str:
    """Format VisualParsingService records into <visual> markdown blocks."""
    blocks_out: list[str] = []
    seen_visuals: set[str] = set()

    for v_id, rec in results.items():
        vis_key = rec.visual_id or v_id
        if vis_key in seen_visuals:
            continue
        seen_visuals.add(vis_key)

        if rec.status in ("success", "partial") and rec.rendered_markdown:
            block = (
                f'<visual id="{vis_key}">\n'
                f"{rec.rendered_markdown.strip()}\n"
                f"</visual>"
            )
            if rec.status == "partial":
                unparsed = list(getattr(rec, "unparsed_element_ids", []) or [])
                if unparsed:
                    block += (
                        f"\n[Partial parse for {vis_key}: unparsed elements remain: "
                        f"{', '.join(unparsed)}. Call parse_visuals(['{vis_key}']) without "
                        f"target_hint for a full parse or request specific element IDs.]"
                    )
                else:
                    block += (
                        f"\n[Partial parse for {vis_key}: call parse_visuals(['{vis_key}']) "
                        f"without target_hint for a full parse.]"
                    )
            blocks_out.append(block)
        else:
            blocks_out.append(f"[{vis_key}: parsing failed]")

    if not blocks_out:
        return "No visual results returned."
    return "\n\n".join(blocks_out)


def _normalize_to_root_visual_ids(visual_ids: list[str]) -> list[str]:
    """Normalize any element-scoped IDs (e.g. 'Scheme_1/mol_1a') to root visual IDs ('Scheme_1')."""
    root_ids: list[str] = []
    for item in visual_ids:
        if not item:
            continue
        raw = str(item).strip()
        if "/" in raw:
            raw = raw.partition("/")[0].strip()
        elif "#" in raw:
            raw = raw.partition("#")[0].strip()
        if raw and raw not in root_ids:
            root_ids.append(raw)
    return root_ids


def make_visual_parsing_tool(
    doc_id: str,
    service: Optional[VisualParsingService] = None,
    mode: Literal["inference", "semantic_optimization"] = "inference",
) -> BaseTool:
    """Factory creating a LangChain tool bound to doc_id and VisualParsingService.

    Args:
        doc_id: Normalized document identifier.
        service: Optional VisualParsingService instance.
        mode: 'inference' (default, supports selective sub-element target_hint) or
              'semantic_optimization' (enforces full visual unpacking without target_hint).
    """
    if service is None:
        try:
            settings = Settings.load(load_env_file=True)
            service = VisualParsingService(
                config=settings.ingestion,
                pdf_dir=settings.paths.pdf_dir,
                ingestion_dir=settings.paths.ingestion_dir,
            )
        except Exception as e:
            logger.warning(f"Could not load global Settings for VisualParsingService: {e}")
            service = VisualParsingService()

    svc = service

    if mode == "semantic_optimization":
        @tool
        def parse_visuals(
            visual_ids: list[str],
        ) -> str:
            """Parse unextracted visual manifest blocks (tables, charts, molecular schemes) in full for contrastive Semantic Prompt Optimization.

            Args:
                visual_ids: List of visual manifest IDs to unpack completely (e.g. ['Table_1', 'Scheme_2']).
                            Pass all Ground Truth-bearing visuals and any competing/ambiguous candidate visuals together.
                            Every requested visual is unpacked in full (all rows, columns, and sub-elements) so both
                            target entities and adjacent distractors/precursors are visible.
            """
            root_ids = _normalize_to_root_visual_ids(visual_ids)
            if not root_ids:
                return "No visual_ids provided."

            logger.info(
                f"Tool parse_visuals (mode='semantic_optimization') invoked for doc '{doc_id}' with visuals: {root_ids}"
            )
            try:
                results = svc.parse_visuals(doc_id, root_ids)
            except Exception as exc:
                logger.error(f"Error during visual parsing of {root_ids}: {exc}")
                return f"Parsing failed for {root_ids}: {exc}"

            return _format_visual_results(results)

        return parse_visuals

    @tool
    def parse_visuals(
        visual_ids: list[str],
        target_hint: Optional[str] = None,
    ) -> str:
        """Parse unextracted visuals from manifest stubs (tables, charts, diagrams/schemes) or specific sub-elements.

        Args:
            visual_ids: List of manifest IDs (e.g. ['Table_1', 'Figure_2']) or specific element IDs
                        (e.g. ['Figure_2/panel_a']). Batch multiple IDs in one call when complementary
                        or when operating under a tight tool-iteration budget.
            target_hint: Optional guidance on which specific items, panels, or metrics are needed
                         (e.g. 'section A items', 'panel c metrics', 'summary rows').
                         When provided, composite figures will selectively parse only matching sub-elements,
                         leaving other elements as requestable stubs. Omit target_hint on a subsequent call
                         to re-parse the full visual if needed.
        """
        if not visual_ids:
            return "No visual_ids provided."

        hint_str = f" (target_hint='{target_hint}')" if target_hint else ""
        logger.info(
            f"Tool parse_visuals invoked for doc '{doc_id}' with visuals: {visual_ids}{hint_str}"
        )
        try:
            if target_hint is not None:
                results = svc.parse_visuals(doc_id, visual_ids, target_hint=target_hint)
            else:
                results = svc.parse_visuals(doc_id, visual_ids)
        except Exception as exc:
            logger.error(f"Error during visual parsing of {visual_ids}: {exc}")
            return f"Parsing failed for {visual_ids}: {exc}"

        return _format_visual_results(results)

    return parse_visuals



