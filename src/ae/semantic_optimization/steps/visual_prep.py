"""Unified Contrastive Visual Unpacking for Reverse Engineering (Phase 1A + 1B)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ae.core.llm.history_logger import extraction_log_context
from ae.core.storage.documents import DocumentRepository
from ae.ingestion.parsers.gemini.stages.render_markdown import (
    extract_unparsed_visual_ids,
    has_unparsed_visuals,
)
from ae.ingestion.tool import _normalize_to_root_visual_ids
from ae.semantic_optimization.context import RunContext

logger = logging.getLogger(__name__)


_CONTRASTIVE_VISUAL_PREP_SYSTEM_PROMPT = """You are preparing a scientific document's visual blocks for Contrastive Reverse Engineering (both Positive Analysis of Ground Truth rows/cells and Negative Analysis of excluded distractor entities, controls, preparatory/intermediate items, and competing measurements).

Inspect the Extraction Schema, Baseline Prompt, Ground Truth table, and the document's unparsed visual manifest stubs (`<visual id="..." type="...">Caption\\n> description</visual>`) or partially unpacked visuals containing `[Unparsed: ...]`.

APPLY THIS STRICT TWO-TIER UNPACKING POLICY:

1. TIER 1 — UNPACK IN FULL (Call `parse_visuals`):
   - **Ground Truth-bearing visuals**: Any table, diagram/schematic, chart, or figure from which Ground Truth entities, structured entity encodings, conditions, or metric values originate.
   - **Competing / Ambiguous / Borderline candidate visuals**: Any visual whose caption and `> description` involve the **same target entity class or target property/assay** defined in the Schema, Baseline Prompt, and Ground Truth (e.g., secondary tables of analogous entities, diagrams or schematics showing preparatory/intermediate items for the target series, or alternative plots of the target property) where full table/schematic contents are needed to formulate precise inclusion/exclusion boundaries or verify potential Ground Truth omissions.
   - **Chained references**: If an unpacked table references entity labels whose definitions, structured encodings, or experimental parameters reside in an unparsed diagram/figure, unpack that diagram/figure as well.

2. TIER 2 — KEEP AS MANIFEST STUB (Do NOT call `parse_visuals`):
   - **Clearly off-target visuals**: Visuals whose irrelevance or distractor role is **already unambiguous from their manifest caption and `> description` alone** — such as clearly off-target/orthogonal experiments, routine diagnostic/calibration plots, or generic schematic illustrations. Leave these as manifest stubs; downstream negative analysis will cite their stub descriptions directly without wasting VLM compute.

Call `parse_visuals` with the list of all Tier 1 `visual_ids`. If no Tier 1 visuals remain unparsed, reply ONLY with `READY`."""


def prepare_document_visuals(
    norm_doc_id: str,
    raw_text: str,
    gt_table_csv: str,
    context: RunContext,
    doc_repo: DocumentRepository,
    force_recompute: bool = False,
    max_unpack_turns: Optional[int] = None,
    extra_context: str = "",
) -> str:
    """Unpack Tier 1 visuals (GT-bearing + competing/ambiguous candidates) in full before 1A & 1B.

    Preserves a stable document markdown across Positive (1A) and Negative (1B) analyses
    to maximize Prompt Prefix Cache reuse and avoid partial sub-element fragmentation.
    """
    if not has_unparsed_visuals(raw_text):
        return raw_text

    artifacts_mgr = getattr(context, "artifacts", None)
    marker_path: Optional[Path] = None
    if artifacts_mgr is not None and hasattr(artifacts_mgr, "get_path"):
        cand_marker = artifacts_mgr.get_path(
            "positive", "visual_prep", f"{norm_doc_id}.json", iteration=None
        )
        if isinstance(cand_marker, Path):
            marker_path = cand_marker

    if not force_recompute and marker_path is not None and marker_path.exists():
        logger.debug(
            f"Visual preparation marker exists for '{norm_doc_id}' ({marker_path}); reusing prepared document."
        )
        return doc_repo.get(norm_doc_id) or raw_text

    try:
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
        from ae.core.config.settings import Settings
        from ae.core.llm.factory import create_chat_model
        from ae.ingestion.service import VisualParsingService
        from ae.ingestion.tool import make_visual_parsing_tool
        from ae.semantic_optimization.client import SemanticOptimizationClient

        raw_cfg = getattr(context, "config_path", None)
        cfg_path = raw_cfg if isinstance(raw_cfg, (str, Path)) else None
        settings = Settings.load(config_path=cfg_path, load_env_file=True)
        if isinstance(getattr(context, "re_client", None), SemanticOptimizationClient):
            chat_model = context.re_client.get_chat_model()
        else:
            chat_model = create_chat_model(settings.semantic_optimization.get_llm_config())

        pdf_dir = getattr(context, "pdf_dir", None) or settings.paths.pdf_dir
        service = VisualParsingService(
            config=settings.ingestion,
            pdf_dir=pdf_dir,
            ingestion_dir=context.ingestion_dir,
        )
        tool = make_visual_parsing_tool(norm_doc_id, service, mode="semantic_optimization")
        bound = chat_model.bind_tools([tool])

        schema_str = getattr(context, "schema", "") or ""
        baseline_str = getattr(context, "baseline_prompt", "") or ""

        user_parts = []
        if schema_str:
            user_parts.append(f"Extraction Schema:\n{schema_str}")
        if baseline_str:
            user_parts.append(f"Baseline Prompt:\n{baseline_str}")
        if extra_context:
            user_parts.append(extra_context.strip())
        user_parts.append(f"Ground Truth Table:\n{gt_table_csv}")
        user_parts.append(f"Document:\n{raw_text}")

        messages: list = [
            SystemMessage(content=_CONTRASTIVE_VISUAL_PREP_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(user_parts)),
        ]

        if max_unpack_turns is None:
            ctx_turns = getattr(context, "max_tool_iterations", None)
            effective_max_turns = ctx_turns if isinstance(ctx_turns, int) and ctx_turns >= 1 else 3
        else:
            effective_max_turns = max(1, max_unpack_turns)

        attempted_root_ids: list[str] = []
        current_text = raw_text
        turns_used = 0

        for turn in range(effective_max_turns):
            unparsed_now = extract_unparsed_visual_ids(current_text)
            remaining_unattempted = [
                vid
                for vid in unparsed_now
                if _normalize_to_root_visual_ids([vid])[0] not in attempted_root_ids
            ]
            if not remaining_unattempted:
                break

            turns_used += 1
            with extraction_log_context(
                doc_id=norm_doc_id,
                source="visual_prep",
                stage="visual_prep",
                node="prepare_document_visuals",
            ):
                msg = bound.invoke(messages)
            tool_calls = getattr(msg, "tool_calls", None) or []
            parse_calls = [tc for tc in tool_calls if tc.get("name") == "parse_visuals"]
            if not parse_calls:
                break

            unpacked_any_in_turn = False
            tool_messages: list[ToolMessage] = []

            for idx, tc in enumerate(parse_calls):
                args = tc.get("args", {}) or {}
                raw_ids = args.get("visual_ids", []) or []
                root_ids = _normalize_to_root_visual_ids(raw_ids)
                new_root_ids = [rid for rid in root_ids if rid not in attempted_root_ids]
                if not new_root_ids:
                    continue

                attempted_root_ids.extend(new_root_ids)
                unpacked_any_in_turn = True
                tool_out = tool.invoke({"visual_ids": new_root_ids})
                tool_messages.append(
                    ToolMessage(
                        content=str(tool_out),
                        tool_call_id=tc.get("id") or f"call_visual_prep_{turn}_{idx}",
                    )
                )

            current_text = doc_repo.get(norm_doc_id) or current_text

            if not unpacked_any_in_turn:
                break

            if isinstance(msg, BaseMessage) and tool_messages:
                messages.append(msg)
                messages.extend(tool_messages)
                messages.append(
                    HumanMessage(
                        content=(
                            "Inspect the newly unpacked visual content above. "
                            "If it references entity identifiers, definitions, or parameters located in another still-unparsed Tier 1 visual "
                            "(e.g. a companion diagram, schematic, or table), call `parse_visuals` with those visual IDs. "
                            "Otherwise reply ONLY with `READY`."
                        )
                    )
                )
            else:
                # Fallback for non-BaseMessage test mocks
                updated_parts = list(user_parts[:-1]) + [f"Document:\n{current_text}"]
                messages = [
                    SystemMessage(content=_CONTRASTIVE_VISUAL_PREP_SYSTEM_PROMPT),
                    HumanMessage(content="\n\n".join(updated_parts)),
                ]

        # Save preparation marker so subsequent cached runs do not re-invoke the LLM
        if marker_path is not None:
            try:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_payload = {
                    "doc_id": norm_doc_id,
                    "unpacked_visual_ids": attempted_root_ids,
                    "remaining_stub_ids": extract_unparsed_visual_ids(current_text),
                    "turns_used": turns_used,
                }
                marker_path.write_text(
                    json.dumps(marker_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as marker_err:
                logger.debug(f"Could not write visual_prep marker for '{norm_doc_id}': {marker_err}")

        return current_text
    except Exception as exc:
        logger.warning(f"Contrastive visual preparation skipped for '{norm_doc_id}': {exc}")
        return raw_text
