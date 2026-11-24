"""VisualParsingService: on-demand visual parsing and decomposition."""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, Optional, Union

import pymupdf as fitz

from ae.core.config.settings import IngestionConfig
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.schemas import (
    ElementLocalizationResult,
    Manifest,
    ManifestVisual,
    PageLocalization,
    PipelineSummary,
    StageUsage,
    VisualBox,
    VisualDecomposition,
    VisualParseRecord,
)
from ae.ingestion.parsers.gemini.stages.context import (
    extract_element_template_context,
    extract_local_text_context,
)
from ae.ingestion.parsers.gemini.stages.parse_visuals import (
    compose_parse_prompt,
    parse_single_visual,
)
from ae.ingestion.parsers.gemini.stages.render_markdown import (
    render_composite_block,
    render_parse_body,
    replace_manifest_with_enrichment,
)
from ae.ingestion.parsers.gemini.stages.segment_and_crop_visuals_elements import (
    compute_crop_rect_and_dpi,
    crop_elements,
    decompose_visual,
    extract_element_localization_from_decomposition,
    generate_debug_overlay,
    localize_elements,
    localize_page,
    select_elements_by_hint,
)

logger = logging.getLogger(__name__)


def strip_pdf_extension(doc_id: str) -> str:
    """Strip .pdf extension if present, preserving dots in document IDs."""
    name = Path(doc_id).name.strip()
    if name.lower().endswith(".pdf"):
        return name[:-4]
    return name


class VisualParsingService:
    """Service for lazy, on-demand decomposition and visual parsing.

    Executes Steps T1-T7 when invoked by an agent tool:
    - Checks cache in 04_parse_visuals/visuals/{visual_id}.json
    - Step T1: Localize visual (if missing)
    - Step T2: High-DPI visual crop
    - Step T3: Decompose visual (single vs composite with composition_template)
    - Step T4: Localize elements (if composite)
    - Step T5: Direct vector PDF crop for elements
    - Step T6: Parallel/profile parsing with RDKit validation
    - Step T7: Render enrichment and permanently update {doc}.md in-place
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        pdf_dir: Optional[Union[str, Path]] = None,
        ingestion_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cfg = config
        if ingestion_dir:
            self.ingestion_dir = Path(ingestion_dir)
        elif config and config.ingestion_dir:
            self.ingestion_dir = Path(config.ingestion_dir)
        else:
            self.ingestion_dir = Path("data/interim/ingestion")
        self.pdf_dir = Path(pdf_dir) if pdf_dir else self._resolve_pdf_dir()
        self._client: Optional[GeminiClient] = None

    def _resolve_pdf_dir(self) -> Path:
        """Resolve directory containing input PDFs."""
        candidates = [
            Path("data/raw/pdfs"),
            Path("data/raw"),
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c
        return Path("data/raw/pdfs")

    @property
    def client(self) -> GeminiClient:
        """Lazy initialization of GeminiClient."""
        if self._client is None:
            if self.cfg is None or self.cfg.gemini is None:
                raise ValueError(
                    "IngestionConfig with gemini section is required for VisualParsingService"
                )
            api_key = self._resolve_api_key()
            self._client = GeminiClient(self.cfg.gemini, api_key)
        return self._client

    @staticmethod
    def _resolve_api_key() -> str:
        """Retrieve Gemini API key from environment or Settings."""
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key
        try:
            from ae.core.config.settings import Settings

            settings = Settings.load(load_env_file=True)
            if settings.gemini_api_key:
                return settings.gemini_api_key.get_secret_value()
        except Exception:
            pass
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Set environment variable or configure in .env",
        )

    def locate_pdf(self, doc_id: str) -> Path:
        """Locate PDF file for doc_id."""
        clean_id = strip_pdf_extension(doc_id)
        candidates = [
            self.pdf_dir / f"{clean_id}.pdf",
            self.pdf_dir / f"{doc_id}.pdf",
            Path("data/raw/pdfs") / f"{clean_id}.pdf",
            Path("data/raw") / f"{clean_id}.pdf",
            Path(doc_id),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()

        # Case-insensitive fallback search
        clean_lower = clean_id.lower()
        target_pdf_name = f"{clean_lower}.pdf"
        search_dirs = [self.pdf_dir, Path("data/raw/pdf"), Path("data/raw/pdfs"), Path("data/raw")]
        for d in search_dirs:
            if d.exists() and d.is_dir():
                for f in d.iterdir():
                    if f.is_file() and f.name.lower() == target_pdf_name:
                        return f.resolve()

        raise FileNotFoundError(
            f"Could not locate PDF file for doc_id='{doc_id}' in {self.pdf_dir}"
        )

    def _resolve_doc_name(self, doc_id: str) -> str:
        """Resolve document folder name in ingestion directory, preserving existing case."""
        clean_id = strip_pdf_extension(doc_id)
        if (self.ingestion_dir / clean_id).is_dir():
            return clean_id
        clean_lower = clean_id.lower()
        if self.ingestion_dir.exists() and self.ingestion_dir.is_dir():
            for p in self.ingestion_dir.iterdir():
                if p.is_dir() and p.name.lower() == clean_lower:
                    return p.name
        return clean_id

    def parse_visuals(
        self,
        doc_id: str,
        visual_ids: list[str],
        target_hint: Optional[str] = None,
    ) -> dict[str, VisualParseRecord]:
        """Parse requested visuals for document, updating {doc}.md permanently.

        Args:
            doc_id: Document identifier (e.g. '10.1021_acs.jmedchem.9b01150')
            visual_ids: List of visual IDs (e.g. ['Figure_3', 'Scheme_1', 'Scheme_2/mol_4'])
            target_hint: Optional guidance on which specific compounds, panels, or metrics
                         are needed. When provided, composite figures will selectively parse
                         only matching sub-elements.

        Returns:
            Dictionary mapping visual_id to VisualParseRecord
        """
        doc_name = self._resolve_doc_name(doc_id)
        artifacts = ArtifactManager(doc_name, base_dir=self.ingestion_dir)
        pdf_path: Optional[Path] = None

        def _get_pdf_path() -> Path:
            nonlocal pdf_path
            if pdf_path is None:
                pdf_path = self.locate_pdf(doc_name)
            return pdf_path

        # Load manifest
        manifest: Optional[Manifest] = None
        if artifacts.manifest_json_path.exists():
            manifest = Manifest.model_validate(artifacts.load_json(artifacts.manifest_json_path))

        # Load caption map
        caption_map: dict[str, str] = {}
        if artifacts.placeholders_json_path.exists():
            raw_ph = artifacts.load_json(artifacts.placeholders_json_path)
            caption_map = {item["id"]: item.get("caption", "") for item in raw_ph}

        # Load existing localizations
        localizations: dict[int, PageLocalization] = artifacts.load_all_page_bboxes()

        # Load current markdown text (fallback to stage 01 text.md for context extraction)
        doc_md_path = artifacts.document_md_path
        current_md = artifacts.load_text(doc_md_path) if doc_md_path.exists() else ""
        context_source_md = current_md
        if not context_source_md and artifacts.text_md_path.exists():
            context_source_md = artifacts.load_text(artifacts.text_md_path)

        records: dict[str, VisualParseRecord] = {}
        pdf_doc: Optional[fitz.Document] = None

        def _get_pdf_doc() -> fitz.Document:
            nonlocal pdf_doc
            if pdf_doc is None:
                pdf_doc = fitz.open(str(_get_pdf_path()))
            return pdf_doc

        def _format_enrichment(v_id: str, cap: str, body: str, v_type: str = "table") -> str:
            inner = f"{cap.strip()}\n\n{body.strip()}" if cap.strip() else body.strip()
            return (
                f'<visual id="{v_id}" type="{v_type}">\n'
                f"{inner}\n"
                f"</visual>"
            )

        # Map root_visual_id -> set of requested element_ids (None represents entire visual or hint-filtered)
        req_by_root: dict[str, set[Optional[str]]] = {}
        for item in visual_ids:
            if not item:
                continue
            item_str = str(item).strip()
            if "/" in item_str:
                r_id, _, e_id = item_str.partition("/")
                req_by_root.setdefault(r_id.strip(), set()).add(e_id.strip())
            elif "#" in item_str:
                r_id, _, e_id = item_str.partition("#")
                req_by_root.setdefault(r_id.strip(), set()).add(e_id.strip())
            else:
                req_by_root.setdefault(item_str, set()).add(None)

        # Snapshot usage counters for this doc_name so we only add deltas to summary.json
        prev_usage: dict[str, StageUsage] = {}
        prev_counters: dict[str, int] = {}
        if self._client is not None and hasattr(self._client, "get_all_stage_usage"):
            raw_u = self._client.get_all_stage_usage(doc_id=doc_name)
            if isinstance(raw_u, dict):
                prev_usage = {k: v.model_copy() for k, v in raw_u.items() if isinstance(v, StageUsage)}
        if self._client is not None and hasattr(self._client, "get_scope_counters"):
            raw_c = self._client.get_scope_counters(doc_id=doc_name)
            if isinstance(raw_c, dict):
                prev_counters = dict(raw_c)

        scope_cm = (
            self._client.doc_scope(doc_name)
            if self._client is not None and hasattr(self._client, "doc_scope")
            else contextlib.nullcontext()
        )
        # Also bind doc_scope dynamically if self.client is lazily instantiated inside the loop
        active_scope_entered = self._client is not None

        try:
          with scope_cm:
            for visual_id, requested_elements in req_by_root.items():
                caption = caption_map.get(visual_id, "")
                cached_file = artifacts.parsed_visuals_dir / f"{visual_id}.json"
                has_root_request = None in requested_elements
                explicit_elements = {e for e in requested_elements if e is not None}

                # 1. Check cached parse
                if cached_file.exists():
                    try:
                        rec = VisualParseRecord.model_validate(artifacts.load_json(cached_file))
                        satisfies_request = False
                        if rec.status == "success":
                            satisfies_request = True
                        elif rec.status == "partial":
                            if explicit_elements and not has_root_request and explicit_elements.issubset(set(rec.parsed_element_ids)):
                                satisfies_request = True

                        if satisfies_request:
                            records[visual_id] = rec
                            for ea in explicit_elements:
                                records[f"{visual_id}/{ea}"] = rec
                            v_type = "table"
                            if manifest and manifest.visuals:
                                for mv in manifest.visuals:
                                    if mv.id == visual_id:
                                        v_type = mv.type
                                        break
                            if rec.rendered_markdown and visual_id in current_md:
                                current_md = replace_manifest_with_enrichment(
                                    current_md,
                                    visual_id,
                                    _format_enrichment(visual_id, caption, rec.rendered_markdown, v_type),
                                )
                            continue
                    except Exception as err:
                        logger.warning(f"Failed to read cached parsed visual for '{visual_id}': {err}")

                # 2. Identify visual and pages from manifest
                visual_obj: Optional[ManifestVisual] = None
                if manifest and manifest.visuals:
                    for vis in manifest.visuals:
                        if vis.id == visual_id:
                            visual_obj = vis
                            break

                visual_type = visual_obj.type if visual_obj else "mixed"
                pages = visual_obj.pages if visual_obj and visual_obj.pages else [1]

                # 3. Step T1: Localize visual on required pages
                visual_crops: list[Path] = []
                visual_clip_rects: dict[int, fitz.Rect] = {}

                for page_num in pages:
                    page_loc = localizations.get(page_num)
                    box: Optional[VisualBox] = None
                    if page_loc:
                        for bx in page_loc.boxes:
                            if bx.id == visual_id and bx.found:
                                box = bx
                                break

                    if box is None:
                        loc_cfg = (
                            self.cfg.gemini.get_stage_config("localize")
                            if self.cfg and getattr(self.cfg, "gemini", None)
                            else None
                        )
                        page_dpi = int(getattr(loc_cfg, "page_dpi", 150) or 150)
                        loc_page_img = (
                            artifacts.visuals_geom_dir / "pages" / f"p{page_num}_dpi{page_dpi}.png"
                        )
                        if not loc_page_img.exists() or loc_page_img.stat().st_size == 0:
                            doc = _get_pdf_doc()
                            p = doc[page_num - 1]
                            pix = p.get_pixmap(dpi=page_dpi, alpha=False)
                            loc_page_img.parent.mkdir(parents=True, exist_ok=True)
                            pix.save(str(loc_page_img))

                        page_img = loc_page_img
                        if not page_img.exists() or page_img.stat().st_size == 0:
                            fallback_img = artifacts.pages_dir / f"p{page_num}.png"
                            if fallback_img.exists():
                                page_img = fallback_img

                        # Localize all manifest visuals on page_num
                        page_visuals: list[ManifestVisual] = []
                        if manifest and manifest.visuals:
                            for mv in manifest.visuals:
                                if page_num in mv.pages:
                                    m_type = (
                                        mv.type
                                        if mv.type in ["table", "chart", "molecular_structure"]
                                        else "molecular_structure"
                                    )
                                    desc = mv.description or caption_map.get(mv.id, "")
                                    page_visuals.append(
                                        ManifestVisual(
                                            id=mv.id,
                                            type=m_type,
                                            pages=[page_num],
                                            description=desc,
                                        )
                                    )
                        if not any(v.id == visual_id for v in page_visuals):
                            dummy_type = (
                                visual_type
                                if visual_type in ["table", "chart", "molecular_structure"]
                                else "molecular_structure"
                            )
                            page_visuals.append(
                                ManifestVisual(
                                    id=visual_id,
                                    type=dummy_type,
                                    pages=[page_num],
                                    description=visual_obj.description if visual_obj else caption,
                                )
                            )

                        page_loc = localize_page(
                            self.client,
                            page_img,
                            page_num,
                            Manifest(visuals=page_visuals),
                            self.cfg,
                        )
                        localizations[page_num] = page_loc
                        artifacts.save_page_bboxes(page_num, page_loc)
                        for bx in page_loc.boxes:
                            if bx.id == visual_id and bx.found:
                                box = bx
                                break

                    box_coords = box.box_2d if box and box.found else [100, 100, 900, 900]

                    # 4. Step T2: High-DPI Vector Crop
                    doc = _get_pdf_doc()
                    p = doc[page_num - 1]
                    crop_cfg = (
                        getattr(self.cfg.gemini, "crop", None)
                        if self.cfg and self.cfg.gemini
                        else None
                    )
                    clip_rect, dpi = compute_crop_rect_and_dpi(box_coords, p.rect, crop_cfg)
                    visual_clip_rects[page_num] = clip_rect
                    crop_path = artifacts.crops_dir / f"{visual_id}__p{page_num}.png"
                    if not (crop_path.exists() and crop_path.stat().st_size > 0):
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        pix = p.get_pixmap(dpi=dpi, clip=clip_rect)
                        pix.save(str(crop_path))
                    visual_crops.append(crop_path)

                local_context = (
                    extract_local_text_context(
                        context_source_md,
                        visual_id,
                        caption_map=caption_map,
                        description=visual_obj.description if visual_obj else "",
                    )
                    if context_source_md
                    else ""
                )

                # Bind doc_scope if client was lazily initialized above
                with (
                    self.client.doc_scope(doc_name)
                    if (not active_scope_entered and hasattr(self.client, "doc_scope"))
                    else contextlib.nullcontext()
                ):
                    # 5. Step T3: Analyze and Decompose Visual (with optional merged T4 boxes)
                    decomp = decompose_visual(
                        self.client,
                        visual_id,
                        visual_crops,
                        caption,
                        visual_type,
                        artifacts,
                        self.cfg,
                        description=visual_obj.description if visual_obj else "",
                        local_context=local_context,
                    )

                invalid_smiles_list: list[dict[str, Any]] = []
                rendered_content = ""
                parsed_element_ids: list[str] = []
                unparsed_element_ids: list[str] = []
                parse_status: str = "success"

                if decomp.mode == "composite" and decomp.elements:
                    all_elem_ids = [e.element_id for e in decomp.elements]
                    if explicit_elements and not has_root_request:
                        target_element_ids = [eid for eid in all_elem_ids if eid in explicit_elements]
                    elif target_hint:
                        target_element_ids = select_elements_by_hint(
                            self.client,
                            visual_id,
                            decomp.elements,
                            target_hint,
                            self.cfg,
                        )
                        if explicit_elements:
                            target_element_ids = list(set(target_element_ids).union(explicit_elements))
                    else:
                        target_element_ids = all_elem_ids

                    # Step T4: Localize elements (use merged T3+T4 boxes if valid, else fallback)
                    merge_t3_t4 = getattr(
                        getattr(self.cfg, "execution", None), "merge_t3_t4", None
                    )
                    if merge_t3_t4 is None:
                        merge_t3_t4 = getattr(
                            getattr(self.cfg, "gemini", None), "merge_t3_t4", True
                        )
                    cached_elem_bbox = artifacts.element_bboxes_dir / f"{visual_id}.json"
                    if cached_elem_bbox.exists() and cached_elem_bbox.stat().st_size > 0:
                        elem_loc = localize_elements(
                            self.client,
                            visual_id,
                            visual_crops,
                            decomp.elements,
                            artifacts,
                            self.cfg,
                        )
                    elif merge_t3_t4:
                        elem_loc, missing_elems = extract_element_localization_from_decomposition(
                            decomp,
                            crop_paths=visual_crops,
                            visual_id=visual_id,
                            artifacts=artifacts,
                        )
                        target_set = set(target_element_ids)
                        missing_targets = [e for e in missing_elems if e.element_id in target_set]
                        if not missing_targets:
                            logger.info(
                                f"Merged T3+T4 succeeded for '{visual_id}': skipped T4 call "
                                f"({len(elem_loc.boxes)} element boxes)."
                            )
                            if hasattr(self.client, "record_event"):
                                self.client.record_event("t4_skipped_calls", doc_id=doc_name)
                        else:
                            logger.info(
                                f"Merged T3+T4 missing {len(missing_targets)} target box(es) for "
                                f"'{visual_id}'; falling back to T4 localize_elements."
                            )
                            if hasattr(self.client, "record_event"):
                                self.client.record_event("t4_fallback_calls", doc_id=doc_name)
                            fallback_loc = localize_elements(
                                self.client,
                                visual_id,
                                visual_crops,
                                missing_elems,
                                artifacts,
                                self.cfg,
                            )
                            merged_boxes_by_id = {b.id: b for b in elem_loc.boxes}
                            for fb in fallback_loc.boxes:
                                merged_boxes_by_id[fb.id] = fb
                            elem_loc = ElementLocalizationResult(
                                boxes=list(merged_boxes_by_id.values())
                            )
                            artifacts.save_json(cached_elem_bbox, elem_loc)
                            if visual_crops and visual_crops[0].exists():
                                element_type_map = {e.element_id: e.type for e in decomp.elements}
                                debug_boxes = [
                                    (b.id, b.box_2d, element_type_map.get(b.id, "molecular_structure"))
                                    for b in elem_loc.boxes
                                    if b.found
                                ]
                                if debug_boxes:
                                    generate_debug_overlay(
                                        visual_crops[0],
                                        debug_boxes,
                                        artifacts.debug_dir / f"{visual_id}_elements_bboxes.png",
                                        padding_ratio=0.05,
                                    )
                    else:
                        elem_loc = localize_elements(
                            self.client,
                            visual_id,
                            visual_crops,
                            decomp.elements,
                            artifacts,
                            self.cfg,
                        )

                    # Step T5: Direct vector crop for elements (grouped by page_index)
                    elem_page_idx_map = {e.element_id: e.page_index for e in decomp.elements}
                    boxes_by_page: dict[int, list[VisualBox]] = {}
                    for bx in elem_loc.boxes:
                        p_idx = elem_page_idx_map.get(bx.id, 0)
                        clamped_idx = min(max(0, p_idx), len(pages) - 1)
                        target_page = pages[clamped_idx]
                        boxes_by_page.setdefault(target_page, []).append(bx)

                    element_crop_paths: dict[str, list[Path]] = {}
                    for target_page, page_boxes in boxes_by_page.items():
                        page_elem_crops = crop_elements(
                            _get_pdf_path(),
                            visual_id,
                            target_page,
                            visual_clip_rects.get(target_page, fitz.Rect(0, 0, 600, 800)),
                            page_boxes,
                            artifacts,
                            self.cfg,
                        )
                        element_crop_paths.update(page_elem_crops)

                    # Step T6: Extract each element (with cache check and parallel execution)
                    element_results: dict[str, Any] = {}
                    pending_elements: list[tuple[Any, ManifestVisual, list[Path], Path, str]] = []

                    for elem in decomp.elements:
                        elem_crop_entry = element_crop_paths.get(elem.element_id)
                        if isinstance(elem_crop_entry, list):
                            elem_crop_list = [p for p in elem_crop_entry if p and p.exists()] or visual_crops
                        elif elem_crop_entry and elem_crop_entry.exists():
                            elem_crop_list = [elem_crop_entry]
                        else:
                            elem_crop_list = visual_crops

                        elem_manifest_visual = ManifestVisual(
                            id=elem.element_id,
                            type=elem.type,
                            description=elem.description,
                            panel_label=elem.label or None,
                        )
                        elem_cached = (
                            artifacts.parsed_elements_dir / f"{visual_id}__{elem.element_id}.json"
                        )
                        if elem_cached.exists() and elem_cached.stat().st_size > 0:
                            try:
                                raw_elem = artifacts.load_json(elem_cached)
                                _, schema_cls = compose_parse_prompt(
                                    elem_manifest_visual, caption, len(elem_crop_list)
                                )
                                cached_res = schema_cls.model_validate(raw_elem)
                                if getattr(cached_res, "status", "success") != "failed":
                                    element_results[elem.element_id] = cached_res
                                    if hasattr(cached_res, "invalid_smiles") and cached_res.invalid_smiles:
                                        for item in cached_res.invalid_smiles:
                                            entry = dict(item) if isinstance(item, dict) else {"smiles": str(item)}
                                            entry.setdefault("visual_id", visual_id)
                                            entry.setdefault("element_id", elem.element_id)
                                            invalid_smiles_list.append(entry)
                                    continue
                            except Exception as err:
                                logger.warning(
                                    f"Failed to load cached element '{elem.element_id}' for '{visual_id}': {err}"
                                )

                        if elem.element_id in target_element_ids:
                            tmpl_ctx = extract_element_template_context(
                                decomp.composition_template or "", elem.element_id
                            )
                            ctx_parts = []
                            if tmpl_ctx:
                                ctx_parts.append(f"Element Placement in Visual:\n{tmpl_ctx}")
                            if local_context:
                                ctx_parts.append(f"Surrounding Article Context:\n{local_context}")
                            elem_context = "\n\n".join(ctx_parts)
                            pending_elements.append(
                                (elem, elem_manifest_visual, elem_crop_list, elem_cached, elem_context)
                            )

                    if pending_elements:
                        max_workers = (
                            getattr(self.cfg.gemini, "max_concurrent_requests", 4)
                            if self.cfg and self.cfg.gemini
                            else 4
                        )
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=min(max_workers, len(pending_elements))
                        ) as executor:
                            future_to_elem = {}
                            for (
                                elem_obj,
                                elem_mv,
                                crop_list,
                                cache_path,
                                elem_context,
                            ) in pending_elements:
                                ctx = contextvars.copy_context()
                                fut = executor.submit(
                                    ctx.run,
                                    parse_single_visual,
                                    self.client,
                                    elem_mv,
                                    crop_list,
                                    caption,
                                    self.cfg,
                                    elem_context,
                                )
                                future_to_elem[fut] = (elem_obj, cache_path)
                            for future in concurrent.futures.as_completed(future_to_elem):
                                elem_obj, cache_path = future_to_elem[future]
                                try:
                                    elem_res = future.result()
                                except Exception as elem_exc:
                                    logger.error(
                                        f"Element '{elem_obj.element_id}' of '{visual_id}' raised an unexpected error: {elem_exc}"
                                    )
                                    continue
                                cache_path.parent.mkdir(parents=True, exist_ok=True)
                                artifacts.save_json(cache_path, elem_res)
                                if getattr(elem_res, "status", "success") == "failed":
                                    logger.warning(
                                        f"Element '{elem_obj.element_id}' of '{visual_id}' failed parsing; leaving unparsed for retry."
                                    )
                                    continue
                                if hasattr(elem_res, "invalid_smiles") and elem_res.invalid_smiles:
                                    for item in elem_res.invalid_smiles:
                                        entry = dict(item) if isinstance(item, dict) else {"smiles": str(item)}
                                        entry.setdefault("visual_id", visual_id)
                                        entry.setdefault("element_id", elem_obj.element_id)
                                        invalid_smiles_list.append(entry)
                                element_results[elem_obj.element_id] = elem_res

                    # Step T7: Assemble composite visual cleanly
                    rendered_content = render_composite_block(decomp, element_results, visual_id=visual_id)
                    parsed_element_ids = [e.element_id for e in decomp.elements if e.element_id in element_results]
                    unparsed_element_ids = [e.element_id for e in decomp.elements if e.element_id not in element_results]
                    parse_status: Literal["success", "partial", "failed"] = (
                        "success"
                        if not unparsed_element_ids
                        else ("partial" if parsed_element_ids else "failed")
                    )

                else:
                    # Single visual
                    single_type = decomp.single_type or (
                        visual_type
                        if visual_type in ["table", "chart", "molecular_structure"]
                        else "table"
                    )
                    single_visual = ManifestVisual(
                        id=visual_id,
                        type=single_type,
                        description=visual_obj.description if visual_obj else caption,
                    )
                    single_res = parse_single_visual(
                        self.client,
                        single_visual,
                        visual_crops,
                        caption,
                        self.cfg,
                        context=local_context,
                    )
                    if hasattr(single_res, "invalid_smiles") and single_res.invalid_smiles:
                        for item in single_res.invalid_smiles:
                            entry = dict(item) if isinstance(item, dict) else {"smiles": str(item)}
                            entry.setdefault("visual_id", visual_id)
                            invalid_smiles_list.append(entry)

                    rendered_content = render_parse_body(single_res)
                    single_status = getattr(single_res, "status", "success")
                    parse_status = (
                        "failed"
                        if single_status != "success" or rendered_content == "*[Parsing failed]*"
                        else "success"
                    )

                full_enrichment = _format_enrichment(
                    visual_id, caption, rendered_content, visual_type
                )

                record = VisualParseRecord(
                    visual_id=visual_id,
                    status=parse_status,
                    mode=decomp.mode,
                    rendered_markdown=rendered_content,
                    parsed_element_ids=parsed_element_ids,
                    unparsed_element_ids=unparsed_element_ids,
                    summary=f"Parsed {visual_id} ({decomp.mode}, status={parse_status}).",
                    invalid_smiles=invalid_smiles_list,
                )
                records[visual_id] = record
                for ea in explicit_elements:
                    records[f"{visual_id}/{ea}"] = record

                # Save parse record
                artifacts.save_json(cached_file, record)

                # In-place permanent update of markdown only when parsing produced usable output
                if parse_status != "failed":
                    current_md = replace_manifest_with_enrichment(current_md, visual_id, full_enrichment)
        finally:
            if pdf_doc is not None:
                pdf_doc.close()

        # Write updated document to disk
        if current_md:
            artifacts.save_text(artifacts.document_md_path, current_md)

        # Update summary.json (merging per-document stage deltas without overwriting stage 01/02)
        if artifacts.summary_json_path.exists():
            try:
                sum_data = artifacts.load_json(artifacts.summary_json_path)
                summary = PipelineSummary.model_validate(sum_data)
                summary.total_parsed_visuals = len(
                    list(artifacts.parsed_visuals_dir.glob("*.json"))
                )
                if self._client is not None and hasattr(self._client, "get_all_stage_usage"):
                    try:
                        cur_usage = self._client.get_all_stage_usage(doc_id=doc_name)
                    except TypeError:
                        cur_usage = self._client.get_all_stage_usage()
                    if isinstance(cur_usage, dict):
                        for stage_key, cur_val in cur_usage.items():
                            if not isinstance(cur_val, StageUsage):
                                continue
                            prev_val = prev_usage.get(stage_key, StageUsage())
                            d_calls = max(0, cur_val.calls - prev_val.calls)
                            d_prompt = max(0, cur_val.prompt_tokens - prev_val.prompt_tokens)
                            d_cached = max(0, cur_val.cached_tokens - prev_val.cached_tokens)
                            d_cand = max(0, cur_val.candidates_tokens - prev_val.candidates_tokens)
                            d_reas = max(0, cur_val.reasoning_tokens - prev_val.reasoning_tokens)
                            d_total = max(0, cur_val.total_tokens - prev_val.total_tokens)
                            if d_calls or d_total:
                                existing = summary.stages_usage.get(stage_key, StageUsage())
                                existing.calls += d_calls
                                existing.prompt_tokens += d_prompt
                                existing.cached_tokens += d_cached
                                existing.candidates_tokens += d_cand
                                existing.reasoning_tokens += d_reas
                                existing.total_tokens += d_total
                                summary.stages_usage[stage_key] = existing
                if self._client is not None and hasattr(self._client, "get_scope_counters"):
                    cur_c = self._client.get_scope_counters(doc_id=doc_name)
                    if isinstance(cur_c, dict):
                        summary.t4_skipped_calls += max(
                            0, int(cur_c.get("t4_skipped_calls", 0)) - prev_counters.get("t4_skipped_calls", 0)
                        )
                        summary.t4_fallback_calls += max(
                            0, int(cur_c.get("t4_fallback_calls", 0)) - prev_counters.get("t4_fallback_calls", 0)
                        )
                        summary.silent_error_retries += max(
                            0, int(cur_c.get("silent_error_retries", 0)) - prev_counters.get("silent_error_retries", 0)
                        )
                artifacts.save_json(artifacts.summary_json_path, summary)
            except Exception as e:
                logger.warning(f"Could not update summary.json: {e}")

        logger.info(
            f"Parsed {len(records)} visual(s) for '{doc_name}'. Document updated."
        )
        return records


