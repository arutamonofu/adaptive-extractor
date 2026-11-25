"""Artifacts management and stage resumption for Gemini Ingestion v2."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from ae.ingestion.parsers.gemini.schemas import PageLocalization

logger = logging.getLogger(__name__)

STAGES = [
    "parse_text",
    "manifest_visuals",
    "segment_and_crop_visuals_elements",
    "parse_visuals",
]


class ArtifactManager:
    """Manages reading, writing, and stage invalidation for ingestion artifacts."""

    def __init__(self, doc_name: str, base_dir: Optional[Path] = None) -> None:
        """Initialize artifact manager for a specific document.

        Args:
            doc_name: PDF stem name (e.g., 'acsomega.2c06142').
            base_dir: Base directory for ingestion outputs. Defaults to
                      'data/interim/ingestion'.
        """
        if base_dir is None:
            base_dir = Path("data/interim/ingestion")
        self.base_dir = Path(base_dir)
        self.doc_name = doc_name
        self.artifact_dir = self.base_dir / doc_name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create subdirectories
        self.parse_text_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.visuals_geom_dir.mkdir(parents=True, exist_ok=True)
        self.visual_bboxes_dir.mkdir(parents=True, exist_ok=True)
        self.elements_geom_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.decompositions_dir.mkdir(parents=True, exist_ok=True)
        self.element_bboxes_dir.mkdir(parents=True, exist_ok=True)
        self.element_crops_dir.mkdir(parents=True, exist_ok=True)
        self.parse_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_visuals_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_elements_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 01: parse_text
    # ------------------------------------------------------------------

    @property
    def parse_text_dir(self) -> Path:
        return self.artifact_dir / "01_parse_text"

    @property
    def chunks_dir(self) -> Path:
        return self.parse_text_dir / "chunks"

    @property
    def text_md_path(self) -> Path:
        return self.parse_text_dir / "text.md"

    @property
    def placeholders_json_path(self) -> Path:
        return self.parse_text_dir / "placeholders.json"

    # ------------------------------------------------------------------
    # Stage 02: manifest_visuals
    # ------------------------------------------------------------------

    @property
    def manifest_dir(self) -> Path:
        return self.artifact_dir / "02_manifest_visuals"

    @property
    def manifest_json_path(self) -> Path:
        return self.manifest_dir / "manifest.json"

    @property
    def manifest_enriched_text_path(self) -> Path:
        return self.manifest_dir / "manifest_enriched_text.md"

    @property
    def manifest_text_path(self) -> Path:
        """Alias for manifest_enriched_text_path."""
        return self.manifest_enriched_text_path

    @property
    def rendered_pages_with_visuals_dir(self) -> Path:
        return self.manifest_dir / "rendered_pages_with_visuals"

    @property
    def pages_dir(self) -> Path:
        """Alias for rendered_pages_with_visuals_dir."""
        return self.rendered_pages_with_visuals_dir

    # ------------------------------------------------------------------
    # Stage 03: segment_and_crop_visuals_elements
    # ------------------------------------------------------------------

    @property
    def segment_and_crop_dir(self) -> Path:
        primary = self.artifact_dir / "03_segment_and_crop_visuals_elements"
        legacy = self.artifact_dir / "03_segment_and_crop_visuals_atoms"
        if legacy.exists() and not primary.exists():
            return legacy
        return primary

    @property
    def visuals_geom_dir(self) -> Path:
        return self.segment_and_crop_dir / "visuals"

    @property
    def elements_geom_dir(self) -> Path:
        primary = self.segment_and_crop_dir / "elements"
        legacy = self.segment_and_crop_dir / "atoms"
        if legacy.exists() and not primary.exists():
            return legacy
        return primary

    @property
    def atoms_geom_dir(self) -> Path:
        """Alias for elements_geom_dir."""
        return self.elements_geom_dir

    @property
    def visual_bboxes_dir(self) -> Path:
        return self.visuals_geom_dir / "bboxes"

    @property
    def bboxes_json_path(self) -> Path:
        """Legacy path for monolithic bboxes.json (maintained for backward compatibility)."""
        return self.visuals_geom_dir / "bboxes.json"

    def page_bboxes_path(self, page_num: int) -> Path:
        return self.visual_bboxes_dir / f"p{page_num}.json"

    def mark_visual_bboxes_empty(self) -> None:
        """Mark visual bboxes stage as complete when document has no visuals."""
        self.visual_bboxes_dir.mkdir(parents=True, exist_ok=True)
        (self.visual_bboxes_dir / ".empty").touch()

    def save_page_bboxes(self, page_num: int, loc: PageLocalization | dict[str, Any]) -> None:
        """Save localization bounding boxes for a single page."""
        self.visual_bboxes_dir.mkdir(parents=True, exist_ok=True)
        self.save_json(self.page_bboxes_path(page_num), loc)

    def load_page_bboxes(self, page_num: int) -> Optional[PageLocalization]:
        """Load localization bounding boxes for a specific page if present."""
        path = self.page_bboxes_path(page_num)
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            data = self.load_json(path)
            return PageLocalization.model_validate(data)
        except Exception as e:
            logger.warning(f"Could not load page bboxes from {path}: {e}")
            return None

    def load_all_page_bboxes(self) -> dict[int, PageLocalization]:
        """Load all page localization bounding boxes.

        Reads from visual_bboxes_dir (p*.json), falling back to legacy bboxes.json.
        """
        results: dict[int, PageLocalization] = {}
        if self.visual_bboxes_dir.exists():
            for p in sorted(self.visual_bboxes_dir.glob("p*.json")):
                if p.stat().st_size > 0:
                    try:
                        data = self.load_json(p)
                        loc = PageLocalization.model_validate(data)
                        results[loc.page] = loc
                    except Exception as e:
                        logger.warning(f"Could not load page bboxes from {p}: {e}")

        if not results and self.bboxes_json_path.exists() and self.bboxes_json_path.stat().st_size > 0:
            try:
                raw = self.load_json(self.bboxes_json_path)
                if isinstance(raw, dict) and "pages" in raw:
                    for p in raw["pages"]:
                        loc = PageLocalization.model_validate(p)
                        results[loc.page] = loc
                elif isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(v, dict):
                            loc = PageLocalization.model_validate(v)
                            results[int(k)] = loc
            except Exception as e:
                logger.warning(f"Could not load legacy bboxes.json from {self.bboxes_json_path}: {e}")

        return results

    @property
    def crops_dir(self) -> Path:
        return self.visuals_geom_dir / "crops"

    @property
    def decompositions_dir(self) -> Path:
        return self.visuals_geom_dir / "segmentations"

    @property
    def element_bboxes_dir(self) -> Path:
        primary = self.elements_geom_dir / "bboxes"
        legacy = self.segment_and_crop_dir / "atoms" / "bboxes"
        if legacy.exists() and not primary.exists():
            return legacy
        return primary

    @property
    def atom_bboxes_dir(self) -> Path:
        """Alias for element_bboxes_dir."""
        return self.element_bboxes_dir

    @property
    def element_crops_dir(self) -> Path:
        primary = self.elements_geom_dir / "crops"
        legacy = self.segment_and_crop_dir / "atoms" / "crops"
        if legacy.exists() and not primary.exists():
            return legacy
        return primary

    @property
    def atom_crops_dir(self) -> Path:
        """Alias for element_crops_dir."""
        return self.element_crops_dir

    # ------------------------------------------------------------------
    # Stage 04: parse_visuals
    # ------------------------------------------------------------------

    @property
    def parse_dir(self) -> Path:
        return self.artifact_dir / "04_parse_visuals"

    @property
    def parsed_visuals_dir(self) -> Path:
        return self.parse_dir / "visuals"

    @property
    def parsed_elements_dir(self) -> Path:
        primary = self.parse_dir / "elements"
        legacy = self.parse_dir / "atoms"
        if legacy.exists() and not primary.exists():
            return legacy
        return primary

    @property
    def parsed_atoms_dir(self) -> Path:
        """Alias for parsed_elements_dir."""
        return self.parsed_elements_dir

    # ------------------------------------------------------------------
    # Document-level root files & Debug
    # ------------------------------------------------------------------

    @property
    def debug_dir(self) -> Path:
        return self.artifact_dir / "debug"

    @property
    def document_md_path(self) -> Path:
        return self.artifact_dir / f"{self.doc_name}.md"

    @property
    def summary_json_path(self) -> Path:
        return self.debug_dir / "summary.json"

    # ------------------------------------------------------------------
    # Stage readiness
    # ------------------------------------------------------------------

    def has_parse_text_artifact(self) -> bool:
        return (
            self.text_md_path.exists()
            and self.text_md_path.stat().st_size > 0
            and self.placeholders_json_path.exists()
            and self.placeholders_json_path.stat().st_size > 0
        )

    def has_manifest_artifact(self) -> bool:
        return self.manifest_json_path.exists() and self.manifest_json_path.stat().st_size > 0

    def has_render_pages_artifact(self, expected_pages: set[int]) -> bool:
        if not expected_pages:
            return True
        for p in expected_pages:
            img_path = self.pages_dir / f"p{p}.png"
            if not img_path.exists() or img_path.stat().st_size == 0:
                return False
        return True

    def has_localize_artifact(self) -> bool:
        if (self.visual_bboxes_dir / ".empty").exists():
            return True
        if self.visual_bboxes_dir.exists():
            json_files = list(self.visual_bboxes_dir.glob("p*.json"))
            if json_files and all(f.stat().st_size > 0 for f in json_files):
                return True
        return self.bboxes_json_path.exists() and self.bboxes_json_path.stat().st_size > 0

    def has_crop_artifact(self, expected_crops: list[str]) -> bool:
        if not expected_crops:
            return True
        for name in expected_crops:
            crop_path = self.crops_dir / name
            if not crop_path.exists() or crop_path.stat().st_size == 0:
                return False
        return True

    def has_parsed_for_object(self, object_id: str) -> bool:
        path = self.parsed_visuals_dir / f"{object_id}.json"
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            data = self.load_json(path)
            return data.get("status") != "failed"
        except Exception:
            return False

    def has_parse_artifact(self, expected_object_ids: list[str]) -> bool:
        if not expected_object_ids:
            return True
        return all(self.has_parsed_for_object(oid) for oid in expected_object_ids)

    def has_render_artifact(self) -> bool:
        return (
            self.document_md_path.exists()
            and self.document_md_path.stat().st_size > 0
            and self.summary_json_path.exists()
            and self.summary_json_path.stat().st_size > 0
        )

    # ------------------------------------------------------------------
    # Invalidation & Resumption
    # ------------------------------------------------------------------

    def invalidate_from_stage(self, stage_name: str) -> None:
        """Delete artifacts for stage_name and all subsequent stages."""
        stage_aliases = {
            "classify": "manifest_visuals",
            "segment_and_crop_visuals_atoms": "segment_and_crop_visuals_elements",
        }
        stage_name = stage_aliases.get(stage_name, stage_name)
        if stage_name not in STAGES:
            raise ValueError(
                f"Unknown stage: {stage_name}. Valid stages: {STAGES}",
            )

        start_idx = STAGES.index(stage_name)
        stages_to_clear = STAGES[start_idx:]
        logger.info(
            f"Invalidating artifacts for doc '{self.doc_name}' from stage '{stage_name}': "
            f"{stages_to_clear}",
        )

        for s in stages_to_clear:
            self._clear_stage_artifacts(s)

    def _clear_stage_artifacts(self, stage: str) -> None:
        if stage == "parse_text":
            if self.parse_text_dir.exists():
                shutil.rmtree(self.parse_text_dir)
            self.parse_text_dir.mkdir(parents=True, exist_ok=True)
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
        elif stage == "manifest_visuals":
            if self.manifest_dir.exists():
                shutil.rmtree(self.manifest_dir)
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            self.pages_dir.mkdir(parents=True, exist_ok=True)
        elif stage in ("segment_and_crop_visuals_elements", "segment_and_crop_visuals_atoms"):
            if self.segment_and_crop_dir.exists():
                shutil.rmtree(self.segment_and_crop_dir)
            legacy_dir = self.artifact_dir / "03_segment_and_crop_visuals_atoms"
            if legacy_dir.exists() and legacy_dir != self.segment_and_crop_dir:
                shutil.rmtree(legacy_dir)
            self.segment_and_crop_dir.mkdir(parents=True, exist_ok=True)
            self.visuals_geom_dir.mkdir(parents=True, exist_ok=True)
            self.visual_bboxes_dir.mkdir(parents=True, exist_ok=True)
            self.elements_geom_dir.mkdir(parents=True, exist_ok=True)
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            self.decompositions_dir.mkdir(parents=True, exist_ok=True)
            self.element_bboxes_dir.mkdir(parents=True, exist_ok=True)
            self.element_crops_dir.mkdir(parents=True, exist_ok=True)
            if self.debug_dir.exists():
                shutil.rmtree(self.debug_dir)
                self.debug_dir.mkdir(parents=True, exist_ok=True)
        elif stage == "parse_visuals":
            if self.parse_dir.exists():
                shutil.rmtree(self.parse_dir)
            self.parse_dir.mkdir(parents=True, exist_ok=True)
            self.parsed_visuals_dir.mkdir(parents=True, exist_ok=True)
            self.parsed_elements_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    @classmethod
    def _to_jsonable(cls, obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()
        if isinstance(obj, dict):
            return {str(k): cls._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._to_jsonable(item) for item in obj]
        return obj

    def save_json(self, path: Path, data: Any) -> None:
        """Write JSON data to path (supporting Pydantic models and dicts)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._to_jsonable(data)
        json_text = json.dumps(payload, indent=2, ensure_ascii=False)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_text)

    def load_json(self, path: Path) -> Any:
        """Load JSON data from path."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_text(self, path: Path, text: str) -> None:
        """Write text to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def load_text(self, path: Path) -> str:
        """Read text from path."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
