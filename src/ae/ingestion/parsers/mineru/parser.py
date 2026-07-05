"""MinerU-based PDF parser with visual extraction pipeline integration."""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Union, Dict, Any, Tuple, Optional

import dspy
from ae.core.config.settings import IngestionConfig
from ae.ingestion.parsers.base import BaseParser
from ae.ingestion.parsers.mineru.client import MinerUClient
from ae.ingestion.parsers.mineru.visual.model_client import get_model_client
from ae.ingestion.parsers.mineru.visual.stages.relevance_filter import (
    collect_visual_candidates,
    evaluate_relevance,
)
from ae.ingestion.parsers.mineru.visual.stages.extract_chart_tables import extract_single_chart
from ae.ingestion.parsers.mineru.visual.stages.parse_html_table import parse_html_table
from ae.ingestion.parsers.mineru.visual.stages.insert_visual_tables import replace_image_tags

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    curr = Path(__file__).resolve()
    while curr != curr.parent:
        if (curr / "pyproject.toml").exists():
            return curr
        curr = curr.parent
    return Path(__file__).resolve().parents[4]  # Default fallback


def find_mineru_outputs(output_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Locate the markdown file, content list json file, and images directory in the output directory.

    MinerU results might be in a subdirectory or at the root of output_dir.
    """
    md_file = None
    json_file = None
    images_dir = None
    
    for p in output_dir.rglob("*"):
        if p.is_file():
            if p.suffix.lower() == ".md" and p.name not in ("result.md", "final_enriched.md"):
                if md_file is None or p.name == "full.md":
                    md_file = p
            elif p.suffix.lower() == ".json" and ("content_list" in p.name or "content_list_v2" in p.name):
                json_file = p
        elif p.is_dir() and p.name == "images":
            images_dir = p
            
    return md_file, json_file, images_dir


class MinerUParser(BaseParser):
    """Parser using MinerU Web API to convert PDF to Markdown, and extracting data from charts."""

    def __init__(self, config: IngestionConfig):
        """Initialize the MinerU parser.

        Args:
            config: The full IngestionConfig instance.
        """
        if config is None:
            raise ValueError("IngestionConfig object is required for MinerUParser")
        self.cfg = config
        self.client = MinerUClient(config.mineru)
        logger.info("Initialized MinerU parser.")

    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a PDF file using MinerU API, extract data from charts, and return enriched Markdown."""
        pdf_path = Path(file_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_stem = pdf_path.stem
        project_root = find_project_root()
        
        # Get ingestion directory (fall back to settings or data/interim/ingestion)
        ingestion_dir = getattr(self.cfg, "ingestion_dir", None)
        if not ingestion_dir:
            try:
                from ae.core.config.settings import Settings
                settings = Settings.load(load_env_file=False)
                ingestion_dir = settings.paths.ingestion_dir
            except Exception:
                ingestion_dir = project_root / "data" / "interim" / "ingestion"
        
        mineru_dir = Path(ingestion_dir) / "mineru_artifacts" / pdf_stem

        logger.info(f"Starting MinerU parsing for {pdf_path.name}")
        self.client.parse_pdf(str(pdf_path), str(mineru_dir))

        md_file, json_file, images_dir = find_mineru_outputs(mineru_dir)
        if not md_file:
            raise RuntimeError(f"Could not find parsed Markdown file in MinerU output directory: {mineru_dir}")

        initial_md = md_file.read_text(encoding="utf-8")

        # Check if chart extraction is configured/enabled
        if not self.cfg.chart_extraction or not self.cfg.chart_extraction.enabled:
            logger.info("Chart extraction is not enabled. Returning raw MinerU markdown.")
            return initial_md

        if not json_file:
            logger.warning("Could not find content list JSON in MinerU output directory. Skipping chart extraction.")
            return initial_md

        # Load content list
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content_list = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load content list JSON from {json_file}: {e}")
            return initial_md

        if isinstance(content_list, dict):
            for k, v in content_list.items():
                if isinstance(v, list):
                    content_list = v
                    break

        if not isinstance(content_list, list):
            logger.warning(f"Unexpected content list JSON structure in {json_file}. Skipping chart extraction.")
            return initial_md

        # Load extraction instruction and settings
        task_name = "nanozymes"
        visual_extractor_cfg = None
        settings = None
        try:
            from ae.core.config.settings import Settings
            settings = Settings.load(load_env_file=False)
            task_name = settings.task.name
            visual_extractor_cfg = settings.llm.visual_extractor
        except Exception:
            pass

        task_dir = project_root / "config" / "tasks" / task_name
        instruction_path = task_dir / "chart_instruction.txt"

        if instruction_path.exists():
            instruction = instruction_path.read_text(encoding="utf-8")
            logger.info(f"Using chart extraction instruction from {instruction_path.name}")
        else:
            instruction = "Locate specific numerical charts and mathematically reverse-engineer their data."
            logger.info("Using default fallback chart extraction instruction")

        # Collect and evaluate relevance of all visual candidates
        candidates = collect_visual_candidates(content_list)
        relevance_map = {}
        if settings:
            relevance_map = evaluate_relevance(
                settings=settings,
                project_root=project_root,
                candidates=candidates,
                mineru_dir=mineru_dir,
            )
        else:
            # Fallback if settings couldn't load: treat all as relevant charts (backward compatibility)
            relevance_map = {
                c.img_path: {"is_relevant": c.type == "chart", "reason": "No settings available; defaulted."}
                for c in candidates
            }

        # Process each visual candidate
        results_by_img_path = {}
        warnings: list[str] = []
        client = get_model_client(client_type="visual_extractor")

        extract_service_dir = mineru_dir / "service" / "extract_chart_tables"
        extract_service_dir.mkdir(parents=True, exist_ok=True)

        for candidate in candidates:
            img_path = candidate.img_path
            
            # 1. Handle Irrelevant candidates
            relevance_info = relevance_map.get(img_path, {"is_relevant": False})
            if not relevance_info.get("is_relevant", False):
                results_by_img_path[img_path] = {
                    "status": "irrelevant",
                    "caption": candidate.caption,
                    "target_id": Path(img_path).name,
                    "tables": []
                }
                continue

            # Resolve absolute path to the image
            img_abs_path = json_file.parent / img_path
            if not img_abs_path.exists() and images_dir:
                img_abs_path = images_dir / Path(img_path).name

            # 2. Handle relevant Table blocks (with HTML body) without VLM call
            if candidate.type == "table" and candidate.table_body and candidate.table_body.strip():
                logger.info(f"Parsing HTML table body directly for table block: {img_path}")
                table_result = parse_html_table(candidate.table_body)
                table_result["caption"] = candidate.caption
                table_result["target_id"] = Path(img_path).name
                results_by_img_path[img_path] = table_result
                continue

            # 3. Handle relevant Chart or Image blocks using VLM
            if not img_abs_path.exists():
                logger.warning(f"Visual candidate image file not found: {img_path}")
                continue

            raw_response_path = extract_service_dir / f"{img_abs_path.stem}.raw_response.txt"

            vlm_result = extract_single_chart(
                cfg={
                    "model": visual_extractor_cfg.model if visual_extractor_cfg else "qwen/qwen3.6-flash",
                    "temperature": visual_extractor_cfg.temperature if visual_extractor_cfg else 0.0,
                    "max_output_tokens": (
                        visual_extractor_cfg.api.max_tokens if visual_extractor_cfg and visual_extractor_cfg.api 
                        else (visual_extractor_cfg.ollama.num_predict if visual_extractor_cfg and visual_extractor_cfg.ollama else 8192)
                    ),
                    "thinking_level": (
                        "high" if visual_extractor_cfg and visual_extractor_cfg.api and visual_extractor_cfg.api.reasoning and visual_extractor_cfg.api.reasoning.get("enabled")
                        else None
                    ),
                },
                client=client,
                image_path=img_abs_path,
                caption=candidate.caption,
                instruction=instruction,
                raw_response_path=raw_response_path
            )
            
            # Ensure VLM result has caption set correctly
            if isinstance(vlm_result, dict):
                vlm_result["caption"] = candidate.caption
                
            results_by_img_path[img_path] = vlm_result

        # Replace image tags in markdown with rendered tables / placeholders
        logger.info("Replacing visual image links in markdown with tables or placeholders...")
        enriched_md = replace_image_tags(initial_md, results_by_img_path, warnings)

        # Save final enriched markdown and enrichment summary for audit/debug
        try:
            (mineru_dir / "final_enriched.md").write_text(enriched_md, encoding="utf-8")
            with open(mineru_dir / "enrichment_summary.json", "w", encoding="utf-8") as f:
                json.dump({
                    "warnings": warnings,
                    "extractions": {
                        k: {
                            "status": v.get("status"),
                            "tables_count": len(v.get("tables", []))
                        } for k, v in results_by_img_path.items()
                    }
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Enrichment completed. Enriched markdown saved to {mineru_dir / 'final_enriched.md'}")
        except Exception as e:
            logger.warning(f"Failed to write final enrichment artifacts: {e}")

        return enriched_md
