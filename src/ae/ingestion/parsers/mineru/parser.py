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
from ae.ingestion.parsers.mineru.visual.stages.extract_chart_tables import extract_single_chart
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
        mineru_dir = project_root / "data" / "parsed" / "service" / "mineru" / pdf_stem

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

        # Load extraction instruction and ingestor config
        task_name = "nanozymes"
        ingestor_cfg = None
        try:
            from ae.core.config.settings import Settings
            settings = Settings.load(load_env_file=False)
            task_name = settings.task.name
            ingestor_cfg = settings.llm.ingestor
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

        # Extract chart data
        results_by_img_path = {}
        warnings: list[str] = []
        client = get_model_client()

        extract_service_dir = mineru_dir / "service" / "extract_chart_tables"
        extract_service_dir.mkdir(parents=True, exist_ok=True)

        normalized_charts = []

        def process_element(item: Any) -> None:
            if not isinstance(item, dict):
                return
            if item.get("type") != "chart":
                return

            img_path = item.get("img_path")
            caption_val = item.get("chart_caption", "")

            # Check for v2 structure where image_source and captions are nested inside "content"
            content_dict = item.get("content")
            if isinstance(content_dict, dict):
                image_source = content_dict.get("image_source")
                if isinstance(image_source, dict):
                    img_path = img_path or image_source.get("path")
                img_path = img_path or content_dict.get("img_path")
                caption_val = caption_val or content_dict.get("chart_caption", "")

            if img_path:
                # Normalize caption to a clean string
                caption_str = ""
                if isinstance(caption_val, list):
                    parts = []
                    for part in caption_val:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            parts.append(part.get("text") or part.get("content") or "")
                    caption_str = " ".join([p.strip() for p in parts if p]).strip()
                elif isinstance(caption_val, str):
                    caption_str = caption_val

                normalized_charts.append({
                    "img_path": img_path,
                    "caption": caption_str
                })

        def flatten_and_extract(obj: Any) -> None:
            if isinstance(obj, list):
                for sub_obj in obj:
                    flatten_and_extract(sub_obj)
            elif isinstance(obj, dict):
                process_element(obj)

        flatten_and_extract(content_list)

        for chart_info in normalized_charts:
            img_path = chart_info["img_path"]
            # Resolve absolute path to the image
            img_abs_path = json_file.parent / img_path
            if not img_abs_path.exists() and images_dir:
                img_abs_path = images_dir / Path(img_path).name

            if not img_abs_path.exists():
                logger.warning(f"Chart image file not found: {img_path}")
                continue

            caption = chart_info["caption"]
            raw_response_path = extract_service_dir / f"{img_abs_path.stem}.raw_response.txt"

            vlm_result = extract_single_chart(
                cfg={
                    "model": ingestor_cfg.model if ingestor_cfg else "qwen/qwen3.6-flash",
                    "temperature": ingestor_cfg.temperature if ingestor_cfg else 0.0,
                    "max_output_tokens": (
                        ingestor_cfg.api.max_tokens if ingestor_cfg and ingestor_cfg.api 
                        else (ingestor_cfg.ollama.num_predict if ingestor_cfg and ingestor_cfg.ollama else 8192)
                    ),
                    "thinking_level": (
                        "high" if ingestor_cfg and ingestor_cfg.api and ingestor_cfg.api.reasoning and ingestor_cfg.api.reasoning.get("enabled")
                        else None
                    ),
                },
                client=client,
                image_path=img_abs_path,
                caption=caption,
                instruction=instruction,
                raw_response_path=raw_response_path
            )
            results_by_img_path[img_path] = vlm_result

        # Replace image tags in markdown with rendered tables
        logger.info("Replacing chart image links in markdown with extracted tables...")
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
