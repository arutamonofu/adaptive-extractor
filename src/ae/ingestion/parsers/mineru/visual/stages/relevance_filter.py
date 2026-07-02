"""Stage for evaluating visual candidate relevance using the Teacher model."""

from __future__ import annotations

import json
import logging
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..model_client import get_model_client, generate_parsed_json

logger = logging.getLogger(__name__)


@dataclass
class VisualCandidate:
    img_path: str
    type: str  # "chart", "image", "table"
    caption: str
    page_idx: int
    table_body: Optional[str] = None


def normalize_caption(item: Dict[str, Any], block_type: str) -> str:
    """Extract and normalize caption from block based on its type."""
    caption_val = None
    if block_type == "chart":
        caption_val = item.get("chart_caption")
    elif block_type == "image":
        caption_val = item.get("image_caption")
    elif block_type == "table":
        caption_val = item.get("table_caption")

    # Fallback to general content dictionary if nested
    content_dict = item.get("content")
    if isinstance(content_dict, dict):
        caption_val = caption_val or content_dict.get("chart_caption") or content_dict.get("image_caption") or content_dict.get("table_caption")

    if not caption_val:
        return ""

    if isinstance(caption_val, list):
        parts = []
        for part in caption_val:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
        return " ".join([p.strip() for p in parts if p]).strip()
    elif isinstance(caption_val, str):
        return caption_val
    return str(caption_val)


def collect_visual_candidates(content_list: List[Any]) -> List[VisualCandidate]:
    """Gather all image-producing layout blocks from MinerU content list."""
    candidates = []

    def process_element(item: Any) -> None:
        if not isinstance(item, dict):
            return
        
        block_type = item.get("type")
        if block_type not in ("chart", "image", "table"):
            return

        img_path = item.get("img_path")
        
        # Check for nested structure inside 'content' (e.g. content_list_v2)
        content_dict = item.get("content")
        if isinstance(content_dict, dict):
            image_source = content_dict.get("image_source")
            if isinstance(image_source, dict):
                img_path = img_path or image_source.get("path")
            img_path = img_path or content_dict.get("img_path")

        if not img_path:
            return

        caption = normalize_caption(item, block_type)
        page_idx = item.get("page_idx", 0)
        table_body = item.get("table_body")

        candidates.append(
            VisualCandidate(
                img_path=img_path,
                type=block_type,
                caption=caption,
                page_idx=page_idx,
                table_body=table_body,
            )
        )

    def flatten_and_collect(obj: Any) -> None:
        if isinstance(obj, list):
            for sub_obj in obj:
                flatten_and_collect(sub_obj)
        elif isinstance(obj, dict):
            process_element(obj)

    flatten_and_collect(content_list)
    return candidates


def evaluate_relevance(
    settings: Any,
    project_root: Path,
    candidates: List[VisualCandidate],
    mineru_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Determine relevance of visual candidates using Teacher LLM."""
    manifest_path = mineru_dir / "visual_manifest.json"
    
    # Try to load existing manifest to avoid duplicate calls
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                logger.info("Loading existing visual manifest relevance decisions.")
                saved_manifest = json.load(f)
                return {item["img_path"]: item for item in saved_manifest}
        except Exception as e:
            logger.warning(f"Failed to read existing manifest at {manifest_path}: {e}")

    if not candidates:
        logger.info("No visual candidates found in document.")
        return {}

    # Load project schema and baseline instructions
    schema_content = ""
    baseline_content = ""

    schema_file = project_root / settings.paths.schema_file
    if schema_file.exists():
        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                schema_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read schema file {schema_file}: {e}")

    baseline_file = project_root / settings.paths.baseline_prompt_file
    if baseline_file.exists():
        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                baseline_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read baseline prompt file {baseline_file}: {e}")

    # Build prompt
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    template_path = prompts_dir / "prompt_relevance.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Relevance prompt template not found: {template_path}")
    
    template = template_path.read_text(encoding="utf-8")

    # Serialize candidates for the prompt
    candidates_data = [
        {
            "img_path": c.img_path,
            "type": c.type,
            "caption": c.caption,
            "page": c.page_idx + 1
        }
        for c in candidates
    ]

    prompt = (
        template.replace("{{SCHEMA_FIELDS}}", schema_content)
        .replace("{{EXTRACTION_INSTRUCTIONS}}", baseline_content)
        .replace("{{CANDIDATES_JSON}}", json.dumps(candidates_data, ensure_ascii=False, indent=2))
    )

    logger.info(f"Evaluating relevance of {len(candidates)} visual candidates via Teacher LLM...")
    
    client = get_model_client(client_type="teacher")
    
    try:
        raw_response_path = mineru_dir / "relevance_raw_response.txt"
        raw, parsed, warnings = generate_parsed_json(
            client,
            raw_response_path=raw_response_path,
            prompt=prompt,
            model=settings.llm.teacher.model,
            temperature=0.0,
            max_output_tokens=4000,
        )
        
        if not isinstance(parsed, list):
            raise ValueError("Relevance LLM response parsed JSON is not a list")

        # Map parsed results back
        results = {}
        for item in parsed:
            if not isinstance(item, dict) or "img_path" not in item:
                continue
            img_path = item["img_path"]
            is_relevant = item.get("is_relevant", False)
            reason = item.get("reason", "")
            results[img_path] = {
                "img_path": img_path,
                "is_relevant": is_relevant,
                "reason": reason
            }

        # Ensure all candidates are in results (fallback to False)
        for c in candidates:
            if c.img_path not in results:
                results[c.img_path] = {
                    "img_path": c.img_path,
                    "is_relevant": False,
                    "reason": "Missing from LLM relevance response; defaulted to False."
                }

        # Save manifest
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
            logger.info(f"Visual manifest saved to {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to write manifest file: {e}")

        return results

    except Exception as exc:
        logger.error(f"Failed to evaluate visual candidates relevance: {exc}", exc_info=True)
        # Fallback to marking all as False or True? Let's default to False to avoid expensive VLM calls on failures
        results = {
            c.img_path: {
                "img_path": c.img_path,
                "is_relevant": False,
                "reason": f"Relevance check failed: {exc}"
            }
            for c in candidates
        }
        return results
