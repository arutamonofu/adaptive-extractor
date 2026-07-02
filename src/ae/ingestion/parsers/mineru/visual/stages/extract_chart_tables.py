"""Stage for extracting tabular data from cropped chart images using VLM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from ..model_client import generate_parsed_json

logger = logging.getLogger(__name__)


def extract_single_chart(
    cfg: Dict[str, Any],
    client: Any,
    image_path: Path,
    caption: str,
    instruction: str,
    raw_response_path: Path,
) -> Dict[str, Any]:
    """Extract tabular data from a single cropped chart image using VLM."""
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    template = (prompts_dir / "prompt_extract.txt").read_text(encoding="utf-8")
    
    target = {
        "target_id": image_path.name,
        "figure": image_path.stem,
        "caption": caption
    }
    
    prompt = (
        template.replace("{{TARGET_JSON}}", json.dumps(target, ensure_ascii=False, indent=2))
        .replace("{{INSTRUCTION}}", instruction)
    )

    logger.info(f"Extracting chart data from image: {image_path.name}")
    
    try:
        raw, parsed, parse_warnings = generate_parsed_json(
            client,
            raw_response_path=raw_response_path,
            prompt=prompt,
            model=cfg.get("model", "gemini-3.5-flash"),
            files=[image_path],
            temperature=cfg.get("temperature", 0.0),
            max_output_tokens=cfg.get("max_output_tokens", 20000),
            thinking_level=cfg.get("thinking_level"),
            thinking_budget=cfg.get("thinking_budget"),
        )
        if not isinstance(parsed, dict):
            raise ValueError("VLM response parsed JSON is not a dictionary object")
        
        # Ensure standard keys are present
        parsed.setdefault("status", "success")
        parsed.setdefault("tables", [])
        parsed.setdefault("warnings", [])
        if parse_warnings:
            parsed["warnings"].extend(parse_warnings)
            
        return parsed
    except Exception as exc:
        logger.error(f"VLM extraction failed for {image_path.name}: {exc}", exc_info=True)
        return {
            "status": "failed",
            "tables": [],
            "warnings": [f"VLM extraction error: {exc}"],
        }
