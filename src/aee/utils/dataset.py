# src/aee/utils/dataset.py

import logging
import random
import dspy
from pathlib import Path
from typing import List, Dict, Any, Optional

from aee.core.types import ProcessedDocument

logger = logging.getLogger(__name__)

def create_dataset_from_ids(
    processed_dir: Path, 
    gt_data: Dict[str, Any], 
    task_conf: Dict[str, Any], 
    allowed_ids: List[str],
    limit: Optional[int] = None,
    seed: int = 42
) -> List[dspy.Example]:
    """
    Loads DSPy Examples strictly for a given list of document IDs.
    Searches recursively in processed_dir to handle nested folders (train/auto, val, etc).
    """
    dataset = []
    OutputModel = task_conf["output_model"]
    
    # Filter candidates: exist in GT AND are in allowed list
    candidates = [doc_id for doc_id in allowed_ids if doc_id in gt_data]
    
    # Shuffle deterministically if we need to limit
    if limit is not None and len(candidates) > limit:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:limit]
    
    logger.info(f"Loading dataset. Candidates: {len(candidates)} (Limit: {limit})")

    file_map = {p.stem: p for p in processed_dir.rglob("*.json")}

    for doc_id in candidates:
        json_path = file_map.get(doc_id)
        
        if not json_path:
            logger.warning(f"File allowed but not found on disk: {doc_id} (Looked in {processed_dir})")
            continue
            
        try:
            doc = ProcessedDocument.model_validate_json(json_path.read_text(encoding="utf-8"))
            
            if not doc.text_content:
                logger.warning(f"Skipping empty document: {doc_id}")
                continue

            example = dspy.Example(
                document_text=doc.text_content,
                extracted_data=OutputModel(experiments=gt_data[doc_id])
            ).with_inputs("document_text")
            
            dataset.append(example)

        except Exception as e:
            logger.warning(f"Failed to load {doc_id}: {e}")
            continue
            
    return dataset