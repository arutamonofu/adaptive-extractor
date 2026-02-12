# src/aee/utils/dataset.py
"""Dataset utilities for AutoEvoExtractor."""

import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import dspy

from aee.data import ProcessedDocument

logger = logging.getLogger(__name__)


def create_dataset_from_ids(
    processed_dir: Path,
    gt_data: Dict[str, Any],
    task_conf: Dict[str, Any],
    allowed_ids: List[str],
    limit: Optional[int] = None,
    seed: int = 42
) -> List[dspy.Example]:
    """Load DSPy Examples strictly for a given list of document IDs.
    
    Searches recursively in processed_dir to handle nested folders (train/auto, val, etc).
    
    Args:
        processed_dir: Directory containing processed documents.
        gt_data: Ground truth data mapping document IDs to their data.
        task_conf: Task configuration containing the output model.
        allowed_ids: List of allowed document IDs to consider.
        limit: Maximum number of examples to load. If None, no limit is applied.
        seed: Random seed for shuffling when limiting examples.
        
    Returns:
        List of DSPy Examples containing document text and extracted data.
        
    Raises:
        ValueError: If required parameters are invalid.
        TypeError: If parameters are of incorrect type.
    """
    # Validate inputs
    if not isinstance(processed_dir, Path):
        raise TypeError("processed_dir must be a Path object")
    
    if not processed_dir.exists():
        raise ValueError(f"Processed directory does not exist: {processed_dir}")
        
    if not isinstance(gt_data, dict):
        raise TypeError("gt_data must be a dictionary")
        
    if not isinstance(task_conf, dict):
        raise TypeError("task_conf must be a dictionary")
        
    if "output_model" not in task_conf:
        raise ValueError("task_conf must contain 'output_model' key")
        
    if not isinstance(allowed_ids, list):
        raise TypeError("allowed_ids must be a list")
        
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
        
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    # Get the output model from task configuration
    output_model = task_conf["output_model"]
    
    # Filter candidates: exist in GT AND are in allowed list
    candidates = [doc_id for doc_id in allowed_ids if doc_id in gt_data]
    
    # Shuffle deterministically if we need to limit
    if limit is not None and len(candidates) > limit:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:limit]
    
    logger.info(f"Loading dataset. Candidates: {len(candidates)} (Limit: {limit})")

    # Create a mapping of document stems to their paths for efficient lookup
    try:
        file_map = {p.stem: p for p in processed_dir.rglob("*.json")}
        logger.debug(f"Found {len(file_map)} JSON files in {processed_dir}")
    except Exception as e:
        logger.error(f"Failed to build file mapping from {processed_dir}: {e}")
        return []

    # Build dataset
    dataset: List[dspy.Example] = []
    missing_files: Set[str] = set()
    load_errors = 0
    
    for doc_id in candidates:
        json_path = file_map.get(doc_id)
        
        if not json_path:
            # Track missing files to avoid repetitive logging
            if doc_id not in missing_files:
                logger.warning(f"File allowed but not found on disk: {doc_id} (Looked in {processed_dir})")
                missing_files.add(doc_id)
            continue
            
        try:
            # Load and validate the processed document
            doc_content = json_path.read_text(encoding="utf-8")
            doc = ProcessedDocument.model_validate_json(doc_content)
            
            # Skip documents without text content
            if not doc.text_content:
                logger.debug(f"Skipping empty document: {doc_id}")
                continue

            # Create DSPy example with the document text and ground truth data
            example = dspy.Example(
                document_text=doc.text_content,
                extracted_data=output_model(experiments=gt_data[doc_id])
            ).with_inputs("document_text")
            
            dataset.append(example)

        except FileNotFoundError:
            if doc_id not in missing_files:
                logger.warning(f"File not found: {doc_id}")
                missing_files.add(doc_id)
        except Exception as e:
            load_errors += 1
            logger.debug(f"Failed to load {doc_id}: {e}")
            continue
    
    if load_errors > 0:
        logger.warning(f"Encountered {load_errors} errors while loading documents")
            
    logger.info(f"Successfully loaded {len(dataset)} examples from {len(candidates)} candidates")
    return dataset