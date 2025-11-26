# src/aee/utils/dataset.py

import random
import dspy
from pathlib import Path
from typing import List, Dict, Any

from aee.core.types import ProcessedDocument

def create_training_set(
    processed_dir: Path, 
    gt_data: Dict[str, Any], 
    task_conf: Dict[str, Any], 
    limit: int
) -> List[dspy.Example]:
    """
    Creates a list of DSPy Examples by matching Processed JSONs with Ground Truth.
    
    Args:
        processed_dir: Path to the folder containing .json processed documents.
        gt_data: Dictionary mapping filenames to ground truth experiments.
        task_conf: Task configuration dictionary (must contain 'output_model').
        limit: Maximum number of examples to return.

    Returns:
        A list of dspy.Example objects ready for the optimizer.
    """
    dataset = []
    
    # Iterate over GT keys to ensure we only take labeled data
    for filename, experiments in gt_data.items():
        # Try to find corresponding JSON
        json_path = processed_dir / f"{filename}.json" 
        
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = ProcessedDocument.model_validate_json(f.read())
            
            if not doc.text_content:
                continue

            # Create DSPy Example
            OutputModel = task_conf["output_model"]
            
            example = dspy.Example(
                document_text=doc.text_content,
                extracted_data=OutputModel(experiments=experiments)
            ).with_inputs("document_text")
            
            dataset.append(example)
        except Exception:
            # Skip corrupted files silently
            continue
            
    # Shuffle and limit
    random.shuffle(dataset)
    return dataset[:limit]