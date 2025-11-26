# src/aee/utils/io.py

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Callable

logger = logging.getLogger(__name__)

def load_ground_truth(csv_path: Path, converter_func: Callable) -> Dict[str, List[Any]]:
    """
    Loads Ground Truth from CSV and groups experiments by filename.
    
    Args:
        csv_path: Path to the CSV file.
        converter_func: Function to convert a pandas row to a Pydantic model.
        
    Returns:
        Dict mapping filename (stem) to a list of experiment objects.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"GT file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Identify filename column
    possible_cols = ["pdf", "filename", "source", "doi"]
    filename_col = next((c for c in possible_cols if c in df.columns), None)
    
    if not filename_col:
        raise ValueError(f"No filename column found in CSV. Available: {list(df.columns)}")

    gt_dict = {}
    
    for identifier, group in df.groupby(filename_col):
        # Normalize key: "File.pdf" -> "file"
        key = str(identifier).replace(".pdf", "").strip().lower()
        
        experiments = []
        for _, row in group.iterrows():
            if exp := converter_func(row):
                experiments.append(exp)
        
        if experiments:
            gt_dict[key] = experiments
            
    return gt_dict

def load_predictions(results_dir: Path, schema_class: Any) -> Dict[str, List[Any]]:
    """
    Loads model predictions from JSON files in a directory.
    
    Args:
        results_dir: Directory containing _result.json files.
        schema_class: Pydantic class to validate the 'experiments' list.
    """
    pred_dict = {}
    files = list(results_dir.glob("*.json"))
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Robust name extraction
            meta_name = data.get("source_metadata", {}).get("filename", "")
            if not meta_name:
                meta_name = f.stem.replace("_result", "")
            
            key = meta_name.replace(".pdf", "").strip().lower()
            
            # Extract and validate
            raw_exps = data.get("extraction", {}).get("experiments", [])
            # Note: We assume schema_class expects kwargs matching the JSON dict
            pred_dict[key] = [schema_class(**exp) for exp in raw_exps]
            
        except Exception as e:
            logger.warning(f"Skipping corrupt prediction {f.name}: {e}")
            # IMPORTANT: We track it as empty to penalize Recall in metrics
            pred_dict[key] = [] 

    return pred_dict

def get_split_files(split_path: Path, split_name: str) -> set:
    """Loads a list of filenames for a specific split (train/test)."""
    if not split_path.exists():
        return set()
    with open(split_path, "r") as f:
        data = json.load(f)
    return set(data.get(split_name, []))