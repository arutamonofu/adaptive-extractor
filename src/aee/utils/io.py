# src/aee/utils/io.py

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Callable, Type

logger = logging.getLogger(__name__)

def load_ground_truth(csv_path: Path, row_converter: Callable) -> Dict[str, List[Any]]:
    """
    Loads Ground Truth from CSV, groups by filename, and converts rows to objects.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"GT file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    
    target_cols = ["pdf", "filename", "source", "doi"]
    id_col = next((c for c in target_cols if c in df.columns), None)
    
    if not id_col:
        raise ValueError(f"Missing ID column. Candidates: {target_cols}. Found: {list(df.columns)}")

    gt_data = {}
    
    for filename, group in df.groupby(id_col):
        # Normalize key: remove extension, lowercase (e.g. "File.pdf" -> "file")
        key = Path(str(filename)).name.lower()
        
        # Convert valid rows only
        experiments = [res for _, row in group.iterrows() if (res := row_converter(row))]
        
        if experiments:
            gt_data[key] = experiments
            
    return gt_data

def load_predictions(results_dir: Path, schema_class: Type) -> Dict[str, List[Any]]:
    """
    Loads and validates predictions from JSON files in a directory.
    """
    preds = {}
    
    for file_path in results_dir.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            
            # Determine document key from metadata or filename
            raw_name = data.get("source_metadata", {}).get("filename") or file_path.name
            # Remove potential suffixes like '_result' if present in filename
            key = Path(raw_name.replace("_result", "")).stem.lower()
            
            raw_exps = data.get("extraction", {}).get("experiments", [])
            
            # Validate against Pydantic schema
            preds[key] = [schema_class(**exp) for exp in raw_exps]
            
        except Exception as e:
            logger.warning(f"Skipping corrupt prediction {file_path.name}: {e}")
            # Track as empty list to correctly penalize Recall
            preds[key] = []

    return preds

def get_split_files(split_path: Path, split_name: str) -> set[str]:
    """Loads a set of filenames for a specific split (train/test)."""
    if not split_path.exists():
        return set()
    
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return set(data.get(split_name, []))