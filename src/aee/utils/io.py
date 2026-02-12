# src/aee/utils/io.py
"""I/O utilities for AutoEvoExtractor."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable, Type, Set

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def load_ground_truth(csv_path: Path, row_converter: Callable) -> Dict[str, List[Any]]:
    """Load Ground Truth from CSV, groups by filename, and converts rows to objects.
    
    Args:
        csv_path: Path to the CSV file.
        row_converter: Function to convert rows to objects.
        
    Returns:
        Dict mapping filenames to lists of objects.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no valid ID column is found in the CSV.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file cannot be parsed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"GT file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"GT file is empty: {csv_path}")
        return {}
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse GT CSV file: {csv_path}") from e
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Identify the ID column
    target_cols = ["pdf", "filename", "source", "doi"]
    id_col = next((c for c in target_cols if c in df.columns), None)
    
    if not id_col:
        raise ValueError(f"Missing ID column. Candidates: {target_cols}. Found: {list(df.columns)}")

    gt_data = {}
    
    # Group by the ID column and process each group
    for filename, group in df.groupby(id_col):
        # Normalize key: remove extension, lowercase (e.g. "File.pdf" -> "file")
        key = Path(str(filename)).stem.lower()
        
        # Convert valid rows only
        experiments = []
        for _, row in group.iterrows():
            try:
                result = row_converter(row)
                if result is not None:
                    experiments.append(result)
            except Exception as e:
                logger.warning(f"Failed to convert row for {filename}: {e}")
                continue
        
        if experiments:
            gt_data[key] = experiments
            
    return gt_data


def load_predictions(results_dir: Path, schema_class: Type[BaseModel]) -> Dict[str, List[Any]]:
    """Load and validate predictions from JSON files in a directory.
    
    Args:
        results_dir: Directory containing prediction JSON files.
        schema_class: Schema class for validation (must be a Pydantic BaseModel).
        
    Returns:
        Dict mapping document keys to lists of predictions.
    """
    if not results_dir.exists():
        logger.warning(f"Predictions directory does not exist: {results_dir}")
        return {}
    
    if not issubclass(schema_class, BaseModel):
        raise TypeError("schema_class must be a Pydantic BaseModel subclass")
    
    preds = {}
    file_count = 0
    error_count = 0
    
    # Process all JSON files in the directory
    for file_path in results_dir.glob("*.json"):
        file_count += 1
        try:
            # Read and parse JSON file
            data = json.loads(file_path.read_text(encoding="utf-8"))
            
            # Determine document key from metadata or filename
            raw_name = data.get("source_metadata", {}).get("filename") or file_path.name
            # Remove potential suffixes like '_result' if present in filename
            key = Path(raw_name.replace("_result", "")).stem.lower()
            
            # Extract experiments from the data
            raw_exps = data.get("extraction", {}).get("experiments", [])
            
            # Validate against Pydantic schema
            validated_exps = []
            for exp in raw_exps:
                try:
                    validated_exps.append(schema_class(**exp))
                except Exception as e:
                    logger.debug(f"Invalid experiment in {file_path.name}: {e}")
                    # Skip invalid experiments but continue processing others
                    continue
            
            preds[key] = validated_exps
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in prediction file {file_path.name}: {e}")
            preds[file_path.stem.lower()] = []
            error_count += 1
        except Exception as e:
            logger.warning(f"Skipping corrupt prediction {file_path.name}: {e}")
            preds[file_path.stem.lower()] = []
            error_count += 1
    
    if error_count > 0:
        logger.info(f"Processed {file_count} prediction files with {error_count} errors")
    
    return preds


def get_split_files(split_path: Path, split_name: str) -> Set[str]:
    """Load a set of filenames for a specific split (train/test).
    
    Args:
        split_path: Path to the split file.
        split_name: Name of the split.
        
    Returns:
        Set of filenames.
    """
    if not split_path.exists():
        logger.warning(f"Split file not found: {split_path}")
        return set()
    
    try:
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in split file {split_path}: {e}")
        return set()
    except Exception as e:
        logger.error(f"Failed to read split file {split_path}: {e}")
        return set()
        
    # Return the requested split or empty set if not found
    return set(data.get(split_name, []))