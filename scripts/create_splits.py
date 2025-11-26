# scripts/create_splits.py

import argparse
import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Set

from aee.core.logging import setup_logging

logger = setup_logging()

def get_unique_filenames(csv_path: Path) -> List[str]:
    """
    Extracts unique filename stems from the Ground Truth CSV.
    
    Args:
        csv_path: Path to the task CSV file.
        
    Returns:
        List of unique filenames (without extensions).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Identify the filename column
    possible_cols = ["pdf", "filename", "source", "doi"]
    filename_col = next((c for c in possible_cols if c in df.columns), None)
    
    if not filename_col:
        raise ValueError(f"Could not find filename column. Available: {list(df.columns)}")

    # Extract, clean, and deduplicate
    filenames = set()
    for raw_name in df[filename_col].unique():
        # Normalize: "Paper.pdf" -> "paper"
        clean_name = str(raw_name).replace(".pdf", "").strip().lower()
        if clean_name:
            filenames.add(clean_name)
            
    return list(filenames)

def main():
    parser = argparse.ArgumentParser(description="Generate train/test splits JSON.")
    parser.add_argument("--gt", type=str, required=True, help="Path to Ground Truth CSV")
    parser.add_argument("--output", type=str, default="data/splits.json", help="Output JSON path")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train set ratio (e.g. 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    # 1. Load Data
    try:
        all_files = get_unique_filenames(Path(args.gt))
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    total_count = len(all_files)
    logger.info(f"Found {total_count} unique documents in GT.")

    # 2. Shuffle and Split
    # Use local random instance for safety
    rng = random.Random(args.seed)
    rng.shuffle(all_files)

    train_count = int(total_count * args.split_ratio)
    
    train_set = all_files[:train_count]
    test_set = all_files[train_count:]

    split_data = {
        "metadata": {
            "source_csv": args.gt,
            "seed": args.seed,
            "ratio": args.split_ratio,
            "total": total_count
        },
        "train": sorted(train_set),
        "test": sorted(test_set)
    }

    # 3. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    logger.info(f"Successfully saved splits to {output_path}")
    logger.info(f"Train: {len(train_set)} | Test: {len(test_set)}")

if __name__ == "__main__":
    main()