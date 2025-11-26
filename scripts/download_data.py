# scripts/download_data.py

import argparse
import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset

from aee.core.logging import setup_logging

logger = setup_logging()

HF_MAPPING = {
    "nanozymes": "ai-chem/Nanozymes",
}

def download_task_data(task: str, output_dir: Path):
    hf_id = HF_MAPPING.get(task, f"ai-chem/{task.capitalize()}")
    logger.info(f"Downloading {hf_id}...")

    try:
        ds = load_dataset(hf_id)
        # Merge splits
        df = pd.concat([d.to_pandas() for d in ds.values()], ignore_index=True)
        
        output_path = output_dir / f"{task}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download ground truth datasets.")
    parser.add_argument("--task", type=str, default="nanozymes")
    parser.add_argument("--output", type=str, default="data/ground_truth")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    download_task_data(args.task, out_dir)

if __name__ == "__main__":
    main()