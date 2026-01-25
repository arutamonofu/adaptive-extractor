# scripts/download_data.py

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

from aee.core.logging import setup_logging

logger = setup_logging()

# Maps task names to specific Hugging Face dataset identifiers
HF_MAPPING = {
    "nanozymes": "ai-chem/Nanozymes",
}

def download_task_data(task: str, output_dir: Path):
    """
    Downloads a dataset from Hugging Face, merges splits, and saves it as a CSV.

    Args:
        task: The name of the task (e.g., "nanozymes").
        output_dir: The directory where the CSV file will be saved.
    """
    # Use specific mapping or fall back to a convention-based name
    hf_id = HF_MAPPING.get(task, f"ai-chem/{task.capitalize()}")
    output_path = output_dir / f"{task}.csv"
    
    logger.info(f"Downloading dataset '{hf_id}' from Hugging Face...")

    try:
        # Load the dataset (all splits)
        dataset = load_dataset(hf_id)
        
        # Merge train, test, etc., into a single DataFrame
        df = pd.concat(
            [split.to_pandas() for split in dataset.values()], 
            ignore_index=True
        )
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Successfully saved {len(df)} rows to {output_path}")

    except DatasetNotFoundError:
        logger.error(
            f"❌ Dataset '{hf_id}' not found on Hugging Face. "
            "Please check the name in HF_MAPPING or the task argument."
        )
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {type(e).__name__}: {e}")

def main():
    """Entrypoint for the data download script."""
    parser = argparse.ArgumentParser(description="Download ground truth datasets from Hugging Face.")
    parser.add_argument(
        "--task", 
        type=str, 
        default="nanozymes", 
        help="Name of the task dataset to download."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/ground_truth",
        help="Directory to save the output CSV file."
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    download_task_data(args.task, out_dir)

if __name__ == "__main__":
    main()