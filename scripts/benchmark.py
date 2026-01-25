# scripts/benchmark.py

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import get_args

from aee.core.logging import setup_logging
from aee.tasks import TASK_REGISTRY
from aee.eval.matcher import ExperimentMatcher
from aee.utils.io import load_ground_truth, load_predictions

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction performance (F1/Precision/Recall).")
    parser.add_argument("--task", type=str, default="nanozymes", help="Task name from registry")
    parser.add_argument("--gt", type=str, default="data/ground_truth/nanozymes.csv", help="Path to Ground Truth CSV")
    parser.add_argument("--results", type=str, required=True, help="Directory containing JSON predictions (from predict.py)")
    parser.add_argument("--split_file", type=str, default="data/splits.json", help="Path to splits file to filter Test set")
    parser.add_argument("--output", type=str, default="benchmark_report.csv", help="Where to save column-wise metrics")
    args = parser.parse_args()

    # 1. Setup Task
    task_conf = TASK_REGISTRY.get(args.task)
    if not task_conf:
        logger.error(f"Task '{args.task}' not found in registry.")
        return

    # 2. Load Data
    logger.info(f"Loading Ground Truth from {args.gt}...")
    try:
        gt_dict = load_ground_truth(Path(args.gt), task_conf["row_converter"])
    except Exception as e:
        logger.error(f"Failed to load GT: {e}")
        return

    logger.info(f"Loading Predictions from {args.results}...")
    # Extract inner type from List[NanozymeExperiment] safely
    # This works for Pydantic fields defined as List[Model]
    try:
        experiments_field = task_conf["output_model"].model_fields['experiments']
        # Handle Optional[List[...]] or just List[...]
        annotation = experiments_field.annotation
        item_schema = get_args(annotation)[0] # Gets NanozymeExperiment class
    except Exception:
        # Fallback if introspection fails (unlikely)
        logger.warning("Could not introspect Pydantic model type. Ensure result format matches.")
        item_schema = None 

    if item_schema is None:
         # Hard fallback if dynamic introspection fails
         from aee.tasks.nanozymes import NanozymeExperiment
         item_schema = NanozymeExperiment

    pred_dict = load_predictions(Path(args.results), item_schema)

    # 3. Filter by Split (Crucial for Benchmarking)
    split_path = Path(args.split_file)
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
            # Support both 'test' and 'test_set' keys
            test_files = set(splits.get("test") or splits.get("test_set", []))
        
        if not test_files:
            logger.warning("Test split is empty in splits.json!")
        
        # Intersect keys: We only evaluate files that are in GT, have Predictions, AND are in Test Split
        common_keys = set(gt_dict.keys()) & set(pred_dict.keys()) & test_files

        logger.info(f"gt_dict: {gt_dict.keys()}")
        logger.info(f"pred_dict: {pred_dict.keys()}")
        logger.info(f"test_files: {test_files}")
        
        # Check for missing predictions
        missing_preds = test_files - set(pred_dict.keys())
        if missing_preds:
            logger.warning(f"Missing predictions for {len(missing_preds)} files in Test set.")

        logger.info(f"🛡️  Evaluating on TEST split ({len(common_keys)} docs).")
    else:
        # Fallback: Evaluate on everything found (WARN: includes train data!)
        common_keys = set(gt_dict.keys()) & set(pred_dict.keys())
        logger.warning(f"⚠️  Split file not found. Evaluating on ALL {len(common_keys)} common docs (DATA LEAKAGE RISK!).")

    if not common_keys:
        logger.error("No common documents found to evaluate.")
        return

    # 4. Prepare Lists for Matcher
    batch_preds = [pred_dict[k] for k in common_keys]
    batch_gts = [gt_dict[k] for k in common_keys]

    # 5. Compute Metrics
    logger.info("Computing metrics via Hungarian Matching...")
    matcher = ExperimentMatcher(
        fields_to_compare=task_conf["compare_fields"],
        float_tolerance=0.05
    )
    
    # Get strict and legacy metrics
    results = matcher.get_full_report(batch_preds, batch_gts)
    aee_stats = results['aee_strict']
    legacy_stats = results['nanominer_legacy']

    # 6. Report Results
    print("\n" + "="*40)
    print(f"📊 BENCHMARK REPORT: {args.task}")
    print("="*40)
    
    print("\n🔹 AEE Strict Metrics (Optimization Target)")
    print(f"   Precision: {aee_stats['precision']:.4f}")
    print(f"   Recall:    {aee_stats['recall']:.4f}")
    print(f"   F1 Score:  {aee_stats['f1']:.4f}")

    print("\n🔸 Legacy Metrics (Text Similarity)")
    print(f"   Avg Levenshtein Dist: {legacy_stats['avg_levenshtein']:.4f} (Lower is better)")
    
    print("\n🔹 Column-wise Performance (Legacy Match)")
    col_data = []
    for field, metrics in legacy_stats['column_metrics'].items():
        print(f"   {field:<15} | P: {metrics['precision']:.2f} | R: {metrics['recall']:.2f}")
        col_data.append({
            "field": field, 
            "precision": metrics['precision'], 
            "recall": metrics['recall']
        })

    # 7. Save Detailed CSV
    if col_data:
        df = pd.DataFrame(col_data)
        df.to_csv(args.output, index=False)
        logger.info(f"\n✅ Column-wise metrics saved to: {Path(args.output).absolute()}")

if __name__ == "__main__":
    main()