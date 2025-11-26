# scripts/benchmark.py

import argparse
from pathlib import Path

from aee.core.logging import setup_logging
from aee.tasks import TASK_REGISTRY
from aee.eval import ExperimentMatcher
from aee.utils.io import load_ground_truth, load_predictions

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Calculate F1/Precision/Recall.")
    parser.add_argument("--task", type=str, default="nanozymes")
    parser.add_argument("--gt", type=str, default="data/ground_truth/nanozymes.csv")
    parser.add_argument("--results", type=str, default="data/results/nanozymes")
    parser.add_argument("--split_file", type=str, help="Evaluate ONLY on 'test' split")
    args = parser.parse_args()

    task = TASK_REGISTRY.get(args.task)
    if not task:
        return

    # 1. Load Data
    logger.info("Loading Ground Truth...")
    gt_dict = load_ground_truth(Path(args.gt), task["row_converter"])
    
    logger.info("Loading Predictions...")
    # Get the inner item type from List[NanozymeExperiment]
    item_schema = task["output_model"].model_fields['experiments'].annotation.__args__[0]
    pred_dict = load_predictions(Path(args.results), item_schema)

    # 2. Filter by Split (Safety)
    if args.split_file:
        import json
        with open(args.split_file) as f:
            test_files = set(json.load(f).get("test", []))
        
        # Intersect with available data
        common_keys = set(gt_dict.keys()) & set(pred_dict.keys()) & test_files
        logger.info(f"🛡️  Evaluating on TEST split only ({len(common_keys)} docs).")
    else:
        common_keys = set(gt_dict.keys()) & set(pred_dict.keys())
        logger.warning(f"⚠️  Evaluating on ALL {len(common_keys)} common docs (Potential Leakage!).")

    if not common_keys:
        logger.error("No overlap between GT, Preds, and Split.")
        return

    # 3. Align Lists
    batch_preds = [pred_dict[k] for k in common_keys]
    batch_gts = [gt_dict[k] for k in common_keys]

    # 4. Compute Metrics
    matcher = ExperimentMatcher(task["compare_fields"])
    metrics = matcher.evaluate_dataset(batch_preds, batch_gts)

    print(f"\nResults for {args.task}:")
    print(f"\tPrecision: {metrics['precision']:.4f}")
    print(f"\tRecall:    {metrics['recall']:.4f}")
    print(f"\tF1-Score:  {metrics['f1']:.4f}")
    print(f"\t(TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']})\n")

if __name__ == "__main__":
    main()