# scripts/optimize.py

import argparse
import random
import dspy
from pathlib import Path
from dspy.teleprompt import MIPROv2

from aee.core.logging import setup_logging
from aee.core.types import ProcessedDocument
from aee.llm import setup_student, setup_teacher
from aee.agents.extractor import UniversalExtractor
from aee.eval import TaskMetric
from aee.tasks import TASK_REGISTRY
from aee.utils.io import load_ground_truth

logger = setup_logging()

def create_training_set(processed_dir: Path, gt_data: dict, task_conf: dict, limit: int):
    """Creates DSPy Examples from processed files and GT."""
    dataset = []
    
    # Iterate over GT keys to ensure we only take labeled data
    # (Simplified logic: taking intersection of files present in both)
    for filename, experiments in gt_data.items():
        # Try to find corresponding JSON
        # Note: This simple lookup assumes filename stems match exactly.
        # In prod, use the sha256 or ID mapping from aee.utils.io
        json_path = processed_dir / f"{filename}.json" 
        
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, "r") as f:
                doc = ProcessedDocument.model_validate_json(f.read())
            
            # Create DSPy Example
            out_model = task_conf["output_model"]
            example = dspy.Example(
                document_text=doc.text_content,
                extracted_data=out_model(experiments=experiments)
            ).with_inputs("document_text")
            
            dataset.append(example)
        except Exception:
            continue
            
    # Shuffle and limit
    random.shuffle(dataset)
    return dataset[:limit]

def main():
    parser = argparse.ArgumentParser(description="Optimize Agent Prompt (MIPROv2).")
    parser.add_argument("--task", type=str, default="nanozymes")
    parser.add_argument("--train_size", type=int, default=20)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--output", type=str, default="data/artifacts/optimized_agent.json")
    # New Argument for Split Safety
    parser.add_argument("--split_file", type=str, default=None, help="Path to splits.json")
    args = parser.parse_args()

    if args.task not in TASK_REGISTRY:
        logger.error(f"Unknown task: {args.task}")
        return
    
    task_conf = TASK_REGISTRY[args.task]
    
    # Setup LLMs
    student = setup_student()
    teacher = setup_teacher()

    # Load Data
    gt_path = Path("data/ground_truth") / f"{args.task}.csv"
    proc_dir = Path("data/processed")
    
    gt_data = load_ground_truth(gt_path, task_conf["row_converter"])
    
    # --- SPLIT SAFETY CHECK ---
    if args.split_file:
        import json
        with open(args.split_file) as f:
            splits = json.load(f)
        train_ids = set(splits.get("train", []))
        # Filter GT data to keep only train files
        gt_data = {k: v for k, v in gt_data.items() if k in train_ids}
        logger.info(f"🛡️  Applied Split: Using {len(gt_data)} training documents.")
    else:
        logger.warning("⚠️  No split file provided! You might be overfitting on test data.")

    trainset = create_training_set(proc_dir, gt_data, task_conf, args.train_size)
    
    if not trainset:
        logger.error("No training data found.")
        return

    # Optimizer
    logger.info(f"Starting optimization with {len(trainset)} examples...")
    teleprompter = MIPROv2(
        prompt_model=teacher,
        task_model=student,
        metric=TaskMetric(task_conf),
        num_candidates=5,
        init_temperature=0.5,
        verbose=True,
        auto=None
    )
    
    agent = UniversalExtractor(task_conf["signature"])
    
    optimized_agent = teleprompter.compile(
        agent,
        trainset=trainset,
        num_trials=args.trials,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        minibatch=False
    )
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    optimized_agent.save(args.output)
    logger.info(f"Saved optimized agent to {args.output}")

if __name__ == "__main__":
    main()