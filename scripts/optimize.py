# scripts/optimize.py

import os
os.environ["DSP_CACHEBOOL"] = "1"

import argparse
import sys
from pathlib import Path
from contextlib import redirect_stdout

import dspy
from dspy.teleprompt import MIPROv2

from aee.agents import UniversalExtractor
from aee.core.config import settings
from aee.core.logging import setup_logging
from aee.eval import TaskMetric
from aee.llm import setup_student, setup_teacher
from aee.tasks import TASK_REGISTRY
from aee.utils import create_dataset_from_ids, load_ground_truth, get_split_files

logger = setup_logging()

def prepare_datasets(task_conf: dict) -> tuple[list, list]:    
    splits_path = settings.paths.splits_file
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found at {splits_path}")
    
    import json
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    train_ids = splits.get("train_auto", [])
    val_ids = splits.get("val", [])

    gt_path = settings.paths.ground_truth_dir / f"{settings.task.name}.csv"
    gt_data = load_ground_truth(gt_path, task_conf["row_converter"])
    
    proc_dir = settings.paths.parsed_dir

    opt_cfg = settings.optimization
    
    trainset = create_dataset_from_ids(
        proc_dir, gt_data, task_conf, 
        allowed_ids=train_ids, 
        limit=opt_cfg.train_split
    )
    valset = create_dataset_from_ids(
        proc_dir, gt_data, task_conf, 
        allowed_ids=val_ids, 
        limit=opt_cfg.total_load
    )
    
    logger.info(f"Datasets loaded: Train (Demos): {len(trainset)} | Val (Metric): {len(valset)}")
    return trainset, valset


def main() -> None:
    parser = argparse.ArgumentParser(description="AEE Optimization Script (MIPROv2)")
    parser.add_argument("--config", type=str, help="Path to custom configuration YAML.")
    parser.add_argument("--output", type=str, help="Override output filename for the agent.")
    args = parser.parse_args()

    if args.config:
        global settings
        settings = settings.load(args.config)

    setup_student() 
    teacher = setup_teacher()

    task_name = settings.task.name
    task_conf = TASK_REGISTRY.get(task_name)
    if not task_conf:
        logger.error(f"Task '{task_name}' not found in registry.")
        sys.exit(1)

    trainset, valset = prepare_datasets(task_conf)

    opt_cfg = settings.optimization
    
    teleprompter = MIPROv2(
        prompt_model=teacher,
        task_model=dspy.settings.lm,
        metric=TaskMetric(task_conf),
        num_candidates=opt_cfg.num_candidates,
        max_bootstrapped_demos=opt_cfg.max_bootstrapped_demos,
        max_labeled_demos=opt_cfg.max_labeled_demos,
        metric_threshold=opt_cfg.metric_threshold,
        verbose=opt_cfg.verbose,
        init_temperature=opt_cfg.init_temperature,
        auto=None
    )

    agent = UniversalExtractor(task_conf["signature"])
    
    logger.info(f"Starting optimization for task: {task_name}")
    try:
        optimized_agent = teleprompter.compile(
            agent,
            trainset=trainset,
            valset=valset,
            num_trials=opt_cfg.num_trials,
            max_bootstrapped_demos=opt_cfg.max_bootstrapped_demos,
            max_labeled_demos=opt_cfg.max_labeled_demos,
            minibatch=opt_cfg.minibatch,
            minibatch_size=opt_cfg.minibatch_size,
            view_data_batch_size=opt_cfg.view_data_batch_size
        )
    except Exception as e:
        logger.error(f"Optimization crashed: {e}", exc_info=True)
        sys.exit(1)

    output_path = Path(args.output) if args.output else settings.paths.agents_dir / f"optimized_{task_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    optimized_agent.save(str(output_path))
    logger.info(f"Optimized agent saved to: {output_path.absolute()}")

    history_path = settings.paths.logs_dir / f"dspy_history_{task_name}.log"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                dspy.settings.lm.inspect_history(n=100)
        logger.info(f"LLM History saved to: {history_path.name}")
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

if __name__ == "__main__":
    main()