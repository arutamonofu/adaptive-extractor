# scripts/optimize.py

import os
os.environ["DSP_CACHEBOOL"] = "1"

import argparse
import json
import logging
from contextlib import redirect_stdout
import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import dspy
from dspy.teleprompt import MIPROv2

from aee.agents import UniversalExtractor
from aee.core.config import settings
from aee.core.logging import setup_logging
from aee.eval import TaskMetric
from aee.llm import setup_student, setup_teacher
from aee.tasks import TASK_REGISTRY
from aee.utils import create_dataset_from_ids, load_ground_truth

logger = setup_logging()

class OptimizationMode(TypedDict):
    description: str
    total_load: int
    train_split: int
    candidates: int
    trials: int
    max_bootstrapped: int
    max_labeled: int
    minibatch: bool
    minibatch_size: int
    view_data_batch: int

OPTIMIZATION_MODES: Dict[str, OptimizationMode] = {
    "test": {
        "description": "Quick pipeline check.",
        "total_load": 3,
        "train_split": 3,
        "candidates": 3,
        "trials": 5,
        "max_labeled": 2,
        "max_bootstrapped": 1,
        "minibatch": True,
        "minibatch_size": 1,
        "view_data_batch": 1,
    },
    "production": {
        "description": "Full-fledged optimization for high quality.",
        "total_load": 20,       
        "train_split": 20,      
        "candidates": 10,       
        "trials": 70,
        "max_labeled": 2,        
        "max_bootstrapped": 2,   
        "minibatch": False,
        "minibatch_size": 1,
        "view_data_batch": 3,
    },
}

def prepare_datasets_strict(
    task_conf: Dict[str, Any], mode_conf: OptimizationMode, task_name: str
) -> tuple[List[dspy.Example], List[dspy.Example]]:
    splits_path = Path("data/splits.json")
    if not splits_path.exists():
        raise FileNotFoundError("splits.json not found.")
    
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    train_ids = splits.get("train_auto", [])
    val_ids = splits.get("val", [])

    gt_path = Path("data/ground_truth") / f"{task_name}.csv"
    gt_data = load_ground_truth(gt_path, task_conf["row_converter"])
    proc_dir = Path("data/parsed")

    trainset = create_dataset_from_ids(
        proc_dir, gt_data, task_conf, 
        allowed_ids=train_ids, 
        limit=mode_conf["train_split"]
    )
    valset = create_dataset_from_ids(
        proc_dir, gt_data, task_conf, 
        allowed_ids=val_ids, 
        limit=mode_conf["total_load"]
    )
    
    logger.info(f"Datasets: Train (Demos Pool): {len(trainset)} | Val (Metric): {len(valset)}")
    return trainset, valset


def main() -> None:
    parser = argparse.ArgumentParser(description="AEE Optimization Script")
    parser.add_argument("--task", type=str, default="nanozymes", choices=TASK_REGISTRY.keys())
    parser.add_argument("--mode", type=str, default="test", choices=OPTIMIZATION_MODES.keys())
    parser.add_argument("--output", type=str, default="data/agents/optimized_agent.json")
    args = parser.parse_args()

    task_conf = TASK_REGISTRY[args.task]
    mode_conf = OPTIMIZATION_MODES[args.mode]

    logger.info(f"Starting Optimization for task '{args.task}' in '{args.mode.upper()}' mode.")

    setup_student()
    teacher = setup_teacher()
    trainset, valset = prepare_datasets_strict(task_conf, mode_conf, args.task)

    teleprompter = MIPROv2(
        prompt_model=teacher,
        task_model=dspy.settings.lm,
        metric=TaskMetric(task_conf),
        auto=None, 
        num_candidates=mode_conf["candidates"],
        max_bootstrapped_demos=mode_conf["max_bootstrapped"],
        max_labeled_demos=mode_conf["max_labeled"],
        metric_threshold=1.0,
        verbose=True,
        init_temperature=0.5
    )

    agent = UniversalExtractor(task_conf["signature"])

    logger.info("Starting MIPROv2 compilation loop...")
    output_path = Path(args.output)
    history_path = output_path.parent / "dspy_llm_history.log"
    logger.info(f"Saving full DSPy LLM history to: {history_path.absolute()}...")

    logger.info("Starting compilation...")
    try:
        optimized_agent = teleprompter.compile(
            agent,
            trainset=trainset,
            valset=valset,
            num_trials=mode_conf["trials"],
            max_bootstrapped_demos=mode_conf["max_bootstrapped"],
            max_labeled_demos=mode_conf["max_labeled"],
            minibatch=mode_conf["minibatch"],
            minibatch_size=mode_conf["minibatch_size"],
            view_data_batch_size=mode_conf["view_data_batch"]
        )
    except Exception as e:
        logger.error(f"Optimization crashed: {e}", exc_info=True)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_agent.save(str(output_path))
    logger.info(f"✅ Optimized agent saved to: {output_path.absolute()}")
    
    history_path = output_path.parent / "dspy_llm_history.log"
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                dspy.settings.lm.inspect_history(n=9999)
        logger.info(f"📜 History saved to: {history_path.name}")
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

if __name__ == "__main__":
    main()