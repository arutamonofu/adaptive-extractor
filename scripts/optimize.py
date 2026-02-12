# scripts/optimize.py

import argparse
import datetime
import json
import os
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dspy
from dspy.teleprompt import MIPROv2

# Set DSPy cache environment variable early
os.environ["DSP_CACHEBOOL"] = "1"

# MLFlow integration
MLFLOW_AVAILABLE: bool = False
MLFLOW_DSPY_AVAILABLE: bool = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
    
    # Try to import mlflow.dspy, but handle if it's not available
    try:
        import mlflow.dspy
        MLFLOW_DSPY_AVAILABLE = True
    except ImportError:
        MLFLOW_DSPY_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False
    MLFLOW_DSPY_AVAILABLE = False
    print("MLFlow not available, skipping MLFlow tracking")

# Import project modules
from aee import UniversalExtractor, settings, setup_logging
from aee.evaluation import TaskMetric
from aee.llm import setup_student, setup_teacher
from aee.tasks import TASK_REGISTRY
from aee.utils import create_dataset_from_ids, load_ground_truth

logger = setup_logging()


def prepare_datasets(task_conf: Dict[str, Any]) -> Tuple[list, list]:
    """Prepare training and validation datasets for optimization.
    
    Args:
        task_conf: Task configuration dictionary
        
    Returns:
        Tuple of (trainset, valset)
        
    Raises:
        FileNotFoundError: If splits file is not found
    """
    splits_path = settings.paths.splits_file
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found at {splits_path}")
    
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


def setup_mlflow_tracking(task_name: str, opt_cfg: Any) -> Optional[str]:
    """Set up MLflow tracking for the optimization process.
    
    Args:
        task_name: Name of the task being optimized
        opt_cfg: Optimization configuration
        
    Returns:
        Timestamp string if MLflow is available, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"AEE_Optimization_{task_name}"
    run_name = f"optimization_{task_name}_{timestamp}"
    
    try:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        
        # Enable DSPy autologging to automatically track traces, compiles, and evaluations
        if MLFLOW_DSPY_AVAILABLE:
            # Use getattr to avoid import errors in type checkers
            dspy_module = getattr(mlflow, 'dspy', None)
            if dspy_module:
                dspy_autolog = getattr(dspy_module, 'autolog', None)
                if dspy_autolog:
                    dspy_autolog(
                        log_traces=True,
                        log_traces_from_compile=True,
                        log_traces_from_eval=True,
                        log_compiles=True,
                        log_evals=True
                    )
        
        # Log parameters
        mlflow.log_params({
            "task_name": task_name,
            "num_trials": opt_cfg.num_trials,
            "num_candidates": opt_cfg.num_candidates,
            "max_bootstrapped_demos": opt_cfg.max_bootstrapped_demos,
            "max_labeled_demos": opt_cfg.max_labeled_demos,
            "metric_threshold": opt_cfg.metric_threshold,
            "init_temperature": opt_cfg.init_temperature,
            "train_split": opt_cfg.train_split,
            "total_load": opt_cfg.total_load,
            "minibatch": opt_cfg.minibatch,
            "minibatch_size": opt_cfg.minibatch_size,
            "view_data_batch_size": opt_cfg.view_data_batch_size,
            "timestamp": timestamp
        })
        
        return timestamp
    except Exception as e:
        logger.warning(f"Failed to set up MLflow tracking: {e}")
        return None


def log_optimization_result(success: bool, error_message: Optional[str] = None) -> None:
    """Log the optimization result to MLflow.
    
    Args:
        success: Whether the optimization was successful
        error_message: Error message if optimization failed
    """
    if not MLFLOW_AVAILABLE:
        return
        
    try:
        mlflow.log_metric("optimization_success", float(success))
        if not success and error_message:
            mlflow.log_param("error_message", error_message)
    except Exception as e:
        logger.warning(f"Failed to log optimization result to MLflow: {e}")


def log_optimized_agent(optimized_agent: Any, task_name: str, output_path: Path) -> None:
    """Log the optimized agent as an artifact and MLflow model.
    
    Args:
        optimized_agent: The optimized DSPy agent
        task_name: Name of the task
        output_path: Path where the agent was saved
    """
    if not MLFLOW_AVAILABLE:
        return
        
    try:
        # Log the optimized agent as a JSON artifact
        mlflow.log_artifact(str(output_path))
        mlflow.log_param("optimized_agent_path", str(output_path))
        
        # Log the optimized agent as an MLflow model for easier loading
        if MLFLOW_DSPY_AVAILABLE:
            try:
                # Use getattr to avoid import errors in type checkers
                dspy_module = getattr(mlflow, 'dspy', None)
                if dspy_module:
                    dspy_log_model = getattr(dspy_module, 'log_model', None)
                    if dspy_log_model:
                        model_name = f"optimized_{task_name}"
                        dspy_log_model(
                            optimized_agent,
                            artifact_path="model",
                            name=model_name,
                            metadata={"task_name": task_name, "model_type": "dspy_agent"}
                        )
                        logger.info(f"Optimized agent logged as MLflow model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to log optimized agent as MLflow model: {e}")
    except Exception as e:
        logger.warning(f"Failed to log optimized agent as MLFlow artifact: {e}")


def log_llm_history(task_name: str) -> None:
    """Log LLM history as an artifact.
    
    Args:
        task_name: Name of the task
    """
    history_path = settings.paths.logs_dir / f"dspy_history_{task_name}.log"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                dspy.settings.lm.inspect_history(n=100)
        logger.info(f"LLM History saved to: {history_path.name}")
        
        # Log history as artifact if MLFlow is available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(str(history_path))
            except Exception as e:
                logger.warning(f"Failed to log history as MLFlow artifact: {e}")
    except Exception as e:
        logger.error(f"Failed to save history: {e}")


def end_mlflow_run() -> None:
    """Safely end the MLflow run if it's active."""
    if not MLFLOW_AVAILABLE:
        return
        
    try:
        # Check if there's an active run before trying to end it
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Failed to end MLflow run: {e}")


def create_output_path(args_output: Optional[str], task_name: str) -> Path:
    """Create the output path for the optimized agent.
    
    Args:
        args_output: Output path from command line arguments
        task_name: Name of the task
        
    Returns:
        Path to the output file
    """
    if args_output:
        output_path = Path(args_output)
    else:
        output_path = settings.paths.agents_dir / f"optimized_{task_name}.json"
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def validate_task_configuration(task_name: str, task_conf: Optional[Dict[str, Any]]) -> bool:
    """Validate that the task configuration is complete.
    
    Args:
        task_name: Name of the task
        task_conf: Task configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    if not task_conf:
        logger.error(f"Task '{task_name}' not found in registry.")
        return False
        
    required_keys = ["signature", "row_converter", "output_model", "compare_fields"]
    missing_keys = [key for key in required_keys if key not in task_conf]
    
    if missing_keys:
        logger.error(f"Task '{task_name}' configuration is missing required keys: {missing_keys}")
        return False
        
    return True


def setup_optimization_environment(opt_cfg: Any) -> None:
    """Set up the optimization environment including random seed.
    
    Args:
        opt_cfg: Optimization configuration
    """
    # Set random seed for reproducibility
    random.seed(opt_cfg.random_seed)
    # Also set numpy and other library seeds if needed
    try:
        import numpy as np
        np.random.seed(opt_cfg.random_seed)
    except ImportError:
        pass


def create_teleprompter(teacher: Any, opt_cfg: Any, task_conf: Dict[str, Any]) -> MIPROv2:
    """Create and configure the MIPROv2 teleprompter.
    
    Args:
        teacher: Teacher language model
        opt_cfg: Optimization configuration
        task_conf: Task configuration
        
    Returns:
        Configured MIPROv2 teleprompter
    """
    return MIPROv2(
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


def main() -> None:
    """Main function for running the AEE optimization process."""
    parser = argparse.ArgumentParser(description="AEE Optimization Script (MIPROv2)")
    parser.add_argument("--config", type=str, help="Path to custom configuration YAML.")
    parser.add_argument("--output", type=str, help="Override output filename for the agent.")
    args = parser.parse_args()

    # Load custom configuration if provided
    if args.config:
        try:
            # We need to reload settings with the custom config
            import importlib
            import aee.config.settings
            importlib.reload(aee.config.settings)
            from aee.config.settings import Settings
            global settings
            settings = Settings.load(args.config)
        except Exception as e:
            logger.error(f"Failed to load custom configuration: {e}")
            sys.exit(1)

    # Set up language models
    try:
        setup_student()
        teacher = setup_teacher()
    except Exception as e:
        logger.error(f"Failed to set up language models: {e}")
        sys.exit(1)

    # Get task configuration
    task_name = settings.task.name
    task_conf = TASK_REGISTRY.get(task_name)
    
    if not validate_task_configuration(task_name, task_conf):
        sys.exit(1)

    # Prepare datasets
    try:
        trainset, valset = prepare_datasets(task_conf)  # type: ignore
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        sys.exit(1)

    opt_cfg = settings.optimization
    
    # Set up optimization environment
    setup_optimization_environment(opt_cfg)
    
    # Create teleprompter
    try:
        teleprompter = create_teleprompter(teacher, opt_cfg, task_conf)  # type: ignore
    except Exception as e:
        logger.error(f"Failed to create teleprompter: {e}")
        sys.exit(1)

    agent = UniversalExtractor(task_conf["signature"])  # type: ignore
    logger.info(f"Starting optimization for task: {task_name}")
    
    # Start MLFlow run if available
    timestamp = setup_mlflow_tracking(task_name, opt_cfg)
    
    # Track if optimization was successful
    optimization_success = False
    error_message = None
    
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
        
        optimization_success = True
        # Log optimization success
        log_optimization_result(success=True)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Optimization crashed: {e}", exc_info=True)
        # Log optimization failure
        log_optimization_result(success=False, error_message=error_message)
        sys.exit(1)
    finally:
        # Ensure MLflow run is ended
        end_mlflow_run()

    # Save optimized agent
    try:
        output_path = create_output_path(args.output, task_name)
        optimized_agent.save(str(output_path))
        logger.info(f"Optimized agent saved to: {output_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save optimized agent: {e}")
        sys.exit(1)
    
    # Log the optimized agent as an artifact
    log_optimized_agent(optimized_agent, task_name, output_path)
    
    # Log LLM history
    log_llm_history(task_name)


if __name__ == "__main__":
    main()