"""CLI command for agent optimization.

This module provides the command-line interface for optimizing agents
using the OptimizeAgentUseCase.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from aee import setup_logging
from aee.application.services import AgentManager, DatasetBuilder, ExperimentTracker
from aee.application.use_cases import OptimizeAgentRequest, OptimizeAgentUseCase
from aee.domain.tasks import load_task_with_instruction
from aee.infrastructure.config.settings import Settings
from aee.infrastructure.storage import (
    AgentRepository,
    DocumentRepository,
    GroundTruthRepository,
)

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for optimize command."""
    parser = argparse.ArgumentParser(
        description="Optimize an extraction agent using MIPROv2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        required=True,
        help="Path to configuration file (required)",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Short name prefix for this MLflow run (e.g., 'A1_high', 'A2_temp1.0'). "
             "Timestamp will be added automatically.",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    return parser


def setup_language_models(config=None, enable_cache: bool = True):
    """Setup student and teacher language models.

    Args:
        config: Optional Settings object to use (defaults to global settings).
        enable_cache: Whether to enable LLM caching (default: True for optimization).

    Returns:
        Tuple of (student_lm, teacher_lm).
    """
    # Import LLM setup functions
    from aee.infrastructure.llm import setup_student, setup_teacher

    student_lm = setup_student(config, enable_cache=enable_cache)
    teacher_lm = setup_teacher(config, enable_cache=enable_cache)

    logger.info(
        f"Configured LMs: Student={type(student_lm).__name__}, "
        f"Teacher={type(teacher_lm).__name__ if teacher_lm else 'None'} "
        f"(cache={enable_cache})"
    )

    return student_lm, teacher_lm


def create_dependencies(args, task, settings):
    """Create dependencies for the use case.

    Args:
        args: Parsed command-line arguments.
        task: Task definition.
        settings: Settings object to use.

    Returns:
        Tuple of (dataset_builder, agent_manager, gt_repo, tracker).
    """
    current_settings = settings

    # Validate parsed_dir exists
    if not current_settings.paths.parsed_dir.exists():
        raise FileNotFoundError(
            f"Parsed directory not found: {current_settings.paths.parsed_dir}\n"
            f"Please ensure documents are parsed before optimization."
        )

    # Create repositories
    doc_repo = DocumentRepository(parsed_dir=current_settings.paths.parsed_dir)
    gt_repo = GroundTruthRepository()
    agent_repo = AgentRepository(agents_dir=current_settings.paths.agents_dir)

    # Create services
    dataset_builder = DatasetBuilder(
        document_repo=doc_repo,
        gt_repo=gt_repo,
    )

    agent_manager = AgentManager(agent_repo=agent_repo)

    # Create experiment tracker (optional)
    tracker = None
    if not args.no_mlflow:
        try:
            tracker = ExperimentTracker(
                experiment_name=f"{task.name}/optimization",
                tracking_uri=current_settings.mlflow_tracking_uri,
                enabled=True,
            )
        except Exception as e:
            logger.warning(f"MLflow tracking disabled: {e}")

    return dataset_builder, agent_manager, gt_repo, tracker


def optimize_command(argv: Optional[List[str]] = None) -> int:
    """Execute the optimize command.

    Args:
        argv: Command-line arguments (None for sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Parse arguments first to get config path
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    # Load settings from config file
    try:
        custom_settings = Settings.load(config_path=args.config)
        logger.info(f"Loaded configuration from CLI argument: {args.config}")
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Setup logging with custom settings
    setup_logging(custom_settings)

    try:
        # Load task definition with instruction
        task_name = custom_settings.task.name
        logger.info(f"Starting optimization for task '{task_name}'")

        task, instruction_metadata = load_task_with_instruction(task_name, custom_settings)

        # Validate task has signature for agent optimization
        if task.get("signature") is None:
            logger.error("Task signature not found - required for agent optimization")
            return 1

        # Setup language models
        student_lm, teacher_lm = setup_language_models(custom_settings)

        # Create dependencies
        dataset_builder, agent_manager, gt_repo, tracker = create_dependencies(
            args, task, custom_settings
        )

        # Create use case (DataValidator created internally if enable_preflight_check=True)
        use_case = OptimizeAgentUseCase(
            dataset_builder=dataset_builder,
            agent_manager=agent_manager,
            gt_repo=gt_repo,
            tracker=tracker,
            enable_preflight_check=True,
        )

        # Build request
        gt_path = custom_settings.paths.ground_truth_dir / f"{task_name}.csv"
        split_path = custom_settings.paths.splits_file

        # Use values from config file only
        num_trials = custom_settings.optimization.num_trials
        train_limit = custom_settings.optimization.train_split
        val_limit = custom_settings.optimization.total_load

        # task is a dict from get_task() with keys: config, experiment_model,
        # output_model, signature, row_converter
        # OptimizeAgentRequest expects task: TaskConfig, so we pass task["config"]
        # But we also need to pass signature separately for agent creation

        # Log all optimization settings for transparency
        logger.info("=" * 60)
        logger.info("OPTIMIZATION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Task: {task_name}")
        logger.info(f"Config file: {args.config}")
        logger.info(
            f"Instruction: {custom_settings.task.initial_instruction_file} "
            f"(hash: {instruction_metadata['instruction_hash']})"
        )
        logger.info("-" * 60)
        logger.info("DATASET:")
        logger.info(f"  Ground truth: {gt_path}")
        logger.info(f"  Data splits: {split_path}")
        logger.info(f"  Train limit: {train_limit}")
        logger.info(f"  Val limit: {val_limit}")
        logger.info("-" * 60)
        logger.info("MIPROv2 PARAMETERS:")
        logger.info(f"  num_trials: {num_trials}")
        logger.info(f"  seed: {custom_settings.optimization.random_seed}")
        logger.info(f"  num_candidates: {custom_settings.optimization.num_candidates}")
        logger.info(f"  max_bootstrapped_demos: {custom_settings.optimization.max_bootstrapped_demos}")
        logger.info(f"  max_labeled_demos: {custom_settings.optimization.max_labeled_demos}")
        logger.info(
            f"  minibatch: {custom_settings.optimization.minibatch} "
            f"(size={custom_settings.optimization.minibatch_size})"
        )
        logger.info(f"  view_data_batch_size: {custom_settings.optimization.view_data_batch_size}")
        logger.info(f"  metric_threshold: {custom_settings.optimization.metric_threshold}")
        logger.info(f"  init_temperature: {custom_settings.optimization.init_temperature}")
        logger.info(f"  verbose: {custom_settings.optimization.verbose}")
        logger.info("-" * 60)
        logger.info("LLM CONFIGURATION:")
        logger.info(f"  Student: {custom_settings.llm.student.model} (temp={custom_settings.llm.student.temperature})")
        logger.info(f"  Teacher: {custom_settings.llm.teacher.model} (temp={custom_settings.llm.teacher.temperature})")
        logger.info(f"  Cache: {'ENABLED' if custom_settings.optimization.use_cache else 'DISABLED'}")
        logger.info("-" * 60)
        logger.info("MLFLOW:")
        logger.info(f"  Enabled: {not args.no_mlflow}")
        logger.info(f"  Run name prefix: {args.run_name if args.run_name else 'auto'}")
        logger.info("=" * 60)

        request = OptimizeAgentRequest(
            task=task["config"],
            signature_class=task["signature"],
            gt_path=gt_path,
            split_path=split_path,
            student_lm=student_lm,
            teacher_lm=teacher_lm,
            num_trials=num_trials,
            train_limit=train_limit,
            val_limit=val_limit,
            model_version=str(student_lm.model),
            description=f"Optimized with {num_trials} trials",
            seed=custom_settings.optimization.random_seed,
            num_candidates=custom_settings.optimization.num_candidates,
            max_bootstrapped_demos=custom_settings.optimization.max_bootstrapped_demos,
            max_labeled_demos=custom_settings.optimization.max_labeled_demos,
            minibatch=custom_settings.optimization.minibatch,
            minibatch_size=custom_settings.optimization.minibatch_size,
            view_data_batch_size=custom_settings.optimization.view_data_batch_size,
            metric_threshold=custom_settings.optimization.metric_threshold,
            init_temperature=custom_settings.optimization.init_temperature,
            verbose=custom_settings.optimization.verbose,
            run_name_prefix=args.run_name,
            initial_instruction_file=custom_settings.task.initial_instruction_file,
            instruction_hash=instruction_metadata["instruction_hash"],
        )

        # Execute optimization
        logger.info(
            f"Starting optimization for task '{task_name}' with {num_trials} trials"
        )

        response = use_case.execute(request)

        # Display results
        if response.success:
            logger.info("Optimization completed successfully")
            logger.info(f"Agent saved: {response.agent_path}")
            logger.info(f"Metrics: {response.final_metrics}")
            logger.info(f"Trials: {response.trial_count}")
            logger.info(f"✓ Success! Agent saved to: {response.agent_path}")
            logger.info(f"✓ Final F1 Score: {response.final_metrics.get('f1', 0):.3f}")  # type: ignore
            return 0
        else:
            logger.error(f"Optimization failed: {response.error_message}")
            return 1

    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    sys.exit(optimize_command())


if __name__ == "__main__":
    main()
