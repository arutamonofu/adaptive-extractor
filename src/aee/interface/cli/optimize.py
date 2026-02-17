"""CLI command for agent optimization.

This module provides the command-line interface for optimizing agents
using the OptimizeAgentUseCase.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import dspy

from aee import setup_logging
from aee.application.services import AgentManager, DatasetBuilder, ExperimentTracker
from aee.application.use_cases import OptimizeAgentRequest, OptimizeAgentUseCase
from aee.domain.tasks import get_task
from aee.infrastructure.config.instruction_loader import InstructionLoader
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


def create_dependencies(args, task, config=None):
    """Create dependencies for the use case.

    Args:
        args: Parsed command-line arguments.
        task: Task definition.
        config: Optional Settings object to use (defaults to global settings).

    Returns:
        Tuple of (dataset_builder, agent_manager, gt_repo, tracker, validator).
    """
    current_settings = config or settings

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

    # Create data validator
    from aee.application.services import DataValidator
    validator = DataValidator(gt_repo=gt_repo)

    # Create experiment tracker (optional)
    tracker = None
    if not args.no_mlflow:
        try:
            tracker = ExperimentTracker(
                experiment_name=f"{task.name}/optimization",
                enabled=True,
            )
        except Exception as e:
            logger.warning(f"MLflow tracking disabled: {e}")

    return dataset_builder, agent_manager, gt_repo, tracker, validator


def load_task_with_instruction(task_name: str, config) -> tuple:
    """Load task with initial instruction from config.

    Args:
        task_name: Name of the task to load.
        config: Settings object containing task configuration.

    Returns:
        Tuple of (task, instruction_metadata_dict).

    Raises:
        MissingConfigError: If instruction file is not found or empty.
        ValueError: If task cannot be created with the instruction.
    """
    # Calculate config directory for instruction loader
    config_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "config"

    # Load instruction using InstructionLoader
    instruction_loader = InstructionLoader(config_dir=config_dir)
    instruction_metadata = instruction_loader.load_with_metadata(config.task.initial_instruction_file)

    logger.info(
        f"Loaded initial instruction: {config.task.initial_instruction_file} "
        f"({instruction_metadata['instruction_length']} chars, hash: {instruction_metadata['instruction_hash']})"
    )

    # Get task class and instantiate with instruction
    # Import the task module to trigger registration
    if task_name == "nanozymes":
        from aee.domain.tasks.nanozymes import NanozymeTask
        task = NanozymeTask(initial_instruction=instruction_metadata["instruction"])
    else:
        # For other tasks that may not support instruction injection yet
        raise ValueError(
            f"Task '{task_name}' requires explicit instruction loading. "
            f"Only 'nanozymes' task supports this feature currently."
        )

    logger.info(f"Task loaded: {task.name} - {task.description}")

    return task, instruction_metadata


def optimize_command(argv: Optional[list] = None) -> int:
    """Execute the optimize command.

    Args:
        argv: Command-line arguments (None for sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Parse arguments first to get config path
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    # Load settings with required config
    from aee.infrastructure.config.settings import Settings
    custom_settings = Settings.load(config_path=args.config)

    # Setup logging with custom settings
    setup_logging(custom_settings)

    try:
        # Load task definition with instruction
        task_name = custom_settings.task.name
        logger.info(f"Starting optimization for task '{task_name}'")

        task, instruction_metadata = load_task_with_instruction(task_name, custom_settings)

        # Setup language models
        student_lm, teacher_lm = setup_language_models(custom_settings)

        # Create dependencies
        dataset_builder, agent_manager, gt_repo, tracker, validator = create_dependencies(
            args, task, custom_settings
        )

        # Create use case
        use_case = OptimizeAgentUseCase(
            dataset_builder=dataset_builder,
            agent_manager=agent_manager,
            gt_repo=gt_repo,
            tracker=tracker,
            validator=validator,
        )

        # Build request
        gt_path = custom_settings.paths.ground_truth_dir / f"{task_name}.csv"
        split_path = custom_settings.paths.splits_file

        # Use values from config file only
        num_trials = custom_settings.optimization.num_trials
        train_limit = custom_settings.optimization.train_split
        val_limit = custom_settings.optimization.total_load

        # Log all optimization settings for transparency
        logger.info("=" * 60)
        logger.info("OPTIMIZATION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Task: {task_name}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Instruction: {custom_settings.task.initial_instruction_file} (hash: {instruction_metadata['instruction_hash']})")
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
        logger.info(f"  minibatch: {custom_settings.optimization.minibatch} (size={custom_settings.optimization.minibatch_size})")
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
            task=task,
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
            print(f"\n✓ Success! Agent saved to: {response.agent_path}")
            print(f"✓ Final F1 Score: {response.final_metrics.get('f1', 0):.3f}") # type: ignore
            return 0
        else:
            logger.error(f"Optimization failed: {response.error_message}")
            print(f"\n✗ Optimization failed: {response.error_message}")
            return 1

    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        print("\n\n⚠ Optimization interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1


def main():
    """Main entry point."""
    sys.exit(optimize_command())


if __name__ == "__main__":
    main()
