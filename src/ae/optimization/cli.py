"""CLI command for agent optimization.

This module provides the command-line interface for optimizing agents
using the OptimizeAgentUseCase.
"""

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import List, Optional

from ae.core.config.settings import Settings
from ae.core.storage import (
    AgentRepository,
    DocumentRepository,
    GroundTruthRepository,
    DataSplitRepository,
)
from ae.extraction.manager import AgentManager
from ae.optimization.dataset_builder import DatasetBuilder
from ae.optimization.use_case import OptimizeAgentRequest, OptimizeAgentUseCase, OptimizeAgentResponse
from ae.optimization.tracking import ExperimentTracker
from ae.core.cli import run_cli_use_case

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
        help="Path to configuration directory (defaults to root config/ directory)",
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

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pre-flight validation checks on datasets and configurations without starting optimization",
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
    import dspy
    if enable_cache:
        dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)
        logger.debug("DSPy cache enabled (disk + memory)")
    else:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        logger.info("DSPy cache disabled for fresh predictions")

    # Import LLM setup functions
    from ae.core.llm import setup_student, setup_teacher

    student_lm = setup_student(config.llm.student, config.circuit_breaker, enable_cache=enable_cache)
    teacher_lm = setup_teacher(config.llm.teacher, config.circuit_breaker, enable_cache=enable_cache)

    logger.info(
        f"Configured LMs: Student={type(student_lm).__name__}, "
        f"Teacher={type(teacher_lm).__name__ if teacher_lm else 'None'} "
        f"(cache={enable_cache})"
    )

    return student_lm, teacher_lm


def optimize_command(argv: Optional[List[str]] = None) -> int:
    """Execute the optimize command."""
    parser = create_argument_parser()

    # Optuna study check, strip 'optimize' subcommand if it exists
    if argv and argv[0] == "optimize":
        argv = argv[1:]

    settings_container = {}
    student_lm_container = {}
    teacher_lm_container = {}
    args_container = {}

    def setup_llms(settings: Settings) -> None:
        settings_container["settings"] = settings
        student_lm, teacher_lm = setup_language_models(settings, enable_cache=settings.optimization.use_cache)
        student_lm_container["lm"] = student_lm
        teacher_lm_container["lm"] = teacher_lm

    def build_request(
        args: argparse.Namespace,
        settings: Settings,
        cancel_event: threading.Event,
    ) -> OptimizeAgentRequest:
        args_container["args"] = args

        # Load schema definition
        from ae.core.schema import load_schema_complete
        task = load_schema_complete(
            yaml_path=settings.paths.schema_file,
            instruction_path=settings.paths.baseline_prompt_file,
        )

        # Validate task has signature for agent optimization
        if task.signature is None:
            raise ValueError("Task signature not found - required for agent optimization")

        # Validate ingestion_dir exists
        if not settings.paths.ingestion_dir.exists():
            raise FileNotFoundError(
                f"Parsed directory not found: {settings.paths.ingestion_dir}\n"
                f"Please ensure documents are parsed before optimization."
            )

        gt_path = settings.paths.ground_truth_file
        split_path = settings.paths.splits_file

        num_trials = settings.optimization.num_trials
        train_limit = settings.optimization.train_split
        val_limit = settings.optimization.total_load

        student_lm = student_lm_container["lm"]
        teacher_lm = teacher_lm_container["lm"]

        # Log all optimization settings for transparency
        logger.info("=" * 60)
        logger.info("OPTIMIZATION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Schema: {task.config.name}")
        logger.info(f"Config file: {args.config}")
        logger.info(
            f"Instruction: {task.config.instruction_file} "
            f"(hash: {task.config.get_instruction_hash()})"
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
        logger.info(f"  seed: {settings.optimization.random_seed}")
        logger.info(f"  num_candidates: {settings.optimization.num_candidates}")
        logger.info(f"  max_bootstrapped_demos: {settings.optimization.max_bootstrapped_demos}")
        logger.info(f"  max_labeled_demos: {settings.optimization.max_labeled_demos}")
        logger.info(
            f"  minibatch: {settings.optimization.minibatch} "
            f"(size={settings.optimization.minibatch_size})"
        )
        logger.info(f"  view_data_batch_size: {settings.optimization.view_data_batch_size}")
        logger.info(f"  metric_threshold: {settings.optimization.metric_threshold}")
        logger.info(f"  init_temperature: {settings.optimization.init_temperature}")
        logger.info(f"  max_errors: {settings.optimization.max_errors}")
        logger.info(f"  verbose: {settings.optimization.verbose}")
        logger.info("-" * 60)
        logger.info("LLM CONFIGURATION:")
        logger.info(f"  Student: {settings.llm.student.model} (temp={settings.llm.student.temperature})")
        if settings.llm.teacher:
            logger.info(f"  Teacher: {settings.llm.teacher.model} (temp={settings.llm.teacher.temperature})")
        else:
            logger.info("  Teacher: None")
        logger.info(f"  Cache: {'ENABLED' if settings.optimization.use_cache else 'DISABLED'}")
        logger.info("-" * 60)
        logger.info("MLFLOW:")
        logger.info(f"  Enabled: {not args.no_mlflow}")
        logger.info(f"  Run name prefix: {args.run_name if args.run_name else 'auto'}")
        logger.info("=" * 60)

        return OptimizeAgentRequest(
            task=task,
            signature_class=task.signature,
            gt_path=gt_path,
            split_path=split_path,
            student_lm=student_lm,
            teacher_lm=teacher_lm,
            num_trials=num_trials,
            train_limit=train_limit,
            val_limit=val_limit,
            model_version=str(student_lm.model),
            description=f"Optimized with {num_trials} trials",
            seed=settings.optimization.random_seed,
            num_candidates=settings.optimization.num_candidates,
            max_bootstrapped_demos=settings.optimization.max_bootstrapped_demos,
            max_labeled_demos=settings.optimization.max_labeled_demos,
            minibatch=settings.optimization.minibatch,
            minibatch_size=settings.optimization.minibatch_size,
            view_data_batch_size=settings.optimization.view_data_batch_size,
            metric_threshold=settings.optimization.metric_threshold,
            init_temperature=settings.optimization.init_temperature,
            max_errors=settings.optimization.max_errors,
            verbose=settings.optimization.verbose,
            run_name_prefix=args.run_name,
            initial_instruction_file=task.config.instruction_file,
            instruction_hash=task.config.get_instruction_hash(),
            cancel_event=cancel_event,
            dry_run=args.dry_run,
        )

    def execute_use_case(request: OptimizeAgentRequest) -> OptimizeAgentResponse:
        settings = settings_container["settings"]
        args = args_container["args"]

        # Create repositories
        doc_repo = DocumentRepository(ingestion_dir=settings.paths.ingestion_dir)
        gt_repo = GroundTruthRepository()
        agent_repo = AgentRepository(agents_dir=settings.paths.agents_dir)
        split_repo = DataSplitRepository()

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
                    experiment_name="optimization",
                    tracking_uri=settings.mlflow_tracking_uri,
                    enabled=True,
                )
            except Exception as e:
                logger.warning(f"MLflow tracking disabled: {e}")

        # Create use case
        use_case = OptimizeAgentUseCase(
            dataset_builder=dataset_builder,
            agent_manager=agent_manager,
            gt_repo=gt_repo,
            split_repo=split_repo,
            tracker=tracker,
            enable_preflight_check=True,
        )

        try:
            return use_case.execute(request)
        finally:
            # Save LLM histories (always, even on error/interrupt)
            if settings.optimization.save_llm_history:
                student_lm = student_lm_container.get("lm")
                teacher_lm = teacher_lm_container.get("lm")
                if student_lm is not None:
                    from ae.core.llm.history_logger import save_optimization_history
                    history_dir = Path(settings.optimization.llm_history_dir)
                    save_optimization_history(student_lm, teacher_lm, history_dir)

    def format_response(response: OptimizeAgentResponse) -> int:
        args = args_container["args"]
        if response.success:
            if args.dry_run:
                logger.info("=" * 60)
                logger.info("DRY RUN VALIDATION COMPLETED SUCCESSFULLY")
                logger.info(f"  Train set size: {response.final_metrics.get('train_size', 0)}")
                logger.info(f"  Val set size:   {response.final_metrics.get('val_size', 0)}")
                logger.info("✓ Success! All dataset files exist, load correctly, and match splits.")
                logger.info("=" * 60)
                print("✓ Success! All dataset files exist, load correctly, and match splits.")
                return 0

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

    return run_cli_use_case(
        argv=argv,
        parser=parser,
        build_request_fn=build_request,
        execute_use_case_fn=execute_use_case,
        format_response_fn=format_response,
        setup_llms_fn=setup_llms,
    )


def main():
    """Main entry point."""
    sys.exit(optimize_command())


if __name__ == "__main__":
    main()
