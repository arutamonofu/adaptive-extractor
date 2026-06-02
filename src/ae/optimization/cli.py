"""CLI command for agent optimization.

This module provides the command-line interface for optimizing agents
using the OptimizeAgentUseCase.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ae import setup_logging
from ae.core.config.settings import Settings
from ae.core.llm.history_logger import save_optimization_history
from ae.core.storage import (
    AgentRepository,
    DocumentRepository,
    GroundTruthRepository,
)
from ae.core.tasks import load_task_with_instruction
from ae.extraction.manager import AgentManager
from ae.optimization.dataset import DatasetBuilder
from ae.optimization.orchestrator import OptimizeAgentRequest, OptimizeAgentUseCase
from ae.optimization.tracking import ExperimentTracker

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
        "--enable-contrastive",
        action="store_true",
        help="Запустить контрастивный анализ перед оптимизацией промпта",
    )

    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=None,
        help="Путь к предрассчитанному JSON результату контрастивного анализа (пропуск фаз Map/Reduce)",
    )

    parser.add_argument(
        "--analysis-batch-size",
        type=int,
        default=10,
        help="Размер пакета документов для контрастивного анализа",
    )

    parser.add_argument(
        "--auto-skip-review",
        action="store_true",
        help="Пропустить интерактивный разбор (использовать только 100%% консенсусные правила)",
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
    from ae.core.llm import setup_student, setup_teacher

    student_lm = setup_student(config, enable_cache=enable_cache)
    teacher_lm = setup_teacher(config, enable_cache=enable_cache)

    logger.info(
        f"Configured LMs: Student={type(student_lm).__name__}, "
        f"Teacher={type(teacher_lm).__name__ if teacher_lm else 'None'} "
        f"(cache={enable_cache})"
    )

    return student_lm, teacher_lm


def create_dependencies(args, task, task_name, settings):
    """Create dependencies for the use case.

    Args:
        args: Parsed command-line arguments.
        task: Task definition.
        task_name: Name of the task.
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
                experiment_name=f"{task_name}/optimization",
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

    student_lm, teacher_lm = None, None

    try:
        # Load task definition with instruction
        task_name = custom_settings.task.name

        task, instruction_metadata = load_task_with_instruction(task_name, custom_settings)

        # Validate task has signature for agent optimization
        if task.get("signature") is None:
            logger.error("Task signature not found - required for agent optimization")
            return 1

        # Setup language models
        student_lm, teacher_lm = setup_language_models(custom_settings)

        # Check if contrastive analysis is enabled or analysis_file is provided
        enable_contrastive = getattr(args, "enable_contrastive", False)
        analysis_file = getattr(args, "analysis_file", None)
        
        schema_in_prompt = False
        contrastive_prompt = None
        analysis_result_path = None
        
        if enable_contrastive or analysis_file:
            try:
                schema_in_prompt = True
                from ae.optimization.contrastive import (
                    AnalysisResult,
                    LocalAnalyzer,
                    ContrastiveMapRunner,
                    StrictAggregator,
                    HumanReviewCLI,
                    merge_review_into_result,
                    build_three_level_prompt,
                    prepare_analysis_inputs
                )
                
                # Prepare folders
                analysis_dir = Path("data/analysis")
                analysis_dir.mkdir(parents=True, exist_ok=True)
                
                if analysis_file:
                    analysis_result_path = Path(analysis_file)
                    logger.info(f"Loading pre-calculated analysis result from {analysis_result_path}")
                    analysis_result = AnalysisResult.from_json(analysis_result_path)
                else:
                    logger.info("Running Map-Reduce contrastive analysis...")
                    import json
                    with open(custom_settings.paths.splits_file, "r") as f:
                        splits = json.load(f)
                    train_ids = splits.get("train", [])[:args.analysis_batch_size]
                    
                    doc_repo = DocumentRepository(parsed_dir=custom_settings.paths.parsed_dir)
                    documents = {}
                    for doc_id in train_ids:
                        try:
                            doc = doc_repo.get(doc_id)
                            if doc:
                                documents[doc_id] = doc.content
                        except Exception as e:
                            logger.warning(f"Failed to load document {doc_id}: {e}")
                    
                    gt_path_loc = custom_settings.paths.ground_truth_dir / f"{task_name}.csv"
                    gt_repo_loc = GroundTruthRepository()
                    gt_data_loc = gt_repo_loc.load(gt_path_loc, task["row_converter"])
                    
                    inputs = prepare_analysis_inputs(task["config"], train_ids, documents, gt_data_loc)
                    
                    if not teacher_lm:
                        logger.error("Teacher LM is required for contrastive analysis")
                        return 1
                    
                    analyzer = LocalAnalyzer(lm=teacher_lm, task_config=task["config"], cache_dir=str(analysis_dir))
                    runner = ContrastiveMapRunner(analyzer=analyzer, max_concurrent=1)
                    
                    import asyncio
                    map_results = asyncio.run(runner.run_batch(inputs))
                    
                    aggregator = StrictAggregator(lm=teacher_lm, task_config=task["config"], cache_dir=str(analysis_dir))
                    analysis_result = aggregator.aggregate(map_results)
                    
                    analysis_result_path = analysis_dir / f"{task_name}_analysis_result.json"
                    analysis_result.to_json(analysis_result_path)
                
                auto_skip_review = getattr(args, "auto_skip_review", False)
                if not auto_skip_review and analysis_result.has_discrepancies():
                    review_cli = HumanReviewCLI(analysis_result)
                    session = review_cli.run()
                    analysis_result = merge_review_into_result(analysis_result, session)
                    analysis_result.to_json(analysis_result_path)
                
                contrastive_prompt = build_three_level_prompt(analysis_result)
                prompt_path = analysis_result_path.with_suffix(".txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(contrastive_prompt)
                logger.info(f"Contrastive prompt compiled and saved to {prompt_path}")
            except Exception as contrastive_err:
                logger.warning(
                    f"Contrastive analysis failed with error: {contrastive_err}. "
                    f"Falling back to default signature/instruction and continuing optimization."
                )
                schema_in_prompt = False
                contrastive_prompt = None
                analysis_result_path = None

        # Create dependencies
        dataset_builder, agent_manager, gt_repo, tracker = create_dependencies(
            args, task, task_name, custom_settings
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
            f"Instruction: {task['config'].initial_instruction_file} "
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
        logger.info(f"  max_errors: {custom_settings.optimization.max_errors}")
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
            max_errors=custom_settings.optimization.max_errors,
            verbose=custom_settings.optimization.verbose,
            run_name_prefix=args.run_name,
            initial_instruction_file=task["config"].initial_instruction_file,
            instruction_hash=instruction_metadata["instruction_hash"],
            schema_in_prompt=schema_in_prompt,
            analysis_result_path=analysis_result_path,
            contrastive_prompt=contrastive_prompt,
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

    finally:
        # Save LLM histories (always, even on error/interrupt)
        if custom_settings.optimization.save_llm_history:
            if student_lm is not None:
                history_dir = Path(custom_settings.optimization.llm_history_dir)
                save_optimization_history(student_lm, teacher_lm, history_dir)




import click

@click.group()
def cli():
    pass

@click.command("analyze")
@click.option("--config", default="config/core.yaml", help="Путь к core.yaml")
@click.option("--batch-size", default=10, help="Количество документов для анализа")
@click.option("--output", default=None, help="Путь для сохранения итогового JSON результатов")
@click.option("--auto-skip-review", is_flag=True, help="Пропустить интерактивный разбор")
def analyze_command(config, batch_size, output, auto_skip_review):
    """Выполняет контрастивный анализ данных Ground Truth и формирует правила."""
    settings = Settings.load(config_path=Path(config))
    setup_logging(settings)
    
    student_lm, teacher_lm = setup_language_models(settings)
    if not teacher_lm:
        logger.error("Teacher LM is required for contrastive analysis")
        sys.exit(1)
        
    task_name = settings.task.name
    task, _ = load_task_with_instruction(task_name, settings)
    task_config = task["config"]
    
    doc_repo = DocumentRepository(parsed_dir=settings.paths.parsed_dir)
    gt_repo = GroundTruthRepository()
    
    from ae.optimization.dataset import DatasetBuilder
    dataset_builder = DatasetBuilder(document_repo=doc_repo, gt_repo=gt_repo)
    
    import json
    with open(settings.paths.splits_file, "r") as f:
        splits = json.load(f)
    train_ids = splits.get("train", [])[:batch_size]
    
    gt_data = gt_repo.load(settings.paths.ground_truth_dir / f"{task_name}.csv", task["row_converter"])
    
    documents = {}
    for doc_id in train_ids:
        try:
            doc = doc_repo.get(doc_id)
            if doc:
                documents[doc_id] = doc.content
        except Exception as e:
            logger.warning(f"Failed to load document {doc_id}: {e}")
            
    from ae.optimization.contrastive import (
        prepare_analysis_inputs,
        LocalAnalyzer,
        ContrastiveMapRunner,
        StrictAggregator,
        HumanReviewCLI,
        merge_review_into_result,
        build_three_level_prompt,
    )
    
    inputs = prepare_analysis_inputs(task_config, train_ids, documents, gt_data)
    
    analyzer = LocalAnalyzer(lm=teacher_lm, task_config=task_config, cache_dir="data/analysis")
    runner = ContrastiveMapRunner(analyzer=analyzer, max_concurrent=1)
    
    import asyncio
    map_results = asyncio.run(runner.run_batch(inputs))
    
    aggregator = StrictAggregator(lm=teacher_lm, task_config=task_config, cache_dir="data/analysis")
    analysis_result = aggregator.aggregate(map_results)
    
    if not auto_skip_review and analysis_result.has_discrepancies():
        review_cli = HumanReviewCLI(analysis_result)
        session = review_cli.run()
        analysis_result = merge_review_into_result(analysis_result, session)
        
    output_path = Path(output) if output else Path(f"data/analysis/{task_name}_analysis_result.json")
    analysis_result.to_json(output_path)
    logger.info(f"Saved analysis results to {output_path}")
    
    prompt = build_three_level_prompt(analysis_result)
    prompt_path = output_path.with_suffix(".txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    logger.info(f"Saved compiled three-level prompt to {prompt_path}")


@click.command("review")
@click.option("--analysis-file", required=True, type=click.Path(exists=True), help="Путь к JSON-файлу результатов анализа")
@click.option("--auto-skip", is_flag=True, help="Автоматически пропускать все расхождения")
def review_command(analysis_file, auto_skip):
    """Запускает интерактивную сессию разрешения расхождений."""
    from ae.optimization.contrastive import AnalysisResult, HumanReviewCLI, merge_review_into_result, build_three_level_prompt
    
    analysis_path = Path(analysis_file)
    analysis_result = AnalysisResult.from_json(analysis_path)
    
    review_cli = HumanReviewCLI(analysis_result)
    session = review_cli.run(auto_skip=auto_skip)
    analysis_result = merge_review_into_result(analysis_result, session)
    
    analysis_result.to_json(analysis_path)
    logger.info(f"Updated analysis results saved to {analysis_path}")
    
    prompt = build_three_level_prompt(analysis_result)
    prompt_path = analysis_path.with_suffix(".txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    logger.info(f"Updated compiled three-level prompt saved to {prompt_path}")

cli.add_command(analyze_command)
cli.add_command(review_command)

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ["analyze", "review"]:
        # Route to Click subcommands
        cli()
    else:
        # Fallback to optimize_command (legacy compatibility)
        args = sys.argv[1:]
        if args and args[0] == "optimize":
            args = args[1:]
        sys.exit(optimize_command(args))

if __name__ == "__main__":
    main()
