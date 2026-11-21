"""CLI command interface for the RE pipeline."""

import argparse
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ae import setup_logging
from ae.core.config.settings import Settings
from ae.semantic_optimization.use_case import (
    SemanticOptimizationUseCase,
    SemanticOptimizationRequest,
    SemanticOptimizationResponse,
)
from ae.core.cli import run_cli_use_case

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ae-optimize-semantic CLI."""
    parser = argparse.ArgumentParser(
        description="Run the Reverse Engineering (RE) pipeline on Ground Truth markup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # Run parser
    run_parser = subparsers.add_parser("run", help="Run the RE pipeline")
    run_parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated steps to run (positive, consolidation, negative, generalization)"
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline from existing cached artifacts (overrides config setting)"
    )
    run_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive terminal prompts and use auto-resolution"
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration directory"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, ground truth dataset, and paths without starting the pipeline"
    )

    # Status parser
    status_parser = subparsers.add_parser("status", help="Show status of semantic prompt optimization artifacts")
    status_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration directory"
    )

    return parser


def status_command(args) -> int:
    """Status subcommand implementation."""
    try:
        custom_settings = Settings.load(config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    import yaml
    task_name = "schema"
    try:
        with open(custom_settings.paths.schema_file, "r") as f:
            schema_data = yaml.safe_load(f)
            if schema_data and "name" in schema_data:
                task_name = schema_data["name"]
    except Exception:
        pass

    artifacts_dir = custom_settings.semantic_optimization.artifacts_dir if custom_settings.semantic_optimization else Path("data/interim/semantic_optimization")
    task_art_dir = artifacts_dir

    print("=" * 60)
    print(f"SEMANTIC PROMPT OPTIMIZATION STATUS FOR SCHEMA: {task_name}")
    print(f"Artifacts Directory: {task_art_dir}")
    print("=" * 60)

    phases = [
        ("Phase 1: Positive Row Analysis", "01_positive/rows"),
        ("Phase 1: Positive Column Analysis", "01_positive/columns"),
        ("Phase 2: Row Consolidation", "02_consolidation"),
        ("Phase 2: Column Consolidation", "02_consolidation/columns"),
        ("Phase 3: Negative Row Analysis", "03_negative/rows"),
        ("Phase 3: Negative Column Analysis", "03_negative/columns"),
        ("Phase 4: Row Generalization", "04_generalization"),
        ("Phase 4: Column Generalization", "04_generalization/columns"),
        ("Phase 5: Iteration Metrics", "05_compilation/iteration_metrics.json"),
    ]

    for label, rel_path in phases:
        path = task_art_dir / rel_path
        if not path.exists():
            print(f"  {label:<40} : MISSING")
            continue

        if path.is_file():
            print(f"  {label:<40} : EXISTS (file, {path.stat().st_size} bytes)")
        else:
            files = list(path.glob("*.json"))
            print(f"  {label:<40} : EXISTS ({len(files)} JSON files)")

    final_prompt_path = custom_settings.paths.generated_prompt_file
    if final_prompt_path.exists():
        print(f"  {'Final Generated Prompt':<40} : EXISTS ({final_prompt_path.stat().st_size} bytes)")
    else:
        print(f"  {'Final Generated Prompt':<40} : MISSING")

    print("=" * 60)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = create_argument_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "status":
        return status_command(args)
    elif args.command == "run":
        settings_container = {}
        re_client_container = {}
        args_container = {}

        def setup_llms(settings: Settings) -> None:
            settings_container["settings"] = settings
            from ae.semantic_optimization.client import SemanticOptimizationClient
            re_client = SemanticOptimizationClient.from_settings(settings)
            re_client_container["client"] = re_client

        def build_request(
            run_args: argparse.Namespace,
            settings: Settings,
            cancel_event: threading.Event,
        ) -> SemanticOptimizationRequest:
            args_container["args"] = run_args
            
            import yaml
            task_name = "schema"
            try:
                with open(settings.paths.schema_file, "r") as f:
                    schema_data = yaml.safe_load(f)
                    if schema_data and "name" in schema_data:
                        task_name = schema_data["name"]
            except Exception:
                pass

            gt_path = settings.paths.ground_truth_file
            split_path = settings.paths.splits_file
            use_split = settings.semantic_optimization.use_split if settings.semantic_optimization else "train"

            from ae.core.storage.splits import load_split_ordered
            doc_ids = load_split_ordered(split_path, use_split)
            logger.info(f"Loaded {len(doc_ids)} document IDs from {split_path} split '{use_split}'")

            val_split = getattr(settings.semantic_optimization, "val_split", "val") if settings.semantic_optimization else "val"
            val_doc_ids: list[str] = []
            try:
                val_doc_ids = load_split_ordered(split_path, val_split)
                logger.info(f"Loaded {len(val_doc_ids)} validation doc IDs from {split_path} split '{val_split}'")
            except Exception:
                pass

            f1_threshold = getattr(settings.semantic_optimization, "f1_threshold", 0.85) if settings.semantic_optimization else 0.85
            max_tool_iterations = getattr(settings.semantic_optimization, "max_tool_iterations", 5) if settings.semantic_optimization else 5
            initial_docs = getattr(settings.semantic_optimization, "initial_docs", 2) if settings.semantic_optimization else 2
            step_size = getattr(settings.semantic_optimization, "step_size", 2) if settings.semantic_optimization else 2
            row_batch_size = getattr(settings.semantic_optimization, "row_batch_size", 5) if settings.semantic_optimization else 5
            max_rows_per_doc = getattr(settings.semantic_optimization, "max_rows_per_doc", 20) if settings.semantic_optimization else 20
            harmonize_prompt = getattr(settings.semantic_optimization, "harmonize_prompt", True) if settings.semantic_optimization else True
            infer_schema_constraints = getattr(settings.semantic_optimization, "infer_schema_constraints", True) if settings.semantic_optimization else True
            enable_semantic_judge = getattr(settings.semantic_optimization, "enable_semantic_judge", True) if settings.semantic_optimization else True
            residual_enabled = getattr(settings.semantic_optimization, "residual_enabled", True) if settings.semantic_optimization else True
            residual_fallback_threshold = getattr(settings.semantic_optimization, "residual_fallback_threshold", 0.35) if settings.semantic_optimization else 0.35
            residual_skip_f1 = getattr(settings.semantic_optimization, "residual_skip_f1", 0.98) if settings.semantic_optimization else 0.98
            residual_max_examples_per_sig = getattr(settings.semantic_optimization, "residual_max_examples_per_sig", 3) if settings.semantic_optimization else 3
            residual_max_fn_rows = getattr(settings.semantic_optimization, "residual_max_fn_rows", 8) if settings.semantic_optimization else 8
            residual_max_fp_rows = getattr(settings.semantic_optimization, "residual_max_fp_rows", 10) if settings.semantic_optimization else 10

            # Configure caching
            import dspy
            enable_cache = settings.semantic_optimization.enable_cache if settings.semantic_optimization else True
            cache_dir = Path.cwd() / ".cache" / "dspy"
            cache_dir.mkdir(parents=True, exist_ok=True)
            dspy.configure_cache(
                enable_disk_cache=enable_cache,
                enable_memory_cache=enable_cache,
                disk_cache_dir=str(cache_dir),
            )

            steps = None
            if run_args.steps:
                steps = [s.strip().lower() for s in run_args.steps.split(",")]

            resume = run_args.resume or (settings.semantic_optimization.resume if settings.semantic_optimization else False)
            interactive = not run_args.non_interactive if hasattr(run_args, "non_interactive") and run_args.non_interactive else (settings.semantic_optimization.interactive if settings.semantic_optimization else True)
            corrected_gt_file = getattr(settings.semantic_optimization, "corrected_gt_file", None) if settings.semantic_optimization else None
            artifacts_dir = settings.semantic_optimization.artifacts_dir if settings.semantic_optimization else Path("data/interim/semantic_optimization")
            baseline_prompt_path = settings.paths.baseline_prompt_file
            schema_path = settings.paths.schema_file

            re_client = re_client_container.get("client")

            return SemanticOptimizationRequest(
                task_name=task_name,
                doc_ids=doc_ids,
                gt_path=gt_path,
                ingestion_dir=settings.paths.ingestion_dir,
                baseline_prompt_path=baseline_prompt_path,
                schema_path=schema_path,
                re_client=re_client,
                output_dir=artifacts_dir,
                val_doc_ids=val_doc_ids,
                f1_threshold=f1_threshold,
                max_tool_iterations=max_tool_iterations,
                initial_docs=initial_docs,
                step_size=step_size,
                row_batch_size=row_batch_size,
                max_rows_per_doc=max_rows_per_doc,
                harmonize_prompt=harmonize_prompt,
                resume=resume,
                interactive=interactive,
                corrected_gt_path=corrected_gt_file,
                steps=steps,
                generated_prompt_path=settings.paths.generated_prompt_file,
                generated_schema_path=settings.paths.generated_schema_file,
                infer_schema_constraints=infer_schema_constraints,
                enable_semantic_judge=enable_semantic_judge,
                residual_enabled=residual_enabled,
                residual_fallback_threshold=residual_fallback_threshold,
                residual_skip_f1=residual_skip_f1,
                residual_max_examples_per_sig=residual_max_examples_per_sig,
                residual_max_fn_rows=residual_max_fn_rows,
                residual_max_fp_rows=residual_max_fp_rows,
                pdf_dir=settings.paths.pdf_dir,
                config_path=run_args.config,
            )

        def execute_use_case(request: SemanticOptimizationRequest) -> SemanticOptimizationResponse:
            settings = settings_container["settings"]
            args = args_container["args"]

            if args.dry_run:
                logger.info("Running dry-run validation checks...")
                if not request.baseline_prompt_path.exists():
                    raise FileNotFoundError(f"Baseline prompt not found: {request.baseline_prompt_path}")
                if not request.schema_path.exists():
                    raise FileNotFoundError(f"Schema file not found: {request.schema_path}")
                if not request.gt_path.exists():
                    raise FileNotFoundError(f"Ground Truth CSV not found: {request.gt_path}")
                
                logger.info("=" * 60)
                logger.info("DRY RUN SEMANTIC PROMPT OPTIMIZATION VALIDATION COMPLETED SUCCESSFULLY")
                logger.info(f"  Task name:       {request.task_name}")
                logger.info(f"  Document IDs:    {len(request.doc_ids)} from split")
                logger.info(f"  Baseline prompt: {request.baseline_prompt_path}")
                logger.info(f"  Schema path:     {request.schema_path}")
                logger.info(f"  Ground truth:    {request.gt_path}")
                logger.info(f"  Artifacts dir:   {request.output_dir}")
                logger.info("✓ Success! All configuration files and datasets are ready for semantic prompt optimization.")
                logger.info("=" * 60)
                print("✓ Success! All configuration files and datasets are ready for semantic prompt optimization.")
                return SemanticOptimizationResponse(
                    success=True,
                    generated_prompt_path=None,
                )

            use_case = SemanticOptimizationUseCase()
            response = use_case.execute(request)

            # Save history if enabled
            if settings.semantic_optimization and settings.semantic_optimization.save_llm_history:
                re_client = re_client_container.get("client")
                if re_client is not None:
                    try:
                        history_dir = Path("research/experiments/_runtime/logs/llm/semantic_optimization")
                        re_client.save_history(history_dir)
                    except Exception as e:
                        logger.warning(f"Failed to save LLM history: {e}")

            return response

        def format_response(response: SemanticOptimizationResponse) -> int:
            args = args_container["args"]
            if args.dry_run:
                return 0 if response.success else 1

            if response.success:
                logger.info("=" * 60)
                logger.info("SEMANTIC PROMPT OPTIMIZATION COMPLETED SUCCESSFULLY")
                logger.info(f"Generated prompt path: {response.generated_prompt_path}")
                logger.info(f"Anomalies found: {len(response.anomalies)}")
                for idx, a in enumerate(response.anomalies[:5]):
                    ref = a.get('source_reference')
                    if isinstance(ref, dict):
                        ref_parts = [f"doc: {ref.get('document_id', 'unknown')}"]
                        if ref.get('positive_reference_ids'):
                            ref_parts.append(f"pos: {','.join(ref['positive_reference_ids'])}")
                        if ref.get('negative_reference_ids'):
                            ref_parts.append(f"neg: {','.join(ref['negative_reference_ids'])}")
                        ref_str = "; ".join(ref_parts)
                    else:
                        ref_str = str(ref)
                    logger.warning(f"  Anomaly {idx+1} ({a.get('scope')}): {a.get('anomaly_description')} [Ref: {ref_str}]")
                if len(response.anomalies) > 5:
                    logger.warning(f"  ... and {len(response.anomalies) - 5} more anomalies.")
                logger.info("=" * 60)
                return 0
            else:
                logger.error(f"Semantic prompt optimization failed: {response.error_message}")
                return 1

        return run_cli_use_case(
            argv=argv,
            parser=parser,
            build_request_fn=build_request,
            execute_use_case_fn=execute_use_case,
            format_response_fn=format_response,
            setup_llms_fn=setup_llms,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
