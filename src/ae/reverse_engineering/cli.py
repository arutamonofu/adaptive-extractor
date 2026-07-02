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
from ae.reverse_engineering.use_case import (
    ReverseEngineeringUseCase,
    ReverseEngineeringRequest,
    ReverseEngineeringResponse,
)
from ae.core.cli import run_cli_use_case

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ae-re CLI."""
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
    status_parser = subparsers.add_parser("status", help="Show status of RE artifacts")
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

    artifacts_dir = custom_settings.re.artifacts_dir if custom_settings.re else Path("data/reverse_engineering")
    task_art_dir = artifacts_dir

    print("=" * 60)
    print(f"RE PIPELINE STATUS FOR SCHEMA: {task_name}")
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
        ("Phase 5: Row Compilation", "05_compilation"),
        ("Phase 5: Column Compilation", "05_compilation/columns"),
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
        teacher_lm_container = {}
        args_container = {}

        def setup_llms(settings: Settings) -> None:
            settings_container["settings"] = settings
            from ae.core.llm.provider import setup_teacher
            teacher_lm = setup_teacher(settings.llm.teacher, settings.circuit_breaker)
            teacher_lm_container["lm"] = teacher_lm

        def build_request(
            run_args: argparse.Namespace,
            settings: Settings,
            cancel_event: threading.Event,
        ) -> ReverseEngineeringRequest:
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
            use_split = settings.re.use_split if settings.re else "train"

            from ae.core.storage.splits import load_split
            doc_ids = sorted(list(load_split(split_path, use_split)))
            logger.info(f"Loaded {len(doc_ids)} document IDs from {split_path} split '{use_split}'")

            # Configure caching
            import dspy
            enable_cache = settings.llm.teacher.enable_cache
            dspy.configure_cache(enable_disk_cache=enable_cache, enable_memory_cache=enable_cache)

            steps = None
            if run_args.steps:
                steps = [s.strip().lower() for s in run_args.steps.split(",")]

            resume = run_args.resume or (settings.re.resume if settings.re else False)
            artifacts_dir = settings.re.artifacts_dir if settings.re else Path("data/reverse_engineering")
            baseline_prompt_path = settings.paths.baseline_prompt_file
            schema_path = settings.paths.schema_file

            teacher_lm = teacher_lm_container["lm"]

            return ReverseEngineeringRequest(
                task_name=task_name,
                doc_ids=doc_ids,
                gt_path=gt_path,
                ingestion_dir=settings.paths.ingestion_dir,
                baseline_prompt_path=baseline_prompt_path,
                schema_path=schema_path,
                teacher_lm=teacher_lm,
                output_dir=artifacts_dir,
                resume=resume,
                steps=steps,
            )

        def execute_use_case(request: ReverseEngineeringRequest) -> ReverseEngineeringResponse:
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
                logger.info("DRY RUN RE PIPELINE VALIDATION COMPLETED SUCCESSFULLY")
                logger.info(f"  Task name:       {request.task_name}")
                logger.info(f"  Document IDs:    {len(request.doc_ids)} from split")
                logger.info(f"  Baseline prompt: {request.baseline_prompt_path}")
                logger.info(f"  Schema path:     {request.schema_path}")
                logger.info(f"  Ground truth:    {request.gt_path}")
                logger.info(f"  Artifacts dir:   {request.output_dir}")
                logger.info("✓ Success! All configuration files and datasets are ready for the RE pipeline.")
                logger.info("=" * 60)
                print("✓ Success! All configuration files and datasets are ready for the RE pipeline.")
                return ReverseEngineeringResponse(
                    success=True,
                    generated_prompt_path=None,
                )

            use_case = ReverseEngineeringUseCase()
            response = use_case.execute(request)

            # Save history if enabled
            if settings.re and settings.re.save_llm_history:
                teacher_lm = teacher_lm_container.get("lm")
                if teacher_lm is not None:
                    try:
                        from datetime import datetime
                        from ae.core.llm.history_logger import save_history
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        history_dir = Path(settings.re.llm_history_dir)
                        save_history(teacher_lm, history_dir / f"re_teacher_lm_{timestamp}.json")
                    except Exception as e:
                        logger.warning(f"Failed to save LLM history: {e}")

            return response

        def format_response(response: ReverseEngineeringResponse) -> int:
            args = args_container["args"]
            if args.dry_run:
                return 0 if response.success else 1

            if response.success:
                logger.info("=" * 60)
                logger.info("RE PIPELINE COMPLETED SUCCESSFULLY")
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
                logger.error(f"RE pipeline failed: {response.error_message}")
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
