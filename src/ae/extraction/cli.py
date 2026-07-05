"""CLI command for batch extraction.

This module provides the command-line interface for running extraction
on documents using trained agents.
"""

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from ae.core.config.settings import Settings
from ae.core.storage import (
    AgentRepository,
    DocumentRepository,
    ExtractionRepository,
)
from ae.extraction.manager import AgentManager
from ae.extraction.use_case import (
    BatchExtractionRequest,
    BatchExtractionResponse,
    BatchExtractionUseCase,
)
from ae.core.cli import run_cli_use_case

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for extract command.

    Returns:
        ArgumentParser configured with --config, --agent, and --dry-run arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run batch extraction on documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration directory (defaults to root config/ directory)",
    )

    parser.add_argument(
        "--agent",
        type=Path,
        required=True,
        help="Path to trained agent JSON file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate agent, configuration, and source documents directory without running extraction predictions",
    )

    return parser


def extract_command(argv: Optional[list] = None) -> int:
    """Execute batch extraction on documents using a trained agent.

    This command loads a trained agent, processes all parsed documents,
    and saves extraction results to JSON files.
    """
    parser = create_argument_parser()
    
    settings_container = {}
    student_lm_container = {}
    args_container = {}

    def setup_llms(settings: Settings) -> None:
        settings_container["settings"] = settings
        from ae.core.llm import setup_student
        student_lm_container["lm"] = setup_student(
            settings.llm.student,
            settings.circuit_breaker,
            enable_cache=settings.extraction.enable_cache,
        )

    def build_request(
        args: argparse.Namespace,
        settings: Settings,
        cancel_event: threading.Event,
    ) -> BatchExtractionRequest:
        args_container["args"] = args

        # Resolve agent path relative to project root if not absolute
        agent_path = Path(args.agent)
        if not agent_path.is_absolute():
            agent_path = Path.cwd() / agent_path
        if not agent_path.exists():
            raise FileNotFoundError(f"Agent not found: {agent_path}")

        # Load schema definition
        from ae.core.schema import load_schema_complete
        instruction_path = settings.paths.generated_prompt_file
        if not instruction_path or not instruction_path.exists():
            instruction_path = settings.paths.baseline_prompt_file
        task = load_schema_complete(
            yaml_path=settings.paths.schema_file,
            instruction_path=instruction_path,
        )
        logger.info(f"Schema loaded: {task.config.name}")

        # Validate task has signature for agent reconstruction
        if task.signature is None:
            raise ValueError("Task signature not found - required for agent reconstruction")

        # Create document repository to retrieve documents
        doc_repo = DocumentRepository(ingestion_dir=settings.paths.ingestion_dir)
        
        # Check if parsed directory exists
        if not settings.paths.ingestion_dir.exists():
            logger.warning(f"Parsed directory does not exist: {settings.paths.ingestion_dir}")
            document_ids = []
        else:
            document_ids = doc_repo.list_document_keys()
            logger.info(f"Found {len(document_ids)} documents to process")

        # Log extraction settings
        logger.info("=" * 60)
        logger.info("EXTRACTION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Schema: {task.config.name}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Agent: {args.agent}")
        logger.info(f"Documents: {len(document_ids)}")
        logger.info(f"Output: {settings.paths.extracted_dir}")
        logger.info(f"LLM cache: {'ENABLED' if settings.extraction.enable_cache else 'DISABLED'}")
        logger.info("=" * 60)

        return BatchExtractionRequest(
            task=task,
            agent_path=agent_path,
            document_ids=document_ids,
            output_dir=settings.paths.extracted_dir,
        )

    def execute_use_case(request: BatchExtractionRequest) -> BatchExtractionResponse:
        settings = settings_container["settings"]
        args = args_container["args"]

        if not request.document_ids:
            logger.warning("No documents to process")
            return BatchExtractionResponse(
                success=True,
                extractions_saved=0,
                total_documents=0,
                failed_documents=0,
                output_dir=request.output_dir,
            )

        agent_repo = AgentRepository(agents_dir=settings.paths.agents_dir)
        agent_manager = AgentManager(agent_repo=agent_repo)

        if args.dry_run:
            logger.info("Running dry-run validation checks...")
            try:
                # Reconstruct and validate the agent
                agent = agent_manager.load_agent_as_object(request.agent_path, task=request.task)
                logger.info(f"✓ Agent reconstructed successfully: {agent}")
            except Exception as e:
                logger.error(f"Failed to reconstruct agent: {e}")
                return BatchExtractionResponse(
                    success=False,
                    error_message=f"Failed to reconstruct agent: {e}",
                )

            logger.info("=" * 60)
            logger.info("DRY RUN VALIDATION COMPLETED SUCCESSFULLY")
            logger.info(f"  Agent file:    {request.agent_path}")
            logger.info(f"  Target schema: {request.task.config.name}")
            logger.info(f"  Parsed docs:   {len(request.document_ids)} files ready")
            logger.info(f"  Output dir:    {request.output_dir}")
            logger.info("✓ Success! Agent is valid, config is correct, and documents are ready.")
            logger.info("=" * 60)
            print("✓ Success! Agent is valid, config is correct, and documents are ready.")
            return BatchExtractionResponse(
                success=True,
                extractions_saved=0,
                total_documents=len(request.document_ids),
                failed_documents=0,
                output_dir=request.output_dir,
            )

        doc_repo = DocumentRepository(ingestion_dir=settings.paths.ingestion_dir)
        pred_repo = ExtractionRepository()

        use_case = BatchExtractionUseCase(
            agent_manager=agent_manager,
            document_repo=doc_repo,
            extraction_repo=pred_repo,
        )

        try:
            logger.info(f"Processing {len(request.document_ids)} documents...")
            return use_case.execute(request)
        finally:
            if settings.extraction.save_llm_history:
                lm = student_lm_container.get("lm")
                if lm is not None:
                    from ae.core.llm.history_logger import save_extraction_history
                    history_dir = Path("logs/llm/extraction")
                    save_extraction_history(lm, history_dir)

    def format_response(response: BatchExtractionResponse) -> int:
        args = args_container["args"]
        if args.dry_run:
            return 0 if response.success else 1

        if response.success:
            logger.info("✓ EXTRACTION COMPLETE")
            logger.info(
                f"✓ Processed: {response.extractions_saved}/{response.total_documents}"
            )
            logger.info(f"✓ Output directory: {response.output_dir}")
            logger.info(
                f"✓ Processed: {response.extractions_saved}/{response.total_documents}, "
                f"Failed: {response.failed_documents}, "
                f"Results: {response.output_dir}"
            )
            if response.failed_documents > 0:
                logger.warning(f"Some documents failed: {response.failed_documents}/{response.total_documents}")
            return 0 if response.failed_documents == 0 else 2
        else:
            logger.error("✗ EXTRACTION FAILED")
            logger.error(f"✗ Error: {response.error_message}")
            if getattr(response, "extractions_saved", 0) > 0:
                logger.warning(
                    f"✗ Part of the batch was processed before failure/abort. "
                    f"Processed: {response.extractions_saved}/{response.total_documents}, "
                    f"Failed/Aborted: {response.failed_documents}"
                )
                return 2
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
    sys.exit(extract_command())


if __name__ == "__main__":
    main()
