"""CLI command for batch extraction.

This module provides the command-line interface for running extraction
on documents using trained agents.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


from aee import setup_logging
from aee.application.services import AgentManager
from aee.application.use_cases import BatchPredictionRequest, BatchPredictionUseCase
from aee.domain.tasks import load_task_with_instruction
from aee.infrastructure.config.settings import Settings
from aee.infrastructure.llm.history_logger import save_extraction_history
from aee.infrastructure.storage import (
    AgentRepository,
    DocumentRepository,
    ExtractionRepository,
)

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for extract command.

    Returns:
        ArgumentParser configured with required --config and --agent arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run batch extraction on documents",
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
        "--agent",
        type=Path,
        required=True,
        help="Path to trained agent JSON file",
    )

    return parser


def extract_command(argv: Optional[list] = None) -> int:
    """Execute batch extraction on documents using a trained agent.

    This command loads a trained agent, processes all parsed documents,
    and saves extraction results to JSON files.

    Args:
        argv: Command-line arguments (None for sys.argv[1:]).
            Required arguments: --config, --agent

    Returns:
        Exit code:
            - 0: Success (all documents processed)
            - 1: Failure (configuration error, agent not found, extraction failed)
            - 130: Interrupted by user (Ctrl+C)

    Example:
        ```python
        # Run extraction
        exit_code = extract_command([
            "--config", "config/systems/dev.yaml",
            "--agent", "data/agents/nanozymes_v1.json"
        ])
        ```
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

    student_lm = None

    try:
        logger.info("Starting batch prediction")

        # Configure LLM and DSPy (setup_student calls dspy.settings.configure internally)
        from aee.infrastructure.llm import setup_student

        student_lm = setup_student(
            custom_settings,
            enable_cache=custom_settings.extraction.enable_cache,
        )

        # Resolve agent path relative to project root if not absolute
        agent_path = Path(args.agent)
        if not agent_path.is_absolute():
            agent_path = Path.cwd() / agent_path
        if not agent_path.exists():
            logger.error(f"Agent not found: {agent_path}")
            return 1

        # Load task definition
        task_name = custom_settings.task.name
        task, instruction_metadata = load_task_with_instruction(task_name, custom_settings)
        logger.info(f"Task loaded: {task['config'].name}")

        # Validate task has signature for agent reconstruction
        if task.get("signature") is None:
            logger.error("Task signature not found - required for agent reconstruction")
            return 1

        # Use output directory from config
        output_dir = custom_settings.paths.extractions_dir

        # Create dependencies
        doc_repo = DocumentRepository(parsed_dir=custom_settings.paths.parsed_dir)
        agent_repo = AgentRepository(agents_dir=custom_settings.paths.agents_dir)
        pred_repo = ExtractionRepository()

        # Check if parsed directory exists
        if not custom_settings.paths.parsed_dir.exists():
            logger.warning(f"Parsed directory does not exist: {custom_settings.paths.parsed_dir}")
            return 0

        # Get all documents from parsed directory
        document_ids = doc_repo.list_document_keys()
        logger.info(f"Found {len(document_ids)} documents to process")

        if not document_ids:
            logger.warning("No documents to process")
            return 0

        # Log extraction settings
        logger.info("=" * 60)
        logger.info("EXTRACTION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Task: {task_name}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Agent: {args.agent}")
        logger.info(f"Documents: {len(document_ids)}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"LLM cache: {'ENABLED' if custom_settings.extraction.enable_cache else 'DISABLED'}")
        logger.info("=" * 60)

        agent_manager = AgentManager(agent_repo=agent_repo)

        # Create use case
        use_case = BatchPredictionUseCase(
            agent_manager=agent_manager,
            document_repo=doc_repo,
            extraction_repo=pred_repo,
        )

        # Build request
        request = BatchPredictionRequest(
            task=task["config"],
            task_dict=task,
            agent_path=agent_path,
            document_ids=document_ids,
            output_dir=output_dir,
        )

        # Execute extraction
        logger.info(f"Processing {len(document_ids)} documents...")

        response = use_case.execute(request)

        # Display results
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

            return 0 if response.failed_documents == 0 else 1

        else:
            logger.error("✗ EXTRACTION FAILED")
            logger.error(f"✗ Error: {response.error_message}")
            return 1

    except KeyboardInterrupt:
        logger.warning("Extraction interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        return 1

    finally:
        # Save LLM history (always, even on error/interrupt)
        if custom_settings.extraction.save_llm_history:
            if student_lm is not None:
                history_dir = Path(custom_settings.extraction.llm_history_dir)
                save_extraction_history(student_lm, history_dir)


def main():
    """Main entry point."""
    sys.exit(extract_command())


if __name__ == "__main__":
    main()
