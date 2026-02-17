"""CLI command for batch prediction.

This module provides the command-line interface for running predictions
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
from aee.domain.tasks import get_task
from aee.infrastructure.storage import (
    AgentRepository,
    DocumentRepository,
    PredictionRepository,
)

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for predict command."""
    parser = argparse.ArgumentParser(
        description="Run batch predictions on documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (default: from config)",
    )

    parser.add_argument(
        "--agent",
        type=Path,
        required=True,
        help="Path to trained agent JSON file",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file (required)",
    )

    parser.add_argument(
        "--enable-cache",
        action="store_true",
        default=False,
        help="Enable LLM response caching (default: disabled)",
    )

    return parser


def predict_command(argv: Optional[list] = None) -> int:
    """Execute the predict command.

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
        logger.info("Starting batch prediction")

        # Configure LLM cache (disabled by default for predictions)
        from aee.infrastructure.llm import create_lm

        create_lm(
            custom_settings.llm.student,
            enable_cache=args.enable_cache,
            enable_circuit_breaker=True,
        )

        # Validate agent path
        if not args.agent.exists():
            logger.error(f"Agent not found: {args.agent}")
            print(f"✗ Error: Agent not found: {args.agent}")
            return 1

        # Load task definition
        task_name = args.task if args.task else custom_settings.task.name
        task = get_task(task_name)
        logger.info(f"Task loaded: {task.name}")

        # Get all documents from parsed directory
        doc_repo = DocumentRepository(parsed_dir=custom_settings.paths.parsed_dir)
        document_ids = doc_repo.list_document_keys()
        logger.info(f"Found {len(document_ids)} documents to process")

        if not document_ids:
            logger.warning("No documents to process")
            print("⚠ No documents found to process")
            return 0

        # Use output directory from config
        output_dir = custom_settings.paths.predictions_dir

        # Log prediction settings
        logger.info("=" * 60)
        logger.info("PREDICTION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Task: {task_name}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Agent: {args.agent}")
        logger.info(f"Documents: {len(document_ids)}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"LLM cache: {'ENABLED' if args.enable_cache else 'DISABLED'}")
        logger.info("=" * 60)

        # Create dependencies
        doc_repo = DocumentRepository(parsed_dir=custom_settings.paths.parsed_dir)
        agent_repo = AgentRepository(agents_dir=custom_settings.paths.agents_dir)
        pred_repo = PredictionRepository()

        agent_manager = AgentManager(agent_repo=agent_repo)

        # Create use case
        use_case = BatchPredictionUseCase(
            agent_manager=agent_manager,
            document_repo=doc_repo,
            prediction_repo=pred_repo,
        )

        # Build request
        request = BatchPredictionRequest(
            task=task,
            agent_path=args.agent,
            document_ids=document_ids,
            output_dir=output_dir,
            batch_size=1,
        )

        # Execute prediction
        logger.info(f"Processing {len(document_ids)} documents...")
        print(f"Processing {len(document_ids)} documents...")

        response = use_case.execute(request)

        # Display results
        if response.success:
            logger.info("✓ PREDICTION COMPLETE")
            logger.info(
                f"✓ Processed: {response.predictions_saved}/{response.total_documents}"
            )
            logger.info(f"✓ Output directory: {response.output_dir}")

            print(f"\n✓ Success!")
            print(f"✓ Processed: {response.predictions_saved}/{response.total_documents}")
            print(f"✓ Failed: {response.failed_documents}")
            print(f"✓ Results saved to: {response.output_dir}")

            return 0 if response.failed_documents == 0 else 2

        else:
            logger.error("✗ PREDICTION FAILED")
            logger.error(f"✗ Error: {response.error_message}")
            print(f"\n✗ Prediction failed: {response.error_message}")
            return 1

    except KeyboardInterrupt:
        logger.warning("Prediction interrupted by user")
        print("\n\n⚠ Prediction interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1


def main():
    """Main entry point."""
    sys.exit(predict_command())


if __name__ == "__main__":
    main()
