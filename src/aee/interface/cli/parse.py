"""CLI command for document parsing.

This module provides the command-line interface for parsing PDF documents.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from aee import setup_logging
from aee.application.use_cases.parse_documents import (
    ParseDocumentsRequest,
    ParseDocumentsUseCase,
)
from aee.infrastructure.config.settings import Settings
from aee.infrastructure.storage import DocumentRepository

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for parse command."""
    parser = argparse.ArgumentParser(
        description="Parse PDF documents into structured format",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing parsed files",
    )

    return parser


def collect_pdf_paths(paths: list[Path]) -> list[Path]:
    """Collect all PDF files from given paths.

    Args:
        paths: List of file or directory paths.

    Returns:
        List of PDF file paths.
    """
    pdf_files: list[Path] = []

    for path in paths:
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            print(f"⚠ Path does not exist: {path}")
            continue

        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files.append(path)
        elif path.is_dir():
            # Recursively find PDFs in directory
            pdf_files.extend(path.rglob("*.pdf"))
            pdf_files.extend(path.rglob("*.PDF"))

    # Remove duplicates and sort
    pdf_files = sorted(set(pdf_files))

    return pdf_files


def parse_command(argv: Optional[list] = None) -> int:
    """Execute the parse command.

    Args:
        argv: Command-line arguments (None for sys.argv[1:]).

    Returns:
        Exit code:
            0 - Success (all documents parsed)
            1 - Failure (error during parsing)
            2 - Partial success (some documents failed)
            130 - Interrupted by user (Ctrl+C)
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
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        logger.error(f"Failed to load configuration: {e}")
        print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
        return 1

    # Setup logging with custom settings
    setup_logging(custom_settings)

    try:
        logger.info("Starting document parsing")

        # Get PDF directory from config
        pdf_dir = custom_settings.paths.pdf_dir

        if not pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {pdf_dir}")
            print(f"⚠ PDF directory does not exist: {pdf_dir}")
            return 0

        # Collect all PDF files from configured directory
        pdf_files = collect_pdf_paths([pdf_dir])

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            print(f"⚠ No PDF files found in {pdf_dir}")
            return 0

        logger.info(f"Found {len(pdf_files)} PDF files to parse")
        print(f"Found {len(pdf_files)} PDF files to parse")

        # Use output directory from config
        output_dir = custom_settings.paths.parsed_dir

        # Log parsing settings
        logger.info("=" * 60)
        logger.info("PARSING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Config file: {args.config}")
        logger.info(f"PDF files: {len(pdf_files)}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Parser: {custom_settings.parsing.parser}")
        logger.info(f"Overwrite: {args.overwrite}")
        logger.info("=" * 60)

        # Create dependencies
        doc_repo = DocumentRepository(parsed_dir=output_dir)

        # Create use case
        use_case = ParseDocumentsUseCase(document_repo=doc_repo)

        # Build request
        request = ParseDocumentsRequest(
            input_paths=pdf_files,
            output_dir=output_dir,
            parser_name=custom_settings.parsing.parser,
            overwrite=args.overwrite,
            parser_config=custom_settings.parsing.gemini
            if custom_settings.parsing.parser == "gemini"
            else custom_settings.parsing.marker,
        )

        # Execute parsing
        response = use_case.execute(request)

        # Display results
        if response.success:
            logger.info("✓ PARSING COMPLETE")
            logger.info(
                f"✓ Parsed: {response.documents_parsed}/{response.total_documents}"
            )

            print("\n✓ Success!")
            print(f"✓ Parsed: {response.documents_parsed}/{response.total_documents}")
            print(f"✓ Failed: {response.failed_documents}")
            print(f"✓ Results saved to: {response.output_dir}")

            return 0 if response.failed_documents == 0 else 2

        else:
            logger.error("✗ PARSING FAILED")
            logger.error(f"✗ Error: {response.error_message}")
            print(f"\n✗ Parsing failed: {response.error_message}")
            return 1

    except KeyboardInterrupt:
        logger.warning("Parsing interrupted by user")
        print("\n\n⚠ Parsing interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Parsing error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1


def main():
    """Main entry point."""
    sys.exit(parse_command())


if __name__ == "__main__":
    main()
