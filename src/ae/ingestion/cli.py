"""CLI command for document parsing.

This module provides the command-line interface for parsing PDF documents.
"""

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from ae.core.config.settings import Settings
from ae.core.storage import DocumentRepository
from ae.ingestion.use_case import (
    ParseDocumentsRequest,
    ParseDocumentsResponse,
    ParseDocumentsUseCase,
)
from ae.core.cli import run_cli_use_case

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
        help="Path to configuration directory (defaults to root config/ directory)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parsed files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check configuration and list PDFs that would be parsed without starting parsing",
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
    """Execute the parse command."""
    parser = create_argument_parser()

    settings_container = {}
    args_container = {}

    def setup_llms(settings: Settings) -> None:
        settings_container["settings"] = settings
        # Configure visual extractor LLM for parsing if chart extraction is enabled
        if settings.parsing.chart_extraction.enabled:
            from ae.core.llm import setup_visual_extractor
            setup_visual_extractor(settings.llm.visual_extractor, settings.circuit_breaker)

    def build_request(
        args: argparse.Namespace,
        settings: Settings,
        cancel_event: threading.Event,
    ) -> ParseDocumentsRequest:
        args_container["args"] = args

        # Get PDF directory from config
        pdf_dir = settings.paths.pdf_dir

        if not pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {pdf_dir}")
            print(f"⚠ PDF directory does not exist: {pdf_dir}")
            pdf_files = []
        else:
            # Collect all PDF files from configured directory
            pdf_files = collect_pdf_paths([pdf_dir])

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            print(f"⚠ No PDF files found in {pdf_dir}")

        logger.info(f"Found {len(pdf_files)} PDF files to parse")

        # Log parsing settings
        logger.info("=" * 60)
        logger.info("PARSING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Config file: {args.config}")
        logger.info(f"PDF files: {len(pdf_files)}")
        logger.info(f"Output: {settings.paths.ingestion_dir}")
        logger.info(f"Parser: mineru")
        logger.info(f"Overwrite: {args.overwrite}")
        logger.info("=" * 60)

        parser_name = "mineru"
        parser_config = settings.parsing

        return ParseDocumentsRequest(
            input_paths=pdf_files,
            output_dir=settings.paths.ingestion_dir,
            parser_name=parser_name,
            overwrite=args.overwrite,
            parser_config=parser_config,
            concurrency=getattr(settings.parsing, "concurrency", 4),
        )

    def execute_use_case(request: ParseDocumentsRequest) -> ParseDocumentsResponse:
        settings = settings_container["settings"]
        args = args_container["args"]

        if not request.input_paths:
            return ParseDocumentsResponse(
                success=True,
                documents_parsed=0,
                documents_skipped=0,
                total_documents=0,
                failed_documents=0,
                output_dir=request.output_dir,
            )

        if args.dry_run:
            logger.info("Running dry-run validation checks...")
            logger.info("=" * 60)
            logger.info("DRY RUN PARSING VALIDATION COMPLETED SUCCESSFULLY")
            logger.info(f"  PDF directory: {settings.paths.pdf_dir}")
            logger.info(f"  PDF files found: {len(request.input_paths)}")
            logger.info(f"  Output directory: {request.output_dir}")
            logger.info(f"  Parser name: {request.parser_name}")
            logger.info("✓ Success! Directory is accessible, and PDF files are ready to parse.")
            logger.info("=" * 60)
            print("✓ Success! Directory is accessible, and PDF files are ready to parse.")
            return ParseDocumentsResponse(
                success=True,
                documents_parsed=0,
                documents_skipped=0,
                total_documents=len(request.input_paths),
                failed_documents=0,
                output_dir=request.output_dir,
            )

        doc_repo = DocumentRepository(ingestion_dir=request.output_dir)
        use_case = ParseDocumentsUseCase(document_repo=doc_repo)
        return use_case.execute(request)

    def format_response(response: ParseDocumentsResponse) -> int:
        args = args_container["args"]
        if args.dry_run:
            return 0 if response.success else 1

        if response.success:
            logger.info("✓ PARSING COMPLETE")
            logger.info(
                f"✓ Parsed: {response.documents_parsed}/{response.total_documents}"
            )

            print("\n✓ Success!")
            print(f"✓ Parsed: {response.documents_parsed}/{response.total_documents}")
            if getattr(response, "documents_skipped", 0) > 0:
                print(f"✓ Skipped (already exist): {response.documents_skipped}")
            print(f"✓ Failed: {response.failed_documents}")
            print(f"✓ Results saved to: {response.output_dir}")

            return 0 if response.failed_documents == 0 else 2
        else:
            logger.error("✗ PARSING FAILED")
            logger.error(f"✗ Error: {response.error_message}")
            print(f"\n✗ Parsing failed: {response.error_message}")
            if getattr(response, "documents_parsed", 0) > 0 or getattr(response, "documents_skipped", 0) > 0:
                print(f"✗ Part of the batch was processed before failure/abort.")
                print(f"  Parsed: {response.documents_parsed}/{response.total_documents}")
                if getattr(response, "documents_skipped", 0) > 0:
                    print(f"  Skipped (already exist): {response.documents_skipped}")
                print(f"  Failed/Aborted: {response.failed_documents}")
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
    sys.exit(parse_command())


if __name__ == "__main__":
    main()
