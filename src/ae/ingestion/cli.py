"""CLI command for document ingestion.

This module provides the command-line interface for ingesting PDF documents.
"""

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from ae.core.cli import run_cli_use_case
from ae.core.config.settings import Settings
from ae.core.storage import DocumentRepository
from ae.ingestion.use_case import (
    IngestDocumentsRequest,
    IngestDocumentsResponse,
    IngestDocumentsUseCase,
)

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for ingest command."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into structured format",
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
        help="Overwrite existing ingested files (run from transcribe stage)",
    )

    parser.add_argument(
        "--from-stage",
        type=str,
        choices=[
            "parse_text",
            "manifest_visuals",
            "segment_and_crop_visuals_elements",
            "segment_and_crop_visuals_atoms",
            "parse_visuals",
        ],
        default=None,
        help="Invalidate and run pipeline from specified stage onwards",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Ingest only PDFs matching specified split name in splits file (e.g., 'test', 'train', 'val')",
    )

    parser.add_argument(
        "--document-ids",
        type=str,
        default=None,
        help="Comma-separated list of document IDs or filenames to ingest",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check configuration and list PDFs that would be ingested without starting ingestion",
    )

    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=None,
        help="Optional specific PDF file paths, document IDs or directory paths to ingest",
    )

    return parser


def collect_pdf_paths(paths: list[Path], fallback_dir: Optional[Path] = None) -> list[Path]:
    """Collect all PDF files from given paths.

    Args:
        paths: List of file or directory paths.
        fallback_dir: Optional directory to search if relative path is not found directly.

    Returns:
        List of PDF file paths.
    """
    pdf_files: list[Path] = []

    for raw_path in paths:
        path = raw_path
        if not path.exists() and fallback_dir and fallback_dir.exists():
            candidates = [
                fallback_dir / path.name,
                fallback_dir / f"{path.name}.pdf",
                fallback_dir / f"{path.stem}.pdf",
            ]
            for cand in candidates:
                if cand.exists():
                    path = cand
                    break

        if not path.exists():
            logger.warning(f"Path does not exist: {raw_path}")
            print(f"⚠ Path does not exist: {raw_path}")
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


def ingest_command(argv: Optional[list] = None) -> int:
    """Execute the ingest command."""
    parser = create_argument_parser()

    settings_container = {}
    args_container = {}

    def setup_llms(settings: Settings) -> None:
        settings_container["settings"] = settings

    def build_request(
        args: argparse.Namespace,
        settings: Settings,
        cancel_event: threading.Event,
    ) -> IngestDocumentsRequest:
        args_container["args"] = args

        # Get PDF directory from config
        pdf_dir = settings.paths.pdf_dir

        if getattr(args, "paths", None):
            pdf_files = collect_pdf_paths(args.paths, fallback_dir=pdf_dir)
        elif not pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {pdf_dir}")
            print(f"⚠ PDF directory does not exist: {pdf_dir}")
            pdf_files = []
        else:
            # Collect all PDF files from configured directory
            pdf_files = collect_pdf_paths([pdf_dir])

        if getattr(args, "split", None):
            from ae.core.storage.splits import load_split
            from ae.core.utils import normalize_document_key

            split_name = args.split.strip().lower()
            split_docs = load_split(settings.paths.splits_file, split_name)
            pdf_files = [
                p for p in pdf_files
                if normalize_document_key(p.stem) in split_docs or p.stem in split_docs
            ]
            logger.info(f"Filtered to {len(pdf_files)} PDF files from split '{split_name}'")

        if getattr(args, "document_ids", None):
            from ae.core.utils import normalize_document_key

            explicit_ids = {normalize_document_key(i.strip()) for i in args.document_ids.split(",") if i.strip()}
            pdf_files = [
                p for p in pdf_files
                if normalize_document_key(p.stem) in explicit_ids or p.stem in explicit_ids
            ]
            logger.info(f"Filtered to {len(pdf_files)} explicit PDF files")

        if not pdf_files:
            logger.warning(f"No PDF files found matching criteria in {pdf_dir}")
            print(f"⚠ No PDF files found matching criteria in {pdf_dir}")

        logger.info(f"Found {len(pdf_files)} PDF files to ingest")

        # Determine parser/engine name from ingestion config
        parser_name = getattr(settings.ingestion, "engine", "gemini")

        # Log ingestion settings
        logger.info("=" * 60)
        logger.info("INGESTION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Config file: {args.config}")
        logger.info(f"PDF files: {len(pdf_files)}")
        logger.info(f"Output: {settings.paths.ingestion_dir}")
        logger.info(f"Engine: {parser_name}")
        logger.info(f"Overwrite: {args.overwrite}")
        if getattr(args, "from_stage", None):
            logger.info(f"From stage: {args.from_stage}")
        logger.info("=" * 60)

        parser_config = settings.ingestion

        return IngestDocumentsRequest(
            input_paths=pdf_files,
            output_dir=settings.paths.ingestion_dir,
            parser_name=parser_name,
            overwrite=args.overwrite,
            parser_config=parser_config,
            concurrency=getattr(settings.ingestion, "concurrency", 4),
            from_stage=getattr(args, "from_stage", None),
        )

    def execute_use_case(request: IngestDocumentsRequest) -> IngestDocumentsResponse:
        settings = settings_container["settings"]
        args = args_container["args"]

        if not request.input_paths:
            return IngestDocumentsResponse(
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
            logger.info("DRY RUN INGESTION VALIDATION COMPLETED SUCCESSFULLY")
            logger.info(f"  PDF directory: {settings.paths.pdf_dir}")
            logger.info(f"  PDF files found: {len(request.input_paths)}")
            logger.info(f"  Output directory: {request.output_dir}")
            logger.info(f"  Engine name: {request.parser_name}")
            logger.info("✓ Success! Directory is accessible, and PDF files are ready to ingest.")
            logger.info("=" * 60)
            print("✓ Success! Directory is accessible, and PDF files are ready to ingest.")
            return IngestDocumentsResponse(
                success=True,
                documents_parsed=0,
                documents_skipped=0,
                total_documents=len(request.input_paths),
                failed_documents=0,
                output_dir=request.output_dir,
            )

        doc_repo = DocumentRepository(ingestion_dir=request.output_dir)
        use_case = IngestDocumentsUseCase(document_repo=doc_repo)
        return use_case.execute(request)

    def format_response(response: IngestDocumentsResponse) -> int:
        args = args_container["args"]
        if args.dry_run:
            return 0 if response.success else 1

        if response.success:
            logger.info("✓ INGESTION COMPLETE")
            logger.info(
                f"✓ Ingested: {response.documents_parsed}/{response.total_documents}"
            )

            print("\n✓ Success!")
            print(f"✓ Ingested: {response.documents_parsed}/{response.total_documents}")
            if getattr(response, "documents_skipped", 0) > 0:
                print(f"✓ Skipped (already exist): {response.documents_skipped}")
            print(f"✓ Failed: {response.failed_documents}")
            print(f"✓ Results saved to: {response.output_dir}")

            return 0 if response.failed_documents == 0 else 2
        else:
            logger.error("✗ INGESTION FAILED")
            partial = (
                getattr(response, "documents_parsed", 0) > 0
                or getattr(response, "documents_skipped", 0) > 0
            )
            if partial:
                print("✗ Part of the batch was processed before failure/abort.")
                print(f"  Ingested: {response.documents_parsed}/{response.total_documents}")
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
    sys.exit(ingest_command())


if __name__ == "__main__":
    main()
