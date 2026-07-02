import concurrent.futures
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ae.core.llm.circuit_breaker import CircuitBreakerError
from ae.core.storage import DocumentRepository
from ae.ingestion.parsers import get_parser

logger = logging.getLogger(__name__)


@dataclass
class ParseDocumentsRequest:
    """Request for document parsing.

    Attributes:
        input_paths: List of PDF file paths to parse.
        output_dir: Directory to save parsed documents.
        parser_name: Name of parser to use (e.g., "marker", "gemini").
        overwrite: Whether to overwrite existing parsed files.
        parser_config: Optional configuration for the parser.
        concurrency: Number of concurrent threads to use for parsing.
    """

    input_paths: List[Path]
    output_dir: Path
    parser_name: str = "marker"
    overwrite: bool = False
    parser_config: Optional[Any] = None
    concurrency: int = 4


@dataclass
class ParseDocumentsResponse:
    """Response from document parsing.

    Attributes:
        success: Whether parsing succeeded overall.
        documents_parsed: Number of documents successfully parsed.
        documents_skipped: Number of documents skipped.
        total_documents: Total documents attempted.
        failed_documents: Number of failed documents.
        output_dir: Directory where documents were saved.
        error_message: Error message if failed.
        aborted: Whether the parsing was aborted due to a fatal error/interruption.
    """

    success: bool
    documents_parsed: int = 0
    documents_skipped: int = 0
    total_documents: int = 0
    failed_documents: int = 0
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None
    aborted: bool = False


class ParseDocumentsUseCase:
    """Use case for parsing PDF documents.

    This use case handles:
    1. Loading parser
    2. Parsing documents
    3. Saving results
    """

    def __init__(self, document_repo: DocumentRepository):
        """Initialize the use case.

        Args:
            document_repo: Repository for saving documents.
        """
        self.document_repo = document_repo
        logger.debug("Initialized ParseDocumentsUseCase")

    def execute(self, request: ParseDocumentsRequest) -> ParseDocumentsResponse:
        """Execute document parsing.

        Args:
            request: Parsing request.

        Returns:
            Response with results.
        """
        try:
            logger.info(
                f"Starting document parsing: {len(request.input_paths)} documents"
            )

            # Create output directory
            request.output_dir.mkdir(parents=True, exist_ok=True)

            # Get parser
            parser = get_parser(request.parser_name, request.parser_config)
            logger.info(f"Using parser: {request.parser_name}")

            # Parse documents
            stats = {
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "total": len(request.input_paths),
            }

            concurrency = getattr(request, "concurrency", 4)
            if concurrency < 1:
                concurrency = 4

            logger.info(f"Concurrency set to: {concurrency}")

            def parse_worker(i: int, pdf_path: Path) -> Tuple[str, str, Optional[Exception]]:
                try:
                    # Generate output path
                    output_path = request.output_dir / f"{pdf_path.stem}.md"

                    # Check if already exists
                    if output_path.exists() and not request.overwrite:
                        logger.debug(f"Skipping existing: {pdf_path.name}")
                        return "skipped", pdf_path.name, None

                    # Only delay if concurrency is 1 (sequential), otherwise provider rate limiter synchronizes it
                    if concurrency == 1 and i > 0 and request.parser_config and hasattr(request.parser_config, "request_delay"):
                        time.sleep(request.parser_config.request_delay)

                    # Parse document
                    logger.info(f"Parsing: {pdf_path.name}")
                    hybrid_text = parser.parse(pdf_path)

                    # Save document
                    self.document_repo.save(hybrid_text, output_path)
                    logger.info(f"✓ Parsed: {pdf_path.name}")
                    return "success", pdf_path.name, None

                except Exception as e:
                    logger.error(f"✗ Failed to parse {pdf_path.name}: {e}")
                    return "failed", pdf_path.name, e

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit tasks
                futures = {
                    executor.submit(parse_worker, i, pdf_path): pdf_path
                    for i, pdf_path in enumerate(request.input_paths)
                }

                aborted = False
                for future in concurrent.futures.as_completed(futures):
                    try:
                        status, file_name, exc = future.result()
                        if status == "success":
                            stats["success"] += 1
                        elif status == "skipped":
                            stats["skipped"] += 1
                        else:
                            stats["failed"] += 1
                            if exc and isinstance(exc, CircuitBreakerError):
                                aborted = True
                                raise exc
                    except Exception as e:
                        pdf_path = futures[future]
                        stats["failed"] += 1
                        logger.error(f"✗ Future threw exception for {pdf_path.name}: {e}")
                        if isinstance(e, CircuitBreakerError):
                            aborted = True
                            for f in futures:
                                f.cancel()
                            break

            if aborted:
                processed_so_far = stats["success"] + stats["skipped"] + stats["failed"]
                remaining = stats["total"] - processed_so_far
                if remaining > 0:
                    stats["failed"] += remaining

                return ParseDocumentsResponse(
                    success=False,
                    aborted=True,
                    documents_parsed=stats["success"],
                    documents_skipped=stats["skipped"],
                    total_documents=stats["total"],
                    failed_documents=stats["failed"],
                    output_dir=request.output_dir,
                    error_message="Batch processing was aborted due to CircuitBreakerError.",
                )

            # Log summary
            logger.info(
                f"Parsing complete: {stats['success']}/{stats['total']} succeeded "
                f"({stats['skipped']} skipped, {stats['failed']} failed)"
            )

            return ParseDocumentsResponse(
                success=True,
                documents_parsed=stats["success"],
                documents_skipped=stats["skipped"],
                total_documents=stats["total"],
                failed_documents=stats["failed"],
                output_dir=request.output_dir,
            )

        except Exception as e:
            logger.error(f"Document parsing failed: {e}", exc_info=True)

            return ParseDocumentsResponse(
                success=False,
                error_message=str(e),
            )
