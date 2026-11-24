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
class IngestDocumentsRequest:
    """Request for document ingestion.

    Attributes:
        input_paths: List of PDF file paths to ingest.
        output_dir: Directory to save ingested documents.
        parser_name: Name of parser/ingester to use (e.g., "gemini").
        overwrite: Whether to overwrite existing files.
        parser_config: Optional configuration for the parser/ingester.
        concurrency: Number of concurrent threads to use.
        from_stage: Optional stage name to invalidate and run from.
    """

    input_paths: List[Path]
    output_dir: Path
    parser_name: str = "gemini"
    overwrite: bool = False
    parser_config: Optional[Any] = None
    concurrency: int = 4
    from_stage: Optional[str] = None


@dataclass
class IngestDocumentsResponse:
    """Response from document ingestion.

    Attributes:
        success: Whether ingestion succeeded overall.
        documents_parsed: Number of documents successfully ingested.
        documents_skipped: Number of documents skipped.
        total_documents: Total documents attempted.
        failed_documents: Number of failed documents.
        output_dir: Directory where documents were saved.
        error_message: Error message if failed.
        aborted: Whether ingestion was aborted due to a fatal error/interruption.
    """

    success: bool
    documents_parsed: int = 0
    documents_skipped: int = 0
    total_documents: int = 0
    failed_documents: int = 0
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None
    aborted: bool = False


class IngestDocumentsUseCase:
    """Use case for ingesting PDF documents into structured format.

    This use case handles:
    1. Loading ingester/parser
    2. Ingesting documents
    3. Saving results
    """

    def __init__(self, document_repo: DocumentRepository):
        """Initialize the use case.

        Args:
            document_repo: Repository for saving documents.
        """
        self.document_repo = document_repo
        logger.debug("Initialized IngestDocumentsUseCase")

    def execute(self, request: IngestDocumentsRequest) -> IngestDocumentsResponse:
        """Execute document ingestion.

        Args:
            request: Ingestion request.

        Returns:
            Response with results.
        """
        try:
            logger.info(
                f"Starting document ingestion: {len(request.input_paths)} documents"
            )

            # Create output directory
            request.output_dir.mkdir(parents=True, exist_ok=True)

            # Configure from_stage if requested
            if (
                request.from_stage
                and request.parser_config
                and hasattr(request.parser_config, "from_stage")
            ):
                request.parser_config.from_stage = request.from_stage

            # Ensure parser writes artifacts into request.output_dir
            if request.parser_config is not None and hasattr(request.parser_config, "ingestion_dir"):
                request.parser_config.ingestion_dir = request.output_dir

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

            use_batch = (
                getattr(
                    getattr(request.parser_config, "execution", None),
                    "use_batch_api",
                    False,
                )
                is True
            )
            if use_batch and hasattr(parser, "parse_batch"):
                pending_paths: list[Path] = []
                for pdf_path in request.input_paths:
                    output_path = request.output_dir / pdf_path.stem / f"{pdf_path.stem}.md"
                    if output_path.exists() and not request.overwrite and not request.from_stage:
                        stats["skipped"] += 1
                    else:
                        pending_paths.append(pdf_path)

                if pending_paths:
                    batch_outputs = parser.parse_batch(pending_paths)
                    for pdf_path in pending_paths:
                        output_path = request.output_dir / pdf_path.stem / f"{pdf_path.stem}.md"
                        hybrid_text = batch_outputs.get(pdf_path.resolve()) or batch_outputs.get(pdf_path)
                        if hybrid_text:
                            self.document_repo.save(hybrid_text, output_path)
                            stats["success"] += 1
                        else:
                            stats["failed"] += 1

                return IngestDocumentsResponse(
                    success=stats["failed"] == 0,
                    documents_parsed=stats["success"],
                    documents_skipped=stats["skipped"],
                    total_documents=stats["total"],
                    failed_documents=stats["failed"],
                    output_dir=request.output_dir,
                )

            concurrency = getattr(request, "concurrency", 4)
            if concurrency < 1:
                concurrency = 4

            logger.info(f"Concurrency set to: {concurrency}")

            def parse_worker(i: int, pdf_path: Path) -> Tuple[str, str, Optional[Exception]]:
                try:
                    # Generate output path
                    output_path = request.output_dir / pdf_path.stem / f"{pdf_path.stem}.md"

                    # Check if already exists
                    if output_path.exists() and not request.overwrite and not request.from_stage:
                        logger.debug(f"Skipping existing: {pdf_path.name}")
                        return "skipped", pdf_path.name, None

                    # Only delay if concurrency is 1 (sequential)
                    if (
                        concurrency == 1
                        and i > 0
                        and request.parser_config
                        and hasattr(request.parser_config, "request_delay")
                    ):
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

                return IngestDocumentsResponse(
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
                f"Ingestion complete: {stats['success']}/{stats['total']} succeeded "
                f"({stats['skipped']} skipped, {stats['failed']} failed)"
            )

            return IngestDocumentsResponse(
                success=True,
                documents_parsed=stats["success"],
                documents_skipped=stats["skipped"],
                total_documents=stats["total"],
                failed_documents=stats["failed"],
                output_dir=request.output_dir,
            )

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)

            return IngestDocumentsResponse(
                success=False,
                error_message=str(e),
            )
