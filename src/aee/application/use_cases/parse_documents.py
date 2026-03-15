"""Parse documents use case.

This use case handles parsing PDF documents into structured format
for downstream processing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from aee.infrastructure.parsers import get_parser
from aee.infrastructure.storage import DocumentRepository

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
    """

    input_paths: List[Path]
    output_dir: Path
    parser_name: str = "marker"
    overwrite: bool = False
    parser_config: Optional[Any] = None


@dataclass
class ParseDocumentsResponse:
    """Response from document parsing.

    Attributes:
        success: Whether parsing succeeded overall.
        documents_parsed: Number of documents successfully parsed.
        total_documents: Total documents attempted.
        failed_documents: Number of failed documents.
        output_dir: Directory where documents were saved.
        error_message: Error message if failed.
    """

    success: bool
    documents_parsed: int = 0
    total_documents: int = 0
    failed_documents: int = 0
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None


class ParseDocumentsUseCase:
    """Use case for parsing PDF documents.

    This use case handles:
    1. Loading parser
    2. Parsing documents
    3. Saving results

    Example:
        ```python
        use_case = ParseDocumentsUseCase(document_repo=doc_repo)

        request = ParseDocumentsRequest(
            input_paths=[Path("doc1.pdf"), Path("doc2.pdf")],
            output_dir=Path("data/parsed"),
            parser_name="marker",
        )

        response = use_case.execute(request)
        ```
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

            for pdf_path in request.input_paths:
                try:
                    # Generate output path
                    output_path = request.output_dir / f"{pdf_path.stem}.md"

                    # Check if already exists
                    if output_path.exists() and not request.overwrite:
                        logger.debug(f"Skipping existing: {pdf_path.name}")
                        stats["skipped"] += 1
                        stats["success"] += 1  # Count as success
                        continue

                    # Parse document
                    logger.info(f"Parsing: {pdf_path.name}")
                    hybrid_text = parser.parse(pdf_path)

                    # Save document
                    self.document_repo.save(hybrid_text, output_path)

                    stats["success"] += 1
                    logger.info(f"✓ Parsed: {pdf_path.name}")

                except Exception as e:
                    stats["failed"] += 1
                    logger.error(f"✗ Failed to parse {pdf_path.name}: {e}")
                    continue

            # Log summary
            logger.info(
                f"Parsing complete: {stats['success']}/{stats['total']} succeeded "
                f"({stats['skipped']} skipped, {stats['failed']} failed)"
            )

            return ParseDocumentsResponse(
                success=True,
                documents_parsed=stats["success"],
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
