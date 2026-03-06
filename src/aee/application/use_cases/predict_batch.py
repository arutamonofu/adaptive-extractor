"""Batch extraction use case.

This use case handles running extraction on multiple documents
using a trained agent.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from aee.application.services import AgentManager
from aee.domain.tasks import TaskConfig
from aee.infrastructure.storage import DocumentRepository, ExtractionRepository

logger = logging.getLogger(__name__)


@dataclass
class BatchPredictionRequest:
    """Request for batch prediction.

    Attributes:
        task: Task definition.
        agent_path: Path to trained agent.
        document_ids: List of document IDs to process.
        output_dir: Directory to save extractions.
        batch_size: Optional batch size for processing.
    """

    task: TaskConfig
    agent_path: Path
    document_ids: List[str]
    output_dir: Path
    batch_size: Optional[int] = None


@dataclass
class BatchPredictionResponse:
    """Response from batch extraction.

    Attributes:
        success: Whether extraction succeeded.
        extractions_saved: Number of extractions saved.
        total_documents: Total documents processed.
        failed_documents: Number of failed documents.
        output_dir: Directory where extractions were saved.
        error_message: Error message if failed.
    """

    success: bool
    extractions_saved: int = 0
    total_documents: int = 0
    failed_documents: int = 0
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None


class BatchPredictionUseCase:
    """Use case for batch extraction.

    This use case handles:
    1. Loading the trained agent
    2. Loading documents
    3. Running extractions
    4. Saving results

    Example:
        ```python
        use_case = BatchPredictionUseCase(
            agent_manager=manager,
            document_repo=doc_repo,
            extraction_repo=ext_repo,
        )

        request = BatchPredictionRequest(
            task=nanozyme_task,
            agent_path=Path("data/agents/agent.json"),
            document_ids=["doc1", "doc2"],
            output_dir=Path("data/extractions"),
        )

        response = use_case.execute(request)
        ```
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        document_repo: DocumentRepository,
        extraction_repo: ExtractionRepository,
    ):
        """Initialize the use case.

        Args:
            agent_manager: Service for managing agents.
            document_repo: Repository for loading documents.
            extraction_repo: Repository for saving extractions.
        """
        self.agent_manager = agent_manager
        self.document_repo = document_repo
        self.extraction_repo = extraction_repo
        logger.debug("Initialized BatchPredictionUseCase")

    def execute(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Execute batch extraction.

        Args:
            request: Extraction request.

        Returns:
            Response with results.
        """
        try:
            logger.info(
                f"Starting batch extraction for {len(request.document_ids)} documents"
            )

            # Load agent
            agent = self.agent_manager.load_agent(request.agent_path)

            # Load documents
            documents = self.document_repo.load_all()

            # Create output directory
            request.output_dir.mkdir(parents=True, exist_ok=True)

            # Run extractions
            stats = {"success": 0, "failed": 0, "total": len(request.document_ids)}

            for doc_id in request.document_ids:
                try:
                    # Get document text
                    doc_text = documents.get(doc_id)
                    if doc_text is None:
                        logger.warning(f"Document not found: {doc_id}")
                        stats["failed"] += 1
                        continue

                    # Run extraction
                    prediction = self._run_extraction(agent, doc_text)

                    # Save extraction
                    output_path = request.output_dir / f"{doc_id}_result.json"
                    self.extraction_repo.save(
                        extractions=prediction.experiments,
                        output_path=output_path,
                        document_metadata={
                            "document_id": doc_id,
                        },
                    )

                    stats["success"] += 1
                    logger.debug(f"Processed {doc_id}: {len(prediction.experiments)} experiments")

                except Exception as e:
                    logger.error(f"Failed to process {doc_id}: {e}")
                    stats["failed"] += 1
                    continue

            logger.info(
                f"Batch extraction complete: {stats['success']}/{stats['total']} succeeded"
            )

            return BatchPredictionResponse(
                success=True,
                extractions_saved=stats["success"],
                total_documents=stats["total"],
                failed_documents=stats["failed"],
                output_dir=request.output_dir,
            )

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}", exc_info=True)

            return BatchPredictionResponse(
                success=False,
                error_message=str(e),
            )

    def _run_extraction(self, agent: Any, document_text: str) -> Any:
        """Run extraction on a single document.

        Args:
            agent: Trained agent.
            document_text: Document text to process.

        Returns:
            Extraction result.
        """
        # Call agent with document text
        result = agent(document_text=document_text)

        # Return extracted data
        return result.extracted_data if hasattr(result, "extracted_data") else result
