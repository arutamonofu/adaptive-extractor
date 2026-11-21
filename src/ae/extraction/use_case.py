"""Batch extraction use case.

This use case handles running extraction on multiple documents
using a trained agent.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ae.core.llm.circuit_breaker import CircuitBreakerError
from ae.core.storage import DocumentRepository, ExtractionRepository
from ae.extraction.manager import AgentManager

logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"<!--\s*document_id:\s*([^\s>]+)\s*-->")


@dataclass
class BatchExtractionRequest:
    """Request for batch extraction.

    Attributes:
        task: Task definition (TaskBundle).
        agent_path: Path to trained agent.
        document_ids: List of document IDs to process.
        output_dir: Directory to save extractions.
    """

    task: Any
    agent_path: Path
    document_ids: List[str]
    output_dir: Path
    task_dict: Optional[Dict[str, Any]] = None
    client: Optional[Any] = None
    history_dir: Optional[Path] = None
    save_llm_history: Optional[bool] = None


@dataclass
class BatchExtractionResponse:
    """Response from batch extraction.

    Attributes:
        success: Whether extraction succeeded.
        extractions_saved: Number of extractions saved.
        total_documents: Total documents processed.
        failed_documents: Number of failed documents.
        output_dir: Directory where extractions were saved.
        error_message: Error message if failed.
        aborted: Whether the extraction was aborted due to a fatal error/interruption.
    """

    success: bool
    extractions_saved: int = 0
    total_documents: int = 0
    failed_documents: int = 0
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None
    aborted: bool = False


class BatchExtractionUseCase:
    """Use case for batch extraction.

    This use case handles:
    1. Loading the trained agent
    2. Loading documents
    3. Running extractions
    4. Saving results
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
        logger.debug("Initialized BatchExtractionUseCase")

    def execute(self, request: BatchExtractionRequest) -> BatchExtractionResponse:
        """Execute batch extraction.

        Args:
            request: Extraction request.

        Returns:
            Response with results.
        """
        history_started_here = False
        try:
            logger.info(
                f"Starting batch extraction for {len(request.document_ids)} documents"
            )

            if request.client is not None and request.history_dir is not None:
                save_history_enabled = (
                    request.save_llm_history
                    if request.save_llm_history is not None
                    else bool(getattr(request.client, "save_llm_history", True))
                )
                if (
                    save_history_enabled
                    and hasattr(request.client, "start_history_session")
                ):
                    recorder = getattr(request.client, "history_recorder", None)
                    was_streaming = bool(recorder and recorder.is_streaming)
                    request.client.start_history_session(request.history_dir)
                    history_started_here = not was_streaming

            # Load agent as callable object (reconstruct from saved state)
            agent = self.agent_manager.load_agent_as_object(
                request.agent_path,
                task=request.task,
                task_dict=request.task_dict,
                client=request.client,
            )
            if (
                getattr(agent, "_ingestion_dir", None) is None
                and self.document_repo.ingestion_dir is not None
            ):
                agent._ingestion_dir = self.document_repo.ingestion_dir

            # Create output directory
            request.output_dir.mkdir(parents=True, exist_ok=True)

            # Log starting extraction
            logger.info("Starting extraction")

            # Run extractions
            stats = {"success": 0, "failed": 0, "total": len(request.document_ids)}
            aborted = False

            for idx, doc_id in enumerate(request.document_ids):
                try:
                    # Get document text
                    doc_text = self.document_repo.get(doc_id)
                    if doc_text is None:
                        logger.warning(f"Document not found: {doc_id}")
                        stats["failed"] += 1
                        continue

                    if not _DOC_ID_RE.search(doc_text):
                        doc_text = f"<!-- document_id: {doc_id} -->\n\n{doc_text}"

                    # Log progress before extraction
                    logger.info(
                        f"[{idx + 1}/{len(request.document_ids)}] Processing document: {doc_id}"
                    )

                    # Run extraction
                    prediction = self._run_extraction(agent, doc_text)

                    # Save extraction
                    recs = getattr(prediction, "records", getattr(prediction, "experiments", []))
                    output_path = request.output_dir / f"{doc_id}.json"
                    self.extraction_repo.save(
                        extractions=recs,
                        output_path=output_path,
                        document_metadata={
                            "document_id": doc_id,
                        },
                    )

                    stats["success"] += 1
                    logger.debug(
                        f"[{idx + 1}/{len(request.document_ids)}] Completed {doc_id}: "
                        f"{len(recs)} records"
                    )

                except CircuitBreakerError as e:
                    logger.error(
                        f"[{idx + 1}/{len(request.document_ids)}] Circuit breaker OPEN for {doc_id}: "
                        f"{e}. Stopping batch processing."
                    )
                    stats["failed"] += len(request.document_ids) - idx
                    aborted = True
                    break
                except Exception as e:
                    logger.error(
                        f"[{idx + 1}/{len(request.document_ids)}] Failed to process {doc_id}: {e}"
                    )
                    stats["failed"] += 1
                    continue

            if aborted:
                return BatchExtractionResponse(
                    success=False,
                    aborted=True,
                    extractions_saved=stats["success"],
                    total_documents=stats["total"],
                    failed_documents=stats["failed"],
                    output_dir=request.output_dir,
                    error_message="Batch processing was aborted due to CircuitBreakerError.",
                )

            logger.info(
                f"Batch extraction complete: {stats['success']}/{stats['total']} succeeded"
            )

            return BatchExtractionResponse(
                success=True,
                extractions_saved=stats["success"],
                total_documents=stats["total"],
                failed_documents=stats["failed"],
                output_dir=request.output_dir,
            )

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}", exc_info=True)

            return BatchExtractionResponse(
                success=False,
                error_message=str(e),
            )
        finally:
            if (
                history_started_here
                and request.client is not None
                and request.history_dir is not None
                and hasattr(request.client, "save_history")
            ):
                try:
                    request.client.save_history(request.history_dir)
                except Exception as hist_exc:
                    logger.warning(f"Failed to save LLM history: {hist_exc}")

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
