"""CLI command for batch extraction.

This module provides the command-line interface for running extraction
on documents using trained agents.
"""

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from ae.core.config.settings import Settings
from ae.core.storage import (
    AgentRepository,
    DocumentRepository,
    ExtractionRepository,
)
from ae.extraction.manager import AgentManager
from ae.extraction.use_case import (
    BatchExtractionRequest,
    BatchExtractionResponse,
    BatchExtractionUseCase,
)
from ae.core.cli import run_cli_use_case

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for extract command.

    Returns:
        ArgumentParser configured with --config, --agent, and --dry-run arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run batch extraction on documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration directory (defaults to root config/ directory)",
    )

    parser.add_argument(
        "--agent",
        type=Path,
        required=True,
        help="Path to trained agent JSON file",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Filter extraction to documents in a specific split (e.g., 'test', 'val', 'train')",
    )

    parser.add_argument(
        "--document-ids",
        type=str,
        default=None,
        help="Comma-separated list of document IDs to process",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate agent, configuration, and source documents directory without running extraction predictions",
    )

    return parser


def extract_command(argv: Optional[list] = None) -> int:
    """Execute batch extraction on documents using a trained agent.

    This command loads a trained agent, processes all parsed documents,
    and saves extraction results to JSON files.
    """
    parser = create_argument_parser()
    
    settings_container = {}
    extraction_client_container = {}
    args_container = {}

    def setup_llms(settings: Settings) -> None:
        settings_container["settings"] = settings
        from ae.extraction.client import ExtractionClient
        client = ExtractionClient.from_settings(settings)
        client.configure_dspy()
        extraction_client_container["client"] = client

    def build_request(
        args: argparse.Namespace,
        settings: Settings,
        cancel_event: threading.Event,
    ) -> BatchExtractionRequest:
        args_container["args"] = args

        # Load schema definition (prefer refined generated_schema_file if compatible)
        from ae.core.schema import load_schema_complete, load_schema_from_yaml
        from ae.core.utils import normalize_document_key

        schema_path = settings.paths.schema_file
        base_hash = load_schema_from_yaml(schema_path).get_schema_hash()
        gen_schema_path = getattr(settings.paths, "generated_schema_file", None)
        if gen_schema_path and gen_schema_path.exists() and gen_schema_path.read_text(encoding="utf-8").strip():
            use_generated_schema = True
            try:
                gen_hash = load_schema_from_yaml(gen_schema_path).get_schema_hash()
            except Exception:
                gen_hash = ""
            if gen_hash != base_hash:
                logger.debug(
                    f"Skipping generated schema {gen_schema_path}: schema hash ({gen_hash}) "
                    f"does not match configured schema {schema_path} ({base_hash})."
                )
                use_generated_schema = False

            explicit_agent = getattr(args, "agent", None)
            if use_generated_schema and explicit_agent:
                agent_p = Path(explicit_agent)
                if not agent_p.is_absolute():
                    agent_p = Path.cwd() / agent_p
                meta_p = agent_p.with_suffix(".meta.json")
                if meta_p.exists():
                    try:
                        import json as _json

                        saved_hash = _json.loads(meta_p.read_text(encoding="utf-8")).get("schema_hash")
                        if saved_hash and saved_hash != gen_hash and saved_hash == base_hash:
                            logger.warning(
                                f"Agent '{agent_p.name}' is bound to base schema hash ({saved_hash}); "
                                f"using {schema_path} instead of {gen_schema_path}."
                            )
                            use_generated_schema = False
                    except Exception:
                        pass
            if use_generated_schema:
                schema_path = gen_schema_path

        instruction_path = settings.paths.generated_prompt_file
        if not instruction_path or not instruction_path.exists() or not instruction_path.read_text(encoding="utf-8").strip():
            instruction_path = settings.paths.baseline_prompt_file
        task = load_schema_complete(
            yaml_path=schema_path,
            instruction_path=instruction_path,
        )
        logger.info(f"Schema loaded: {task.config.name}")

        # Validate task has signature for agent reconstruction
        if task.signature is None:
            raise ValueError("Task signature not found - required for agent reconstruction")

        # Resolve agent path
        if getattr(args, "agent", None):
            agent_path = Path(args.agent)
            if not agent_path.is_absolute():
                agent_path = Path.cwd() / agent_path
            if not agent_path.exists():
                raise FileNotFoundError(f"Agent not found: {agent_path}")
        else:
            # Find latest agent matching schema in agents_dir, or create zero-shot agent
            agent_repo = AgentRepository(agents_dir=settings.paths.agents_dir)
            agent_manager = AgentManager(agent_repo=agent_repo)
            current_schema_hash = task.config.get_schema_hash()

            latest_matching_agent = None
            if settings.paths.agents_dir.exists():
                for json_file in sorted(settings.paths.agents_dir.glob("*.json"), reverse=True):
                    if json_file.name.endswith(".meta.json"):
                        continue
                    meta_file = json_file.with_suffix(".meta.json")
                    if meta_file.exists():
                        try:
                            import json
                            meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                            if meta_data.get("schema_hash") == current_schema_hash:
                                latest_matching_agent = json_file
                                break
                        except Exception:
                            pass

            if latest_matching_agent is not None:
                agent_path = latest_matching_agent
                logger.info(f"Auto-selected latest matching agent: {agent_path}")
            else:
                from ae.extraction.agent import UniversalExtractor
                extractor = UniversalExtractor(signature_class=task.signature)
                agent_path = agent_manager.save_agent(
                    agent=extractor,
                    schema_config=task.config,
                    metrics={},
                    config={},
                    model_version="zero-shot",
                    description=f"Auto-generated zero-shot agent for {task.config.name}",
                    initial_instruction_file=str(instruction_path),
                    instruction_hash=task.config.get_instruction_hash(),
                )
                logger.info(f"Generated new zero-shot agent: {agent_path}")

        # Create document repository to retrieve documents
        doc_repo = DocumentRepository(ingestion_dir=settings.paths.ingestion_dir)
        
        # Check if parsed directory exists
        if not settings.paths.ingestion_dir.exists():
            logger.warning(f"Parsed directory does not exist: {settings.paths.ingestion_dir}")
            document_ids = []
        else:
            document_ids = doc_repo.list_document_keys()
            logger.info(f"Found {len(document_ids)} documents to process")

        if getattr(args, "split", None):
            from ae.core.storage.splits import load_split
            split_name = args.split.strip().lower()
            split_docs = load_split(settings.paths.splits_file, split_name)
            document_ids = [
                d for d in document_ids
                if d in split_docs or normalize_document_key(d) in split_docs
            ]
            logger.info(f"Filtered to {len(document_ids)} documents from split '{split_name}'")

        if getattr(args, "document_ids", None):
            explicit_ids = {normalize_document_key(i.strip()) for i in args.document_ids.split(",") if i.strip()}
            document_ids = [
                d for d in document_ids
                if normalize_document_key(d) in explicit_ids or d in explicit_ids
            ]
            logger.info(f"Filtered to {len(document_ids)} explicit document IDs")

        # Log extraction settings
        logger.info("=" * 60)
        logger.info("EXTRACTION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Schema: {task.config.name}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Agent: {agent_path}")
        logger.info(f"Documents: {len(document_ids)}")
        logger.info(f"Output: {settings.paths.extracted_dir}")
        logger.info(f"LLM cache: {'ENABLED' if settings.extraction.enable_cache else 'DISABLED'}")
        logger.info("=" * 60)

        client = extraction_client_container.get("client")
        history_dir = (
            Path("research/experiments/_runtime/logs/llm/extraction")
            if settings.extraction.save_llm_history
            else None
        )
        return BatchExtractionRequest(
            task=task,
            agent_path=agent_path,
            document_ids=document_ids,
            output_dir=settings.paths.extracted_dir,
            client=client,
            history_dir=history_dir,
            save_llm_history=settings.extraction.save_llm_history,
        )

    def execute_use_case(request: BatchExtractionRequest) -> BatchExtractionResponse:
        settings = settings_container["settings"]
        args = args_container["args"]

        if not request.document_ids:
            logger.warning("No documents to process")
            return BatchExtractionResponse(
                success=True,
                extractions_saved=0,
                total_documents=0,
                failed_documents=0,
                output_dir=request.output_dir,
            )

        agent_repo = AgentRepository(agents_dir=settings.paths.agents_dir)
        agent_manager = AgentManager(agent_repo=agent_repo)

        if args.dry_run:
            logger.info("Running dry-run validation checks...")
            try:
                # Reconstruct and validate the agent
                agent = agent_manager.load_agent_as_object(request.agent_path, task=request.task)
                logger.info(f"✓ Agent reconstructed successfully: {agent}")
            except Exception as e:
                logger.error(f"Failed to reconstruct agent: {e}")
                return BatchExtractionResponse(
                    success=False,
                    error_message=f"Failed to reconstruct agent: {e}",
                )

            logger.info("=" * 60)
            logger.info("DRY RUN VALIDATION COMPLETED SUCCESSFULLY")
            logger.info(f"  Agent file:    {request.agent_path}")
            logger.info(f"  Target schema: {request.task.config.name}")
            logger.info(f"  Parsed docs:   {len(request.document_ids)} files ready")
            logger.info(f"  Output dir:    {request.output_dir}")
            logger.info("✓ Success! Agent is valid, config is correct, and documents are ready.")
            logger.info("=" * 60)
            print("✓ Success! Agent is valid, config is correct, and documents are ready.")
            return BatchExtractionResponse(
                success=True,
                extractions_saved=0,
                total_documents=len(request.document_ids),
                failed_documents=0,
                output_dir=request.output_dir,
            )

        doc_repo = DocumentRepository(ingestion_dir=settings.paths.ingestion_dir)
        pred_repo = ExtractionRepository()

        use_case = BatchExtractionUseCase(
            agent_manager=agent_manager,
            document_repo=doc_repo,
            extraction_repo=pred_repo,
        )

        try:
            if settings.extraction.save_llm_history:
                client = extraction_client_container.get("client")
                if client is not None and hasattr(client, "start_history_session"):
                    client.start_history_session(Path("research/experiments/_runtime/logs/llm/extraction"))
            logger.info(f"Processing {len(request.document_ids)} documents...")
            return use_case.execute(request)
        finally:
            if settings.extraction.save_llm_history:
                client = extraction_client_container.get("client")
                if client is not None:
                    history_dir = Path("research/experiments/_runtime/logs/llm/extraction")
                    client.save_history(history_dir)

    def format_response(response: BatchExtractionResponse) -> int:
        args = args_container["args"]
        if args.dry_run:
            return 0 if response.success else 1

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
            return 0 if response.failed_documents == 0 else 2
        else:
            logger.error("✗ EXTRACTION FAILED")
            logger.error(f"✗ Error: {response.error_message}")
            if getattr(response, "extractions_saved", 0) > 0:
                logger.warning(
                    f"✗ Part of the batch was processed before failure/abort. "
                    f"Processed: {response.extractions_saved}/{response.total_documents}, "
                    f"Failed/Aborted: {response.failed_documents}"
                )
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
    sys.exit(extract_command())


if __name__ == "__main__":
    main()
