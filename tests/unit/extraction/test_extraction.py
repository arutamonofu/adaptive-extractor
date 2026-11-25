from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import dspy
import pytest

from ae.core.exceptions import UseCaseExecutionError
from ae.core.llm.circuit_breaker import CircuitBreakerError
from ae.core.storage import AgentMetadata, AgentRepository, DocumentRepository, ExtractionRepository
from ae.extraction.agent import UniversalExtractor
from ae.extraction.manager import AgentManager
from ae.extraction.use_case import BatchExtractionRequest, BatchExtractionUseCase


@pytest.mark.unit
class TestAgentManager:
    """Consolidated unit tests for AgentManager service and serialization."""

    def test_load_agent_as_object_and_demos(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # 1. Setup: Create agent with demos
        agent = UniversalExtractor(nanozyme_task["signature"])
        demo = dspy.Example(document_text="Test doc", extracted_data="Test res").with_inputs("document_text")
        agent.prog.predict.demos = [demo]

        metadata = AgentMetadata(
            created_at=datetime.now(),
            model_version="test-model",
            metrics={"f1": 0.85},
            config_snapshot={},
        )

        agent_dict = agent.dump_state()
        agent_path = repo.save(agent=agent_dict, metadata=metadata)

        # 2. Reconstruct agent as object
        reconstructed = manager.load_agent_as_object(agent_path=agent_path, task_dict=nanozyme_task)
        assert reconstructed is not None
        assert isinstance(reconstructed, UniversalExtractor)
        assert callable(reconstructed)
        assert len(reconstructed.prog.predict.demos) == 1


    def test_create_agent_with_demos(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        demo = dspy.Example(document_text="Demo doc", extracted_data="Demo res").with_inputs("document_text")
        agent = manager.create_agent_with_demos(signature_class=nanozyme_task["signature"], demos=[demo])

        assert isinstance(agent, UniversalExtractor)
        assert len(agent.prog.predict.demos) == 1
        assert agent.prog.predict.demos[0].document_text == "Demo doc"

    def test_load_agent_as_object_missing_signature(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        agent = UniversalExtractor(nanozyme_task["signature"])
        metadata = AgentMetadata(
            created_at=datetime.now(),
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        agent_path = repo.save(agent=agent.dump_state(), metadata=metadata)

        bad_task_dict = {"config": nanozyme_task["config"]}
        with pytest.raises(UseCaseExecutionError, match="signature"):
            manager.load_agent_as_object(agent_path=agent_path, task_dict=bad_task_dict)

    def test_serialization(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Dict serialization (passthrough)
        agent_dict = {"lm": {"model": "test"}, "traces": []}
        assert manager._serialize_agent(agent_dict) == agent_dict

        # UniversalExtractor serialization
        agent = UniversalExtractor(nanozyme_task["signature"])
        serialized = manager._serialize_agent(agent)
        assert isinstance(serialized, dict)
        assert len(serialized) > 0

        # Invalid agent serialization
        class InvalidAgent:
            pass
        with pytest.raises(UseCaseExecutionError, match="serialize"):
            manager._serialize_agent(InvalidAgent())  # type: ignore[arg-type]

    def test_save_and_load_roundtrip(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        original_agent = UniversalExtractor(nanozyme_task["signature"])
        saved_path = manager.save_agent(
            agent=original_agent,
            schema_config=nanozyme_task["config"],
            metrics={"f1": 0.90},
            config={"num_trials": 10},
            model_version="test-v1",
        )

        # Load as dict
        loaded_dict = manager.load_agent(saved_path)
        assert isinstance(loaded_dict, dict)

        # Load as object
        loaded_object = manager.load_agent_as_object(saved_path, nanozyme_task)
        assert isinstance(loaded_object, UniversalExtractor)
        assert callable(loaded_object)

    def test_schema_hash_mismatch_validation(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        original_agent = UniversalExtractor(nanozyme_task["signature"])
        
        saved_path = manager.save_agent(
            agent=original_agent,
            schema_config=nanozyme_task["config"],
            metrics={"f1": 0.90},
            config={"num_trials": 10},
            model_version="test-v1",
        )

        from copy import deepcopy
        from ae.core.schema.config import SchemaConfig, FieldSpec
        
        orig_config = nanozyme_task["config"]
        modified_fields = deepcopy(orig_config.experiment_fields)
        modified_fields["new_dummy_field"] = FieldSpec(type=str, description="dummy field")
        
        modified_config = SchemaConfig(
            name=orig_config.name,
            experiment_fields=modified_fields,
            compare_fields=orig_config.compare_fields + ["new_dummy_field"],
            float_tolerance=orig_config.float_tolerance,
            instruction_file=orig_config.instruction_file,
            row_converter=orig_config.row_converter,
            base_class=orig_config.base_class,
        )
        
        modified_task = {
            "config": modified_config,
            "signature": nanozyme_task["signature"]
        }

        with pytest.raises(UseCaseExecutionError) as exc_info:
            manager.load_agent_as_object(saved_path, modified_task)
            
        assert "Schema hash mismatch" in str(exc_info.value)


@pytest.mark.unit
class TestUniversalExtractorFallback:
    """Tests for preserving visual enrichment across ExtractionGraph failures and fallbacks."""

    def test_fallback_uses_refreshed_disk_enrichment(
        self, tmp_path: Path, nanozyme_task: Dict[str, Any]
    ):
        doc_id = "acs.jafc.1c02545"
        doc_file = tmp_path / doc_id / f"{doc_id}.md"
        doc_file.parent.mkdir(parents=True, exist_ok=True)

        stale_text = (
            f"<!-- document_id: {doc_id} -->\n\n"
            '<visual id="Table_1" type="table">\nTable 1\n> MIC values\n</visual>'
        )
        enriched_text = (
            f"<!-- document_id: {doc_id} -->\n\n"
            '<visual id="Table_1" type="table">\nTable 1\n\n<table><tr><td>7d</td><td>0.006</td></tr></table>\n</visual>'
        )
        doc_file.write_text(stale_text, encoding="utf-8")

        agent = UniversalExtractor(
            nanozyme_task["signature"],
            ingestion_dir=tmp_path,
        )
        mock_graph = MagicMock()
        mock_graph.ingestion_dir = tmp_path
        mock_graph.document_repo = DocumentRepository(tmp_path)
        mock_graph.get_last_document_text.return_value = None

        def _simulate_parse_then_crash(**kwargs):
            doc_file.write_text(enriched_text, encoding="utf-8")
            raise RuntimeError("ReadError: Connection reset by peer")

        mock_graph.run.side_effect = _simulate_parse_then_crash
        agent._graph = mock_graph

        fallback_pred = dspy.Prediction(extracted_data=nanozyme_task["output_model"](records=[]))
        agent.prog = MagicMock(return_value=fallback_pred)

        res = agent.forward(document_text=stale_text)
        assert res is fallback_pred
        called_doc = agent.prog.call_args.kwargs["document_text"]
        assert "<table><tr><td>7d</td><td>0.006</td></tr></table>" in called_doc
        assert "> MIC values" not in called_doc

    def test_fallback_uses_in_memory_graph_enrichment_when_disk_missing(
        self, tmp_path: Path, nanozyme_task: Dict[str, Any]
    ):
        doc_id = "paper_mem"
        stale_text = (
            f"<!-- document_id: {doc_id} -->\n\n"
            '<visual id="Scheme_1" type="molecular_structure">\nScheme 1\n> Compounds\n</visual>'
        )
        enriched_mem = (
            f"<!-- document_id: {doc_id} -->\n\n"
            '<visual id="Scheme_1" type="molecular_structure">\nScheme 1\n\n| Label | SMILES |\n| :--- | :--- |\n| **7d** | `CCO` |\n</visual>'
        )

        agent = UniversalExtractor(
            nanozyme_task["signature"],
            ingestion_dir=tmp_path,
        )
        mock_graph = MagicMock()
        mock_graph.ingestion_dir = tmp_path
        mock_graph.document_repo = DocumentRepository(tmp_path)
        mock_graph.get_last_document_text.return_value = enriched_mem
        mock_graph.run.side_effect = RuntimeError("Structured output failed")
        agent._graph = mock_graph

        fallback_pred = dspy.Prediction(extracted_data=nanozyme_task["output_model"](records=[]))
        agent.prog = MagicMock(return_value=fallback_pred)

        agent.forward(document_text=stale_text)
        called_doc = agent.prog.call_args.kwargs["document_text"]
        assert "**7d**" in called_doc
        assert "> Compounds" not in called_doc

    def test_circuit_breaker_error_not_swallowed_by_fallback(
        self, tmp_path: Path, nanozyme_task: Dict[str, Any]
    ):
        agent = UniversalExtractor(
            nanozyme_task["signature"],
            ingestion_dir=tmp_path,
        )
        mock_graph = MagicMock()
        mock_graph.run.side_effect = CircuitBreakerError("OPEN")
        agent._graph = mock_graph
        agent.prog = MagicMock()

        with pytest.raises(CircuitBreakerError):
            agent.forward(document_text="<!-- document_id: doc_cb -->\n\nText")
        agent.prog.assert_not_called()

    def test_batch_use_case_aligns_ingestion_dir_and_doc_id_header(
        self, tmp_path: Path, nanozyme_task: Dict[str, Any]
    ):
        agents_dir = tmp_path / "agents"
        ingestion_dir = tmp_path / "ingestion"
        output_dir = tmp_path / "extracted"
        ingestion_dir.mkdir()

        doc_file = ingestion_dir / "doc_01.md"
        doc_file.write_text("Plain text without doc_id comment.", encoding="utf-8")

        agent_repo = AgentRepository(agents_dir=agents_dir)
        manager = AgentManager(agent_repo=agent_repo)
        orig_agent = UniversalExtractor(nanozyme_task["signature"])
        agent_path = manager.save_agent(
            agent=orig_agent,
            schema_config=nanozyme_task["config"],
            metrics={"f1": 0.9},
            config={},
        )

        mock_reconstructed = UniversalExtractor(nanozyme_task["signature"])
        mock_graph = MagicMock()
        mock_graph.run.return_value = {
            "extracted_data": nanozyme_task["output_model"](records=[]),
            "rationale": "",
        }
        mock_reconstructed._graph = mock_graph
        manager.load_agent_as_object = MagicMock(return_value=mock_reconstructed)

        use_case = BatchExtractionUseCase(
            agent_manager=manager,
            document_repo=DocumentRepository(ingestion_dir=ingestion_dir),
            extraction_repo=ExtractionRepository(),
        )
        resp = use_case.execute(
            BatchExtractionRequest(
                task=nanozyme_task,
                agent_path=agent_path,
                document_ids=["doc_01"],
                output_dir=output_dir,
            )
        )
        assert resp.success is True
        assert resp.extractions_saved == 1
        assert mock_reconstructed._ingestion_dir == ingestion_dir
        passed_text = mock_graph.run.call_args.kwargs["document_text"]
        assert passed_text.startswith("<!-- document_id: doc_01 -->")

    def test_batch_use_case_flushes_history_on_circuit_breaker_abort(
        self, tmp_path: Path, nanozyme_task: Dict[str, Any]
    ):
        import json
        from ae.core.config.optimization_settings import ExtractionConfig
        from ae.extraction.client import ExtractionClient

        agents_dir = tmp_path / "agents"
        ingestion_dir = tmp_path / "ingestion"
        output_dir = tmp_path / "extracted"
        history_dir = tmp_path / "logs" / "llm" / "extraction"
        ingestion_dir.mkdir()

        (ingestion_dir / "doc_cb.md").write_text("Text", encoding="utf-8")

        agent_repo = AgentRepository(agents_dir=agents_dir)
        manager = AgentManager(agent_repo=agent_repo)
        agent_path = manager.save_agent(
            agent=UniversalExtractor(nanozyme_task["signature"]),
            schema_config=nanozyme_task["config"],
            metrics={"f1": 0.9},
            config={},
        )

        client = ExtractionClient(
            config=ExtractionConfig(
                execution={"save_llm_history": True},
                model={"provider": "gemini", "model": "gemini-3.8-flash"},
            )
        )

        mock_reconstructed = UniversalExtractor(nanozyme_task["signature"], client=client)

        def _record_then_abort(**kwargs):
            client.history_recorder.record(
                {
                    "doc_id": "doc_cb",
                    "node": "graph",
                    "stage": "graph_failure",
                    "status": "error",
                    "model": "gemini-3.8-flash",
                    "messages": [],
                    "outputs": [],
                    "error": CircuitBreakerError("OPEN"),
                }
            )
            raise CircuitBreakerError("OPEN")

        mock_graph = MagicMock()
        mock_graph.run.side_effect = _record_then_abort
        mock_reconstructed._graph = mock_graph
        manager.load_agent_as_object = MagicMock(return_value=mock_reconstructed)

        use_case = BatchExtractionUseCase(
            agent_manager=manager,
            document_repo=DocumentRepository(ingestion_dir=ingestion_dir),
            extraction_repo=ExtractionRepository(),
        )
        resp = use_case.execute(
            BatchExtractionRequest(
                task=nanozyme_task,
                agent_path=agent_path,
                document_ids=["doc_cb"],
                output_dir=output_dir,
                client=client,
                history_dir=history_dir,
            )
        )
        assert resp.aborted is True
        jsonl_files = list(history_dir.glob("extraction_*.jsonl"))
        assert len(jsonl_files) == 1
        entries = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        assert entries[0]["doc_id"] == "doc_cb"
        assert entries[0]["error"]["type"] == "CircuitBreakerError"

    def test_universal_extractor_fallback_binds_dspy_context_doc_id(
        self, tmp_path: Path, nanozyme_task: Dict[str, Any]
    ):
        from ae.core.llm.history_logger import get_extraction_log_context

        observed_ctx = {}
        agent = UniversalExtractor(nanozyme_task["signature"], ingestion_dir=tmp_path)
        mock_graph = MagicMock()
        mock_graph.ingestion_dir = tmp_path
        mock_graph.document_repo = DocumentRepository(tmp_path)
        mock_graph.get_last_document_text.return_value = None
        mock_graph.run.side_effect = RuntimeError("graph boom")
        agent._graph = mock_graph

        fallback_pred = dspy.Prediction(extracted_data=nanozyme_task["output_model"](records=[]))

        def _capture_prog(document_text: str):
            observed_ctx.update(get_extraction_log_context())
            return fallback_pred

        agent.prog = MagicMock(side_effect=_capture_prog)

        res = agent.forward(document_text="<!-- document_id: acs.jafc.1c02545 -->\n\nBody")
        assert res is fallback_pred
        assert observed_ctx["doc_id"] == "acs.jafc.1c02545"
        assert observed_ctx["source"] == "dspy_fallback"
        assert observed_ctx["stage"] == "dspy_fallback"


    def test_universal_extractor_forwards_agent_thinking_level(self, tmp_path: Path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from pydantic import BaseModel

        class DummyOut(BaseModel):
            summary: str

        mock_client = MagicMock()
        mock_client.get_chat_model.return_value = MagicMock()
        mock_client.history_recorder = None
        mock_client.extraction_config = SimpleNamespace(
            max_tool_iterations=3,
            api_client=SimpleNamespace(max_retries=2, rate_limit_delay=0.0),
            structured_output_batch_size=0,
            max_structured_output_batches=5,
            enable_sanity_checks=True,
            agent_thinking_level="minimal",
        )

        sig = MagicMock()
        sig.output_fields = {"extracted_data": MagicMock(annotation=DummyOut)}
        extractor = UniversalExtractor(
            signature_class=sig,
            client=mock_client,
            ingestion_dir=tmp_path,
        )
        with patch("ae.core.config.settings.Settings.load", return_value=MagicMock()), patch(
            "ae.extraction.graph.ExtractionGraph"
        ) as mock_graph_cls:
            extractor._get_or_create_graph()
            mock_graph_cls.assert_called_once()
            assert mock_graph_cls.call_args.kwargs["agent_thinking_level"] == "minimal"

    def test_universal_extractor_forwards_config_path_to_settings_load(self, tmp_path: Path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from pydantic import BaseModel

        class DummyOut(BaseModel):
            summary: str

        custom_cfg = tmp_path / "custom_cfg"
        custom_cfg.mkdir()

        mock_client = MagicMock()
        mock_client.config_path = custom_cfg
        mock_client.get_chat_model.return_value = MagicMock()
        mock_client.history_recorder = None
        mock_client.extraction_config = SimpleNamespace(
            max_tool_iterations=3,
            api_client=SimpleNamespace(max_retries=2, rate_limit_delay=0.0),
            structured_output_batch_size=0,
            max_structured_output_batches=5,
            enable_sanity_checks=True,
            agent_thinking_level="low",
        )

        sig = MagicMock()
        sig.output_fields = {"extracted_data": MagicMock(annotation=DummyOut)}

        extractor = UniversalExtractor(signature_class=sig, client=mock_client, ingestion_dir=tmp_path)
        with patch("ae.core.config.settings.Settings.load", return_value=MagicMock()) as mock_load, patch(
            "ae.extraction.graph.ExtractionGraph"
        ) as mock_graph_cls:
            extractor._get_or_create_graph()
            mock_load.assert_called_once_with(config_path=custom_cfg, load_env_file=True)
            assert mock_graph_cls.call_args.kwargs["config_path"] == custom_cfg

        explicit_cfg = tmp_path / "explicit_cfg"
        explicit_cfg.mkdir()
        extractor2 = UniversalExtractor(
            signature_class=sig,
            client=mock_client,
            ingestion_dir=tmp_path,
            config_path=explicit_cfg,
        )
        with patch("ae.core.config.settings.Settings.load", return_value=MagicMock()) as mock_load2, patch(
            "ae.extraction.graph.ExtractionGraph"
        ) as mock_graph_cls2:
            extractor2._get_or_create_graph()
            mock_load2.assert_called_once_with(config_path=explicit_cfg, load_env_file=True)
            assert mock_graph_cls2.call_args.kwargs["config_path"] == explicit_cfg

    def test_universal_extractor_forwards_enable_record_dedup(self, tmp_path: Path):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from pydantic import BaseModel

        class DummyOut(BaseModel):
            summary: str

        sig = MagicMock()
        sig.output_fields = {"extracted_data": MagicMock(annotation=DummyOut)}

        mock_client = MagicMock()
        mock_client.get_chat_model.return_value = MagicMock()
        mock_client.history_recorder = None
        mock_client.extraction_config = SimpleNamespace(
            max_tool_iterations=3,
            api_client=SimpleNamespace(max_retries=2, rate_limit_delay=0.0),
            structured_output_batch_size=0,
            max_structured_output_batches=5,
            enable_sanity_checks=True,
            agent_thinking_level="low",
            enable_record_dedup=False,
        )
        extractor = UniversalExtractor(signature_class=sig, client=mock_client, ingestion_dir=tmp_path)
        with patch("ae.core.config.settings.Settings.load", return_value=MagicMock()), patch(
            "ae.extraction.graph.ExtractionGraph"
        ) as mock_graph_cls:
            extractor._get_or_create_graph()
            assert mock_graph_cls.call_args.kwargs["enable_record_dedup"] is False

        # Fallback default True when omitted
        mock_client.extraction_config = SimpleNamespace(
            max_tool_iterations=3,
            api_client=SimpleNamespace(max_retries=2, rate_limit_delay=0.0),
        )
        extractor2 = UniversalExtractor(signature_class=sig, client=mock_client, ingestion_dir=tmp_path)
        with patch("ae.core.config.settings.Settings.load", return_value=MagicMock()), patch(
            "ae.extraction.graph.ExtractionGraph"
        ) as mock_graph_cls2:
            extractor2._get_or_create_graph()
            assert mock_graph_cls2.call_args.kwargs["enable_record_dedup"] is True
