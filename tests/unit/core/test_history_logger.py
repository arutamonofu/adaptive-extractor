"""Unit tests for LLMHistoryRecorder and ExtractionHistoryCallbackHandler."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterator, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel

from ae.core.llm.history_logger import (
    ExtractionHistoryCallbackHandler,
    LLMHistoryRecorder,
    extraction_log_context,
)


class _SampleRecord(BaseModel):
    name: str
    value: int


@pytest.mark.unit
class TestLLMHistoryRecorder:
    def test_streaming_jsonl_append_and_o1_memory(self, tmp_path: Path):
        recorder = LLMHistoryRecorder(max_in_memory_entries=2)
        stream_path = tmp_path / "logs" / "extraction_test.jsonl"
        recorder.start_stream(stream_path)

        for i in range(5):
            recorder.record(
                {
                    "doc_id": f"doc_{i}",
                    "node": "agent_node",
                    "status": "success",
                    "model": "gemini-3.8-flash",
                    "messages": [
                        SystemMessage(content="sys"),
                        HumanMessage(content=f"hello {i}"),
                        AIMessage(
                            content="",
                            tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Table_1"]}, "id": f"tc_{i}", "type": "tool_call"}],
                        ),
                        ToolMessage(content="<table>...</table>", tool_call_id=f"tc_{i}", name="parse_visuals"),
                    ],
                    "outputs": [_SampleRecord(name="x", value=i)],
                }
            )
        recorder.close()

        assert recorder.recorded_count == 5
        assert recorder.get_entries() == []  # Streaming mode does not accumulate in RAM

        lines = stream_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert first["doc_id"] == "doc_0"
        assert first["messages"][0]["role"] == "system"
        assert first["messages"][2]["tool_calls"][0]["name"] == "parse_visuals"
        assert first["messages"][3]["role"] == "tool"
        assert first["outputs"][0] == {"name": "x", "value": 0}

    def test_max_entry_bytes_truncates_safely(self, tmp_path: Path):
        recorder = LLMHistoryRecorder(max_entry_bytes=2048)
        stream_path = tmp_path / "bounded.jsonl"
        recorder.start_stream(stream_path)

        recorder.record(
            {
                "doc_id": "big_doc",
                "status": "success",
                "model": "test-model",
                "messages": [{"role": "user", "content": "A" * 10000}],
                "outputs": ["B" * 10000],
            }
        )
        recorder.close()

        raw = stream_path.read_text(encoding="utf-8").strip()
        assert len(raw.encode("utf-8")) <= 2048
        parsed = json.loads(raw)
        assert parsed["doc_id"] == "big_doc"

    def test_max_file_bytes_stops_appending(self, tmp_path: Path):
        recorder = LLMHistoryRecorder(max_entry_bytes=4096, max_file_bytes=1024)
        stream_path = tmp_path / "capped.jsonl"
        recorder.start_stream(stream_path)

        for i in range(10):
            recorder.record(
                {
                    "doc_id": f"doc_{i}",
                    "status": "success",
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "X" * 300}],
                    "outputs": ["Y" * 300],
                }
            )
        recorder.close()

        assert stream_path.stat().st_size <= 1024
        assert recorder.recorded_count < 10

    def test_surrogate_characters_and_io_error_do_not_crash(self, tmp_path: Path):
        recorder = LLMHistoryRecorder()
        stream_path = tmp_path / "surrogates.jsonl"
        recorder.start_stream(stream_path)

        recorder.record(
            {
                "doc_id": "bad_utf",
                "status": "error",
                "model": "test",
                "messages": [{"role": "user", "content": "bad \ud800 text"}],
                "outputs": [],
                "error": ValueError("broken \udfff msg"),
            }
        )
        recorder.close()

        lines = stream_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["status"] == "error"
        assert data["error"]["type"] == "ValueError"

    def test_empty_stream_does_not_create_file(self, tmp_path: Path):
        recorder = LLMHistoryRecorder()
        stream_path = tmp_path / "unused" / "empty_session.jsonl"
        recorder.start_stream(stream_path)
        assert recorder.is_streaming is True
        recorder.close()

        assert recorder.recorded_count == 0
        assert not stream_path.exists()
        assert not stream_path.parent.exists()


class _FakeStreamErrorChatModel(BaseChatModel):
    model_name: str = "fake-gemini"

    @property
    def _llm_type(self) -> str:
        return "fake-gemini"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=[
                            {"type": "thinking", "thinking": "checking table"},
                            {"type": "text", "text": "READY"},
                        ],
                        tool_calls=[
                            {
                                "name": "parse_visuals",
                                "args": {"visual_ids": ["Table_1"]},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                        usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
                        response_metadata={"finish_reason": "STOP"},
                    )
                )
            ]
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content='{"records":[{"id":'))
        raise ConnectionResetError("peer reset mid-stream")


@pytest.mark.unit
class TestExtractionHistoryCallbackHandler:
    def test_captures_thinking_tool_calls_and_usage(self, tmp_path: Path):
        recorder = LLMHistoryRecorder()
        stream_path = tmp_path / "cb.jsonl"
        recorder.start_stream(stream_path)
        handler = ExtractionHistoryCallbackHandler(recorder)

        model = _FakeStreamErrorChatModel(callbacks=[handler])
        with extraction_log_context(doc_id="doc_cb", node="agent_node", stage="agent_node", attempt=1):
            model.invoke([HumanMessage(content="extract")])

        recorder.close()
        entries = [json.loads(line) for line in stream_path.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        e = entries[0]
        assert e["doc_id"] == "doc_cb"
        assert e["node"] == "agent_node"
        assert e["status"] == "success"
        assert e["outputs"] == ["READY"]
        assert e["tool_calls"][0]["name"] == "parse_visuals"
        assert e["usage"]["total_tokens"] == 17
        assert e["finish_reason"] == "STOP"
        assert e["kwargs"]["thinking"] == "checking table"

    def test_captures_partial_output_on_stream_error(self, tmp_path: Path):
        recorder = LLMHistoryRecorder()
        stream_path = tmp_path / "stream_err.jsonl"
        recorder.start_stream(stream_path)
        handler = ExtractionHistoryCallbackHandler(recorder)

        model = _FakeStreamErrorChatModel(callbacks=[handler])
        with extraction_log_context(doc_id="doc_stream", node="structured_output_node", stage="structured_output", attempt=2):
            with pytest.raises(ConnectionResetError):
                list(model.stream([HumanMessage(content="extract")]))

        recorder.close()
        entries = [json.loads(line) for line in stream_path.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        e = entries[0]
        assert e["status"] == "error"
        assert e["doc_id"] == "doc_stream"
        assert e["attempt"] == 2
        assert e["partial_output"] == '{"records":[{"id":'
        assert e["error"]["type"] == "ConnectionResetError"

    def test_deduplicates_identical_run_id(self, tmp_path: Path):
        recorder = LLMHistoryRecorder()
        handler = ExtractionHistoryCallbackHandler(recorder)
        run_id = uuid.uuid4()

        handler.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=run_id)
        handler.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=run_id)
        res = ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
        handler.on_llm_end(res, run_id=run_id)
        handler.on_llm_end(res, run_id=run_id)

        assert recorder.recorded_count == 1


    def test_record_deduplicates_identical_run_id_status_stage(self):
        recorder = LLMHistoryRecorder()
        recorder.record({"llm_run_id": "r1", "status": "success", "stage": "agent_node", "outputs": ["a"]})
        recorder.record({"llm_run_id": "r1", "status": "success", "stage": "agent_node", "outputs": ["a"]})
        recorder.record({"llm_run_id": "r1", "status": "error", "stage": "graph_failure", "outputs": []})
        assert recorder.recorded_count == 2
