"""Unit tests for ExtractionGraph (ReAct workflow)."""

from unittest.mock import MagicMock
from pathlib import Path
import json
import httpx
import pytest
from pydantic import BaseModel, Field

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from ae.core.llm.circuit_breaker import CircuitBreakerError
from ae.extraction.graph import ExtractionGraph, ExtractionState
from ae.ingestion.service import VisualParsingService
from ae.ingestion.parsers.gemini.schemas import VisualParseRecord


class DummyOutput(BaseModel):
    summary: str = Field(description="Summary")
    count: int = Field(default=0)


def test_extraction_graph_prepare_input(tmp_path):
    mock_llm = MagicMock()
    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
    )

    doc_id = "paper_01"
    doc_file = tmp_path / doc_id / f"{doc_id}.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text("<!-- document_id: paper_01 -->\n\nFull article text here.", encoding="utf-8")

    initial_state: ExtractionState = {
        "doc_id": "",
        "document_text": "<!-- document_id: paper_01 -->\n\nStale text",
        "system_instruction": "Extract all data",
        "messages": [],
        "tool_iterations": 0,
        "extracted_data": None,
        "rationale": None,
    }

    res = graph._prepare_input(initial_state)
    assert res["doc_id"] == "paper_01"
    assert "Full article text here" in res["document_text"]
    assert len(res["messages"]) == 2


def test_extraction_graph_prepare_input_dotted_doc_id(tmp_path):
    mock_llm = MagicMock()
    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
    )

    doc_id = "acsomega.2c06142"
    doc_file = tmp_path / doc_id / f"{doc_id}.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text("<!-- document_id: acsomega.2c06142 -->\n\nFull article text here.", encoding="utf-8")

    initial_state: ExtractionState = {
        "doc_id": doc_id,
        "document_text": "",
        "system_instruction": "Extract all data",
        "messages": [],
        "tool_iterations": 0,
        "extracted_data": None,
        "rationale": None,
    }

    res = graph._prepare_input(initial_state)
    assert res["doc_id"] == "acsomega.2c06142"
    assert "Full article text here" in res["document_text"]



def test_extraction_graph_workflow_with_tool_call(tmp_path):
    # Prepare service mock
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Figure_1": VisualParseRecord(
            visual_id="Figure_1",
            status="success",
            mode="single",
            rendered_markdown="| compound | IC50 |\n|---|---|\n| A1 | 12 nM |",
        )
    }

    # Prepare LLM mock:
    # Call 1: model returns tool call for Figure_1
    # Call 2 (after tool execution): structured output invocation
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound

    tool_call_msg = AIMessage(
        content="I need to inspect Figure 1 to obtain IC50 values.",
        tool_calls=[
            {
                "name": "parse_visuals",
                "args": {"visual_ids": ["Figure_1"]},
                "id": "call_123",
                "type": "tool_call",
            }
        ],
    )
    mock_bound.invoke.side_effect = [
        tool_call_msg,
        AIMessage(content="I have extracted the necessary data from Figure 1."),
    ]

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="Extracted A1", count=1)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=2,
        ingestion_dir=tmp_path,
    )

    # Put initial doc
    doc_file = tmp_path / "doc_test" / "doc_test.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text("<!-- document_id: doc_test -->\n\n<visual id=\"Figure_1\" type=\"chart\">\nFigure 1\n> Chart description\n</visual>", encoding="utf-8")

    result = graph.run(
        document_text="<!-- document_id: doc_test -->\n\n<visual id=\"Figure_1\" type=\"chart\">\nFigure 1\n> Chart description\n</visual>",
        doc_id="doc_test",
        system_instruction="Extract experiments",
    )

    mock_service.parse_visuals.assert_called_once_with("doc_test", ["Figure_1"])
    assert result["extracted_data"] is not None
    assert result["extracted_data"].summary == "Extracted A1"
    assert result["extracted_data"].count == 1


def test_extraction_graph_short_circuit_no_visuals(tmp_path):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="No visuals", count=0)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        max_tool_iterations=3,
        ingestion_dir=tmp_path,
    )

    doc_text = "<!-- document_id: text_only -->\n\nPure article text without any visual stubs."
    res = graph.run(document_text=doc_text, doc_id="text_only")

    # Short-circuit: bind_tools should never be invoked
    mock_llm.bind_tools.assert_not_called()
    mock_llm.with_structured_output.assert_called_once_with(DummyOutput)
    assert res["extracted_data"].summary == "No visuals"


def test_extraction_graph_system_prompt_policies(tmp_path):
    mock_llm = MagicMock()
    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
    )

    state: ExtractionState = {
        "doc_id": "test_policy",
        "document_text": "<!-- document_id: test_policy -->\n\n<visual id=\"T1\">\n> table\n</visual>",
        "system_instruction": "Base prompt",
        "messages": [],
        "tool_iterations": 0,
        "attempted_visuals": [],
        "extracted_data": None,
        "rationale": None,
    }

    res = graph._prepare_input(state)
    sys_msg = res["messages"][0].content
    assert "TARGET SCHEMA FIELDS:" in sys_msg
    assert "summary" in sys_msg
    assert "Priority Cascade" in sys_msg
    assert "Manifest Plausibility Gate" in sys_msg


def test_extraction_graph_prevents_duplicate_visual_calls(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Fig_1": VisualParseRecord(
            visual_id="Fig_1",
            status="success",
            mode="single",
            rendered_markdown="| X | Y |\n|---|---|\n| 1 | 2 |",
        )
    }

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound

    # Call 1 asks for Fig_1, Call 2 tries to ask for Fig_1 again
    call1 = AIMessage(
        content="Parsing Fig_1",
        tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Fig_1"]}, "id": "c1", "type": "tool_call"}],
    )
    mock_bound.invoke.return_value = call1

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="Deduplicated", count=1)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=2,
        ingestion_dir=tmp_path,
    )

    # Initial state with Fig_1 already in attempted_visuals
    state: ExtractionState = {
        "doc_id": "doc_dup",
        "document_text": "text",
        "system_instruction": "",
        "messages": [call1],
        "tool_iterations": 0,
        "attempted_visuals": ["Fig_1"],
        "extracted_data": None,
        "rationale": None,
    }

    # _should_continue should route to structured_output because Fig_1 is already attempted
    assert graph._should_continue(state) == "structured_output"


def test_extraction_graph_tool_call_with_target_hint(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Scheme_2": VisualParseRecord(
            visual_id="Scheme_2",
            status="partial",
            mode="composite",
            rendered_markdown="**4** + **5a-5f** --> **6a-6f**\n| **6a** | `CC` |",
            parsed_element_ids=["mol_6a_6f"],
            unparsed_element_ids=["mol_4", "mol_5a_5f"],
        )
    }

    mock_llm = MagicMock()
    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=2,
        ingestion_dir=tmp_path,
    )

    tool_call_msg = AIMessage(
        content="Parsing only products of Scheme 2",
        tool_calls=[
            {
                "name": "parse_visuals",
                "args": {"visual_ids": ["Scheme_2"], "target_hint": "products 6a-6f"},
                "id": "call_hint",
                "type": "tool_call",
            }
        ],
    )

    state: ExtractionState = {
        "doc_id": "doc_hint",
        "document_text": "<!-- document_id: doc_hint -->\n<visual id=\"Scheme_2\">> scheme</visual>",
        "system_instruction": "",
        "messages": [tool_call_msg],
        "tool_iterations": 0,
        "attempted_visuals": [],
        "extracted_data": None,
        "rationale": None,
    }

    res = graph._tool_node(state)
    assert res["attempted_visuals"] == []
    assert res["partial_visuals"] == ["Scheme_2"]
    assert res["attempted_hints"] == {"Scheme_2": ["products 6a-6f"]}
    mock_service.parse_visuals.assert_called_once_with(
        "doc_hint", ["Scheme_2"], target_hint="products 6a-6f"
    )


def test_extraction_graph_selective_sub_element_invocation(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_llm = MagicMock()
    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=3,
        ingestion_dir=tmp_path,
    )

    # State where Scheme_2 was already attempted, but Scheme_2/mol_4 is now requested
    call_subelement = AIMessage(
        content="Requesting unparsed reactant element mol_4",
        tool_calls=[
            {
                "name": "parse_visuals",
                "args": {"visual_ids": ["Scheme_2/mol_4"]},
                "id": "c_sub",
                "type": "tool_call",
            }
        ],
    )

    state: ExtractionState = {
        "doc_id": "doc_sub",
        "document_text": "text",
        "system_instruction": "",
        "messages": [call_subelement],
        "tool_iterations": 1,
        "attempted_visuals": ["Scheme_2"],
        "extracted_data": None,
        "rationale": None,
    }

    # _should_continue routes to tools because Scheme_2/mol_4 has not yet been attempted
    assert graph._should_continue(state) == "tools"


class ItemRecord(BaseModel):
    item_id: str
    value: str


class CollectionOutput(BaseModel):
    records: list[ItemRecord] = Field(default_factory=list)


def test_structured_output_uses_clean_single_turn_enriched_document(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Figure_1": VisualParseRecord(
            visual_id="Figure_1",
            status="success",
            mode="single",
            rendered_markdown="| compound | MIC |\n|---|---|\n| 7d | 0.006 mM |",
        )
    }

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    mock_bound.invoke.side_effect = [
        AIMessage(
            content="Calling tool",
            tool_calls=[
                {
                    "name": "parse_visuals",
                    "args": {"visual_ids": ["Figure_1"]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="READY"),
    ]

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="7d: 0.006 mM", count=1)

    doc_file = tmp_path / "doc_clean" / "doc_clean.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    initial_doc = (
        "<!-- document_id: doc_clean -->\n\n"
        "<visual id=\"Figure_1\" type=\"table\">\nTable 1\n> MIC table\n</visual>"
    )
    doc_file.write_text(initial_doc, encoding="utf-8")

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=3,
        ingestion_dir=tmp_path,
        retry_delay=0.0,
    )

    res = graph.run(
        document_text=initial_doc,
        doc_id="doc_clean",
        system_instruction="Extract all records.",
    )

    assert res["extracted_data"].summary == "7d: 0.006 mM"
    passed_msgs = mock_structured.invoke.call_args[0][0]
    assert len(passed_msgs) == 3
    assert isinstance(passed_msgs[0], SystemMessage)
    assert passed_msgs[0].content.startswith("Extract all records.")
    assert "TARGET SCHEMA FIELDS:" in passed_msgs[0].content
    assert isinstance(passed_msgs[1], HumanMessage)
    assert "| 7d | 0.006 mM |" in passed_msgs[1].content
    assert "> MIC table" not in passed_msgs[1].content


def test_tool_node_merges_visuals_when_disk_unchanged(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Table_1": VisualParseRecord(
            visual_id="Table_1",
            status="success",
            mode="single",
            rendered_markdown="| id | val |\n|---|---|\n| 1a | 5 uM |",
        )
    }

    graph = ExtractionGraph(
        llm=MagicMock(),
        output_model=DummyOutput,
        service=mock_service,
        ingestion_dir=tmp_path,
    )

    stub_doc = (
        "<!-- document_id: doc_mem -->\n\n"
        "<visual id=\"Table_1\" type=\"table\">\nTable 1\n> Activity\n</visual>"
    )
    tool_msg = AIMessage(
        content="Call",
        tool_calls=[
            {"name": "parse_visuals", "args": {"visual_ids": ["Table_1"]}, "id": "c1", "type": "tool_call"}
        ],
    )
    state: ExtractionState = {
        "doc_id": "doc_mem",
        "document_text": stub_doc,
        "system_instruction": "",
        "messages": [tool_msg],
        "tool_iterations": 0,
        "attempted_visuals": [],
        "extracted_data": None,
        "rationale": None,
    }

    res = graph._tool_node(state)
    assert "| 1a | 5 uM |" in res["document_text"]
    assert "> Activity" not in res["document_text"]
    assert graph.get_last_document_text("doc_mem") == res["document_text"]


def test_structured_output_retries_on_transient_read_error(tmp_path):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.side_effect = [
        httpx.ReadError("Connection reset by peer"),
        httpx.ReadError("Connection reset by peer"),
        DummyOutput(summary="Recovered", count=2),
    ]

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        max_retries=3,
        retry_delay=0.0,
    )

    res = graph.run(
        document_text="<!-- document_id: doc_retry -->\n\nPlain text.",
        doc_id="doc_retry",
    )
    assert res["extracted_data"].summary == "Recovered"
    assert mock_structured.invoke.call_count == 3


def test_agent_node_retries_on_transient_connection_error(tmp_path):
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    mock_bound.invoke.side_effect = [
        ConnectionError("Connection reset by peer"),
        AIMessage(content="READY"),
    ]
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="OK", count=1)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        max_retries=2,
        retry_delay=0.0,
    )

    res = graph.run(
        document_text="<!-- document_id: doc_agent_retry -->\n\n<visual id=\"T1\">\n> desc\n</visual>",
        doc_id="doc_agent_retry",
    )
    assert res["extracted_data"].summary == "OK"
    assert mock_bound.invoke.call_count == 2


def test_circuit_breaker_error_is_not_retried(tmp_path):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.side_effect = CircuitBreakerError("Circuit OPEN")

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        max_retries=5,
        retry_delay=0.0,
    )

    with pytest.raises(CircuitBreakerError):
        graph.run(
            document_text="<!-- document_id: doc_cb -->\n\nPlain text.",
            doc_id="doc_cb",
        )
    assert mock_structured.invoke.call_count == 1


def test_batched_extraction_terminates_on_empty_batch_not_under_limit(tmp_path):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    # Batch 1 returns 3 records (< batch_size=5), so it still checks batch 2;
    # Batch 2 returns 1 more record, and batch 3 returns empty -> stops cleanly.
    mock_structured.invoke.side_effect = [
        CollectionOutput(
            records=[
                ItemRecord(item_id="1", value="a"),
                ItemRecord(item_id="2", value="b"),
                ItemRecord(item_id="3", value="c"),
            ]
        ),
        CollectionOutput(records=[ItemRecord(item_id="4", value="d")]),
        CollectionOutput(records=[]),
    ]

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=CollectionOutput,
        ingestion_dir=tmp_path,
        structured_output_batch_size=5,
        max_structured_output_batches=5,
        retry_delay=0.0,
    )

    res = graph.run(
        document_text="<!-- document_id: doc_small -->\n\nSmall doc.",
        doc_id="doc_small",
    )
    assert [r.item_id for r in res["extracted_data"].records] == ["1", "2", "3", "4"]
    assert mock_structured.invoke.call_count == 3


def test_batched_extraction_continuation_dedup_and_convergence(tmp_path):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    rec_a = ItemRecord(item_id="A", value="1")
    rec_b = ItemRecord(item_id="B", value="2")
    rec_c = ItemRecord(item_id="C", value="3")
    rec_d = ItemRecord(item_id="D", value="4")

    mock_structured.invoke.side_effect = [
        CollectionOutput(records=[rec_a, rec_b]),
        CollectionOutput(records=[rec_b, rec_c, rec_d]),
        CollectionOutput(records=[rec_d]),  # 0 new unique records -> stops
    ]

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=CollectionOutput,
        ingestion_dir=tmp_path,
        structured_output_batch_size=2,
        max_structured_output_batches=5,
        retry_delay=0.0,
    )

    res = graph.run(
        document_text="<!-- document_id: doc_dense -->\n\nDense doc.",
        doc_id="doc_dense",
    )
    records = res["extracted_data"].records
    assert [r.item_id for r in records] == ["A", "B", "C", "D"]
    assert mock_structured.invoke.call_count == 3

    second_call_prompt = mock_structured.invoke.call_args_list[1][0][0][-1].content
    assert '"item_id":"A"' in second_call_prompt
    assert '"item_id":"B"' in second_call_prompt


def test_batched_extraction_adaptive_batch_size_and_partial_preservation(tmp_path):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    batch1 = [ItemRecord(item_id=f"id_{i}", value=str(i)) for i in range(10)]
    mock_structured.invoke.side_effect = [
        CollectionOutput(records=batch1),
        OutputParserException("Unterminated JSON"),
        httpx.ReadError("Connection reset by peer"),
    ]

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=CollectionOutput,
        ingestion_dir=tmp_path,
        structured_output_batch_size=10,
        max_structured_output_batches=4,
        max_retries=1,
        retry_delay=0.0,
    )

    res = graph.run(
        document_text="<!-- document_id: doc_partial -->\n\nDense doc.",
        doc_id="doc_partial",
    )
    assert len(res["extracted_data"].records) == 10
    # After OutputParserException on batch 2, retry prompt reduced batch size 10 -> 5
    retry_prompt = mock_structured.invoke.call_args_list[2][0][0][-1].content
    assert "up to 5 remaining unique records" in retry_prompt


def test_structured_output_streaming_path_for_base_chat_model(tmp_path):
    class FakeChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda msgs: DummyOutput(summary="Streamed", count=7))

    graph = ExtractionGraph(
        llm=FakeChatModel(),
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        use_streaming=True,
        retry_delay=0.0,
    )

    res = graph.run(
        document_text="<!-- document_id: doc_stream -->\n\nPlain text.",
        doc_id="doc_stream",
    )
    assert res["extracted_data"].summary == "Streamed"
    assert res["extracted_data"].count == 7


def test_extraction_graph_retries_failed_visual_then_succeeds(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.side_effect = [
        {"Figure_1": VisualParseRecord(visual_id="Figure_1", status="failed", mode="single")},
        {
            "Figure_1": VisualParseRecord(
                visual_id="Figure_1",
                status="success",
                mode="single",
                rendered_markdown="| item | val |\n|---|---|\n| X | 42 |",
            )
        },
    ]

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    call_fig1 = AIMessage(
        content="Call Figure_1",
        tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Figure_1"]}, "id": "c1", "type": "tool_call"}],
    )
    call_fig1_retry = AIMessage(
        content="Retry Figure_1",
        tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Figure_1"]}, "id": "c2", "type": "tool_call"}],
    )
    mock_bound.invoke.side_effect = [call_fig1, call_fig1_retry]

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="X=42", count=1)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=3,
        max_visual_attempts=2,
        ingestion_dir=tmp_path,
        retry_delay=0.0,
    )

    doc_text = "<!-- document_id: doc_retry_vis -->\n\n<visual id=\"Figure_1\" type=\"table\">\nTable 1\n> Data\n</visual>"
    res = graph.run(document_text=doc_text, doc_id="doc_retry_vis")

    assert mock_service.parse_visuals.call_count == 2
    assert res["extracted_data"].summary == "X=42"
    assert "Figure_1" in res["completed_visuals"]
    assert "| X | 42 |" in res["document_text"]


def test_extraction_graph_caps_repeated_visual_failures(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Figure_1": VisualParseRecord(visual_id="Figure_1", status="failed", mode="single")
    }

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    mock_bound.invoke.side_effect = [
        AIMessage(
            content="Try 1",
            tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Figure_1"]}, "id": "c1", "type": "tool_call"}],
        ),
        AIMessage(
            content="Try 2",
            tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Figure_1"]}, "id": "c2", "type": "tool_call"}],
        ),
        AIMessage(
            content="Try 3 (should be blocked)",
            tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Figure_1"]}, "id": "c3", "type": "tool_call"}],
        ),
    ]

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="Fallback", count=0)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=5,
        max_visual_attempts=2,
        ingestion_dir=tmp_path,
        retry_delay=0.0,
    )

    doc_text = "<!-- document_id: doc_cap -->\n\n<visual id=\"Figure_1\" type=\"table\">\nTable 1\n> Data\n</visual>"
    res = graph.run(document_text=doc_text, doc_id="doc_cap")

    # Capped at max_visual_attempts=2 despite max_tool_iterations=5
    assert mock_service.parse_visuals.call_count == 2
    assert "Figure_1" in res["attempted_visuals"]
    assert res["extracted_data"].summary == "Fallback"


def test_extraction_graph_target_hint_allows_new_hint_and_full_reparse_blocks_duplicate_hint(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.side_effect = [
        {
            "Figure_2": VisualParseRecord(
                visual_id="Figure_2",
                status="partial",
                mode="composite",
                rendered_markdown="| item | val |\n|---|---|\n| A | 1 |\n<element id=\"Figure_2/panel_b\">[Unparsed: Panel B]</element>",
                parsed_element_ids=["panel_a"],
                unparsed_element_ids=["panel_b"],
            )
        },
        {
            "Figure_2": VisualParseRecord(
                visual_id="Figure_2",
                status="success",
                mode="composite",
                rendered_markdown="| item | val |\n|---|---|\n| A | 1 |\n| B | 2 |",
                parsed_element_ids=["panel_a", "panel_b"],
                unparsed_element_ids=[],
            )
        },
    ]

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    mock_bound.invoke.side_effect = [
        AIMessage(
            content="Scoped parse",
            tool_calls=[
                {
                    "name": "parse_visuals",
                    "args": {"visual_ids": ["Figure_2"], "target_hint": "panel a"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="Full re-parse",
            tool_calls=[
                {
                    "name": "parse_visuals",
                    "args": {"visual_ids": ["Figure_2"]},
                    "id": "c2",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="A+B", count=2)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=4,
        ingestion_dir=tmp_path,
        retry_delay=0.0,
    )

    doc_text = "<!-- document_id: doc_reparse -->\n\n<visual id=\"Figure_2\" type=\"chart\">\nFig 2\n> Panels A and B\n</visual>"
    res = graph.run(document_text=doc_text, doc_id="doc_reparse")

    assert mock_service.parse_visuals.call_count == 2
    assert mock_service.parse_visuals.call_args_list[0].kwargs == {"target_hint": "panel a"}
    assert mock_service.parse_visuals.call_args_list[1].kwargs == {}
    assert "Figure_2" in res["completed_visuals"]
    assert "Figure_2" not in res["partial_visuals"]
    assert "| B | 2 |" in res["document_text"]

    # Also verify duplicate exact hint is blocked while a different hint is allowed on partial state
    dup_state: ExtractionState = {
        "doc_id": "doc_reparse",
        "document_text": "text",
        "system_instruction": "",
        "messages": [
            AIMessage(
                content="dup hint",
                tool_calls=[
                    {
                        "name": "parse_visuals",
                        "args": {"visual_ids": ["Figure_2"], "target_hint": "Panel A"},
                        "id": "c_dup",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "tool_iterations": 1,
        "attempted_visuals": [],
        "completed_visuals": [],
        "partial_visuals": ["Figure_2"],
        "attempted_hints": {"Figure_2": ["panel a"]},
        "visual_attempt_counts": {},
        "extracted_data": None,
        "rationale": None,
    }
    assert graph._should_continue(dup_state) == "structured_output"

    new_hint_state = dict(dup_state)
    new_hint_state["messages"] = [
        AIMessage(
            content="new hint",
            tool_calls=[
                {
                    "name": "parse_visuals",
                    "args": {"visual_ids": ["Figure_2"], "target_hint": "panel b"},
                    "id": "c_new",
                    "type": "tool_call",
                }
            ],
        )
    ]
    assert graph._should_continue(new_hint_state) == "tools"


def test_extraction_graph_budget_feedback_and_domain_independence(tmp_path):
    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Table_1": VisualParseRecord(
            visual_id="Table_1",
            status="success",
            mode="single",
            rendered_markdown="| k | v |\n|---|---|\n| 1 | 2 |",
        )
    }

    graph = ExtractionGraph(
        llm=MagicMock(),
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=2,
        ingestion_dir=tmp_path,
    )

    doc_text = (
        "<!-- document_id: doc_budget -->\n\n"
        "<visual id=\"Table_1\" type=\"table\">\nT1\n> Table 1\n</visual>\n"
        "<visual id=\"Table_2\" type=\"table\">\nT2\n> Table 2\n</visual>"
    )
    prep = graph._prepare_input(
        {
            "doc_id": "doc_budget",
            "document_text": doc_text,
            "system_instruction": "Extract items.",
            "messages": [],
            "tool_iterations": 0,
            "attempted_visuals": [],
            "extracted_data": None,
            "rationale": None,
        }
    )
    sys_msg = prep["messages"][0].content
    assert "TOOL ITERATION BUDGET: You have at most 2 tool-calling iteration(s)" in sys_msg
    assert "unparsed visual stub(s)/element(s) present" not in sys_msg
    assert "products 6a-6f" not in sys_msg
    assert "kinetics" not in sys_msg

    # Iteration 1 of 2 -> should include FINAL iteration warning
    tool_res = graph._tool_node(
        {
            "doc_id": "doc_budget",
            "document_text": doc_text,
            "system_instruction": "",
            "messages": [
                AIMessage(
                    content="call",
                    tool_calls=[
                        {"name": "parse_visuals", "args": {"visual_ids": ["Table_1"]}, "id": "c1", "type": "tool_call"}
                    ],
                )
            ],
            "tool_iterations": 0,
            "attempted_visuals": [],
            "completed_visuals": [],
            "partial_visuals": [],
            "attempted_hints": {},
            "visual_attempt_counts": {},
            "extracted_data": None,
            "rationale": None,
        }
    )
    tool_msg_content = tool_res["messages"][0].content
    assert "1/2 tool iteration(s) used, 1 FINAL iteration remaining" in tool_msg_content
    assert "1 eligible unparsed/partial visual(s)" in tool_msg_content


def test_extraction_graph_multiturn_in_memory_merge_with_unchanged_disk_file(tmp_path):
    doc_file = tmp_path / "doc_multi" / "doc_multi.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    initial_doc = (
        "<!-- document_id: doc_multi -->\n\n"
        "<visual id=\"T1\" type=\"table\">\nT1\n> Desc 1\n</visual>\n\n"
        "<visual id=\"T2\" type=\"table\">\nT2\n> Desc 2\n</visual>"
    )
    doc_file.write_text(initial_doc, encoding="utf-8")

    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.side_effect = [
        {"T1": VisualParseRecord(visual_id="T1", status="success", mode="single", rendered_markdown="| a |\n|---|\n| 1 |")},
        {"T2": VisualParseRecord(visual_id="T2", status="success", mode="single", rendered_markdown="| b |\n|---|\n| 2 |")},
    ]

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_llm.bind_tools.return_value = mock_bound
    mock_bound.invoke.side_effect = [
        AIMessage(content="t1", tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["T1"]}, "id": "c1", "type": "tool_call"}]),
        AIMessage(content="t2", tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["T2"]}, "id": "c2", "type": "tool_call"}]),
    ]
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = DummyOutput(summary="Both", count=2)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        service=mock_service,
        max_tool_iterations=3,
        ingestion_dir=tmp_path,
        retry_delay=0.0,
    )

    res = graph.run(document_text=initial_doc, doc_id="doc_multi")
    assert "| 1 |" in res["document_text"]
    assert "| 2 |" in res["document_text"]
    assert "> Desc 1" not in res["document_text"]
    assert "> Desc 2" not in res["document_text"]


def test_extraction_graph_logs_tool_calls_and_structured_output(tmp_path):
    """Verify BaseChatModel tool calls and structured output are streamed to LLMHistoryRecorder."""
    import json
    from typing import Iterator, List, Optional
    from langchain_core.messages import AIMessageChunk, BaseMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.outputs import ChatGenerationChunk
    from ae.core.llm.history_logger import LLMHistoryRecorder

    class FakeReActChatModel(BaseChatModel):
        model_name: str = "fake-gemini-flash"
        step: int = 0

        @property
        def _llm_type(self) -> str:
            return "fake-react"

        def bind_tools(self, tools, **kwargs):
            return self

        def with_structured_output(self, schema, **kwargs):
            return self | PydanticOutputParser(pydantic_object=schema)

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:
            self.step += 1
            if self.step == 1:
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "parse_visuals",
                                        "args": {"visual_ids": ["Table_1"]},
                                        "id": "call_t1",
                                        "type": "tool_call",
                                    }
                                ],
                                usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                            )
                        )
                    ]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content='{"summary": "Extracted ok", "count": 3}',
                            usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
                        )
                    )
                ]
            )

        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs,
        ) -> Iterator[ChatGenerationChunk]:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content='{"summary": "Extracted ok", "count": 3}',
                    usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
                    response_metadata={"finish_reason": "STOP"},
                ),
                generation_info={"finish_reason": "STOP"},
            )

    doc_id = "doc_log_ok"
    doc_file = tmp_path / doc_id / f"{doc_id}.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text(
        '<visual id="Table_1" type="table">\nTable 1\n> MIC table\n</visual>',
        encoding="utf-8",
    )

    mock_service = MagicMock(spec=VisualParsingService)
    mock_service.parse_visuals.return_value = {
        "Table_1": VisualParseRecord(
            visual_id="Table_1",
            status="success",
            mode="single",
            rendered_markdown="| cmp | mic |\n|---|---|\n| 7d | 0.006 |",
        )
    }

    recorder = LLMHistoryRecorder()
    log_file = tmp_path / "logs" / "extraction.jsonl"
    recorder.start_stream(log_file)

    graph = ExtractionGraph(
        llm=FakeReActChatModel(),
        output_model=DummyOutput,
        service=mock_service,
        ingestion_dir=tmp_path,
        retry_delay=0.0,
        history_recorder=recorder,
    )
    res = graph.run(document_text=doc_file.read_text(encoding="utf-8"), doc_id=doc_id)
    recorder.close()

    assert res["extracted_data"] == DummyOutput(summary="Extracted ok", count=3)

    entries = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert entries[0]["doc_id"] == doc_id
    assert entries[0]["node"] == "agent_node"
    assert entries[0]["status"] == "success"
    assert entries[0]["tool_calls"][0]["name"] == "parse_visuals"
    assert entries[1]["doc_id"] == doc_id
    assert entries[1]["node"] == "structured_output_node"
    assert entries[1]["status"] == "success"
    assert entries[1]["usage"]["total_tokens"] == 40
    assert entries[1]["finish_reason"] == "STOP"


def test_extraction_graph_logs_parse_error_stream_cutoff_and_graph_failure(tmp_path):
    """Verify truncated stream errors, parse errors, and graph_failure are logged."""
    import json
    from typing import Iterator, List, Optional
    from langchain_core.messages import AIMessageChunk, BaseMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.outputs import ChatGenerationChunk
    from ae.core.llm.history_logger import LLMHistoryRecorder

    class FailingStructuredChatModel(BaseChatModel):
        model_name: str = "fake-gemini-fail"

        @property
        def _llm_type(self) -> str:
            return "fake-fail"

        def with_structured_output(self, schema, **kwargs):
            return self | PydanticOutputParser(pydantic_object=schema)

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:
            # Return invalid/truncated JSON missing required field -> OutputParserException
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content='{"wrong_field": "cut off',
                            response_metadata={"finish_reason": "MAX_TOKENS"},
                        )
                    )
                ]
            )

        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager=None,
            **kwargs,
        ) -> Iterator[ChatGenerationChunk]:
            yield ChatGenerationChunk(message=AIMessageChunk(content='{"summary": "partial'))
            raise ConnectionResetError("stream aborted")

    recorder = LLMHistoryRecorder()
    log_file = tmp_path / "logs" / "failures.jsonl"
    recorder.start_stream(log_file)

    # 1. Stream cutoff error -> retry -> second stream cutoff -> graph_failure
    graph_stream_fail = ExtractionGraph(
        llm=FailingStructuredChatModel(),
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        max_retries=1,
        retry_delay=0.0,
        use_streaming=True,
        history_recorder=recorder,
    )
    with pytest.raises(ConnectionResetError):
        graph_stream_fail.run(document_text="<!-- document_id: doc_cutoff -->\n\nText")

    # 2. Invoke parse/truncation error (use_streaming=False) -> on_llm_end raw output + OutputParserException + graph_failure
    graph_parse_fail = ExtractionGraph(
        llm=FailingStructuredChatModel(),
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        max_retries=0,
        retry_delay=0.0,
        use_streaming=False,
        history_recorder=recorder,
    )
    with pytest.raises(OutputParserException):
        graph_parse_fail.run(document_text="<!-- document_id: doc_parse_err -->\n\nText")

    recorder.close()
    entries = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]

    cutoff_entries = [e for e in entries if e.get("doc_id") == "doc_cutoff"]
    assert any(
        e["status"] == "error" and e.get("partial_output") == '{"summary": "partial'
        for e in cutoff_entries
    )
    assert any(e.get("stage") == "graph_failure" and e["status"] == "error" for e in cutoff_entries)

    parse_entries = [e for e in entries if e.get("doc_id") == "doc_parse_err"]
    assert any(
        e["status"] == "success"
        and e.get("outputs") == ['{"wrong_field": "cut off']
        and e.get("finish_reason") == "MAX_TOKENS"
        for e in parse_entries
    )
    assert any(
        e["status"] == "error" and e.get("error", {}).get("type") == "OutputParserException"
        for e in parse_entries
    )
    assert any(e.get("stage") == "graph_failure" for e in parse_entries)


def test_structured_output_node_includes_schema_summary_in_system_message(tmp_path):
    from typing import Literal

    class ConstrainedOutput(BaseModel):
        relation: Literal["=", ">", "<"] = Field(description="Relation operator")
        score: float = Field(ge=0.0, le=100.0, description="Score")

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = ConstrainedOutput(relation="=", score=95.0)

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=ConstrainedOutput,
        ingestion_dir=tmp_path,
    )

    state: ExtractionState = {
        "doc_id": "doc_schema_summary",
        "document_text": "<!-- document_id: doc_schema_summary -->\n\nPlain text.",
        "system_instruction": "Extract structured data.",
        "messages": [],
        "tool_iterations": 0,
        "extracted_data": None,
        "rationale": None,
    }

    graph._structured_output_node(state)

    mock_structured.invoke.assert_called_once()
    sent_messages = mock_structured.invoke.call_args[0][0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert "TARGET SCHEMA FIELDS:" in sent_messages[0].content
    assert "allowed: ['=', '>', '<']" in sent_messages[0].content
    assert "min=0.0, max=100.0" in sent_messages[0].content


def test_prepare_input_resets_from_stage02_manifest_while_preserving_vlm_cache(tmp_path):
    mock_llm = MagicMock()
    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
    )

    doc_id = "paper_reset"
    doc_dir = tmp_path / doc_id
    manifest_dir = doc_dir / "02_manifest_visuals"
    vlm_cache_dir = doc_dir / "04_parse_visuals" / "visuals"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    vlm_cache_dir.mkdir(parents=True, exist_ok=True)

    clean_manifest_md = (
        f"<!-- document_id: {doc_id} -->\n\n"
        '<visual id="Table_1" type="table">\nTable 1\n> MIC values\n</visual>\n\n'
        '<visual id="Scheme_Distractor" type="molecular_structure">\nScheme 1\n> Precursors\n</visual>'
    )
    (manifest_dir / "manifest_enriched_text.md").write_text(clean_manifest_md, encoding="utf-8")

    # Simulate dirty doc.md previously enriched during RE (including distractor scheme)
    doc_file = doc_dir / f"{doc_id}.md"
    doc_file.write_text(
        f"<!-- document_id: {doc_id} -->\n\n"
        '<visual id="Table_1" type="table">\n| Cpd | MIC |\n|---|---|\n| 1a | 2 |\n</visual>\n\n'
        '<visual id="Scheme_Distractor" type="molecular_structure">\n| Precursor | SMILES |\n|---|---|\n| P1 | CC |\n</visual>',
        encoding="utf-8",
    )
    cached_table_json = vlm_cache_dir / "Table_1.json"
    cached_table_json.write_text('{"visual_id": "Table_1", "status": "success"}', encoding="utf-8")

    state: ExtractionState = {
        "doc_id": doc_id,
        "document_text": "",
        "system_instruction": "Extract records.",
        "messages": [],
        "tool_iterations": 0,
        "extracted_data": None,
        "rationale": None,
    }
    res = graph._prepare_input(state)

    assert res["document_text"] == clean_manifest_md
    assert doc_file.read_text(encoding="utf-8") == clean_manifest_md
    assert cached_table_json.exists()
    assert graph._route_after_prepare(res) == "agent"


def test_extraction_graph_runs_schema_domain_tool_without_unparsed_visuals(tmp_path):
    """Domain tools declared on the output model run when applicable even if no visuals are unparsed."""
    from langchain_core.tools import StructuredTool
    from ae.core.schema.tools import ExtractionToolSpec, DEFAULT_TOOL_REGISTRY, RegisteredToolProvider

    def _compute_norm(code: str, val: str) -> str:
        return json.dumps({"code": code, "normalized": f"{code}:{val}"})

    provider = RegisteredToolProvider(
        name="dummy_domain_calculator",
        build_tool=lambda **_: StructuredTool.from_function(
            func=_compute_norm,
            name="dummy_domain_calculator",
            description="Normalize coded values",
        ),
        is_applicable=lambda txt: "CALC_NEEDED" in txt,
        policy_hint="Call dummy_domain_calculator when CALC_NEEDED appears.",
    )
    DEFAULT_TOOL_REGISTRY.register("dummy_domain_calculator", provider)
    try:
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "dummy_domain_calculator",
                        "args": {"code": "X1", "val": "99"},
                        "id": "tc_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="READY"),
        ]

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = DummyOutput(summary="X1:99", count=1)

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=DummyOutput,
            ingestion_dir=tmp_path,
            max_tool_iterations=3,
            tools=[ExtractionToolSpec(name="dummy_domain_calculator")],
        )

        # Document with CALC_NEEDED -> routes through agent -> calls tool -> READY -> structured output
        res = graph.run(
            document_text="<!-- document_id: doc_calc -->\n\nCALC_NEEDED for X1 = 99.",
            doc_id="doc_calc",
        )
        assert res["extracted_data"].summary == "X1:99"
        assert len(res["tool_results"]) == 1
        assert "X1:99" in res["tool_results"][0]["output"]
        sent_human = mock_structured.invoke.call_args[0][0][-2].content
        assert "COMPUTED TOOL RESULTS (AUTHORITATIVE)" in sent_human
        assert "X1:99" in sent_human

        # Document WITHOUT CALC_NEEDED -> skips agent_node completely (0 extra LLM calls)
        mock_bound.invoke.reset_mock()
        mock_structured.invoke.reset_mock()
        res2 = graph.run(
            document_text="<!-- document_id: doc_plain -->\n\nPlain text without trigger.",
            doc_id="doc_plain",
        )
        assert mock_bound.invoke.call_count == 0
        assert res2["extracted_data"].summary == "X1:99"
    finally:
        DEFAULT_TOOL_REGISTRY.unregister("dummy_domain_calculator")


def test_extraction_graph_activates_domain_tool_after_visual_parsing(tmp_path):
    """If parsing a visual introduces a domain-tool trigger, _route_after_tool routes back to agent."""
    from langchain_core.tools import StructuredTool
    from ae.core.schema.tools import ExtractionToolSpec, DEFAULT_TOOL_REGISTRY, RegisteredToolProvider

    provider = RegisteredToolProvider(
        name="post_visual_assembler",
        build_tool=lambda **_: StructuredTool.from_function(
            func=lambda token: f"ASSEMBLED_{token}",
            name="post_visual_assembler",
            description="Assemble post-visual fragments",
        ),
        is_applicable=lambda txt: "[*:1]" in txt,
        policy_hint="Use post_visual_assembler whenever [*:1] appears.",
    )
    DEFAULT_TOOL_REGISTRY.register("post_visual_assembler", provider)
    try:
        mock_service = MagicMock(spec=VisualParsingService)
        mock_service.parse_visuals.return_value = {
            "Scheme_1": VisualParseRecord(
                visual_id="Scheme_1",
                status="success",
                mode="single",
                rendered_markdown="Scaffold: c1ccccc1[*:1]",
            )
        }

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "parse_visuals",
                        "args": {"visual_ids": ["Scheme_1"]},
                        "id": "v1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "post_visual_assembler",
                        "args": {"token": "5a"},
                        "id": "d1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="READY"),
        ]

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = DummyOutput(summary="ASSEMBLED_5a", count=1)

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=DummyOutput,
            service=mock_service,
            ingestion_dir=tmp_path,
            max_tool_iterations=4,
            tools=[ExtractionToolSpec(name="post_visual_assembler")],
        )

        doc_text = (
            "<!-- document_id: doc_two_stage -->\n\n"
            '<visual id="Scheme_1" type="molecular_structure">\nScheme 1\n> Compounds 5a\n</visual>'
        )
        res = graph.run(document_text=doc_text, doc_id="doc_two_stage")
        assert mock_service.parse_visuals.call_count == 1
        assert mock_bound.invoke.call_count == 3
        assert len(res["tool_results"]) == 1
        assert res["tool_results"][0]["output"] == "ASSEMBLED_5a"
    finally:
        DEFAULT_TOOL_REGISTRY.unregister("post_visual_assembler")






def test_agent_node_passes_thinking_level_only_when_supported(tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ThinkingChatModel(BaseChatModel):
        thinking_level: str = "medium"
        bind_calls: list[dict] = []

        @property
        def _llm_type(self) -> str:
            return "fake-thinking"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="READY"))])

        def bind_tools(self, tools, **kwargs):
            self.bind_calls.append(kwargs)
            bound = MagicMock()
            bound.invoke.return_value = AIMessage(content="READY")
            return bound

    class PlainChatModel(BaseChatModel):
        bind_calls: list[dict] = []

        @property
        def _llm_type(self) -> str:
            return "fake-plain"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="READY"))])

        def bind_tools(self, tools, **kwargs):
            self.bind_calls.append(kwargs)
            bound = MagicMock()
            bound.invoke.return_value = AIMessage(content="READY")
            return bound

    thinking_llm = ThinkingChatModel()
    g1 = ExtractionGraph(
        llm=thinking_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        agent_thinking_level="low",
    )
    g1._agent_node({"doc_id": "d1", "document_text": "hello", "messages": [HumanMessage(content="hi")]})
    assert thinking_llm.bind_calls == [{"thinking_level": "low"}]

    plain_llm = PlainChatModel()
    g2 = ExtractionGraph(
        llm=plain_llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        agent_thinking_level="low",
    )
    g2._agent_node({"doc_id": "d2", "document_text": "hello", "messages": [HumanMessage(content="hi")]})
    assert plain_llm.bind_calls == [{}]


def test_route_after_tool_skips_idle_ready_and_retries_on_tool_error(tmp_path):
    graph = ExtractionGraph(
        llm=MagicMock(),
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        max_tool_iterations=4,
    )
    # After a non-visual tool call with remaining budget, re-enter agent so sequential/multi-series calls work
    non_visual_state = {
        "document_text": "plain text",
        "tool_iterations": 1,
        "attempted_visuals": [],
        "completed_visuals": [],
        "partial_visuals": [],
        "attempted_hints": {},
        "visual_attempt_counts": {},
        "non_visual_tools_called_since_doc_change": ["some_tool"],
        "last_turn_had_non_visual": True,
        "last_turn_had_tool_error": False,
    }
    assert graph._route_after_tool(non_visual_state) == "agent"

    # At budget limit -> go straight to structured_output
    budget_exhausted = dict(non_visual_state, tool_iterations=4)
    assert graph._route_after_tool(budget_exhausted) == "structured_output"

    # When no non-visual tool ran and no visuals/errors remain -> go straight to structured_output
    idle_state = dict(non_visual_state, last_turn_had_non_visual=False)
    assert graph._route_after_tool(idle_state) == "structured_output"

    error_state = dict(idle_state, last_turn_had_tool_error=True)
    assert graph._route_after_tool(error_state) == "agent"


def test_build_structured_messages_separates_static_prefix_and_dynamic_instruction(tmp_path):
    graph = ExtractionGraph(
        llm=MagicMock(),
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
    )
    state = {
        "doc_id": "doc_cache",
        "document_text": "Static document body.",
        "system_instruction": "System prompt.",
        "tool_results": [{"tool": "calc", "args": {}, "output": "42"}],
    }
    msgs = graph._build_structured_messages(state, "Dynamic batch 1 instruction")
    assert len(msgs) == 3
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)
    assert "Static document body." in msgs[1].content
    assert "42" in msgs[1].content
    assert "Dynamic batch 1 instruction" not in msgs[1].content
    assert isinstance(msgs[2], HumanMessage)
    assert msgs[2].content == "Dynamic batch 1 instruction"


def test_invoke_with_retry_does_not_double_count_latency_on_post_validation_error(tmp_path):
    from ae.core.llm.history_logger import extraction_log_context, update_extraction_log_context
    from ae.extraction.sanity import StructuredOutputSanityError

    recorded: list[dict] = []
    mock_recorder = MagicMock()
    mock_recorder.record.side_effect = lambda entry: recorded.append(entry)

    graph = ExtractionGraph(
        llm=MagicMock(),
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        max_retries=0,
        retry_delay=0.0,
        history_recorder=mock_recorder,
    )

    def _fn():
        update_extraction_log_context(callback_invoked=True, last_run_id="run-123")
        raise StructuredOutputSanityError("bad field", reasons=["bad field"])

    with extraction_log_context(doc_id="doc_lat", node="structured_output_node", stage="structured_output"):
        with pytest.raises(StructuredOutputSanityError):
            graph._invoke_with_retry("structured_output[doc_lat]", _fn)

    assert len(recorded) == 1
    assert recorded[0]["stage"] == "post_validation_error"
    assert recorded[0]["latency_s"] == 0.0
    assert recorded[0]["llm_run_id"] == "run-123"


def test_local_record_repair_patches_only_failed_index_and_falls_back_above_threshold(tmp_path):
    from ae.core.schema import FieldSpec, SchemaConfig, create_all_models

    cfg = SchemaConfig(
        name="chem_repair_test",
        fields={
            "compound_id": FieldSpec(type=str, description="Compound ID"),
            "smiles": FieldSpec(type=str, description="SMILES", required=False, format="smiles"),
        },
        compare_fields=["compound_id"],
    )
    rec_model, out_model = create_all_models(cfg)

    # Case 1: 1 broken record out of 5 (index 2) -> local repair patches only index 2
    initial_out = out_model(
        records=[
            rec_model(compound_id="1a", smiles="C"),
            rec_model(compound_id="1b", smiles="CC"),
            rec_model(compound_id="1c", smiles="c1ccc([*:1])cc1"),
            rec_model(compound_id="1d", smiles="CCCC"),
            rec_model(compound_id="1e", smiles="CCCCC"),
        ]
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    def _with_structured_output(schema):
        if schema is out_model:
            return mock_structured
        # Dynamic PatchedRecordsOutput
        repair_mock = MagicMock()
        item_cls = schema.model_fields["repairs"].annotation.__args__[0]
        repair_mock.invoke.return_value = schema(
            repairs=[item_cls(index=2, record=rec_model(compound_id="1c", smiles="c1ccccc1"))]
        )
        return repair_mock

    mock_llm.with_structured_output.side_effect = _with_structured_output
    mock_structured.invoke.return_value = initial_out

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=out_model,
        ingestion_dir=tmp_path,
        max_retries=1,
        retry_delay=0.0,
    )
    res = graph.run(
        document_text="<!-- document_id: doc_rep -->\n\nCompounds 1a..1e.",
        doc_id="doc_rep",
    )
    # Full extraction called only once (no full retry!)
    assert mock_structured.invoke.call_count == 1
    recs = res["extracted_data"].records
    assert [r.smiles for r in recs] == ["C", "CC", "c1ccccc1", "CCCC", "CCCCC"]

    # Case 2: 2 broken records out of 5 (> threshold 5 // 3 == 1) -> full retry
    bad_two = out_model(
        records=[
            rec_model(compound_id="1a", smiles="c1ccc([*:1])cc1"),
            rec_model(compound_id="1b", smiles="CC"),
            rec_model(compound_id="1c", smiles="c1ccc([*:1])cc1"),
            rec_model(compound_id="1d", smiles="CCCC"),
            rec_model(compound_id="1e", smiles="CCCCC"),
        ]
    )
    good_all = out_model(
        records=[
            rec_model(compound_id="1a", smiles="C"),
            rec_model(compound_id="1b", smiles="CC"),
            rec_model(compound_id="1c", smiles="CCC"),
            rec_model(compound_id="1d", smiles="CCCC"),
            rec_model(compound_id="1e", smiles="CCCCC"),
        ]
    )
    mock_structured.invoke.reset_mock()
    mock_structured.invoke.side_effect = [bad_two, good_all]
    res2 = graph.run(
        document_text="<!-- document_id: doc_rep2 -->\n\nCompounds 1a..1e.",
        doc_id="doc_rep2",
    )
    assert mock_structured.invoke.call_count == 2
    assert [r.smiles for r in res2["extracted_data"].records] == ["C", "CC", "CCC", "CCCC", "CCCCC"]


def test_finalize_evidence_static_binding_and_no_dynamic_counter(tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ToolChoiceChatModel(BaseChatModel):
        bind_calls: list[dict] = []

        @property
        def _llm_type(self) -> str:
            return "chat-google-generative-ai"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            self.bind_calls.append({"tool_names": [t.name for t in tools], "tool_choice": tool_choice})
            bound = MagicMock()
            bound.invoke.return_value = AIMessage(
                content="",
                tool_calls=[{"name": "finalize_evidence", "args": {"confirm": True}, "id": "fc_1"}],
            )
            return bound

        def with_structured_output(self, schema, **kwargs):
            mock_s = MagicMock()
            mock_s.invoke.return_value = DummyOutput(summary="Done")
            return mock_s

    llm = ToolChoiceChatModel()
    graph = ExtractionGraph(
        llm=llm,
        output_model=DummyOutput,
        ingestion_dir=tmp_path,
        tools=["assemble_markush_smiles"],
    )

    doc_text = '<!-- document_id: doc_static -->\n\n<visual id="T1">\n> table\n</visual>'
    prep = graph._prepare_input({"doc_id": "doc_static", "document_text": doc_text, "system_instruction": "Base"})
    sys_msg = prep["messages"][0].content
    assert "unparsed visual stub(s)/element(s) present" not in sys_msg
    assert "finalize_evidence" in sys_msg
    assert "TOOL `assemble_markush_smiles`:" in sys_msg

    res = graph.run(document_text=doc_text, doc_id="doc_static", system_instruction="Base")
    assert res["extracted_data"].summary == "Done"
    assert len(llm.bind_calls) == 1
    assert llm.bind_calls[0]["tool_choice"] == "any"
    assert llm.bind_calls[0]["tool_names"] == [
        "parse_visuals",
        "assemble_markush_smiles",
        "finalize_evidence",
    ]


def test_crane_excludes_pattern_from_json_schema_but_keeps_in_summary(tmp_path):
    from ae.core.schema import FieldSpec, SchemaConfig, create_all_models
    from ae.core.schema.dynamic_models import extract_schema_field_constraints

    cfg = SchemaConfig(
        name="crane_test",
        fields={
            "code": FieldSpec(type=str, description="Compound code", pattern=r"^[A-Za-z0-9_-]+$"),
        },
        compare_fields=["code"],
    )
    _, out_model = create_all_models(cfg)
    json_schema = out_model.model_json_schema()
    assert '"pattern"' not in json.dumps(json_schema)

    constraints = extract_schema_field_constraints(out_model)
    assert constraints["code"]["pattern"] == r"^[A-Za-z0-9_-]+$"

    graph = ExtractionGraph(llm=MagicMock(), output_model=out_model, ingestion_dir=tmp_path)
    assert "pattern: '^[A-Za-z0-9_-]+$'" in graph._format_schema_summary()


def test_retry_temperature_raised_only_on_retry_and_repair(tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from ae.core.schema import FieldSpec, SchemaConfig, create_all_models

    cfg = SchemaConfig(
        name="temp_retry_test",
        fields={
            "compound_id": FieldSpec(type=str, required=True),
            "smiles": FieldSpec(type=str, required=True, validators=["smiles"]),
        },
        compare_fields=["compound_id", "smiles"],
    )
    rec_model, out_model = create_all_models(cfg)
    temps_seen: list[float] = []

    class FakeStructuredRunnable:
        def __init__(self, temp=0.0):
            self.temp = temp

        def bind(self, **kwargs):
            return FakeStructuredRunnable(temp=kwargs.get("temperature", self.temp))

        def invoke(self, _messages):
            temps_seen.append(self.temp)
            if len(temps_seen) == 1:
                # 1st attempt: invalid placeholder SMILES -> triggers local repair
                return out_model(
                    records=[
                        rec_model(compound_id="1a", smiles="c1ccc([*:1])cc1"),
                        rec_model(compound_id="1b", smiles="CC"),
                        rec_model(compound_id="1c", smiles="CCC"),
                        rec_model(compound_id="1d", smiles="CCCC"),
                    ]
                )
            # Repair call
            from pydantic import BaseModel as _BM
            return type("_Repair", (_BM,), {})()

    class TempChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-temp"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        def with_structured_output(self, schema, **kwargs):
            if "PatchOutput" in schema.__name__:
                class RepairRunnable:
                    def __init__(self, temp=0.0):
                        self.temp = temp

                    def bind(self, **kw):
                        return RepairRunnable(temp=kw.get("temperature", self.temp))

                    def invoke(self, _msgs):
                        temps_seen.append(self.temp)
                        return schema(repairs=[{"index": 0, "record": {"compound_id": "1a", "smiles": "C"}}])

                return RepairRunnable()
            return FakeStructuredRunnable()

    graph = ExtractionGraph(
        llm=TempChatModel(),
        output_model=out_model,
        ingestion_dir=tmp_path,
        use_streaming=False,
        retry_delay=0.0,
    )
    res = graph.run(
        document_text="<!-- document_id: doc_t -->\n\nCompounds 1a 1b 1c 1d.",
        doc_id="doc_t",
    )
    # First repair attempt runs at temperature 0.0
    assert temps_seen == [0.0, 0.0]
    assert [r.smiles for r in res["extracted_data"].records] == ["C", "CC", "CCC", "CCCC"]

    # If the first repair attempt fails with a parse error, the repair retry raises temperature to 0.7
    temps_retry: list[float] = []

    class RetryRepairChatModel(BaseChatModel):
        repair_calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "fake-temp-retry"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        def with_structured_output(self, schema, **kwargs):
            outer = self
            if "PatchOutput" in schema.__name__:
                class RepairRetryRunnable:
                    def __init__(self, temp=0.0):
                        self.temp = temp

                    def bind(self, **kw):
                        return RepairRetryRunnable(temp=kw.get("temperature", self.temp))

                    def invoke(self, _msgs):
                        outer.repair_calls += 1
                        temps_retry.append(self.temp)
                        if outer.repair_calls == 1:
                            raise OutputParserException("Truncated repair JSON")
                        return schema(repairs=[{"index": 0, "record": {"compound_id": "1a", "smiles": "C"}}])

                return RepairRetryRunnable()

            class InitRunnable:
                def __init__(self, temp=0.0):
                    self.temp = temp

                def bind(self, **kw):
                    return InitRunnable(temp=kw.get("temperature", self.temp))

                def invoke(self, _msgs):
                    temps_retry.append(self.temp)
                    return out_model(
                        records=[
                            rec_model(compound_id="1a", smiles="c1ccc([*:1])cc1"),
                            rec_model(compound_id="1b", smiles="CC"),
                            rec_model(compound_id="1c", smiles="CCC"),
                            rec_model(compound_id="1d", smiles="CCCC"),
                        ]
                    )

            return InitRunnable()

    graph_retry = ExtractionGraph(
        llm=RetryRepairChatModel(),
        output_model=out_model,
        ingestion_dir=tmp_path,
        use_streaming=False,
        max_retries=2,
        retry_delay=0.0,
    )
    res_retry = graph_retry.run(
        document_text="<!-- document_id: doc_t_retry -->\n\nCompounds 1a 1b 1c 1d.",
        doc_id="doc_t_retry",
    )
    assert temps_retry == [0.0, 0.0, 0.7]
    assert [r.smiles for r in res_retry["extracted_data"].records] == ["C", "CC", "CCC", "CCCC"]


def test_canonical_dedup_in_single_shot_preserves_distinct_strains(tmp_path):
    from ae.core.schema import FieldSpec, SchemaConfig, create_all_models

    cfg = SchemaConfig(
        name="dedup_test",
        fields={
            "compound_id": FieldSpec(type=str, required=True),
            "smiles": FieldSpec(type=str, required=True),
            "target_value": FieldSpec(type=str, required=True),
            "target_units": FieldSpec(type=str, required=True),
            "bacteria": FieldSpec(type=str, required=True),
        },
        compare_fields=["compound_id", "smiles", "target_value", "target_units", "bacteria"],
    )
    rec_model, out_model = create_all_models(cfg)

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = out_model(
        records=[
            rec_model(
                compound_id="8b",
                smiles="Cc1ccccc1",
                target_value="8",
                target_units="µg/mL",
                bacteria="S. aureus ATCC 29213",
            ),
            # Canonical duplicate (different SMILES order, 8.0, ug/mL) -> should be removed
            rec_model(
                compound_id="8b",
                smiles="c1ccc(C)cc1",
                target_value="8.0",
                target_units="ug/mL",
                bacteria="S. aureus ATCC 29213",
            ),
            # Different strain with same MIC -> must NOT be merged!
            rec_model(
                compound_id="8b",
                smiles="Cc1ccccc1",
                target_value="8",
                target_units="µg/mL",
                bacteria="S. aureus ATCC 43300",
            ),
        ]
    )
    mock_llm.with_structured_output.return_value = mock_structured

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=out_model,
        ingestion_dir=tmp_path,
        enable_record_dedup=True,
    )
    res = graph.run(
        document_text="<!-- document_id: doc_dedup -->\n\n8b Cc1ccccc1 8 µg/mL S. aureus ATCC 29213 ATCC 43300.",
        doc_id="doc_dedup",
    )
    recs = res["extracted_data"].records
    assert len(recs) == 2
    assert [r.bacteria for r in recs] == ["S. aureus ATCC 29213", "S. aureus ATCC 43300"]


def test_resolve_iupac_to_smiles_tool_and_missing_java(monkeypatch):
    from ae.core.schema.domain_tools.chemistry import resolve_iupac_batch
    from ae.core.schema.tools import DEFAULT_TOOL_REGISTRY

    provider = DEFAULT_TOOL_REGISTRY.resolve("resolve_iupac_to_smiles")
    assert provider.auto_remind is False

    monkeypatch.setattr("ae.core.schema.domain_tools.chemistry.shutil.which", lambda _cmd: None)
    res_no_java = resolve_iupac_batch(["ethanol"])
    assert res_no_java["resolved"] == []
    assert "Java runtime" in res_no_java["errors"][0]["error"]

    monkeypatch.setattr("ae.core.schema.domain_tools.chemistry.shutil.which", lambda _cmd: "/usr/bin/java")
    import sys
    import types

    fake_py2opsin = types.SimpleNamespace(py2opsin=lambda names, **_kw: ["CCO" for _ in names])
    monkeypatch.setitem(sys.modules, "py2opsin", fake_py2opsin)
    res_ok = resolve_iupac_batch(["ethanol"])
    assert res_ok == {"resolved": [{"iupac_name": "ethanol", "smiles": "CCO"}], "errors": []}


def test_finalize_evidence_respects_domain_reminder_guard_and_propagates_structured_hint(tmp_path):
    from ae.core.schema.tools import RegisteredToolProvider, register_extraction_tool, unregister_extraction_tool

    tool_name = "test_hinted_markush_tool"
    passive_tool = "test_passive_hinted_tool"
    register_extraction_tool(
        tool_name,
        RegisteredToolProvider(
            name=tool_name,
            build_tool=lambda **_: _make_dummy_tool(tool_name),
            is_applicable=lambda txt: "[*:1]" in txt,
            policy_hint="Agent policy hint.",
            auto_remind=True,
            structured_output_hint="STRUCTURED CROSS-CHECK HINT",
        ),
    )
    register_extraction_tool(
        passive_tool,
        RegisteredToolProvider(
            name=passive_tool,
            build_tool=lambda **_: _make_dummy_tool(passive_tool),
            is_applicable=lambda txt: "PASSIVE" in txt,
            policy_hint="Passive policy hint.",
            auto_remind=False,
            structured_output_hint="PASSIVE STRUCTURED HINT",
        ),
    )
    try:
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = AIMessage(
            content="",
            tool_calls=[{"name": "finalize_evidence", "args": {"confirm": True}, "id": "fe_1"}],
        )
        mock_llm.bind_tools.return_value = mock_bound
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = DummyOutput(summary="OK")
        mock_llm.with_structured_output.return_value = mock_structured

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=DummyOutput,
            ingestion_dir=tmp_path,
            tools=[tool_name],
        )
        res = graph.run(
            document_text="<!-- document_id: doc_fe -->\n\nScaffold [*:1] and final SMILES.",
            doc_id="doc_fe",
        )
        assert res["extracted_data"].summary == "OK"
        # One-time reminder guard: 1st finalize_evidence -> reminder -> 2nd finalize_evidence -> structured_output
        assert mock_bound.invoke.call_count == 2
        structured_msgs = mock_structured.invoke.call_args[0][0]
        assert "STRUCTURED CROSS-CHECK HINT" in structured_msgs[0].content

        # auto_remind=False exits immediately on first finalize_evidence
        mock_bound.invoke.reset_mock()
        graph_passive = ExtractionGraph(
            llm=mock_llm,
            output_model=DummyOutput,
            ingestion_dir=tmp_path,
            tools=[passive_tool],
        )
        res2 = graph_passive.run(
            document_text="<!-- document_id: doc_fe_passive -->\n\nPASSIVE doc.",
            doc_id="doc_fe_passive",
        )
        assert res2["extracted_data"].summary == "OK"
        assert mock_bound.invoke.call_count == 1
    finally:
        unregister_extraction_tool(tool_name)
        unregister_extraction_tool(passive_tool)


def _make_dummy_tool(name: str):
    from langchain_core.tools import tool

    @tool(name)
    def _dummy(x: str = "ok") -> str:
        """Dummy tool."""
        return x

    return _dummy


def test_canonical_dedup_preserves_distinct_phenotypes_with_collection_id():
    from ae.extraction.graph import _record_dedup_key

    base = {
        "compound_id": "8b",
        "smiles": "Cc1ccccc1",
        "target_value": "8",
        "target_units": "µg/mL",
        "bacteria": "S. aureus ATCC 29213",
    }
    mssa = {**base, "bacteria": "MSSA S. aureus ATCC 29213"}
    mssa_pipe = {**base, "bacteria": "MSSA | S. aureus ATCC-29213"}
    mrsa = {**base, "bacteria": "MRSA S. aureus ATCC 29213"}
    mdr = {**base, "bacteria": "MDR S. aureus ATCC 29213"}
    vre = {**base, "bacteria": "VRE E. faecalis ATCC 29212"}
    plain_ef = {**base, "bacteria": "E. faecalis ATCC 29212"}
    diff_atcc = {**base, "bacteria": "MRSA S. aureus ATCC 43300"}
    no_coll_mrsa = {**base, "bacteria": "MRSA N315"}
    no_coll_mssa = {**base, "bacteria": "MSSA N315"}

    assert _record_dedup_key(mssa) == _record_dedup_key(mssa_pipe)
    assert _record_dedup_key(base) != _record_dedup_key(mssa)
    assert _record_dedup_key(base) != _record_dedup_key(mrsa)
    assert _record_dedup_key(base) != _record_dedup_key(mdr)
    assert _record_dedup_key(mssa) != _record_dedup_key(mrsa)
    assert _record_dedup_key(vre) != _record_dedup_key(plain_ef)
    assert _record_dedup_key(mrsa) != _record_dedup_key(diff_atcc)
    assert _record_dedup_key(no_coll_mrsa) != _record_dedup_key(no_coll_mssa)
