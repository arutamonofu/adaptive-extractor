"""Unit tests for Contour 3 (EXP-02) audit fixes in graph.py and chemistry.py."""

from __future__ import annotations

import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from unittest.mock import MagicMock

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ae.core.schema import FieldSpec, SchemaConfig, create_all_models
from ae.core.schema.domain_tools import chemistry
from ae.core.schema.tools import (
    DEFAULT_TOOL_REGISTRY,
    RegisteredToolProvider,
    register_extraction_tool,
    unregister_extraction_tool,
)
from ae.extraction.graph import (
    ExtractionGraph,
    _get_unit_registry,
    _record_dedup_key,
)


class _Record(BaseModel):
    compound_id: Optional[str] = None
    smiles: Optional[str] = None
    target_relation: Optional[str] = None
    target_value: Optional[str] = None
    target_units: Optional[str] = None
    bacteria: Optional[str] = None


class _RecordOutput(BaseModel):
    records: list[_Record] = Field(default_factory=list)


def test_unit_registry_singleton_and_magnitude_preserved():
    assert _get_unit_registry() is _get_unit_registry()

    base = {
        "compound_id": "1a",
        "smiles": "CCO",
        "target_value": "8",
        "target_units": "µg/mL",
        "bacteria": "S. aureus ATCC 29213",
    }
    eq_ug = {**base, "target_units": "ug/mL"}
    eq_mg_l = {**base, "target_units": "mg/L"}
    diff_mg_ml = {**base, "target_units": "mg/mL"}
    um_rec = {**base, "target_units": "µM"}
    um_ascii = {**base, "target_units": "uM"}
    mm_rec = {**base, "target_units": "mM"}

    assert _record_dedup_key(base) == _record_dedup_key(eq_ug)
    assert _record_dedup_key(base) == _record_dedup_key(eq_mg_l)
    assert _record_dedup_key(base) != _record_dedup_key(diff_mg_ml)
    assert _record_dedup_key(um_rec) == _record_dedup_key(um_ascii)
    assert _record_dedup_key(um_rec) != _record_dedup_key(mm_rec)


def test_dedup_normalization_nulls_relations_and_bacteria_case_and_collection_id():
    r_none = {"compound_id": "1a", "target_relation": None, "bacteria": None}
    r_empty = {"compound_id": "1a", "target_relation": "", "bacteria": ""}
    r_nan = {"compound_id": "1a", "target_relation": "NaN", "bacteria": "nan"}
    assert _record_dedup_key(r_none) == _record_dedup_key(r_empty)
    assert _record_dedup_key(r_none) == _record_dedup_key(r_nan)

    r_ge_sym = {"compound_id": "1a", "target_relation": "≥"}
    r_ge_asc = {"compound_id": "1a", "target_relation": ">="}
    r_le_sym = {"compound_id": "1a", "target_relation": "≤"}
    r_le_asc = {"compound_id": "1a", "target_relation": "<="}
    assert _record_dedup_key(r_ge_sym) == _record_dedup_key(r_ge_asc)
    assert _record_dedup_key(r_le_sym) == _record_dedup_key(r_le_asc)
    assert _record_dedup_key(r_ge_sym) != _record_dedup_key(r_le_sym)

    b1 = {"bacteria": "Staphylococcus aureus"}
    b2 = {"bacteria": "staphylococcus aureus"}
    assert _record_dedup_key(b1) == _record_dedup_key(b2)

    c1 = {"bacteria": "S. aureus ATCC 29213"}
    c2 = {"bacteria": "S. aureus ATCC-29213"}
    c3 = {"bacteria": "S. aureus ATCC#29213"}
    c4 = {"bacteria": "MSSA S. aureus atcc-29213"}
    c5 = {"bacteria": "MRSA S. aureus atcc-29213"}
    assert _record_dedup_key(c1) == _record_dedup_key(c2) == _record_dedup_key(c3)
    assert _record_dedup_key(c4) != _record_dedup_key(c1)
    assert _record_dedup_key(c4) != _record_dedup_key(c5)


def test_batched_mode_does_not_call_deduplicate_output_twice(tmp_path, monkeypatch):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = [
        _RecordOutput(records=[_Record(compound_id="1a", smiles="CCO")]),
        _RecordOutput(records=[]),
    ]
    mock_llm.with_structured_output.return_value = mock_structured

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=_RecordOutput,
        ingestion_dir=tmp_path,
        structured_output_batch_size=5,
    )

    dedup_calls = 0
    orig_dedup = graph._deduplicate_output

    def _spy_dedup(result):
        nonlocal dedup_calls
        dedup_calls += 1
        return orig_dedup(result)

    monkeypatch.setattr(graph, "_deduplicate_output", _spy_dedup)
    res = graph.run(document_text="<!-- document_id: doc_b -->\n\nCompound 1a.", doc_id="doc_b")
    assert len(res["extracted_data"].records) == 1
    assert dedup_calls == 0


def test_finalize_evidence_triggers_one_time_domain_reminder_and_budget_note(tmp_path):
    tool_name = "test_markush_guard_tool"

    @tool(tool_name)
    def _dummy_markush(cid: str = "1a") -> str:
        """Assemble markush."""
        return json.dumps({"assembled": [{"compound_id": cid, "smiles": "CCO"}], "errors": []})

    register_extraction_tool(
        tool_name,
        RegisteredToolProvider(
            name=tool_name,
            build_tool=lambda **_: _dummy_markush,
            is_applicable=lambda txt: "[*:1]" in txt,
            policy_hint="Call test_markush_guard_tool.",
            auto_remind=True,
        ),
    )
    try:
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"name": "finalize_evidence", "args": {"confirm": True}, "id": "fe_1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": {"cid": "1a"}, "id": "mk_1"}],
            ),
        ]
        mock_llm.bind_tools.return_value = mock_bound
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _RecordOutput(
            records=[_Record(compound_id="1a", smiles="CCO")]
        )
        mock_llm.with_structured_output.return_value = mock_structured

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=_RecordOutput,
            ingestion_dir=tmp_path,
            max_tool_iterations=2,
            tools=[tool_name],
        )
        res = graph.run(
            document_text="<!-- document_id: doc_guard -->\n\nScaffold [*:1].",
            doc_id="doc_guard",
        )
        assert mock_bound.invoke.call_count == 2
        assert len(res["tool_results"]) == 1
        # Check tool messages for reminder and budget note text (no "reply READY")
        tool_msgs = [m for m in res["messages"] if m.type == "tool"]
        assert any("ACTION REQUIRED" in m.content for m in tool_msgs)
        assert any("call finalize_evidence." in m.content for m in tool_msgs)
        assert not any("reply READY" in m.content for m in tool_msgs)
    finally:
        unregister_extraction_tool(tool_name)


def test_retry_temperature_stays_zero_on_transient_errors_and_rises_on_parse_errors(tmp_path):
    temps_transient: list[float] = []
    temps_parse: list[float] = []

    class _ModeChatModel(BaseChatModel):
        mode: str = "transient"
        attempt: int = 0

        @property
        def _llm_type(self) -> str:
            return "fake-temp-check"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        def with_structured_output(self, schema, **kwargs):
            outer = self

            class _Runnable:
                def __init__(self, temp: float = 0.0):
                    self.temp = temp

                def bind(self, **kw):
                    return _Runnable(temp=kw.get("temperature", self.temp))

                def invoke(self, _msgs):
                    outer.attempt += 1
                    bucket = temps_transient if outer.mode == "transient" else temps_parse
                    bucket.append(self.temp)
                    if outer.attempt == 1:
                        if outer.mode == "transient":
                            raise TimeoutError("HTTP 429 rate limit timed out")
                        raise OutputParserException("Truncated JSON")
                    return _RecordOutput(records=[_Record(compound_id="1a", smiles="CCO")])

            return _Runnable(temp=0.0)

    g_trans = ExtractionGraph(
        llm=_ModeChatModel(mode="transient"),
        output_model=_RecordOutput,
        ingestion_dir=tmp_path,
        use_streaming=False,
        max_retries=2,
        retry_delay=0.0,
    )
    g_trans.run(document_text="<!-- document_id: doc_t1 -->\n\nText.", doc_id="doc_t1")
    assert temps_transient == [0.0, 0.0]

    g_parse = ExtractionGraph(
        llm=_ModeChatModel(mode="parse"),
        output_model=_RecordOutput,
        ingestion_dir=tmp_path,
        use_streaming=False,
        max_retries=2,
        retry_delay=0.0,
    )
    g_parse.run(document_text="<!-- document_id: doc_t2 -->\n\nText.", doc_id="doc_t2")
    assert temps_parse == [0.0, 0.7]


def test_local_record_repair_retries_transient_error_and_checks_repair_llm(tmp_path):
    cfg = SchemaConfig(
        name="repair_retry_test",
        fields={
            "compound_id": FieldSpec(type=str, required=True),
            "smiles": FieldSpec(type=str, required=True, validators=["smiles"]),
        },
        compare_fields=["compound_id", "smiles"],
    )
    rec_model, out_model = create_all_models(cfg)

    main_llm = MagicMock()
    main_structured = MagicMock()
    main_structured.invoke.return_value = out_model(
        records=[
            rec_model(compound_id="1a", smiles="c1ccc([*:1])cc1"),
            rec_model(compound_id="1b", smiles="CC"),
            rec_model(compound_id="1c", smiles="CCC"),
            rec_model(compound_id="1d", smiles="CCCC"),
        ]
    )
    main_llm.with_structured_output.return_value = main_structured

    stream_checked_on_repair = False
    repair_attempts = 0

    class _RepairChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-repair"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        def with_structured_output(self, schema, **kwargs):
            from langchain_core.runnables import RunnableLambda

            def _run_repair(_msgs, **_kw):
                nonlocal repair_attempts, stream_checked_on_repair
                repair_attempts += 1
                if repair_attempts == 1:
                    raise TimeoutError("transient 503 during repair")
                return schema(repairs=[{"index": 0, "record": {"compound_id": "1a", "smiles": "C"}}])

            class _StreamableRepair(RunnableLambda):
                def stream(self, input, config=None, **kwargs):
                    nonlocal stream_checked_on_repair
                    stream_checked_on_repair = True
                    yield self.invoke(input, config=config, **kwargs)

            return _StreamableRepair(_run_repair)

    graph = ExtractionGraph(
        llm=main_llm,
        repair_llm=_RepairChatModel(),
        output_model=out_model,
        ingestion_dir=tmp_path,
        use_streaming=True,
        max_retries=3,
        retry_delay=0.0,
    )
    res = graph.run(
        document_text="<!-- document_id: doc_rep_retry -->\n\nCompounds 1a..1d.",
        doc_id="doc_rep_retry",
    )
    assert main_structured.invoke.call_count == 1
    assert repair_attempts == 2
    assert stream_checked_on_repair is True
    assert [r.smiles for r in res["extracted_data"].records] == ["C", "CC", "CCC", "CCCC"]


def test_failed_domain_tool_not_appended_to_tool_results_while_partial_kept(tmp_path):
    fail_tool_name = "test_failing_domain_tool"
    partial_tool_name = "test_partial_domain_tool"

    @tool(fail_tool_name)
    def _fail_tool(q: str = "x") -> str:
        """Fail completely."""
        return json.dumps({"resolved": [], "errors": [{"iupac_name": q, "error": "no java"}]})

    @tool(partial_tool_name)
    def _partial_tool(q: str = "x") -> str:
        """Succeed partially."""
        return json.dumps(
            {
                "resolved": [{"iupac_name": "ethanol", "smiles": "CCO"}],
                "errors": [{"iupac_name": q, "error": "bad name"}],
            }
        )

    register_extraction_tool(
        fail_tool_name,
        RegisteredToolProvider(
            name=fail_tool_name,
            build_tool=lambda **_: _fail_tool,
            is_applicable=lambda _: True,
            auto_remind=False,
        ),
    )
    register_extraction_tool(
        partial_tool_name,
        RegisteredToolProvider(
            name=partial_tool_name,
            build_tool=lambda **_: _partial_tool,
            is_applicable=lambda _: True,
            auto_remind=False,
        ),
    )
    try:
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": fail_tool_name, "args": {"q": "bad"}, "id": "c_fail"},
                    {"name": partial_tool_name, "args": {"q": "mix"}, "id": "c_part"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "finalize_evidence", "args": {"confirm": True}, "id": "c_fin"}],
            ),
        ]
        mock_llm.bind_tools.return_value = mock_bound
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _RecordOutput(
            records=[_Record(compound_id="1a", smiles="CCO")]
        )
        mock_llm.with_structured_output.return_value = mock_structured

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=_RecordOutput,
            ingestion_dir=tmp_path,
            max_tool_iterations=3,
            tools=[fail_tool_name, partial_tool_name],
        )
        res = graph.run(document_text="<!-- document_id: doc_tr -->\n\nBody.", doc_id="doc_tr")
        assert len(res["tool_results"]) == 1
        assert res["tool_results"][0]["tool"] == partial_tool_name
        assert "CCO" in res["tool_results"][0]["output"]
    finally:
        unregister_extraction_tool(fail_tool_name)
        unregister_extraction_tool(partial_tool_name)


def test_resolve_iupac_cache_no_io_failure_caching_and_thread_safety(monkeypatch):
    chemistry._IUPAC_CACHE.clear()
    monkeypatch.setattr("ae.core.schema.domain_tools.chemistry.shutil.which", lambda _cmd: "/usr/bin/java")

    call_count = 0
    seen_tmp_paths: list[str] = []
    lock = threading.Lock()

    def _flaky_py2opsin(names, tmp_fpath=None):
        nonlocal call_count
        with lock:
            call_count += 1
            if tmp_fpath:
                seen_tmp_paths.append(tmp_fpath)
            current = call_count
        if current == 1:
            raise OSError("transient I/O failure")
        return ["CCO" if n == "ethanol" else "CC" for n in names]

    monkeypatch.setitem(sys.modules, "py2opsin", types.SimpleNamespace(py2opsin=_flaky_py2opsin))

    try:
        # 1st call: I/O error must NOT be cached
        res_err = chemistry.resolve_iupac_batch(["ethanol"])
        assert res_err["resolved"] == []
        assert "OPSIN conversion failed" in res_err["errors"][0]["error"]

        # 2nd call: succeeds and caches "ethanol"
        res_ok = chemistry.resolve_iupac_batch(["ethanol"])
        assert res_ok == {"resolved": [{"iupac_name": "ethanol", "smiles": "CCO"}], "errors": []}
        assert call_count == 2

        # 3rd call: "ethanol" served from cache without calling py2opsin
        res_cached = chemistry.resolve_iupac_batch(["ethanol"])
        assert res_cached == res_ok
        assert call_count == 2

        # Concurrent calls on uncached name use unique NamedTemporaryFile paths cleanly
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: chemistry.resolve_iupac_batch([f"ethane_{i}"]), range(4)))
        assert all(len(r["resolved"]) == 1 for r in results)
        assert len(seen_tmp_paths) == len(set(seen_tmp_paths))
    finally:
        chemistry._IUPAC_CACHE.clear()


def test_resolve_iupac_is_applicable_heuristic_bypasses_non_iupac_docs(tmp_path):
    provider = DEFAULT_TOOL_REGISTRY.resolve("resolve_iupac_to_smiles")

    assert provider.is_applicable("") is False
    assert provider.is_applicable("Plain text without chemical names.") is False
    assert provider.is_applicable("Compound 1a had MIC = 8 ug/mL against S. aureus ATCC 29213.") is False
    assert provider.is_applicable("2-fold serial dilution in 96-well plates; 1H-NMR confirmed purity.") is False

    assert provider.is_applicable("Synthesized 2-chloro-6-fluorophenyl derivative 3a.") is True
    assert provider.is_applicable("Contains 6-methylpyridin-3-yl moiety.") is True
    assert provider.is_applicable("5-fluoro-1H-benzimidazole scaffold.") is True

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _RecordOutput(records=[])
    mock_llm.with_structured_output.return_value = mock_structured

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=_RecordOutput,
        ingestion_dir=tmp_path,
        tools=["resolve_iupac_to_smiles"],
    )
    graph.run(
        document_text="<!-- document_id: doc_no_iupac -->\n\nCompound 1a MIC = 8 ug/mL (2-fold dilution).",
        doc_id="doc_no_iupac",
    )
    mock_llm.bind_tools.assert_not_called()


def test_dedup_unit_superscripts_narrow_unit_key_and_batched_flag(tmp_path):
    # 1. µg mL-1 / µg·mL-1 / µg mL⁻¹ normalize to µg/mL before pint
    base = {"compound_id": "1a", "target_units": "µg/mL"}
    for variant in ("µg mL-1", "µg·mL-1", "µg mL⁻¹", "ug mL-1", "ug·mL-1"):
        assert _record_dedup_key(base) == _record_dedup_key({**base, "target_units": variant})

    # 2. Narrow unit key check: "subunit" / "community" must NOT be parsed as units (e.g. "m" vs "meter")
    assert _record_dedup_key({"subunit": "ug/mL"}) != _record_dedup_key({"subunit": "mg/L"})
    assert _record_dedup_key({"unit": "ug/mL"}) == _record_dedup_key({"unit": "mg/L"})
    assert _record_dedup_key({"dose_unit": "ug/mL"}) == _record_dedup_key({"dose_unit": "mg/L"})
    assert _record_dedup_key({"units": "ug/mL"}) == _record_dedup_key({"units": "mg/L"})

    # 3. _extract_records_batched respects enable_record_dedup=False
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    dup_rec = _Record(compound_id="1a", smiles="CCO", target_value="8", target_units="µg/mL")
    mock_structured.invoke.side_effect = [
        _RecordOutput(records=[dup_rec, dup_rec]),
        _RecordOutput(records=[]),
    ]
    mock_llm.with_structured_output.return_value = mock_structured

    graph_no_dedup = ExtractionGraph(
        llm=mock_llm,
        output_model=_RecordOutput,
        ingestion_dir=tmp_path,
        structured_output_batch_size=5,
        enable_record_dedup=False,
        retry_delay=0.0,
    )
    res = graph_no_dedup.run(
        document_text="<!-- document_id: doc_nodedup -->\n\nCompound 1a CCO 8 µg/mL.",
        doc_id="doc_nodedup",
    )
    assert len(res["extracted_data"].records) == 2


def test_route_after_tool_reenters_agent_on_doc_change_for_auto_remind_false_tool(tmp_path):
    from ae.ingestion.parsers.gemini.schemas import VisualParseRecord
    from ae.ingestion.service import VisualParsingService

    passive_tool = "test_passive_crosscheck_tool"
    seen_calls: list[str] = []

    @tool(passive_tool)
    def _passive(name: str = "x") -> str:
        """Cross-check passive."""
        seen_calls.append(name)
        return json.dumps({"resolved": [{"iupac_name": name, "smiles": "CCO"}], "errors": []})

    register_extraction_tool(
        passive_tool,
        RegisteredToolProvider(
            name=passive_tool,
            build_tool=lambda **_: _passive,
            is_applicable=lambda txt: "Scheme_1" in txt or "2-chloro" in txt,
            auto_remind=False,
        ),
    )
    try:
        mock_service = MagicMock(spec=VisualParsingService)
        mock_service.parse_visuals.return_value = {
            "Scheme_1": VisualParseRecord(
                visual_id="Scheme_1",
                status="success",
                mode="single",
                rendered_markdown="| cmp | smiles |\n|---|---|\n| 1a | CCO |",
            )
        }
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"name": "parse_visuals", "args": {"visual_ids": ["Scheme_1"]}, "id": "v1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": passive_tool, "args": {"name": "ethanol"}, "id": "p1"}],
            ),
        ]
        mock_llm.bind_tools.return_value = mock_bound
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _RecordOutput(records=[_Record(compound_id="1a", smiles="CCO")])
        mock_llm.with_structured_output.return_value = mock_structured

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=_RecordOutput,
            service=mock_service,
            ingestion_dir=tmp_path,
            max_tool_iterations=4,
            tools=[passive_tool],
        )
        res = graph.run(
            document_text='<!-- document_id: doc_passive_re -->\n\n<visual id="Scheme_1">\n> Scheme 1\n</visual>\n2-chloro derivative.',
            doc_id="doc_passive_re",
        )
        assert mock_bound.invoke.call_count == 2
        assert seen_calls == ["ethanol"]
        assert len(res["tool_results"]) == 1
    finally:
        unregister_extraction_tool(passive_tool)


def test_failed_domain_tool_allows_exact_retry_and_reminder_preserves_iteration_budget(tmp_path):
    retry_tool = "test_retryable_domain_tool"
    calls = 0

    @tool(retry_tool)
    def _flaky(cid: str = "1a") -> str:
        """Flaky domain tool."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps({"resolved": [], "errors": [{"iupac_name": cid, "error": "transient"}]})
        return json.dumps({"resolved": [{"iupac_name": cid, "smiles": "CCO"}], "errors": []})

    register_extraction_tool(
        retry_tool,
        RegisteredToolProvider(
            name=retry_tool,
            build_tool=lambda **_: _flaky,
            is_applicable=lambda txt: "TRIGGER" in txt,
            policy_hint="Call test_retryable_domain_tool.",
            auto_remind=True,
        ),
    )
    try:
        # 1. With max_tool_iterations=1, finalize_evidence reminder must NOT burn the single iteration
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"name": "finalize_evidence", "args": {"confirm": True}, "id": "fe_1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": retry_tool, "args": {"cid": "ethanol"}, "id": "rt_1"}],
            ),
        ]
        mock_llm.bind_tools.return_value = mock_bound
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _RecordOutput(records=[_Record(compound_id="1a", smiles="CCO")])
        mock_llm.with_structured_output.return_value = mock_structured

        calls = 0
        graph_budget1 = ExtractionGraph(
            llm=mock_llm,
            output_model=_RecordOutput,
            ingestion_dir=tmp_path,
            max_tool_iterations=1,
            tools=[retry_tool],
        )
        # First call fails completely -> test exact same sig can be retried when budget allows
        res1 = graph_budget1.run(document_text="<!-- document_id: doc_b1 -->\n\nTRIGGER.", doc_id="doc_b1")
        assert mock_bound.invoke.call_count == 2
        assert calls == 1
        assert res1["executed_tool_signatures"] == []
        assert res1["non_visual_tools_called_since_doc_change"] == []

        # 2. With max_tool_iterations=3, exact same failed signature can be retried and succeeds
        calls = 0
        mock_bound.invoke.reset_mock()
        mock_bound.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"name": retry_tool, "args": {"cid": "ethanol"}, "id": "rt_1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": retry_tool, "args": {"cid": "ethanol"}, "id": "rt_2"}],
            ),
        ]
        graph_retry = ExtractionGraph(
            llm=mock_llm,
            output_model=_RecordOutput,
            ingestion_dir=tmp_path,
            max_tool_iterations=3,
            tools=[retry_tool],
        )
        res2 = graph_retry.run(document_text="<!-- document_id: doc_b2 -->\n\nTRIGGER.", doc_id="doc_b2")
        assert calls == 2
        assert len(res2["tool_results"]) == 1
        assert "CCO" in res2["tool_results"][0]["output"]
        assert retry_tool in res2["non_visual_tools_called_since_doc_change"]
    finally:
        unregister_extraction_tool(retry_tool)


def test_structured_output_fast_path_requires_ai_message(tmp_path):
    from langchain_core.messages import ToolMessage

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    expected = _RecordOutput(records=[_Record(compound_id="1a", smiles="CCO")])
    mock_structured.invoke.return_value = expected
    mock_llm.with_structured_output.return_value = mock_structured

    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=_RecordOutput,
        ingestion_dir=tmp_path,
    )

    # ToolMessage with JSON (e.g. from domain tool) must NOT trigger fast-path
    tool_state = {
        "doc_id": "doc_fp1",
        "document_text": "<!-- document_id: doc_fp1 -->\n\nCompound 1a.",
        "system_instruction": "",
        "messages": [ToolMessage(content='{"resolved": []}', tool_call_id="t1")],
    }
    out1 = graph._structured_output_node(tool_state)
    assert out1["extracted_data"] == expected
    assert mock_structured.invoke.call_count == 1

    # AIMessage with valid JSON DOES trigger fast-path without calling structured_model
    ai_state = {
        "doc_id": "doc_fp2",
        "document_text": "<!-- document_id: doc_fp2 -->\n\nCompound 1a.",
        "system_instruction": "",
        "messages": [AIMessage(content='{"records": [{"compound_id": "1a", "smiles": "CCO"}]}')],
    }
    out2 = graph._structured_output_node(ai_state)
    assert out2["extracted_data"].records[0].smiles == "CCO"
    assert mock_structured.invoke.call_count == 1


def test_iupac_cache_lru_eviction_and_key_normalization(monkeypatch):
    chemistry._IUPAC_CACHE.clear()
    monkeypatch.setattr("ae.core.schema.domain_tools.chemistry.shutil.which", lambda _cmd: "/usr/bin/java")
    monkeypatch.setattr(chemistry, "_IUPAC_CACHE_MAX_SIZE", 2)

    seen_calls: list[list[str]] = []

    def _fake_py2opsin(names, **_kw):
        seen_calls.append(list(names))
        return ["CCO" for _ in names]

    monkeypatch.setitem(sys.modules, "py2opsin", types.SimpleNamespace(py2opsin=_fake_py2opsin))

    try:
        # Dash and whitespace normalization hits the same cache entry
        r1 = chemistry.resolve_iupac_batch(["2\u2013chloro   ethanol"])
        r2 = chemistry.resolve_iupac_batch(["2-chloro ethanol"])
        assert r1["resolved"][0]["smiles"] == "CCO"
        assert r2["resolved"][0]["smiles"] == "CCO"
        assert len(seen_calls) == 1
        assert seen_calls[0] == ["2-chloro ethanol"]

        # Add a second entry, touch the first, then add a third -> second (LRU) is evicted, first survives
        chemistry.resolve_iupac_batch(["propane"])
        chemistry.resolve_iupac_batch(["2-chloro ethanol"])  # refresh LRU order
        chemistry.resolve_iupac_batch(["butane"])

        assert len(chemistry._IUPAC_CACHE) == 2
        assert "2-chloro ethanol" in chemistry._IUPAC_CACHE
        assert "butane" in chemistry._IUPAC_CACHE
        assert "propane" not in chemistry._IUPAC_CACHE
    finally:
        chemistry._IUPAC_CACHE.clear()


def test_is_iupac_applicable_excludes_solvents_and_markush_hint_is_clean(tmp_path):
    provider = DEFAULT_TOOL_REGISTRY.resolve("resolve_iupac_to_smiles")
    solvent_doc = (
        "Dissolved in N,N-dimethylformamide, 1,2-dichloroethane, "
        "2-mercaptoethanol, dimethyl sulfoxide, and tetrahydrofuran."
    )
    assert provider.is_applicable(solvent_doc) is False
    assert provider.is_applicable(f"{solvent_doc} Product: 2-chloro-6-fluorophenyl analog.") is True

    # Markush-only graph should not instruct about resolve_iupac_to_smiles
    mock_llm = MagicMock()
    graph = ExtractionGraph(
        llm=mock_llm,
        output_model=_RecordOutput,
        ingestion_dir=tmp_path,
        tools=["assemble_markush_smiles"],
    )
    msgs = graph._build_structured_messages(
        {"doc_id": "doc_mk", "document_text": "Scaffold [*:1].", "system_instruction": "Extract."},
        "Go.",
    )
    assert "resolve_iupac_to_smiles" not in msgs[0].content
