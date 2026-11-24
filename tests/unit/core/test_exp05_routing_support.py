"""Unit tests for EXP-05 model routing, parameter filtering, and split-tier support."""

from unittest.mock import MagicMock

import dspy
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, SecretStr

from ae.core.config.llm_settings import ApiConfig, LLMInstanceConfig
from ae.core.llm.factory import create_chat_model
from ae.core.llm.provider import (
    MODEL_PRICING,
    OpenRouterLM,
    model_supports_sampling_params,
)
from ae.extraction.graph import ExtractionGraph
from ae.semantic_optimization.context import RunContext


def test_model_supports_sampling_params():
    assert model_supports_sampling_params("openai/gpt-6-luna") is False
    assert model_supports_sampling_params("openai/gpt-6.1-terra") is False
    assert model_supports_sampling_params("openai/gpt-6.1-sol") is False
    assert model_supports_sampling_params("openai/gpt-5.4") is False
    assert model_supports_sampling_params("anthropic/claude-haiku-5.5") is False
    assert model_supports_sampling_params("anthropic/claude-sonnet-5.5") is False
    assert model_supports_sampling_params("anthropic/claude-sonnet-4.6") is True
    assert model_supports_sampling_params("deepseek/deepseek-v4.1-flash") is True
    assert model_supports_sampling_params("google/gemini-4-hydrogen") is True
    assert model_supports_sampling_params("qwen/qwen3.8-flash") is True


def test_openrouter_lm_omits_sampling_params_and_forwards_cache_options():
    cfg = LLMInstanceConfig(
        provider="api",
        model="openai/gpt-6.1-sol",
        timeout=60,
        max_retries=1,
        temperature=0.0,
        rate_limit_delay=0.0,
        top_p=0.95,
        enable_cache=False,
        api=ApiConfig(
            api_key=SecretStr("sk-test"),
            max_tokens=4096,
            base_url="https://openrouter.ai/api/v1",
            reasoning={"effort": "low"},
            prompt_cache_options={"mode": "explicit"},
        ),
    )
    lm = OpenRouterLM(cfg)
    payload = lm._prepare_payload([{"role": "user", "content": "test"}])
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert payload["prompt_cache_options"] == {"mode": "explicit"}
    assert payload["reasoning"] == {"effort": "low"}


def test_create_chat_model_omits_sampling_params_for_gpt6():
    cfg = LLMInstanceConfig(
        provider="api",
        model="openai/gpt-6-luna",
        timeout=60,
        max_retries=1,
        temperature=0.0,
        rate_limit_delay=0.0,
        top_p=0.95,
        enable_cache=False,
        api=ApiConfig(
            api_key=SecretStr("sk-test"),
            max_tokens=4096,
            base_url="https://openrouter.ai/api/v1",
            reasoning={"effort": "low"},
            prompt_cache_options={"mode": "explicit"},
        ),
    )
    chat = create_chat_model(cfg)
    assert "temperature" not in chat._default_params
    assert "top_p" not in chat._default_params
    assert chat.stream_usage is True
    assert chat.extra_body["prompt_cache_options"] == {"mode": "explicit"}


def test_model_pricing_contains_exp05_candidates():
    for slug in (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "deepseek/deepseek-v4.1-flash",
        "deepseek/deepseek-v4-pro",
        "openai/gpt-6-luna",
        "openai/gpt-6.1-terra",
        "openai/gpt-6.1-sol",
        "qwen/qwen3.8-flash",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "z-ai/glm-5.3-flash",
        "inclusionai/ling-3.0-flash-vl",
        "google/gemma-4-31b-it",
        "anthropic/claude-sonnet-4.6",
    ):
        assert slug in MODEL_PRICING


def test_extraction_graph_supports_split_agent_and_repair_llms():
    class Rec(BaseModel):
        compound_id: str = Field(...)

    class Out(BaseModel):
        records: list[Rec] = Field(default_factory=list)

    main_llm = MagicMock()
    agent_llm = MagicMock()
    repair_llm = MagicMock()
    agent_bound = MagicMock()
    agent_bound.invoke.return_value = AIMessage(content="", tool_calls=[])
    agent_llm.bind_tools.return_value = agent_bound

    graph = ExtractionGraph(
        llm=main_llm,
        output_model=Out,
        agent_llm=agent_llm,
        repair_llm=repair_llm,
        enable_sanity_checks=False,
    )
    assert graph.agent_llm is agent_llm
    assert graph.repair_llm is repair_llm

    state = {
        "doc_id": "doc1",
        "document_text": "text",
        "system_instruction": "inst",
        "messages": [],
        "tool_iterations": 0,
        "attempted_visuals": [],
        "completed_visuals": [],
        "partial_visuals": [],
        "attempted_hints": {},
        "visual_attempt_counts": {},
        "executed_tool_signatures": [],
        "non_visual_tools_called_since_doc_change": [],
        "last_turn_had_non_visual": False,
        "last_turn_had_tool_error": False,
        "extracted_data": None,
        "rationale": None,
    }
    graph._agent_node(state)
    agent_llm.bind_tools.assert_called_once()
    main_llm.bind_tools.assert_not_called()


def test_run_context_tier_4a_context_switching(tmp_path):
    baseline = tmp_path / "base.txt"
    baseline.write_text("inst", encoding="utf-8")
    schema = tmp_path / "schema.yaml"
    schema.write_text("name: test\nfields:\n  compound_id:\n    type: string\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    gt.write_text("document_id,compound_id\ndoc1,1a\n", encoding="utf-8")
    ing = tmp_path / "ing"
    ing.mkdir()

    lm_4b = dspy.LM("openai/mock-4b")
    lm_4a = dspy.LM("openai/mock-4a")

    ctx = RunContext(
        task_name="test",
        doc_ids=["doc1"],
        gt_path=gt,
        ingestion_dir=ing,
        baseline_prompt_path=baseline,
        schema_path=schema,
        teacher_lm=lm_4b,
        artifacts_dir=tmp_path / "art",
        interactive=False,
        tier_4a_lm=lm_4a,
    )

    with dspy.context(lm=lm_4b):
        assert dspy.settings.lm is lm_4b
        with ctx.tier_4a_context():
            assert dspy.settings.lm is lm_4a
        assert dspy.settings.lm is lm_4b
