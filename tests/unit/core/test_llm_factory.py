"""Unit tests for create_chat_model LLM factory."""

import pytest
from pydantic import SecretStr, ValidationError

from ae.core.config.llm_settings import ApiConfig, GeminiInstanceConfig, LLMInstanceConfig
from ae.core.llm.factory import create_chat_model


def test_llm_instance_config_empty_model_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMInstanceConfig(
            provider="gemini",
            model="",
            max_retries=3,
            temperature=0.0,
            top_p=1.0,
            enable_cache=False,
        )


def test_create_chat_model_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    cfg = LLMInstanceConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        max_retries=2,
        temperature=0.1,
        top_p=0.9,
        enable_cache=False,
        timeout=60,
        gemini=GeminiInstanceConfig(
            api_key=SecretStr("test-gemini-key"),
            max_output_tokens=8192,
            thinking_budget=0,
        ),
    )
    chat_model = create_chat_model(cfg)
    assert isinstance(chat_model, ChatGoogleGenerativeAI)
    assert chat_model.model == "gemini-2.5-flash"
    assert chat_model.temperature == 0.1


def test_create_chat_model_gemini_thinking_level():
    from langchain_google_genai import ChatGoogleGenerativeAI

    cfg = LLMInstanceConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        max_retries=2,
        temperature=0.0,
        top_p=0.95,
        enable_cache=False,
        timeout=60,
        gemini=GeminiInstanceConfig(
            api_key=SecretStr("test-gemini-key"),
            max_output_tokens=65536,
            thinking_level="medium",
        ),
    )
    chat_model = create_chat_model(cfg)
    assert isinstance(chat_model, ChatGoogleGenerativeAI)
    assert chat_model.model == "gemini-3.8-flash"
    assert chat_model.reasoning_effort == "medium"


def test_gemini_lm_provider_registry():
    from ae.core.llm.provider import LLM_PROVIDER_REGISTRY, GeminiLM

    cfg = LLMInstanceConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        max_retries=2,
        temperature=0.0,
        top_p=0.95,
        enable_cache=False,
        timeout=60,
        gemini=GeminiInstanceConfig(
            api_key=SecretStr("test-gemini-key"),
            max_output_tokens=65536,
            thinking_level="medium",
        ),
    )
    lm = LLM_PROVIDER_REGISTRY.create_lm(cfg)
    assert isinstance(lm, GeminiLM)
    assert lm.model == "gemini-3.8-flash"
    assert lm.thinking_level == "medium"


def test_create_chat_model_openrouter():
    from langchain_openai import ChatOpenAI

    cfg = LLMInstanceConfig(
        provider="api",
        model="qwen/qwen3.5-27b",
        max_retries=3,
        temperature=0.0,
        top_p=0.95,
        enable_cache=True,
        timeout=120,
        rate_limit_delay=1.0,
        api=ApiConfig(
            api_key=SecretStr("test-openrouter-key"),
            max_tokens=4096,
            base_url="https://openrouter.ai/api/v1",
            openrouter_cache=True,
            reasoning={"enabled": False},
        ),
    )
    chat_model = create_chat_model(cfg)
    assert isinstance(chat_model, ChatOpenAI)
    assert chat_model.model_name == "qwen/qwen3.5-27b"
    assert "X-OpenRouter-Cache" in chat_model.default_headers


def test_create_chat_model_attaches_callbacks():
    from langchain_core.callbacks import BaseCallbackHandler

    class DummyHandler(BaseCallbackHandler):
        pass

    handler = DummyHandler()
    cfg = LLMInstanceConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        max_retries=2,
        temperature=0.0,
        top_p=0.95,
        enable_cache=False,
        gemini=GeminiInstanceConfig(api_key=SecretStr("test-gemini-key")),
    )
    chat_model = create_chat_model(cfg, callbacks=[handler])
    assert chat_model.callbacks == [handler]


def test_create_chat_model_max_retries_override():
    cfg = LLMInstanceConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        max_retries=5,
        temperature=0.0,
        top_p=0.95,
        enable_cache=False,
        gemini=GeminiInstanceConfig(api_key=SecretStr("test-gemini-key")),
    )
    default_model = create_chat_model(cfg)
    assert default_model.max_retries == 5

    overridden_model = create_chat_model(cfg, max_retries=1)
    assert overridden_model.max_retries == 1


def test_create_chat_model_prompt_caching_injects_cache_control():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    cfg = LLMInstanceConfig(
        provider="api",
        model="qwen/qwen3.5-397b-a17b",
        max_retries=2,
        temperature=0.0,
        top_p=0.95,
        enable_cache=False,
        timeout=60,
        rate_limit_delay=0.0,
        api=ApiConfig(
            api_key=SecretStr("test-openrouter-key"),
            max_tokens=4096,
            base_url="https://openrouter.ai/api/v1",
            prompt_caching=True,
        ),
    )
    chat_model = create_chat_model(cfg)
    assert isinstance(chat_model, ChatOpenAI)

    payload = chat_model._get_request_payload(
        [
            SystemMessage(content="Static system prompt"),
            HumanMessage(content="Dynamic user prompt"),
        ]
    )
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][0]["content"][0]["text"] == "Static system prompt"
    assert payload["messages"][1]["content"] == "Dynamic user prompt"

