"""Unified factory for creating LangChain BaseChatModel instances."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from ae.core.config.llm_settings import LLMInstanceConfig
from ae.core.llm.provider import model_supports_sampling_params

logger = logging.getLogger(__name__)


def create_chat_model(
    config: LLMInstanceConfig,
    callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    max_retries: Optional[int] = None,
) -> BaseChatModel:
    """Create a LangChain BaseChatModel from LLMInstanceConfig.

    Supports:
    - provider: "gemini" -> ChatGoogleGenerativeAI
    - provider: "api" (OpenRouter / OpenAI-compatible) -> ChatOpenAI
    """
    if not config.model or not config.model.strip():
        raise ValueError("Model name must not be empty.")

    effective_max_retries = config.max_retries if max_retries is None else max_retries

    if config.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key_str: Optional[str] = None
        if config.gemini and config.gemini.api_key:
            api_key_str = config.gemini.api_key.get_secret_value()

        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_retries": effective_max_retries,
        }
        if api_key_str:
            kwargs["google_api_key"] = api_key_str
        if config.timeout:
            kwargs["request_timeout"] = float(config.timeout)

        if config.gemini:
            if config.gemini.max_output_tokens is not None:
                kwargs["max_output_tokens"] = config.gemini.max_output_tokens
            if config.gemini.thinking_level is not None:
                kwargs["thinking_level"] = config.gemini.thinking_level
            elif config.gemini.thinking_budget is not None:
                kwargs["thinking_budget"] = config.gemini.thinking_budget
        if callbacks:
            kwargs["callbacks"] = list(callbacks)

        logger.info(f"Instantiating ChatGoogleGenerativeAI (model={config.model})")
        return ChatGoogleGenerativeAI(**kwargs)

    elif config.provider == "api":
        from langchain_openai import ChatOpenAI

        if not config.api or not config.api.api_key:
            raise ValueError(
                f"API config and key required for provider='api' (model={config.model})"
            )

        api_key_str = config.api.api_key.get_secret_value()
        base_url = config.api.base_url or "https://openrouter.ai/api/v1"

        headers: dict[str, str] = {
            "HTTP-Referer": "https://github.com/adaptive-extractor/adaptive-extractor",
            "X-Title": "Adaptive Extractor",
        }
        if config.api.openrouter_cache:
            headers["X-OpenRouter-Cache"] = "true"

        extra_body: dict[str, Any] = {}
        if config.api.reasoning is not None:
            extra_body["reasoning"] = config.api.reasoning
        if getattr(config.api, "prompt_cache_options", None) is not None:
            extra_body["prompt_cache_options"] = config.api.prompt_cache_options
        if config.api.session_id is not None:
            extra_body["session_id"] = config.api.session_id
        if config.api.provider is not None:
            if hasattr(config.api.provider, "model_dump"):
                extra_body["provider"] = config.api.provider.model_dump(
                    by_alias=True, exclude_none=True
                )
            elif isinstance(config.api.provider, dict):
                extra_body["provider"] = config.api.provider

        chat_kwargs: dict[str, Any] = {
            "model": config.model,
            "api_key": api_key_str,
            "base_url": base_url,
            "max_tokens": config.api.max_tokens,
            "max_retries": effective_max_retries,
            "default_headers": headers,
            "stream_usage": True,
        }
        if model_supports_sampling_params(config.model):
            chat_kwargs["temperature"] = config.temperature
            chat_kwargs["top_p"] = config.top_p
        if config.timeout:
            chat_kwargs["timeout"] = float(config.timeout)
        if extra_body:
            chat_kwargs["extra_body"] = extra_body
        if callbacks:
            chat_kwargs["callbacks"] = list(callbacks)

        logger.info(f"Instantiating ChatOpenAI (model={config.model}, base_url={base_url})")
        return ChatOpenAI(**chat_kwargs)

    raise ValueError(f"Unsupported provider for create_chat_model: '{config.provider}'")
