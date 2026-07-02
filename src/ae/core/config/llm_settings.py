"""LLM configuration settings."""

import logging
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

logger = logging.getLogger(__name__)

# Valid ID column names
KNOWN_OPENROUTER_PROVIDERS = {
    "arcee-ai", "sambanova", "reka", "cerebras", "morph", "stealth", "akashml",
    "moonshot", "openai", "z-ai", "wandb", "featherless", "fireworks", "groq",
    "crusoe", "ncompass", "cohere", "deepseek", "siliconflow", "perplexity",
    "infermatic", "x-ai", "alibaba", "novita", "deepinfra", "digitalocean",
    "phala", "parasail", "gmicloud", "anthropic", "atlas-cloud", "together",
    "venice", "streamlake", "google", "chutes", "lepton", "runpod", "openrouter"
}


class OllamaConfig(BaseModel):
    """Ollama-specific configuration."""
    ollama_base_url: Optional[str] = Field(
        default=None,
        description="Ollama base URL (from OLLAMA_*_BASE_URL env var only)"
    )
    num_ctx: int = Field(
        ...,
        description="Context window size for Ollama model"
    )
    num_predict: int = Field(
        ...,
        description="Maximum number of tokens to predict"
    )
    repeat_penalty: float = Field(
        ...,
        description="Penalty for repeated tokens"
    )
    repeat_last_n: int = Field(
        ...,
        description="Number of tokens to consider for repeat penalty"
    )
    stream: bool = Field(
        ...,
        description="Enable streaming responses"
    )

    @field_validator("ollama_base_url", mode="after")
    @classmethod
    def validate_ollama_base_url(cls, v: Optional[str]) -> str:
        """Validate that Ollama base URL is set via environment variable."""
        if v is None or v.strip() == "":
            raise ValueError(
                "OLLAMA_*_BASE_URL environment variable must be set in .env file. "
                "Set OLLAMA_STUDENT_BASE_URL, OLLAMA_TEACHER_BASE_URL, or OLLAMA_VISUAL_EXTRACTOR_BASE_URL as appropriate."
            )
        return v.strip()


class OllamaStudentConfig(OllamaConfig):
    """Ollama configuration for student model with dedicated env var."""
    num_ctx: int = Field(
        ...,
        description="Context window size for student model"
    )
    num_predict: int = Field(
        ...,
        description="Maximum tokens to predict for student model"
    )
    repeat_penalty: float = Field(
        ...,
        description="Repeat penalty for student model"
    )
    repeat_last_n: int = Field(
        ...,
        description="Number of tokens for repeat penalty for student model"
    )
    stream: bool = Field(
        ...,
        description="Enable streaming for student model"
    )


class OllamaTeacherConfig(OllamaConfig):
    """Ollama configuration for teacher model with dedicated env var."""
    num_ctx: int = Field(
        ...,
        description="Context window size for teacher model"
    )
    num_predict: int = Field(
        ...,
        description="Maximum tokens to predict for teacher model"
    )
    repeat_penalty: float = Field(
        ...,
        description="Repeat penalty for teacher model"
    )
    repeat_last_n: int = Field(
        ...,
        description="Number of tokens for repeat penalty for teacher model"
    )
    stream: bool = Field(
        ...,
        description="Enable streaming for teacher model"
    )


class OpenRouterServiceProviderPreferences(BaseModel):
    """Provider routing preferences for OpenRouter."""
    priority_order: Optional[list[str]] = Field(
        default=None,
        alias="order",
        serialization_alias="order",
        description="Prioritized list of provider lowercase slugs (e.g., ['chutes', 'deepinfra'])"
    )
    require_parameter_support: Optional[bool] = Field(
        default=None,
        alias="require_parameter_support",
        serialization_alias="require_parameters",
        description="Only route to providers supporting all payload parameters (e.g. cache_control)"
    )
    allow_fallbacks: Optional[bool] = Field(
        default=None,
        alias="allow_fallbacks",
        serialization_alias="allow_fallbacks",
        description="Allow fallback providers if preferred ones are offline"
    )

    model_config = {
        "populate_by_name": True,
        "serialize_by_alias": True,
    }

    @field_validator("priority_order", mode="after")
    @classmethod
    def validate_provider_slugs(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate and clean provider slugs to ensure correct lowercase format."""
        if v is not None:
            cleaned = []
            for slug in v:
                if not isinstance(slug, str):
                    raise ValueError(f"Provider slug must be a string, got {type(slug)}")
                
                if slug != slug.lower():
                    logger.warning(
                        f"Provider slug '{slug}' is not lowercase. "
                        f"Canonical OpenRouter slugs are lowercase (e.g. '{slug.lower()}'). "
                        "Converting to lowercase."
                    )
                    slug = slug.lower()
                
                base_slug = slug.split("/", 1)[0]
                
                if base_slug not in KNOWN_OPENROUTER_PROVIDERS:
                    logger.warning(
                        f"Provider slug '{slug}' (base: '{base_slug}') is not recognized as a standard OpenRouter provider. "
                        "Double check spelling to avoid routing issues."
                    )
                cleaned.append(slug)
            return cleaned
        return v


class ApiConfig(BaseModel):
    """API provider configuration."""
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for API providers (from environment)"
    )
    max_tokens: int = Field(
        ...,
        description="Maximum tokens for API providers"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (e.g., https://openrouter.ai/api/v1 for OpenRouter)"
    )
    reasoning: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Reasoning configuration for OpenRouter reasoning models (e.g., {'enabled': True})"
    )
    openrouter_cache: Optional[bool] = Field(
        default=None,
        description="Enable OpenRouter response caching (adds X-OpenRouter-Cache header)"
    )
    prompt_caching: Optional[bool] = Field(
        default=None,
        description="Enable native prompt caching for models that support it"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional OpenRouter session ID for sticky routing"
    )
    provider: Optional[OpenRouterServiceProviderPreferences] = Field(
        default=None,
        description="Optional OpenRouter provider routing preferences"
    )

    @field_validator("api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        """Validate that API key is set for API providers."""
        return v


class LLMInstanceConfig(BaseModel):
    """Configuration for a single LLM instance."""
    provider: Literal["ollama", "api"] = Field(
        ...,
        description="LLM provider type: 'ollama' or 'api'"
    )
    model: str = Field(
        ...,
        description="Model name/identifier"
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Request timeout in seconds (required for ollama/api)"
    )
    max_retries: int = Field(
        ...,
        description="Maximum number of retry attempts"
    )
    temperature: float = Field(
        ...,
        description="Sampling temperature for generation (>= 0)"
    )
    rate_limit_delay: Optional[float] = Field(
        default=None,
        description="Delay in seconds between API calls (required for ollama/api)"
    )
    top_p: float = Field(
        ...,
        description="Nucleus sampling top-p parameter (0.0 < value <= 1.0)"
    )
    enable_cache: bool = Field(
        ...,
        description="Enable LLM response caching"
    )
    input_price_per_1m: Optional[float] = Field(
        default=None,
        description="Input price per 1M tokens ($)"
    )
    output_price_per_1m: Optional[float] = Field(
        default=None,
        description="Output price per 1M tokens ($)"
    )
    cache_read_price_per_1m: Optional[float] = Field(
        default=None,
        description="Cache read price per 1M tokens ($)"
    )

    ollama: Optional[OllamaConfig] = Field(
        default=None,
        description="Ollama-specific configuration (required when provider='ollama')"
    )
    api: Optional[ApiConfig] = Field(
        default=None,
        description="API provider configuration (required when provider='api')"
    )

    @field_validator("temperature", mode="after")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Temperature must be non-negative."""
        if v < 0:
            raise ValueError(f"temperature must be >= 0, got {v}")
        return v

    @field_validator("top_p", mode="after")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """top_p must be in range (0.0, 1.0]."""
        if v <= 0.0 or v > 1.0:
            raise ValueError(f"top_p must be in range (0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def validate_provider_config(self) -> "LLMInstanceConfig":
        """Validate provider-specific requirements."""
        if self.provider == "ollama":
            if self.timeout is None:
                raise ValueError("timeout is required when provider='ollama'")
            if self.rate_limit_delay is None:
                raise ValueError("rate_limit_delay is required when provider='ollama'")
            if self.ollama is None:
                raise ValueError(
                    "ollama configuration is required when provider='ollama'. "
                    "Add an 'ollama' section with num_ctx, num_predict, "
                    "repeat_penalty, repeat_last_n, stream, and set the "
                    "OLLAMA_*_BASE_URL environment variable."
                )
            if not self.ollama.ollama_base_url:
                raise ValueError(
                    "Ollama URL must be set via OLLAMA_*_BASE_URL env var when provider='ollama'"
                )
        elif self.provider == "api":
            if self.timeout is None:
                raise ValueError("timeout is required when provider='api'")
            if self.rate_limit_delay is None:
                raise ValueError("rate_limit_delay is required when provider='api'")
            if self.api is None:
                raise ValueError(
                    "api configuration is required when provider='api'. "
                    "Add an 'api' section with max_tokens and set the "
                    "appropriate API key environment variable."
                )
            if self.api.api_key is None:
                raise ValueError(
                    "API key must be set for API provider. "
                    "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in .env file."
                )
        return self


class LLMConfig(BaseModel):
    """Configuration for LLM instances."""
    visual_extractor: Optional[LLMInstanceConfig] = Field(
        default=None,
        description="VisualExtractor LLM configuration"
    )
    student: LLMInstanceConfig = Field(
        ...,
        description="Student LLM configuration"
    )
    teacher: LLMInstanceConfig = Field(
        ...,
        description="Teacher LLM configuration"
    )

    @model_validator(mode="after")
    def set_default_visual_extractor(self) -> "LLMConfig":
        """Set default visual extractor configuration to student model if not provided."""
        if self.visual_extractor is None:
            self.visual_extractor = self.student
        return self
