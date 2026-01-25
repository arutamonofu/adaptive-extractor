# src/aee/core/config.py

from typing import Literal
from pydantic import Field, SecretStr, HttpUrl, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Global application configuration.
    
    Manages switching between Local (University Server) and Cloud (Gemini) modes,
    security credential handling, and model selection strategies.
    """

    use_local_llm: bool = Field(
        default=True, 
        description="Switch between University Server (Ollama) and Cloud (Gemini)."
    )

    ollama_base_url: HttpUrl = Field(
        default="https://aicltr.itmo.ru/ollama",
        description="Base URL for the university server."
    ) # type: ignore
    local_student_model: str = "mistral-small3.1-24b-128k:latest"
    local_teacher_model: str = "gpt-oss:120b"

    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    cloud_student_model: str = "gemini/gemini-2.5-flash"
    cloud_teacher_model: str = "gemini/gemini-2.5-pro"

    temperature: float = 0.0
    max_tokens: int = 100000
    rate_limit_delay: float = 10.0  
    num_threads: int = 1            
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True
    )

    @computed_field
    @property
    def student_model_name(self) -> str:
        """Returns the active student model identifier based on current mode."""
        return self.local_student_model if self.use_local_llm else self.cloud_student_model

    @computed_field
    @property
    def teacher_model_name(self) -> str:
        """Returns the active teacher model identifier based on current mode."""
        return self.local_teacher_model if self.use_local_llm else self.cloud_teacher_model

    @property
    def active_api_key(self) -> str:
        """
        Returns the appropriate API key string.
        - For Local: Returns dummy "ollama".
        - For Cloud: returns unmasked secret key (raises error if missing).
        """
        if self.use_local_llm:
            return "ollama"
        
        if not self.gemini_api_key:
            raise ValueError(
                "Cloud mode is enabled (use_local_llm=False), but GEMINI_API_KEY is missing. "
                "Please check your .env file."
            )
        return self.gemini_api_key.get_secret_value()

    @property
    def active_api_base(self) -> str | None:
        """
        Returns the clean string URL for local models or None for cloud.
        Used by llm.py factory to decide which client to instantiate.
        """
        return str(self.ollama_base_url) if self.use_local_llm else None

settings = Settings()