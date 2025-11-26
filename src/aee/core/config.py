# src/aee/core/config.py

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Global application configuration loaded from environment variables.
    
    Attributes:
        student_model: Identifier for the inference model.
        student_api_key: API key for the student model (defaults to GEMINI_API_KEY).
        teacher_model: Identifier for the optimizer model (optional).
        teacher_api_key: API key for the teacher model (defaults to GEMINI_API_KEY).
        temperature: Sampling temperature for generation.
        max_tokens: Maximum output token limit.
        rate_limit_delay: forced delay between API calls in seconds.
        log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    """

    # --- Model Configuration ---
    student_model: str = "gemini/gemini-2.5-flash-lite"
    student_api_key: str = Field(..., alias="GEMINI_API_KEY")
    
    teacher_model: Optional[str] = "gemini/gemini-2.5-flash-lite"
    # By default, falls back to the same key as student if not explicitly set
    teacher_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")

    # --- Generation Parameters ---
    temperature: float = 0.0
    max_tokens: int = 8192

    # --- Infrastructure ---
    rate_limit_delay: float = 30.0
    log_level: str = "ERROR"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()