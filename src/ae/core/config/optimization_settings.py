"""Optimization and other configuration settings."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator

logger = logging.getLogger(__name__)


class MinerUParserConfig(BaseModel):
    """MinerU PDF parser configuration."""
    api_url: str = Field(
        default="https://mineru.net/api/v4",
        description="MinerU API base URL"
    )
    model_version: str = Field(
        default="vlm",
        description="MinerU model version to use"
    )
    poll_interval: int = Field(
        default=3,
        description="Interval in seconds between status polls"
    )
    poll_timeout: int = Field(
        default=600,
        description="Timeout in seconds for parsing to complete"
    )


class ChartExtractionConfig(BaseModel):
    """Configuration for data extraction from charts using VLM."""
    enabled: bool = Field(
        default=True,
        description="Whether chart extraction is enabled using the ingestor model"
    )


class IngestionConfig(BaseModel):
    """Document ingestion configuration."""
    overwrite: bool = Field(
        ...,
        description="Overwrite existing parsed files"
    )
    concurrency: int = Field(
        default=1,
        description="Number of concurrent files to parse"
    )
    mineru: MinerUParserConfig = Field(
        default_factory=MinerUParserConfig,
        description="MinerU parser configuration"
    )
    chart_extraction: ChartExtractionConfig = Field(
        default_factory=ChartExtractionConfig,
        description="Chart extraction configuration"
    )


class OptimizationConfig(BaseModel):
    """Optimization and training configuration."""
    total_load: int = Field(
        ...,
        description="Total number of samples to load for optimization"
    )
    train_split: int = Field(
        ...,
        description="Number of samples for training split"
    )
    num_candidates: int = Field(
        ...,
        description="Number of candidate instructions to generate"
    )
    num_trials: int = Field(
        ...,
        description="Number of optimization trials to run"
    )
    max_bootstrapped_demos: int = Field(
        ...,
        description="Maximum number of bootstrapped demonstrations (0 for zero-shot mode)"
    )
    max_labeled_demos: int = Field(
        ...,
        description="Maximum number of labeled demonstrations (0 for zero-shot mode)"
    )

    @field_validator('max_bootstrapped_demos')
    @classmethod
    def validate_max_bootstrapped_demos(cls, v):
        """Validate max_bootstrapped_demos is non-negative."""
        if v < 0:
            raise ValueError('max_bootstrapped_demos must be >= 0 (use 0 for zero-shot mode)')
        return v

    @field_validator('max_labeled_demos')
    @classmethod
    def validate_max_labeled_demos(cls, v):
        """Validate max_labeled_demos is non-negative."""
        if v < 0:
            raise ValueError('max_labeled_demos must be >= 0 (use 0 for zero-shot mode)')
        return v
    minibatch: bool = Field(
        ...,
        description="Use minibatch evaluation during optimization"
    )
    minibatch_size: int = Field(
        ...,
        description="Size of minibatch for evaluation"
    )
    view_data_batch_size: int = Field(
        ...,
        description="Batch size for viewing data samples"
    )
    metric_threshold: float = Field(
        ...,
        description="Threshold metric value for optimization stopping"
    )
    init_temperature: float = Field(
        ...,
        description="Initial temperature for candidate generation"
    )
    random_seed: int = Field(
        ...,
        description="Random seed for reproducibility"
    )
    use_cache: bool = Field(
        ...,
        validation_alias=AliasChoices("enable_cache", "use_cache"),
        serialization_alias="use_cache",
        description="Enable caching during optimization"
    )
    verbose: bool = Field(
        ...,
        description="Enable verbose logging during optimization"
    )
    max_errors: int = Field(
        default=5,
        description=(
            "Maximum number of errors allowed before stopping optimization "
            "(DSPy parallelizer setting). Increase from default (5) to allow "
            "more faults during trials."
        )
    )
    save_llm_history: bool = Field(
        default=True,
        description="Save LLM call histories after optimization"
    )
    llm_history_dir: str = Field(
        default="logs/llm_history",
        description="Directory for LLM history files"
    )


class ExtractionConfig(BaseModel):
    """Extraction configuration."""
    enable_cache: bool = Field(
        ...,
        description="Enable LLM response caching during extraction"
    )
    save_llm_history: bool = Field(
        default=False,
        description="Save LLM call histories after extraction"
    )
    llm_history_dir: str = Field(
        default="logs/llm_history",
        description="Directory for LLM history files"
    )


class CacheConfig(BaseModel):
    """Disk cache configuration."""
    disk_size_limit_bytes: int = Field(
        ...,
        description="Maximum disk cache size in bytes"
    )
    memory_max_entries: int = Field(
        ...,
        description="Maximum number of entries in memory cache"
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = Field(
        ...,
        description="Number of failures before opening circuit"
    )
    reset_timeout: float = Field(
        ...,
        description="Seconds to wait before attempting reset (half-open state)"
    )
    half_open_max_calls: int = Field(
        ...,
        description="Maximum test calls allowed in half-open state"
    )


class REConfig(BaseModel):
    """Configuration for reverse engineering pipeline."""
    use_split: str = Field(
        default="train",
        description="Dataset split to use for RE"
    )
    artifacts_dir: Path = Field(
        default=Path("data/reverse_engineering"),
        description="Directory for intermediate RE artifacts"
    )
    resume: bool = Field(
        default=False,
        description="Whether to resume from existing intermediate artifacts"
    )
    save_llm_history: bool = Field(
        default=True,
        description="Whether to save the LLM history log for RE calls"
    )
    llm_history_dir: Path = Field(
        default=Path("logs/re_llm_history"),
        description="Directory to save RE LLM logs"
    )
