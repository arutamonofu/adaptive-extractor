# src/aee/core/config.py

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class ProjectConfig(BaseModel):
    name: str = "autoevoextractor"
    log_level: str = "INFO"

class PathsConfig(BaseModel):
    pdf_dir: Path = Path("data/pdf")
    parsed_dir: Path = Path("data/parsed")
    ground_truth_dir: Path = Path("data/ground_truth")
    splits_file: Path = Path("data/splits.json")
    agents_dir: Path = Path("data/agents")
    predictions_dir: Path = Path("data/predictions")
    logs_dir: Path = Path("logs")

    @field_validator("*", mode="before")
    @classmethod
    def cast_to_path(cls, v):
        return Path(v) if v else v

class OllamaConfig(BaseModel):
    ollama_base_url: str = "https://aicltr.itmo.ru/ollama"
    num_ctx: int = 32000
    num_predict: int = 4096
    repeat_penalty: float = 1.2
    repeat_last_n: int = 64
    stream: bool = True

class NonOllamaConfig(BaseModel):
    api_key: Optional[SecretStr] = None
    max_tokens: int = 4096

class LLMInstanceConfig(BaseModel):
    use_ollama: bool = True
    model: str = "mistral-small3.1-24b-128k:latest"
    timeout: int = 600
    max_retries: int = 3
    temperature: float = 0.0
    rate_limit_delay: float = 1.0
    top_p: float = 0.9
    
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    non_ollama: NonOllamaConfig = Field(default_factory=NonOllamaConfig)

class LLMConfig(BaseModel):
    student: LLMInstanceConfig = Field(default_factory=LLMInstanceConfig)
    teacher: LLMInstanceConfig = Field(default_factory=LLMInstanceConfig)

class DoclingConfig(BaseModel):
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_threads: int = 4
    do_ocr: bool = True
    do_table_structure: bool = True

class MarkerConfig(BaseModel):
    device: Literal["cpu", "cuda"] = "cpu"

class IngestionConfig(BaseModel):
    parser: Literal["docling", "marker"] = "docling"
    overwrite: bool = False
    
    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    marker: MarkerConfig = Field(default_factory=MarkerConfig)

class OptimizationConfig(BaseModel):
    total_load: int = 20
    train_split: int = 20
    num_candidates: int = 10
    num_trials: int = 50
    max_bootstrapped_demos: int = 2
    max_labeled_demos: int = 2
    minibatch: bool = False
    minibatch_size: int = 1
    view_data_batch_size: int = 3
    metric_threshold: float = 1.0
    init_temperature: float = 0.5
    random_seed: int = 42
    use_cache: bool = True
    verbose: bool = True

class EvaluationConfig(BaseModel):
    float_tolerance: float = 0.05
    compare_fields: List[str] = Field(default_factory=list)

class TaskConfig(BaseModel):
    name: str = "nanozymes"
    evaluation: EvaluationConfig

class Settings(BaseSettings):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    parsing: IngestionConfig = Field(default_factory=IngestionConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    task: TaskConfig

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "Settings":
        """
        Loads settings in the following priority:
        1. Default YAML (relative to this file)
        2. Custom YAML (if provided)
        3. Environment variables (handled by Pydantic)
        """
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        default_path = base_dir / "config" / "default.yaml"
        
        config_data = {}
        
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Default config not found at {default_path}. Using internal defaults.")

        if config_path:
            custom_path = Path(config_path)
            if custom_path.exists():
                with open(custom_path, "r", encoding="utf-8") as f:
                    custom_data = yaml.safe_load(f) or {}
                    cls._deep_update(config_data, custom_data)
            else:
                print(f"Warning: Custom config {config_path} not found. Using defaults.")

        return cls(**config_data)

    @staticmethod
    def _deep_update(base_dict: dict, update_with: dict):
        for k, v in update_with.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                Settings._deep_update(base_dict[k], v)
            else:
                base_dict[k] = v

settings = Settings.load()