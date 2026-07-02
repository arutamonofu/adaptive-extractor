"""Paths and project configuration settings."""

import logging
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    """Project-level configuration settings."""
    log_level: str = Field(
        ...,
        description="Logging level (from YAML config)"
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if isinstance(v, str):
            v = v.upper()
            if v not in valid_levels:
                raise ValueError(
                    f"Invalid log level: {v}. Must be one of {valid_levels}"
                )
        return v


class PathsConfig(BaseModel):
    """File system paths configuration."""
    pdf_dir: Path = Field(
        default=Path("data/raw/pdfs"),
        description="Directory containing PDF files to process"
    )
    ingestion_dir: Path = Field(
        default=Path("data/interim/ingestion"),
        description="Directory for ingested/parsed document outputs"
    )
    ground_truth_file: Path = Field(
        default=Path("data/raw/ground_truth.csv"),
        description="Path to ground truth CSV file"
    )
    splits_file: Path = Field(
        default=Path("data/processed/splits.json"),
        description="Path to JSON file with data splits (train/val/test)"
    )
    agents_dir: Path = Field(
        default=Path("data/processed/agents"),
        description="Directory for storing trained agents"
    )
    extracted_dir: Path = Field(
        default=Path("data/processed/extractions"),
        description="Directory for final extraction outputs"
    )
    config_dir: Path = Field(
        default=Path("config"),
        description="Directory for configuration files"
    )
    schema_file: Path = Field(
        default=Path("config/schema.yaml"),
        description="Path to extraction schema YAML file"
    )
    baseline_prompt_file: Path = Field(
        default=Path("config/baseline_instruction.txt"),
        description="Path to baseline instruction text file"
    )
    generated_prompt_file: Path = Field(
        default=Path("data/interim/reverse_engineering/generated_instruction.txt"),
        description="Path to generated/optimized instruction text file"
    )
    chart_prompt_file: Path = Field(
        default=Path("config/chart_instruction.txt"),
        description="Path to optional chart prompt instruction file"
    )

    @field_validator("*", mode="before")
    @classmethod
    def cast_to_path(cls, v: Any) -> Path:
        """Cast input value to Path object."""
        return Path(v) if v else v
