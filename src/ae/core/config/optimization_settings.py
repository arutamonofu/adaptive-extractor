"""Optimization and other configuration settings."""

import logging
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from ae.core.config.llm_settings import (
    ConfigurableModelSpec,
    LLMConfig,
    LLMInstanceConfig,
    StageApiClientConfig,
)

logger = logging.getLogger(__name__)


class ExecutionConcurrencyConfig(BaseModel):
    """Concurrency settings for document ingestion pipeline."""

    max_documents: int = Field(
        default=2,
        ge=1,
        description="Maximum number of PDF documents processed concurrently",
    )
    max_requests_per_document: int = Field(
        default=8,
        ge=1,
        description="Maximum concurrent API requests within a single document",
    )


class ExecutionConfig(BaseModel):
    """Execution and run control settings for ingestion."""

    overwrite: bool = Field(
        default=False,
        description="Overwrite existing parsed files (rerun from transcribe stage)",
    )
    from_stage: Optional[str] = Field(
        default=None,
        description="Stage to run pipeline from",
    )
    save_llm_history: bool = Field(
        default=False,
        description="Save LLM call histories after ingestion",
    )
    concurrency: ExecutionConcurrencyConfig = Field(
        default_factory=ExecutionConcurrencyConfig,
        description="Concurrency settings for document and request parallelism",
    )


class ApiClientConfig(BaseModel):
    """API client and network transport parameters."""

    timeout: int = Field(
        default=300,
        ge=1,
        description="API request timeout in seconds",
    )
    max_retries: int = Field(
        default=5,
        ge=0,
        description="Maximum API call retries on failure",
    )
    retry_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Base delay between retries in seconds",
    )


class GeminiDefaultsConfig(BaseModel):
    """Legacy default generation parameters for backward compatibility."""

    model: str = Field(
        default="",
        description="Default model name",
    )
    temperature: float = Field(default=0.0)
    max_output_tokens: int = Field(default=65536)
    media_resolution: str = Field(default="high")
    thinking_level: Optional[str] = Field(default="low")
    thinking_budget: Optional[int] = Field(
        default=None,
        ge=0,
        description="Max reasoning tokens budget",
    )


class StageModelConfig(BaseModel):
    """Explicit model configuration for an individual pipeline stage."""

    provider: str = Field(
        default="gemini",
        description="LLM provider name (e.g. 'gemini', 'openrouter', 'ollama')",
    )
    model: str = Field(
        default="gemini-3.8-flash",
        description="Model identifier",
    )
    temperature: float = Field(default=0.0)
    max_output_tokens: int = Field(default=65536)
    media_resolution: str = Field(default="high")
    thinking_level: Optional[str] = Field(default="low")
    thinking_budget: Optional[int] = Field(
        default=None,
        ge=0,
        description="Max reasoning tokens budget (0 disables reasoning entirely)",
    )
    page_dpi: Optional[int] = Field(
        default=None,
        description="DPI for page rendering passed to visual localization (if applicable)",
    )

    @field_validator("model", mode="after")
    @classmethod
    def validate_stage_model(cls, v: str) -> str:
        if v is not None and not v.strip():
            raise ValueError("Stage model cannot be empty string")
        return v.strip()


class _StageWithModelConfig(BaseModel):
    """Base class proxying model fields for stage configs."""

    model: StageModelConfig

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "model" and hasattr(self, "model") and hasattr(self.model, name):
            setattr(self.model, name, value)
        else:
            super().__setattr__(name, value)


class ParseTextStageConfig(_StageWithModelConfig):
    """Configuration for text parsing stage."""

    model: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(model="gemini-3.8-flash", thinking_level="low"),
        description="Model settings for text parsing",
    )


class ClassifyStageConfig(_StageWithModelConfig):
    """Configuration for visual placeholder classification stage."""

    model: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(model="gemini-3.8-flash", thinking_level="low"),
        description="Model settings for classification",
    )


class LocalizeStageConfig(_StageWithModelConfig):
    """Configuration for visual bounding box localization stage."""

    page_dpi: int = Field(
        default=150,
        ge=72,
        description="DPI for page rendering passed to visual localization",
    )
    model: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(model="gemini-3.8-flash", thinking_level="low"),
        description="Model settings for localization",
    )


class CropUpscaleConfig(BaseModel):
    """Upscaling configuration for visual cropping."""

    enabled: bool = Field(default=True)
    min_long_side_px: int = Field(default=2000)
    max_dpi: int = Field(default=1200)


class CropConfig(BaseModel):
    """Crop stage configuration (PyMuPDF high-DPI cropping and upscaling)."""

    padding_ratio: float = Field(default=0.02)
    base_dpi: int = Field(default=300)
    upscale: CropUpscaleConfig = Field(default_factory=CropUpscaleConfig)


class DecomposeStageConfig(_StageWithModelConfig):
    """Configuration for composite figure decomposition stage."""

    model: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(model="gemini-3.8-flash", thinking_level="medium"),
        description="Model settings for decomposition",
    )


class TableExtractStageConfig(_StageWithModelConfig):
    """Configuration for table extraction."""

    model: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(model="gemini-3.8-flash", thinking_level="medium"),
        description="Model settings for table extraction",
    )


class ChartExtractStageConfig(_StageWithModelConfig):
    """Configuration for chart extraction."""

    model: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(model="gemini-3.8-flash", thinking_level="medium"),
        description="Model settings for chart extraction",
    )


class MoleculeValidationConfig(BaseModel):
    """Configuration for RDKit molecular validation."""

    validate_rdkit: bool = Field(default=True, description="Validate SMILES via RDKit")
    max_retries: int = Field(
        default=1,
        ge=0,
        description="Maximum self-correction retries when RDKit validation fails",
    )


# Alias for backward compatibility
MoleculesConfig = MoleculeValidationConfig


class MoleculeExtractStageConfig(_StageWithModelConfig):
    """Configuration for molecular structure extraction."""

    model: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(model="gemini-3.8-flash", thinking_level="medium"),
        description="Model settings for molecule extraction",
    )
    validation: MoleculeValidationConfig = Field(
        default_factory=MoleculeValidationConfig,
        description="Molecular validation settings",
    )


class ParseVisualsStagesConfig(BaseModel):
    """Configuration for modality-specific visual parsing stages."""

    table: TableExtractStageConfig = Field(default_factory=TableExtractStageConfig)
    chart: ChartExtractStageConfig = Field(default_factory=ChartExtractStageConfig)
    molecule: MoleculeExtractStageConfig = Field(default_factory=MoleculeExtractStageConfig)


class IngestionStagesConfig(BaseModel):
    """Pipeline stages configuration for document ingestion."""

    parse_text: ParseTextStageConfig = Field(default_factory=ParseTextStageConfig)
    classify: ClassifyStageConfig = Field(default_factory=ClassifyStageConfig)
    localize: LocalizeStageConfig = Field(default_factory=LocalizeStageConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    decompose: DecomposeStageConfig = Field(default_factory=DecomposeStageConfig)
    parse_visuals: ParseVisualsStagesConfig = Field(default_factory=ParseVisualsStagesConfig)

    def get_stage_config(
        self, stage_name: str, sub_stage: Optional[str] = None
    ) -> StageModelConfig:
        """Resolve stage model configuration."""
        if stage_name in ("parse_text", "transcribe"):
            return self.parse_text.model
        if stage_name == "classify":
            return self.classify.model
        if stage_name == "localize":
            m = self.localize.model.model_copy()
            m.page_dpi = self.localize.page_dpi
            return m
        if stage_name == "decompose":
            return self.decompose.model
        if stage_name in ("parse_visuals", "parse", "extract"):
            if sub_stage in ("molecule", "molecular_structure"):
                return self.parse_visuals.molecule.model
            if sub_stage == "chart":
                return self.parse_visuals.chart.model
            if sub_stage == "table":
                return self.parse_visuals.table.model
            raise ValueError(f"Unknown parse sub_stage: {sub_stage}")
        raise ValueError(f"Unknown stage_name: {stage_name}")


class GeminiStagesConfig(IngestionStagesConfig):
    """Legacy alias for stages config."""
    pass


class GeminiParserConfig(BaseModel):
    """Legacy configuration container for backward compatibility."""

    max_concurrent_requests: int = Field(default=4)
    max_retries: int = Field(default=5)
    retry_delay: float = Field(default=2.0)
    timeout: int = Field(default=300)
    defaults: GeminiDefaultsConfig = Field(default_factory=GeminiDefaultsConfig)
    stages: Optional[Any] = None
    crop: CropConfig = Field(default_factory=CropConfig)
    molecules: MoleculeValidationConfig = Field(default_factory=MoleculeValidationConfig)

    def get_stage_config(
        self, stage_name: str, sub_stage: Optional[str] = None
    ) -> StageModelConfig:
        if self.stages is not None and hasattr(self.stages, "get_stage_config"):
            return self.stages.get_stage_config(stage_name, sub_stage=sub_stage)
        return StageModelConfig(
            provider="gemini",
            model=self.defaults.model or "gemini-3.8-flash",
            temperature=self.defaults.temperature,
            max_output_tokens=self.defaults.max_output_tokens,
            media_resolution=self.defaults.media_resolution,
            thinking_level=self.defaults.thinking_level,
            thinking_budget=self.defaults.thinking_budget,
        )


class _GeminiLegacyAdapter:
    """Backward compatibility adapter for legacy code accessing config.gemini."""

    def __init__(self, config: "IngestionConfig") -> None:
        self._config = config

    @property
    def defaults(self) -> GeminiDefaultsConfig:
        m = self._config.stages.parse_text.model
        return GeminiDefaultsConfig(
            model=m.model,
            temperature=m.temperature,
            max_output_tokens=m.max_output_tokens,
            media_resolution=m.media_resolution,
            thinking_level=m.thinking_level,
            thinking_budget=m.thinking_budget,
        )

    @property
    def max_concurrent_requests(self) -> int:
        return self._config.execution.concurrency.max_requests_per_document

    @property
    def max_retries(self) -> int:
        return self._config.api_client.max_retries

    @property
    def retry_delay(self) -> float:
        return self._config.api_client.retry_delay

    @property
    def timeout(self) -> int:
        return self._config.api_client.timeout

    @property
    def crop(self) -> CropConfig:
        return self._config.stages.crop

    @property
    def molecules(self) -> MoleculeValidationConfig:
        return self._config.stages.parse_visuals.molecule.validation

    @property
    def stages(self) -> IngestionStagesConfig:
        return self._config.stages

    def get_stage_config(
        self, stage_name: str, sub_stage: Optional[str] = None
    ) -> StageModelConfig:
        return self._config.stages.get_stage_config(stage_name, sub_stage=sub_stage)


class IngestionConfig(BaseModel):
    """Document ingestion configuration (Ingestion v2)."""

    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Pipeline execution and concurrency control",
    )
    api_client: ApiClientConfig = Field(
        default_factory=ApiClientConfig,
        description="API client and network transport parameters",
    )
    stages: IngestionStagesConfig = Field(
        default_factory=IngestionStagesConfig,
        description="Configurations for all pipeline stages",
    )
    ingestion_dir: Optional[Path] = Field(
        default=None,
        description="Directory for ingested/parsed document outputs",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_structure(cls, data: Any) -> Any:
        """Migrate legacy dictionary structures if encountered."""
        if not isinstance(data, dict):
            return data

        # If flat legacy fields or legacy "gemini" container are present
        if "stages" not in data and "gemini" in data:
            gemini_raw = data.get("gemini", {})
            if isinstance(gemini_raw, GeminiParserConfig):
                gemini_dict = gemini_raw.model_dump()
            elif isinstance(gemini_raw, dict):
                gemini_dict = gemini_raw
            else:
                gemini_dict = {}

            stages_raw = gemini_dict.get("stages", {}) or {}
            defaults_raw = gemini_dict.get("defaults", {}) or {}
            default_model = (
                defaults_raw.get("model")
                if isinstance(defaults_raw, dict)
                else getattr(defaults_raw, "model", "gemini-3.8-flash")
            ) or "gemini-3.8-flash"

            def _build_stage_model(stage_item: Any, default_thinking: str = "low") -> dict:
                s_dict = (
                    stage_item
                    if isinstance(stage_item, dict)
                    else getattr(stage_item, "__dict__", {})
                )
                d_dict = (
                    defaults_raw
                    if isinstance(defaults_raw, dict)
                    else getattr(defaults_raw, "__dict__", {})
                )
                return {
                    "provider": s_dict.get("provider", "gemini"),
                    "model": s_dict.get("model") or default_model,
                    "temperature": s_dict.get("temperature", d_dict.get("temperature", 0.0)),
                    "max_output_tokens": s_dict.get(
                        "max_output_tokens", d_dict.get("max_output_tokens", 65536)
                    ),
                    "media_resolution": s_dict.get(
                        "media_resolution", d_dict.get("media_resolution", "high")
                    ),
                    "thinking_level": s_dict.get(
                        "thinking_level", d_dict.get("thinking_level", default_thinking)
                    ),
                    "thinking_budget": s_dict.get(
                        "thinking_budget", d_dict.get("thinking_budget", None)
                    ),
                }

            parse_text_raw = (
                stages_raw.get("parse_text", stages_raw.get("transcribe", {}))
                if isinstance(stages_raw, dict)
                else {}
            )
            classify_raw = stages_raw.get("classify", {}) if isinstance(stages_raw, dict) else {}
            localize_raw = stages_raw.get("localize", {}) if isinstance(stages_raw, dict) else {}
            decompose_raw = stages_raw.get("decompose", {}) if isinstance(stages_raw, dict) else {}
            extract_raw = (
                stages_raw.get("parse_visuals", stages_raw.get("extract", {}))
                if isinstance(stages_raw, dict)
                else {}
            )

            crop_raw = gemini_dict.get("crop", {})
            molecules_raw = gemini_dict.get("molecules", {})

            data["stages"] = {
                "parse_text": {
                    "model": _build_stage_model(parse_text_raw, "low"),
                },
                "classify": {
                    "model": _build_stage_model(classify_raw, "low"),
                },
                "localize": {
                    "page_dpi": localize_raw.get("page_dpi", 150),
                    "model": _build_stage_model(localize_raw, "low"),
                },
                "crop": crop_raw or {},
                "decompose": {
                    "model": _build_stage_model(decompose_raw, "medium"),
                },
                "parse_visuals": {
                    "table": {
                        "model": _build_stage_model(
                            extract_raw.get("table", {}) if isinstance(extract_raw, dict) else {},
                            "medium",
                        ),
                    },
                    "chart": {
                        "model": _build_stage_model(
                            extract_raw.get("chart", {}) if isinstance(extract_raw, dict) else {},
                            "medium",
                        ),
                    },
                    "molecule": {
                        "model": _build_stage_model(
                            extract_raw.get("molecule", {}) if isinstance(extract_raw, dict) else {},
                            "medium",
                        ),
                        "validation": {
                            "validate_rdkit": molecules_raw.get("validate_rdkit", True)
                            if isinstance(molecules_raw, dict)
                            else getattr(molecules_raw, "validate_rdkit", True),
                            "max_retries": molecules_raw.get(
                                "max_rdkit_retries",
                                molecules_raw.get("max_retries", 1),
                            )
                            if isinstance(molecules_raw, dict)
                            else getattr(molecules_raw, "max_retries", 1),
                        },
                    },
                },
            }
            data["api_client"] = {
                "timeout": gemini_dict.get("timeout", 300),
                "max_retries": gemini_dict.get("max_retries", 5),
                "retry_delay": gemini_dict.get("retry_delay", 2.0),
            }
            data["execution"] = {
                "overwrite": data.get("overwrite", False),
                "from_stage": data.get("from_stage", None),
                "save_llm_history": data.get("save_llm_history", False),
                "concurrency": {
                    "max_documents": data.get("concurrency", 2),
                    "max_requests_per_document": gemini_dict.get("max_concurrent_requests", 8),
                },
            }

        return data

    # -------------------------------------------------------------------------
    # Backward compatibility properties and methods
    # -------------------------------------------------------------------------
    @property
    def overwrite(self) -> bool:
        return self.execution.overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        self.execution.overwrite = value

    @property
    def from_stage(self) -> Optional[str]:
        return self.execution.from_stage

    @from_stage.setter
    def from_stage(self, value: Optional[str]) -> None:
        self.execution.from_stage = value

    @property
    def concurrency(self) -> int:
        return self.execution.concurrency.max_documents

    @concurrency.setter
    def concurrency(self, value: int) -> None:
        self.execution.concurrency.max_documents = value

    @property
    def save_llm_history(self) -> bool:
        return self.execution.save_llm_history

    @property
    def engine(self) -> str:
        return self.stages.parse_text.model.provider

    def get_stage_config(
        self, stage_name: str, sub_stage: Optional[str] = None
    ) -> StageModelConfig:
        return self.stages.get_stage_config(stage_name, sub_stage=sub_stage)

    @property
    def gemini(self) -> Any:
        return _GeminiLegacyAdapter(self)


class SyntacticOptimizationConfig(BaseModel):
    """Syntactic prompt optimization configuration (DSPy MIPROv2)."""

    total_load: int = Field(..., description="Total number of samples to load for optimization")
    train_split: int = Field(..., description="Number of samples for training split")
    num_candidates: int = Field(..., description="Number of candidate instructions to generate")
    num_trials: int = Field(..., description="Number of optimization trials to run")
    max_bootstrapped_demos: int = Field(
        ..., description="Maximum number of bootstrapped demonstrations (0 for zero-shot mode)"
    )
    max_labeled_demos: int = Field(
        ..., description="Maximum number of labeled demonstrations (0 for zero-shot mode)"
    )

    @field_validator("max_bootstrapped_demos")
    @classmethod
    def validate_max_bootstrapped_demos(cls, v):
        """Validate max_bootstrapped_demos is non-negative."""
        if v < 0:
            raise ValueError("max_bootstrapped_demos must be >= 0 (use 0 for zero-shot mode)")
        return v

    @field_validator("max_labeled_demos")
    @classmethod
    def validate_max_labeled_demos(cls, v):
        """Validate max_labeled_demos is non-negative."""
        if v < 0:
            raise ValueError("max_labeled_demos must be >= 0 (use 0 for zero-shot mode)")
        return v

    minibatch: bool = Field(..., description="Use minibatch evaluation during optimization")
    minibatch_size: int = Field(..., description="Size of minibatch for evaluation")
    view_data_batch_size: int = Field(..., description="Batch size for viewing data samples")
    metric_threshold: float = Field(
        ..., description="Threshold metric value for optimization stopping"
    )
    init_temperature: float = Field(..., description="Initial temperature for candidate generation")
    random_seed: int = Field(..., description="Random seed for reproducibility")
    use_cache: bool = Field(
        ...,
        validation_alias=AliasChoices("enable_cache", "use_cache"),
        serialization_alias="use_cache",
        description="Enable caching during optimization",
    )
    verbose: bool = Field(..., description="Enable verbose logging during optimization")
    max_errors: int = Field(
        default=5,
        description=(
            "Maximum number of errors allowed before stopping optimization "
            "(DSPy parallelizer setting). Increase from default (5) to allow "
            "more faults during trials."
        ),
    )
    save_llm_history: bool = Field(
        default=True, description="Save LLM call histories after optimization"
    )
    enable_semantic_judge: bool = Field(
        default=True,
        description="Enable LLM Semantic Judge during optimization evaluation",
    )
    llm: Optional[LLMConfig] = Field(
        default=None,
        description="MIPROv2 Student & Teacher Language Models",
    )


class ExtractionExecutionConfig(BaseModel):
    """Execution and logging settings for extraction."""
    max_tool_iterations: int = Field(
        default=5, ge=0, description="Max tool iterations for ReAct extraction graph"
    )
    enable_cache: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_cache", "use_cache"),
        description="Enable LLM response caching during extraction",
    )
    save_llm_history: bool = Field(
        default=True, description="Save LLM call histories after extraction"
    )
    structured_output_batch_size: int = Field(
        default=0,
        ge=0,
        description="Maximum records per structured output call (0 disables batching)",
    )
    max_structured_output_batches: int = Field(
        default=10,
        ge=1,
        description="Maximum continuation batches for dense documents",
    )
    enable_sanity_checks: bool = Field(
        default=True,
        description="Enable domain-independent decoding sanity checks and retry on structured output anomalies",
    )
    agent_thinking_level: Optional[Literal["minimal", "low", "medium", "high"]] = Field(
        default="low",
        description="Thinking level override for agent tool-routing node (null uses model default)",
    )


class ExtractionConfig(BaseModel):
    """Extraction configuration."""

    execution: ExtractionExecutionConfig = Field(
        default_factory=ExtractionExecutionConfig,
        description="Pipeline execution and logging control",
    )
    api_client: StageApiClientConfig = Field(
        default_factory=lambda: StageApiClientConfig(timeout=600, max_retries=5, rate_limit_delay=2.0),
        description="API client and network transport parameters",
    )
    model: ConfigurableModelSpec = Field(
        default_factory=lambda: ConfigurableModelSpec(
            provider="gemini",
            model="gemini-3.8-flash",
            temperature=0.0,
            top_p=0.95,
        ),
        description="Model configuration for extraction",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_structure(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "execution" not in data and ("enable_cache" in data or "use_cache" in data or "max_tool_iterations" in data or "save_llm_history" in data):
            res = dict(data)
            res["execution"] = {
                "enable_cache": data.get("enable_cache", data.get("use_cache", False)),
                "max_tool_iterations": data.get("max_tool_iterations", 5),
                "save_llm_history": data.get("save_llm_history", True),
            }
            if "model" not in res:
                res["model"] = {}
            if "api_client" not in res:
                res["api_client"] = {}
            return res
        return data

    @property
    def enable_cache(self) -> bool:
        return self.execution.enable_cache

    @enable_cache.setter
    def enable_cache(self, value: bool) -> None:
        self.execution.enable_cache = value

    @property
    def max_tool_iterations(self) -> int:
        return self.execution.max_tool_iterations

    @max_tool_iterations.setter
    def max_tool_iterations(self, value: int) -> None:
        self.execution.max_tool_iterations = value

    @property
    def save_llm_history(self) -> bool:
        return self.execution.save_llm_history

    @save_llm_history.setter
    def save_llm_history(self, value: bool) -> None:
        self.execution.save_llm_history = value

    @property
    def structured_output_batch_size(self) -> int:
        return self.execution.structured_output_batch_size

    @structured_output_batch_size.setter
    def structured_output_batch_size(self, value: int) -> None:
        self.execution.structured_output_batch_size = value

    @property
    def max_structured_output_batches(self) -> int:
        return self.execution.max_structured_output_batches

    @max_structured_output_batches.setter
    def max_structured_output_batches(self, value: int) -> None:
        self.execution.max_structured_output_batches = value

    @property
    def enable_sanity_checks(self) -> bool:
        return self.execution.enable_sanity_checks

    @enable_sanity_checks.setter
    def enable_sanity_checks(self, value: bool) -> None:
        self.execution.enable_sanity_checks = value

    @property
    def agent_thinking_level(self) -> Optional[Literal["minimal", "low", "medium", "high"]]:
        return self.execution.agent_thinking_level

    @agent_thinking_level.setter
    def agent_thinking_level(self, value: Optional[Literal["minimal", "low", "medium", "high"]]) -> None:
        self.execution.agent_thinking_level = value

    def get_llm_config(self) -> LLMInstanceConfig:
        return self.model.to_instance_config(
            api_client=self.api_client,
            enable_cache=self.execution.enable_cache,
        )


class CacheConfig(BaseModel):
    """Disk cache configuration."""

    disk_size_limit_bytes: int = Field(..., description="Maximum disk cache size in bytes")
    memory_max_entries: int = Field(..., description="Maximum number of entries in memory cache")


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = Field(..., description="Number of failures before opening circuit")
    reset_timeout: float = Field(
        ..., description="Seconds to wait before attempting reset (half-open state)"
    )
    half_open_max_calls: int = Field(
        ..., description="Maximum test calls allowed in half-open state"
    )


class SemanticOptimizationExecutionConfig(BaseModel):
    """Pipeline execution and control settings for Semantic Prompt Optimization."""
    use_split: str = Field(default="train", description="Dataset split to use for semantic prompt optimization")
    val_split: str = Field(default="val", description="Validation split to evaluate F1")
    f1_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="F1 threshold for early stopping"
    )
    initial_docs: int = Field(
        default=2, ge=1, description="Number of documents processed in the first iteration"
    )
    step_size: int = Field(
        default=2, ge=1, description="Number of documents added per subsequent iteration"
    )
    row_batch_size: int = Field(
        default=5, ge=1, description="Number of GT rows grouped per column analysis LLM call"
    )
    max_rows_per_doc: int = Field(
        default=20,
        ge=0,
        description="Maximum representative GT rows analyzed per document (0 for unlimited)",
    )
    harmonize_prompt: bool = Field(
        default=True, description="Whether to run a final LLM prompt harmonization step"
    )
    infer_schema_constraints: bool = Field(
        default=True,
        description="Infer and persist refined schema constraints (choices, defaults, descriptions) during generalization",
    )
    enable_semantic_judge: bool = Field(
        default=True,
        description="Enable LLM Semantic Judge during validation and residual alignment",
    )
    residual_enabled: bool = Field(
        default=True,
        description="Enable Hybrid Anchor (iter 1) + Error-Driven Residual (iter >= 2) optimization mode",
    )
    residual_fallback_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Fallback to full stratified analysis if doc F1 or Recall is below this threshold",
    )
    residual_skip_f1: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Zero-Delta Fast-Forward F1 threshold to skip reasoning analysis when discrepancies are absent/negligible",
    )
    residual_max_examples_per_sig: int = Field(
        default=3,
        ge=1,
        description="Max representative rows analyzed per column mismatch signature in residual mode",
    )
    residual_max_fn_rows: int = Field(
        default=8,
        ge=1,
        description="Max representative false-negative rows analyzed per document in residual mode",
    )
    residual_max_fp_rows: int = Field(
        default=10,
        ge=1,
        description="Max representative false-positive rows passed to NegativeRowAnalysis in residual mode",
    )
    max_tool_iterations: int = Field(
        default=5, ge=0, description="Max tool iterations for ReAct nodes"
    )
    artifacts_dir: Path = Field(
        default=Path("data/interim/semantic_optimization"),
        description="Directory for intermediate semantic optimization artifacts",
    )
    resume: bool = Field(
        default=False, description="Whether to resume from existing intermediate artifacts"
    )
    interactive: bool = Field(
        default=True, description="Enable interactive human-in-the-loop conflict resolution"
    )
    corrected_gt_file: Optional[Path] = Field(
        default=None, description="Path to save the corrected copy of Ground Truth"
    )
    enable_cache: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_cache", "use_cache"),
        description="Enable LLM response caching during semantic prompt optimization",
    )
    save_llm_history: bool = Field(
        default=True, description="Whether to save the LLM history log for semantic prompt optimization calls"
    )


class SemanticOptimizationConfig(BaseModel):
    """Configuration for semantic prompt optimization pipeline."""

    execution: SemanticOptimizationExecutionConfig = Field(
        default_factory=SemanticOptimizationExecutionConfig,
        description="Pipeline execution and control parameters",
    )
    api_client: StageApiClientConfig = Field(
        default_factory=lambda: StageApiClientConfig(timeout=300, max_retries=5, rate_limit_delay=10.0),
        description="API client and network transport parameters",
    )
    model: ConfigurableModelSpec = Field(
        default_factory=lambda: ConfigurableModelSpec(
            provider="api",
            model="qwen/qwen3.5-397b-a17b",
            temperature=1.0,
            top_p=0.9,
        ),
        description="Model configuration for Semantic Prompt Optimization reasoning",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_structure(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "execution" not in data and any(
            k in data for k in [
                "use_split", "val_split", "f1_threshold", "initial_docs", "step_size",
                "row_batch_size", "max_rows_per_doc", "harmonize_prompt", "infer_schema_constraints",
                "enable_semantic_judge", "residual_enabled", "residual_fallback_threshold",
                "residual_skip_f1", "residual_max_examples_per_sig",
                "residual_max_fn_rows", "residual_max_fp_rows", "max_tool_iterations",
                "artifacts_dir", "resume", "interactive", "corrected_gt_file",
                "save_llm_history", "enable_cache", "use_cache",
            ]
        ):
            res = dict(data)
            res["execution"] = {
                "use_split": data.get("use_split", "train"),
                "val_split": data.get("val_split", "val"),
                "f1_threshold": data.get("f1_threshold", 0.85),
                "initial_docs": data.get("initial_docs", 2),
                "step_size": data.get("step_size", 2),
                "row_batch_size": data.get("row_batch_size", 5),
                "max_rows_per_doc": data.get("max_rows_per_doc", 20),
                "harmonize_prompt": data.get("harmonize_prompt", True),
                "infer_schema_constraints": data.get("infer_schema_constraints", True),
                "enable_semantic_judge": data.get("enable_semantic_judge", True),
                "residual_enabled": data.get("residual_enabled", True),
                "residual_fallback_threshold": data.get("residual_fallback_threshold", 0.35),
                "residual_skip_f1": data.get("residual_skip_f1", 0.98),
                "residual_max_examples_per_sig": data.get("residual_max_examples_per_sig", 3),
                "residual_max_fn_rows": data.get("residual_max_fn_rows", 8),
                "residual_max_fp_rows": data.get("residual_max_fp_rows", 10),
                "max_tool_iterations": data.get("max_tool_iterations", 5),
                "artifacts_dir": data.get("artifacts_dir", Path("data/interim/semantic_optimization")),
                "resume": data.get("resume", False),
                "interactive": data.get("interactive", True),
                "corrected_gt_file": data.get("corrected_gt_file", None),
                "enable_cache": data.get("enable_cache", data.get("use_cache", True)),
                "save_llm_history": data.get("save_llm_history", True),
            }
            if "model" not in res:
                res["model"] = {}
            if "api_client" not in res:
                res["api_client"] = {}
            return res
        return data

    @property
    def use_split(self) -> str:
        return self.execution.use_split

    @property
    def val_split(self) -> str:
        return self.execution.val_split

    @property
    def f1_threshold(self) -> float:
        return self.execution.f1_threshold

    @property
    def initial_docs(self) -> int:
        return self.execution.initial_docs

    @property
    def step_size(self) -> int:
        return self.execution.step_size

    @property
    def row_batch_size(self) -> int:
        return self.execution.row_batch_size

    @property
    def max_rows_per_doc(self) -> int:
        return self.execution.max_rows_per_doc

    @property
    def harmonize_prompt(self) -> bool:
        return self.execution.harmonize_prompt

    @property
    def infer_schema_constraints(self) -> bool:
        return self.execution.infer_schema_constraints

    @property
    def enable_semantic_judge(self) -> bool:
        return self.execution.enable_semantic_judge

    @property
    def residual_enabled(self) -> bool:
        return self.execution.residual_enabled

    @property
    def residual_fallback_threshold(self) -> float:
        return self.execution.residual_fallback_threshold

    @property
    def residual_skip_f1(self) -> float:
        return self.execution.residual_skip_f1

    @property
    def residual_max_examples_per_sig(self) -> int:
        return self.execution.residual_max_examples_per_sig

    @property
    def residual_max_fn_rows(self) -> int:
        return self.execution.residual_max_fn_rows

    @property
    def residual_max_fp_rows(self) -> int:
        return self.execution.residual_max_fp_rows

    @property
    def max_tool_iterations(self) -> int:
        return self.execution.max_tool_iterations

    @property
    def artifacts_dir(self) -> Path:
        return self.execution.artifacts_dir

    @property
    def resume(self) -> bool:
        return self.execution.resume

    @resume.setter
    def resume(self, value: bool) -> None:
        self.execution.resume = value

    @property
    def interactive(self) -> bool:
        return self.execution.interactive

    @property
    def corrected_gt_file(self) -> Optional[Path]:
        return self.execution.corrected_gt_file

    @property
    def enable_cache(self) -> bool:
        return self.execution.enable_cache

    @property
    def save_llm_history(self) -> bool:
        return self.execution.save_llm_history

    def get_llm_config(self) -> LLMInstanceConfig:
        return self.model.to_instance_config(
            api_client=self.api_client,
            enable_cache=self.execution.enable_cache,
        )
