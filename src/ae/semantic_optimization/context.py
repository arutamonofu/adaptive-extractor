"""Pipeline run context and artifact manager for the RE pipeline."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

from ae.semantic_optimization.hitl import (
    GTEditor,
    InteractiveConflictResolver,
    MarkdownEditor,
    RulingsStore,
)

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages reading, writing, and caching of intermediate RE artifacts."""

    def __init__(self, base_dir: Path, resume: bool = False):
        self.base_dir = Path(base_dir)
        self.resume = resume

    def get_phase_dir(self, phase: str, category: str = "", iteration: Optional[int] = None) -> Path:
        """Get the directory path for a phase.

        Maps phase names to renamed folders.
        If iteration is specified, scopes to iterations/iter_{k}/<phase>.
        """
        phase_mapping = {
            "positive": "01_positive",
            "consolidation": "02_consolidation",
            "negative": "03_negative",
            "generalization": "04_generalization",
            "compilation": "05_compilation",
        }
        mapped_phase = phase_mapping.get(phase, phase)
        root = self.base_dir
        if iteration is not None:
            root = self.base_dir / "iterations" / f"iter_{iteration}"

        if category:
            return root / mapped_phase / category
        return root / mapped_phase

    def get_path(self, phase: str, category: str, name: str, iteration: Optional[int] = None) -> Path:
        """Get the absolute path for an artifact.

        Args:
            phase: 'positive', 'consolidation', 'negative', 'generalization', or 'compilation'
            category: 'rows', 'columns', etc. (or empty string/None)
            name: Filename (e.g. '{doc_id}.json' or '{field_name}.json')
            iteration: Optional iteration number to scope under iterations/iter_{k}
        """
        phase_dir = self.get_phase_dir(phase, category, iteration=iteration)
        return phase_dir / name

    def load_or_compute(
        self,
        artifact_path: Path,
        compute_fn: Callable[[], Any],
        output_model: Optional[Any] = None,
        force_recompute: bool = False,
        cache_key: Optional[str] = None,
    ) -> Any:
        """Load artifact from disk if cache is valid and resume is active; otherwise compute and save."""
        hash_path = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
        cache_key_valid = True
        if cache_key is not None and artifact_path.exists():
            if not hash_path.exists() or hash_path.read_text(encoding="utf-8").strip() != cache_key:
                cache_key_valid = False
                logger.info(f"Cache key mismatch for {artifact_path}; invalidating stale artifact.")

        if not force_recompute and self.resume and artifact_path.exists() and cache_key_valid:
            try:
                content = artifact_path.read_text(encoding="utf-8")
                if output_model:
                    logger.info(f"Loaded cached artifact from {artifact_path} as {output_model.__name__}")
                    return output_model.model_validate_json(content)
                logger.info(f"Loaded cached artifact from {artifact_path}")
                return json.loads(content)
            except Exception as e:
                logger.warning(f"Failed to load cached artifact {artifact_path}: {e}. Recomputing.")

        result = compute_fn()

        # Serialize result
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(result, "model_dump_json"):
            content = result.model_dump_json(indent=2)
        elif hasattr(result, "json"):
            content = result.json(indent=2)
        else:
            content = json.dumps(result, ensure_ascii=False, indent=2)

        artifact_path.write_text(content, encoding="utf-8")
        if cache_key is not None:
            hash_path.write_text(cache_key, encoding="utf-8")
        logger.info(f"Saved artifact to {artifact_path}")
        return result


class RunContext:
    """Context for a single execution of the RE pipeline."""

    def __init__(
        self,
        task_name: str,
        doc_ids: List[str],
        gt_path: Path,
        ingestion_dir: Path,
        baseline_prompt_path: Path,
        schema_path: Path,
        re_client: Optional[Any] = None,
        teacher_lm: Optional[Any] = None,
        artifacts_dir: Optional[Path] = None,
        resume: bool = False,
        interactive: bool = True,
        corrected_gt_path: Optional[Path] = None,
        pdf_dir: Optional[Path] = None,
        initial_docs: int = 2,
        step_size: int = 2,
        row_batch_size: int = 5,
        max_rows_per_doc: int = 20,
        harmonize_prompt: bool = True,
        max_tool_iterations: int = 5,
        generated_prompt_path: Optional[Path] = None,
        generated_schema_path: Optional[Path] = None,
        infer_schema_constraints: bool = True,
        min_enum_doc_support: int = 4,
        pareto_regression_threshold: float = 0.01,
        enable_semantic_judge: bool = True,
        residual_enabled: bool = True,
        residual_fallback_threshold: float = 0.35,
        residual_skip_f1: float = 0.98,
        residual_max_examples_per_sig: int = 3,
        residual_max_fn_rows: int = 8,
        residual_max_fp_rows: int = 10,
        selection_metric: str = "micro_f1",
        config_path: Optional[Path] = None,
        tier_4a_lm: Optional[Any] = None,
    ):
        self.task_name = task_name
        self.doc_ids = doc_ids
        self.gt_path = Path(gt_path)
        self.ingestion_dir = Path(ingestion_dir)
        self.baseline_prompt_path = Path(baseline_prompt_path)
        self.schema_path = Path(schema_path)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else None
        self.config_path = Path(config_path) if config_path else None
        self.initial_docs = initial_docs
        self.step_size = step_size
        self.row_batch_size = row_batch_size
        self.max_rows_per_doc = max_rows_per_doc
        self.harmonize_prompt = harmonize_prompt
        self.max_tool_iterations = max_tool_iterations
        self.infer_schema_constraints = infer_schema_constraints
        self.min_enum_doc_support = min_enum_doc_support
        self.pareto_regression_threshold = pareto_regression_threshold
        self.enable_semantic_judge = enable_semantic_judge
        self.residual_enabled = residual_enabled
        self.residual_fallback_threshold = residual_fallback_threshold
        self.residual_skip_f1 = residual_skip_f1
        self.residual_max_examples_per_sig = residual_max_examples_per_sig
        self.residual_max_fn_rows = residual_max_fn_rows
        self.residual_max_fp_rows = residual_max_fp_rows
        self.selection_metric = (
            selection_metric if selection_metric in ("micro_f1", "macro_f1") else "micro_f1"
        )

        # Support both new re_client and legacy teacher_lm
        if re_client is not None:
            self.re_client = re_client
            if hasattr(re_client, "get_lm_provider"):
                self.teacher_lm = re_client.get_lm_provider()
            else:
                self.teacher_lm = re_client
        elif teacher_lm is not None:
            self.teacher_lm = teacher_lm
            self.re_client = teacher_lm
        else:
            raise ValueError("Either re_client or teacher_lm must be provided to RunContext")

        self.tier_4a_lm = tier_4a_lm
        self._tier_4a_adapter: Optional[Any] = None

        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path("data/interim/semantic_optimization")
        self.generated_prompt_path = (
            Path(generated_prompt_path) if generated_prompt_path else (self.artifacts_dir / "generated_instruction.txt")
        )
        self.generated_schema_path = (
            Path(generated_schema_path) if generated_schema_path else (self.artifacts_dir / "generated_schema.yaml")
        )
        self.resume = resume
        self.interactive = interactive

        # Initialize artifact manager directly in artifacts_dir
        self.artifacts = ArtifactManager(self.artifacts_dir, resume=resume)

        # Initialize corrected GT editor
        if corrected_gt_path is not None:
            self.corrected_gt_path = Path(corrected_gt_path)
        else:
            self.corrected_gt_path = self.artifacts_dir / "ground_truth_corrected.csv"

        self.gt_editor = GTEditor(
            source_gt_path=self.gt_path,
            corrected_gt_path=self.corrected_gt_path,
            resume=resume,
        )

        # Initialize MarkdownEditor for inspecting and patching ingested documents
        self.md_editor = MarkdownEditor(
            ingestion_dir=self.ingestion_dir,
            pdf_dir=pdf_dir,
        )

        # Initialize Expert Rulings Store
        self.rulings_store = RulingsStore(storage_path=self.artifacts_dir / "expert_rulings.json")

        # Initialize Interactive Conflict Resolver
        self.resolver = InteractiveConflictResolver(
            gt_editor=self.gt_editor,
            rulings_store=self.rulings_store,
            md_editor=self.md_editor,
            interactive=self.interactive,
            artifacts=self.artifacts,
        )

        # Current iteration tracker (for iteration-scoped artifacts)
        self.current_iteration: Optional[int] = None

        # Lazy loaded inputs
        self._baseline_prompt: Optional[str] = None
        self._schema: Optional[str] = None

    def tier_4a_context(self) -> Any:
        """Return a DSPy LM context manager for Tier 4A worker signatures if configured."""
        if self.tier_4a_lm is None:
            return contextlib.nullcontext()
        import dspy

        if self._tier_4a_adapter is None:
            if isinstance(self.tier_4a_lm, dspy.LM):
                self._tier_4a_adapter = self.tier_4a_lm
            elif hasattr(self.tier_4a_lm, "get_dspy_adapter"):
                self._tier_4a_adapter = self.tier_4a_lm.get_dspy_adapter()
            else:
                from ae.core.llm import DSPyLMAdapter

                self._tier_4a_adapter = DSPyLMAdapter(self.tier_4a_lm)
        return dspy.context(lm=self._tier_4a_adapter)

    @property
    def active_gt_path(self) -> Path:
        """Always return path to corrected Ground Truth file."""
        return self.corrected_gt_path

    @property
    def baseline_prompt(self) -> str:
        """Load baseline instruction from file."""
        if self._baseline_prompt is None:
            if not self.baseline_prompt_path.exists():
                raise FileNotFoundError(f"Baseline prompt file not found at {self.baseline_prompt_path}")
            self._baseline_prompt = self.baseline_prompt_path.read_text(encoding="utf-8")
        return self._baseline_prompt

    @property
    def schema(self) -> str:
        """Load task schema YAML config as a raw string."""
        if self._schema is None:
            if not self.schema_path.exists():
                raise FileNotFoundError(f"Schema YAML file not found at {self.schema_path}")
            self._schema = self.schema_path.read_text(encoding="utf-8")
        return self._schema

    @property
    def schema_fields(self) -> List[str]:
        """Parse schema YAML and return list of defined field names (lowercased, stripped)."""
        import yaml
        try:
            schema_data = yaml.safe_load(self.schema) or {}
            fields_dict = schema_data.get("fields", {})
            return [str(f).strip().lower() for f in fields_dict.keys()]
        except Exception as e:
            logger.error(f"Failed to parse schema YAML: {e}")
            return []

    def get_gt_columns_to_keep(self, df_columns: List[str]) -> List[str]:
        """Return list of DataFrame columns that correspond to schema fields or row_converter mappings."""
        import yaml
        fields = set(self.schema_fields)
        try:
            schema_data = yaml.safe_load(self.schema) or {}
            converter_map = schema_data.get("row_converter", {})
            if isinstance(converter_map, dict):
                mapping = converter_map.get("mapping", converter_map)
                if isinstance(mapping, dict):
                    for f_name, cols in mapping.items():
                        if isinstance(cols, list):
                            fields.update(str(c).strip().lower() for c in cols)
                        elif isinstance(cols, str):
                            fields.add(cols.strip().lower())
        except Exception:
            pass
        return [c for c in df_columns if c in fields]
