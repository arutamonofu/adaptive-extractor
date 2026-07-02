"""Pipeline run context and artifact manager for the RE pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional
import dspy

from ae.core.llm import LMProvider

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages reading, writing, and caching of intermediate RE artifacts."""

    def __init__(self, base_dir: Path, resume: bool = False):
        self.base_dir = Path(base_dir)
        self.resume = resume

    def get_phase_dir(self, phase: str, category: str = "") -> Path:
        """Get the directory path for a phase.

        Maps phase names to renamed folders.
        """
        phase_mapping = {
            "positive": "01_positive",
            "consolidation": "02_consolidation",
            "negative": "03_negative",
            "generalization": "04_generalization",
        }
        mapped_phase = phase_mapping.get(phase, phase)
        if category:
            return self.base_dir / mapped_phase / category
        return self.base_dir / mapped_phase

    def get_path(self, phase: str, category: str, name: str) -> Path:
        """Get the absolute path for an artifact.

        Args:
            phase: 'positive', 'consolidation', 'negative', or 'generalization'
            category: 'rows', 'columns', etc. (or empty string/None)
            name: Filename (e.g. '{doc_id}.json' or '{field_name}.json')
        """
        phase_dir = self.get_phase_dir(phase, category)
        return phase_dir / name

    def load_or_compute(self, artifact_path: Path, compute_fn: Callable[[], Any], output_model: Optional[Any] = None) -> Any:
        """Load artifact from disk if cache is valid and resume is active; otherwise compute and save."""
        if self.resume and artifact_path.exists():
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
        teacher_lm: LMProvider,
        artifacts_dir: Path,
        resume: bool = False,
    ):
        self.task_name = task_name
        self.doc_ids = doc_ids
        self.gt_path = Path(gt_path)
        self.ingestion_dir = Path(ingestion_dir)
        self.baseline_prompt_path = Path(baseline_prompt_path)
        self.schema_path = Path(schema_path)
        self.teacher_lm = teacher_lm
        self.artifacts_dir = Path(artifacts_dir)
        self.resume = resume

        # Initialize artifact manager directly in artifacts_dir without task_name subfolder
        self.artifacts = ArtifactManager(self.artifacts_dir, resume=resume)

        # Lazy loaded inputs
        self._baseline_prompt: Optional[str] = None
        self._schema: Optional[str] = None

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
