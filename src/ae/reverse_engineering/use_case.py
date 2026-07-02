"""RE UseCase orchestrating the Reverse Engineering pipeline execution."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import dspy

from ae.core.llm import LMProvider, DSPyLMAdapter
from ae.reverse_engineering.context import RunContext
from ae.reverse_engineering.steps import (
    run_positive_analysis,
    run_consolidation,
    run_negative_analysis,
    run_generalization,
)

logger = logging.getLogger(__name__)


@dataclass
class ReverseEngineeringRequest:
    task_name: str
    doc_ids: List[str]           # list of document IDs from train split
    gt_path: Path                 # path to Ground Truth CSV
    ingestion_dir: Path           # directory with MD files
    baseline_prompt_path: Path    # path to baseline prompt TXT
    schema_path: Path             # path to task schema YAML
    teacher_lm: LMProvider        # Teacher LM
    output_dir: Path              # data/reverse_engineering (base dir)
    resume: bool = False          # resume from cache
    steps: Optional[List[str]] = None  # specific steps to run (e.g. ['positive', 'consolidation'])


@dataclass
class ReverseEngineeringResponse:
    success: bool
    generated_prompt_path: Optional[Path] = None
    anomalies: List[Dict] = field(default_factory=list)
    error_message: Optional[str] = None
    step_artifacts: Dict[str, Path] = field(default_factory=dict)


class ReverseEngineeringUseCase:
    """Orchestrates execution of the Reverse Engineering (RE) pipeline."""

    def execute(self, request: ReverseEngineeringRequest) -> ReverseEngineeringResponse:
        """Execute the RE pipeline request."""
        logger.info("=" * 60)
        logger.info(f"STARTING RE PIPELINE FOR TASK: {request.task_name}")
        logger.info("=" * 60)

        # Wrap teacher_lm with DSPyLMAdapter for DSPy integration
        teacher_lm_adapter = DSPyLMAdapter(request.teacher_lm)

        # 2. Build RunContext
        context = RunContext(
            task_name=request.task_name,
            doc_ids=request.doc_ids,
            gt_path=request.gt_path,
            ingestion_dir=request.ingestion_dir,
            baseline_prompt_path=request.baseline_prompt_path,
            schema_path=request.schema_path,
            teacher_lm=request.teacher_lm,
            artifacts_dir=request.output_dir,
            resume=request.resume,
        )

        # 3. Determine steps to run
        all_steps = ["positive", "consolidation", "negative", "generalization"]
        steps_to_run = request.steps if request.steps else all_steps

        # Validate steps
        invalid_steps = [s for s in steps_to_run if s not in all_steps]
        if invalid_steps:
            return ReverseEngineeringResponse(
                success=False,
                error_message=f"Invalid steps specified: {invalid_steps}. Allowed steps: {all_steps}"
            )

        step_artifacts = {}
        anomalies = []

        try:
            with dspy.context(
                lm=teacher_lm_adapter,
                provide_traceback=True,
                num_threads=1,
                async_max_workers=1,
            ):
                # Step: positive
                if "positive" in steps_to_run:
                    logger.info("Running Step: positive")
                    run_positive_analysis(context)
                    step_artifacts["positive"] = context.artifacts.get_phase_dir("positive")

                # Step: consolidation
                if "consolidation" in steps_to_run:
                    logger.info("Running Step: consolidation")
                    run_consolidation(context)
                    step_artifacts["consolidation"] = context.artifacts.get_phase_dir("consolidation")

                # Step: negative
                if "negative" in steps_to_run:
                    logger.info("Running Step: negative")
                    run_negative_analysis(context)
                    step_artifacts["negative"] = context.artifacts.get_phase_dir("negative")

                # Step: generalization
                if "generalization" in steps_to_run:
                    logger.info("Running Step: generalization")
                    run_generalization(context)
                    step_artifacts["generalization"] = context.artifacts.get_phase_dir("generalization")

                # Load any anomalies produced during generalization step (if run or available)
                anomalies = self._collect_anomalies(context)

                # Locate generated prompt if generalization step was run
                from ae.core.config.settings import Settings
                settings = Settings.load()
                generated_prompt_path = Path(settings.paths.generated_prompt_file).resolve()
                prompt_path = generated_prompt_path if generated_prompt_path.exists() else None

                return ReverseEngineeringResponse(
                    success=True,
                    generated_prompt_path=prompt_path,
                    anomalies=anomalies,
                    step_artifacts=step_artifacts,
                )

        except Exception as e:
            logger.error(f"RE pipeline execution failed: {e}", exc_info=True)
            return ReverseEngineeringResponse(
                success=False,
                error_message=str(e),
                anomalies=anomalies,
                step_artifacts=step_artifacts,
            )

    def _collect_anomalies(self, context: RunContext) -> List[Dict]:
        """Collect anomalies from the generalization rules artifacts."""
        anomalies = []
        gen_dir = context.artifacts.get_phase_dir("generalization")
        if not gen_dir.exists():
            return anomalies

        # Row generalization anomalies
        row_rules_path = gen_dir / "rows.json"
        if row_rules_path.exists():
            try:
                data = json.loads(row_rules_path.read_text(encoding="utf-8"))
                for a in data.get("Anomalies", []):
                    a["scope"] = "rows"
                    anomalies.append(a)
            except Exception as e:
                logger.warning(f"Failed to read anomalies from row rules: {e}")

        # Column generalization anomalies
        col_dir = gen_dir / "columns"
        if col_dir.exists():
            for file_path in col_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    field_name = file_path.stem
                    for a in data.get("Anomalies", []):
                        a["scope"] = f"column:{field_name}"
                        anomalies.append(a)
                except Exception as e:
                    logger.warning(f"Failed to read anomalies from {file_path}: {e}")

        return anomalies
