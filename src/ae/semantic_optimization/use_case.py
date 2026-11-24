"""RE UseCase orchestrating the Reverse Engineering pipeline execution."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import dspy

from ae.core.llm import LMProvider, DSPyLMAdapter
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.steps import (
    run_positive_analysis,
    run_consolidation,
    run_negative_analysis,
    run_generalization,
)

logger = logging.getLogger(__name__)


@dataclass
class SemanticOptimizationRequest:
    task_name: str
    doc_ids: List[str]           # list of document IDs from train split
    gt_path: Path                 # path to Ground Truth CSV
    ingestion_dir: Path           # directory with MD files
    baseline_prompt_path: Path    # path to baseline prompt TXT
    schema_path: Path             # path to task schema YAML
    teacher_lm: Optional[Any] = None  # Legacy Teacher LM (optional if re_client provided)
    output_dir: Path = Path("data/interim/semantic_optimization")  # base dir
    re_client: Optional[Any] = None   # Dedicated SemanticOptimizationClient
    val_doc_ids: Optional[List[str]] = None
    f1_threshold: float = 0.95
    max_tool_iterations: int = 5
    initial_docs: int = 2
    step_size: int = 2
    row_batch_size: int = 5
    max_rows_per_doc: int = 20
    harmonize_prompt: bool = True
    resume: bool = False          # resume from cache
    interactive: bool = True      # interactive human-in-the-loop resolution
    corrected_gt_path: Optional[Path] = None  # path for saving corrected ground truth copy
    steps: Optional[List[str]] = None  # specific steps to run (e.g. ['positive', 'consolidation'])
    generated_prompt_path: Optional[Path] = None
    generated_schema_path: Optional[Path] = None
    infer_schema_constraints: bool = True
    min_enum_doc_support: int = 4
    pareto_regression_threshold: float = 0.01
    enable_semantic_judge: bool = True
    residual_enabled: bool = True
    residual_fallback_threshold: float = 0.35
    residual_skip_f1: float = 0.98
    residual_max_examples_per_sig: int = 3
    residual_max_fn_rows: int = 8
    residual_max_fp_rows: int = 10
    pdf_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    tier_4a_lm: Optional[Any] = None


@dataclass
class SemanticOptimizationResponse:
    success: bool
    generated_prompt_path: Optional[Path] = None
    generated_schema_path: Optional[Path] = None
    corrected_gt_path: Optional[Path] = None
    anomalies: List[Dict] = field(default_factory=list)
    expert_rulings_count: int = 0
    error_message: Optional[str] = None
    step_artifacts: Dict[str, Path] = field(default_factory=dict)


class SemanticOptimizationUseCase:
    """Orchestrates execution of the Reverse Engineering (RE) pipeline."""

    def execute(self, request: SemanticOptimizationRequest) -> SemanticOptimizationResponse:
        """Execute the RE pipeline request."""
        logger.info("=" * 60)
        logger.info(f"STARTING SEMANTIC PROMPT OPTIMIZATION FOR TASK: {request.task_name}")
        logger.info("=" * 60)

        client = request.re_client if request.re_client is not None else request.teacher_lm
        if client is None:
            raise ValueError("Either re_client or teacher_lm must be specified in SemanticOptimizationRequest")

        if hasattr(client, "get_dspy_adapter"):
            teacher_lm_adapter = client.get_dspy_adapter()
        else:
            teacher_lm_adapter = DSPyLMAdapter(client)

        # 2. Build RunContext
        context = RunContext(
            task_name=request.task_name,
            doc_ids=request.doc_ids,
            gt_path=request.gt_path,
            ingestion_dir=request.ingestion_dir,
            baseline_prompt_path=request.baseline_prompt_path,
            schema_path=request.schema_path,
            re_client=client,
            artifacts_dir=request.output_dir,
            resume=request.resume,
            interactive=request.interactive,
            corrected_gt_path=request.corrected_gt_path,
            pdf_dir=request.pdf_dir,
            initial_docs=request.initial_docs,
            step_size=request.step_size,
            row_batch_size=request.row_batch_size,
            max_rows_per_doc=request.max_rows_per_doc,
            harmonize_prompt=request.harmonize_prompt,
            max_tool_iterations=request.max_tool_iterations,
            generated_prompt_path=request.generated_prompt_path,
            generated_schema_path=request.generated_schema_path,
            infer_schema_constraints=request.infer_schema_constraints,
            min_enum_doc_support=request.min_enum_doc_support,
            pareto_regression_threshold=request.pareto_regression_threshold,
            enable_semantic_judge=request.enable_semantic_judge,
            residual_enabled=request.residual_enabled,
            residual_fallback_threshold=request.residual_fallback_threshold,
            residual_skip_f1=request.residual_skip_f1,
            residual_max_examples_per_sig=request.residual_max_examples_per_sig,
            residual_max_fn_rows=request.residual_max_fn_rows,
            residual_max_fp_rows=request.residual_max_fp_rows,
            config_path=request.config_path,
            tier_4a_lm=request.tier_4a_lm,
        )

        # 3. Determine steps to run
        all_steps = ["positive", "consolidation", "negative", "generalization"]
        steps_to_run = request.steps if request.steps else all_steps

        invalid_steps = [s for s in steps_to_run if s not in all_steps]
        if invalid_steps:
            return SemanticOptimizationResponse(
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
                if request.steps is not None:
                    if "positive" in steps_to_run:
                        logger.info("Running Step: positive")
                        run_positive_analysis(context)
                        step_artifacts["positive"] = context.artifacts.get_phase_dir("positive")

                    if "consolidation" in steps_to_run:
                        logger.info("Running Step: consolidation")
                        run_consolidation(context)
                        step_artifacts["consolidation"] = context.artifacts.get_phase_dir("consolidation")

                    if "negative" in steps_to_run:
                        logger.info("Running Step: negative")
                        run_negative_analysis(context)
                        step_artifacts["negative"] = context.artifacts.get_phase_dir("negative")

                    if "generalization" in steps_to_run:
                        logger.info("Running Step: generalization")
                        run_generalization(context)
                        step_artifacts["generalization"] = context.artifacts.get_phase_dir("generalization")
                else:
                    from ae.semantic_optimization.graph import SemanticOptimizationGraph

                    logger.info("Executing Iterative SemanticOptimizationGraph across training documents...")
                    graph = SemanticOptimizationGraph(
                        context=context,
                        train_doc_ids=request.doc_ids,
                        val_doc_ids=request.val_doc_ids,
                        f1_threshold=request.f1_threshold,
                        max_tool_iterations=request.max_tool_iterations,
                        pareto_regression_threshold=request.pareto_regression_threshold,
                    )
                    final_state = graph.run()
                    step_artifacts = final_state.get("step_artifacts", {})

                anomalies = self._collect_anomalies(context)

                prompt_path = context.generated_prompt_path if context.generated_prompt_path.exists() else None
                schema_out_path = context.generated_schema_path if context.generated_schema_path.exists() else None

                return SemanticOptimizationResponse(
                    success=True,
                    generated_prompt_path=prompt_path,
                    generated_schema_path=schema_out_path,
                    corrected_gt_path=context.active_gt_path,
                    anomalies=anomalies,
                    expert_rulings_count=len(context.rulings_store.rulings),
                    step_artifacts=step_artifacts,
                )

        except Exception as e:
            logger.error(f"Semantic prompt optimization failed: {e}", exc_info=True)
            return SemanticOptimizationResponse(
                success=False,
                error_message=str(e),
                anomalies=anomalies,
                step_artifacts=step_artifacts,
            )

    def _collect_anomalies(self, context: RunContext) -> List[Dict]:
        """Collect anomalies and conflicts from generalization artifacts."""
        anomalies = []
        gen_dir = context.artifacts.get_phase_dir("generalization", iteration=None)
        if not gen_dir.exists():
            return anomalies

        row_rules_path = gen_dir / "rows.json"
        if row_rules_path.exists():
            try:
                data = json.loads(row_rules_path.read_text(encoding="utf-8"))
                for a in data.get("Anomalies", []):
                    a["scope"] = "rows"
                    anomalies.append(a)
                for c in data.get("Conflicts", []):
                    c_dict = dict(c) if isinstance(c, dict) else (c.model_dump() if hasattr(c, "model_dump") else dict(c))
                    c_dict["scope"] = "rows"
                    if "anomaly_description" not in c_dict:
                        c_dict["anomaly_description"] = c_dict.get("explanation", f"Conflict: {c_dict.get('conflict_type')}")
                    if "source_reference" not in c_dict:
                        c_dict["source_reference"] = {
                            "document_id": c_dict.get("document_id", "unknown"),
                            "positive_reference_ids": [c_dict.get("row_id")] if c_dict.get("row_id") else [],
                            "negative_reference_ids": [],
                        }
                    anomalies.append(c_dict)
            except Exception as e:
                logger.warning(f"Failed to read anomalies from row rules: {e}")

        col_dir = gen_dir / "columns"
        if col_dir.exists():
            for file_path in col_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    field_name = file_path.stem
                    for a in data.get("Anomalies", []):
                        a["scope"] = f"column:{field_name}"
                        anomalies.append(a)
                    for c in data.get("Conflicts", []):
                        c_dict = dict(c) if isinstance(c, dict) else (c.model_dump() if hasattr(c, "model_dump") else dict(c))
                        c_dict["scope"] = f"column:{field_name}"
                        if "anomaly_description" not in c_dict:
                            c_dict["anomaly_description"] = c_dict.get("explanation", f"Conflict: {c_dict.get('conflict_type')}")
                        if "source_reference" not in c_dict:
                            c_dict["source_reference"] = {
                                "document_id": c_dict.get("document_id", "unknown"),
                                "positive_reference_ids": [c_dict.get("row_id")] if c_dict.get("row_id") else [],
                                "negative_reference_ids": [],
                            }
                        anomalies.append(c_dict)
                except Exception as e:
                    logger.warning(f"Failed to read anomalies from {file_path}: {e}")

        return anomalies
