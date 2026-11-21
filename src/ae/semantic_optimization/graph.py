"""SemanticOptimizationGraph: Iterative LangGraph for Semantic Prompt Engineering."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from ae.core.config.settings import Settings
from ae.core.storage.documents import DocumentRepository
from ae.core.utils import normalize_document_key
from ae.semantic_optimization.context import RunContext
from ae.semantic_optimization.hitl import _get_id_column
from ae.semantic_optimization.steps.generalization import run_generalization
from ae.semantic_optimization.steps.negative_analysis import run_negative_analysis
from ae.semantic_optimization.steps.positive_analysis import run_positive_analysis
from ae.semantic_optimization.steps.residual_analysis import run_residual_analysis

logger = logging.getLogger(__name__)


class SemanticOptimizationState(TypedDict, total=False):
    """State for iterative Reverse Engineering LangGraph."""

    current_iteration: int  # 1-based index (k)
    train_doc_ids: list[str]  # Ordered training doc IDs
    val_doc_ids: list[str]  # Validation doc IDs
    active_doc_ids: list[str]  # Active slice of training doc IDs
    new_doc_ids: list[str]  # Newly added doc IDs in the current iteration batch
    current_doc_id: str  # Last doc ID in current batch

    context: Any  # RunContext
    f1_threshold: float
    max_tool_iterations: int

    # Hybrid Anchor + Residual tracking
    iteration_mode: str  # "anchor" | "residual"
    fallback_doc_ids: list[str]
    changed_scopes: Optional[list[str]]
    zero_delta_iteration: bool

    # Metrics
    val_f1_history: list[float]
    best_f1: float
    best_prompt: str
    best_iteration: int
    early_stopped: bool
    step_artifacts: dict[str, Path]
    _gt_modified_in_generalization: bool


class SemanticOptimizationGraph:
    """Iterative LangGraph for Reverse Engineering with Human-in-the-Loop."""

    def __init__(
        self,
        context: RunContext,
        train_doc_ids: list[str],
        val_doc_ids: Optional[list[str]] = None,
        f1_threshold: float = 0.95,
        max_tool_iterations: int = 5,
        initial_docs: Optional[int] = None,
        step_size: Optional[int] = None,
    ) -> None:
        self.context = context
        self.train_doc_ids = train_doc_ids
        self.val_doc_ids = val_doc_ids or []
        self.f1_threshold = f1_threshold
        self.max_tool_iterations = max_tool_iterations

        ctx_init = getattr(context, "initial_docs", None)
        ctx_step = getattr(context, "step_size", None)
        self.initial_docs = (
            initial_docs
            if initial_docs is not None
            else (ctx_init if isinstance(ctx_init, int) and ctx_init >= 1 else 1)
        )
        self.step_size = (
            step_size
            if step_size is not None
            else (ctx_step if isinstance(ctx_step, int) and ctx_step >= 1 else 1)
        )
        if not isinstance(getattr(self.context, "max_tool_iterations", None), int):
            self.context.max_tool_iterations = self.max_tool_iterations

        self.app = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(SemanticOptimizationState)

        workflow.add_node("init_iteration", self._init_iteration)
        workflow.add_node("residual_analysis", self._residual_analysis)
        workflow.add_node("positive_analysis", self._positive_analysis)
        workflow.add_node("negative_analysis", self._negative_analysis)
        workflow.add_node("generalization", self._generalization)
        workflow.add_node("evaluate_val", self._evaluate_val)
        workflow.add_node("next_iteration", self._next_iteration)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "init_iteration")
        workflow.add_conditional_edges(
            "init_iteration",
            self._route_after_init,
            {
                "positive_analysis": "positive_analysis",
                "residual_analysis": "residual_analysis",
            },
        )
        workflow.add_conditional_edges(
            "residual_analysis",
            self._route_after_residual,
            {
                "fallback_positive": "positive_analysis",
                "generalization": "generalization",
                "zero_delta": "evaluate_val",
            },
        )
        workflow.add_edge("positive_analysis", "negative_analysis")
        workflow.add_edge("negative_analysis", "generalization")

        workflow.add_conditional_edges(
            "generalization",
            self._route_after_generalization,
            {
                "recompute_positive": "positive_analysis",
                "recompute_residual": "residual_analysis",
                "proceed_to_evaluate": "evaluate_val",
            },
        )

        workflow.add_conditional_edges(
            "evaluate_val",
            self._should_continue,
            {
                "next_iteration": "next_iteration",
                "finalize": "finalize",
                "early_stop": "finalize",
            },
        )

        workflow.add_edge("next_iteration", "init_iteration")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _resolve_prompt_path(self, ctx: RunContext) -> Path:
        if getattr(ctx, "generated_prompt_path", None) is not None:
            return Path(ctx.generated_prompt_path).resolve()
        raw_cfg = getattr(ctx, "config_path", None)
        cfg_path = raw_cfg if isinstance(raw_cfg, (str, Path)) else None
        settings = Settings.load(config_path=cfg_path, load_env_file=False)
        return Path(settings.paths.generated_prompt_file).resolve()

    def _route_after_init(self, state: SemanticOptimizationState) -> str:
        """Select Anchor full analysis (Iter 1) or Error-Driven Residual (Iter >= 2)."""
        ctx: RunContext = state["context"]
        k = state["current_iteration"]
        if (
            k >= 2
            and getattr(ctx, "residual_enabled", True) is not False
            and bool(state.get("new_doc_ids"))
        ):
            prompt_path = self._resolve_prompt_path(ctx)
            if prompt_path.exists() and prompt_path.read_text(encoding="utf-8").strip():
                return "residual_analysis"
        return "positive_analysis"

    def _init_iteration(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Initialize iteration k with configurable batching (initial_docs + step_size)."""
        k = state["current_iteration"]
        train_ids = state["train_doc_ids"]
        total_docs = len(train_ids)

        if total_docs == 0:
            active_ids: list[str] = []
            new_ids: list[str] = []
            current_doc = ""
        elif k == 1:
            end_idx = min(total_docs, self.initial_docs)
            active_ids = train_ids[:end_idx]
            new_ids = active_ids
            current_doc = active_ids[-1]
        else:
            prev_end = min(total_docs, self.initial_docs + (k - 2) * self.step_size)
            end_idx = min(total_docs, prev_end + self.step_size)
            active_ids = train_ids[:end_idx]
            new_ids = train_ids[prev_end:end_idx]
            current_doc = new_ids[-1] if new_ids else active_ids[-1]

        logger.info(
            f"=== Starting Semantic Prompt Optimization Iteration {k} | "
            f"New Batch: {new_ids} | Active Set: {len(active_ids)}/{total_docs} docs ==="
        )

        ctx: RunContext = state["context"]
        ctx.current_iteration = k

        return {
            "current_doc_id": current_doc,
            "new_doc_ids": new_ids,
            "active_doc_ids": active_ids,
            "iteration_mode": "anchor" if k == 1 else "residual",
            "fallback_doc_ids": [],
            "changed_scopes": None,
            "zero_delta_iteration": False,
        }

    def _residual_analysis(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Execute Error-Driven / Residual analysis for new_doc_ids or recompute_doc_ids in Iteration >= 2."""
        ctx: RunContext = state["context"]
        recompute_ids = getattr(ctx, "_recompute_doc_ids", None)
        raw_targets = recompute_ids if recompute_ids else (state.get("new_doc_ids") or [])
        target_ids = [d for d in raw_targets if d]
        if recompute_ids is not None:
            setattr(ctx, "_recompute_doc_ids", None)
        ctx.doc_ids = target_ids

        logger.info(
            f"Iteration {state['current_iteration']}: Running Error-Driven Residual Analysis for docs {ctx.doc_ids}"
        )
        summary = run_residual_analysis(ctx)

        step_artifacts = dict(state.get("step_artifacts", {}))
        step_artifacts["residual"] = ctx.artifacts.get_phase_dir("positive", "residual")

        if summary.fallback_doc_ids:
            return {
                "step_artifacts": step_artifacts,
                "fallback_doc_ids": list(summary.fallback_doc_ids),
                "changed_scopes": None,
                "zero_delta_iteration": False,
            }

        if summary.is_zero_delta:
            logger.info(
                f"Iteration {state['current_iteration']}: Zero-Delta Fast-Forward — skipping rule synthesis and validation."
            )
            return {
                "step_artifacts": step_artifacts,
                "fallback_doc_ids": [],
                "changed_scopes": [],
                "zero_delta_iteration": True,
            }

        return {
            "step_artifacts": step_artifacts,
            "fallback_doc_ids": [],
            "changed_scopes": sorted(summary.affected_scopes),
            "zero_delta_iteration": False,
        }

    def _route_after_residual(self, state: dict[str, Any]) -> str:
        if state.get("fallback_doc_ids"):
            return "fallback_positive"
        if state.get("zero_delta_iteration"):
            return "zero_delta"
        return "generalization"

    def _positive_analysis(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Execute Step 1A: Positive Analysis for new_doc_ids, fallback_doc_ids, or recompute_doc_ids."""
        ctx: RunContext = state["context"]
        recompute_ids = getattr(ctx, "_recompute_doc_ids", None)
        fallback_ids = state.get("fallback_doc_ids") or []
        target_ids = (
            recompute_ids
            if recompute_ids
            else (fallback_ids if fallback_ids else (state.get("new_doc_ids") or [state["current_doc_id"]]))
        )
        ctx.doc_ids = [d for d in target_ids if d]

        logger.info(
            f"Iteration {state['current_iteration']}: Running 1A Positive Analysis for docs {ctx.doc_ids}"
        )
        run_positive_analysis(ctx)

        step_artifacts = dict(state.get("step_artifacts", {}))
        step_artifacts["positive"] = ctx.artifacts.get_phase_dir("positive")
        return {"step_artifacts": step_artifacts}

    def _negative_analysis(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Execute Step 1B: Negative Analysis immediately after Positive Analysis for the same docs."""
        ctx: RunContext = state["context"]
        recompute_ids = getattr(ctx, "_recompute_doc_ids", None)
        fallback_ids = state.get("fallback_doc_ids") or []
        target_ids = (
            recompute_ids
            if recompute_ids
            else (fallback_ids if fallback_ids else (state.get("new_doc_ids") or [state["current_doc_id"]]))
        )
        ctx.doc_ids = [d for d in target_ids if d]

        if recompute_ids is not None:
            setattr(ctx, "_recompute_doc_ids", None)

        logger.info(
            f"Iteration {state['current_iteration']}: Running 1B Negative Analysis for docs {ctx.doc_ids}"
        )
        run_negative_analysis(ctx)

        step_artifacts = dict(state.get("step_artifacts", {}))
        step_artifacts["negative"] = ctx.artifacts.get_phase_dir("negative")
        return {"step_artifacts": step_artifacts}

    def _generalization(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Execute Unified Rule Synthesis (Consolidation + Generalization), HITL Triage & Prompt Harmonization."""
        active_ids = state["active_doc_ids"]
        ctx: RunContext = state["context"]
        ctx.doc_ids = active_ids

        logger.info(
            f"Iter {state['current_iteration']}: Unified Rule Synthesis over {len(active_ids)} active docs"
        )
        raw_changed = state.get("changed_scopes")
        if (
            raw_changed is not None
            and "changed_scopes" in inspect.signature(run_generalization).parameters
        ):
            gt_modified = run_generalization(ctx, changed_scopes=set(raw_changed))
        else:
            gt_modified = run_generalization(ctx)

        generated_path = self._resolve_prompt_path(ctx)
        candidate_prompt = ""
        if generated_path.exists():
            candidate_prompt = generated_path.read_text(encoding="utf-8")

        step_artifacts = dict(state.get("step_artifacts", {}))
        step_artifacts["consolidation"] = ctx.artifacts.get_phase_dir(
            "consolidation", iteration=state["current_iteration"]
        )
        step_artifacts["generalization"] = ctx.artifacts.get_phase_dir(
            "generalization", iteration=state["current_iteration"]
        )

        out: dict[str, Any] = {
            "step_artifacts": step_artifacts,
            "best_prompt": candidate_prompt if not state["best_prompt"] else state["best_prompt"],
            "_gt_modified_in_generalization": gt_modified,
        }
        if gt_modified:
            out["changed_scopes"] = None
            out["fallback_doc_ids"] = []
        return out

    def _route_after_generalization(self, state: dict[str, Any]) -> str:
        if state.get("_gt_modified_in_generalization"):
            ctx: Optional[RunContext] = state.get("context")
            if (
                ctx is not None
                and state.get("iteration_mode") == "residual"
                and getattr(ctx, "residual_enabled", True) is not False
            ):
                prompt_path = self._resolve_prompt_path(ctx)
                if prompt_path.exists() and prompt_path.read_text(encoding="utf-8").strip():
                    logger.info(
                        "Document .md or GT rows modified during HITL Triage in residual mode. "
                        "Recomputing Residual Analysis..."
                    )
                    return "recompute_residual"
            logger.info("Document .md or GT rows modified during HITL Triage. Recomputing Document Analysis...")
            return "recompute_positive"
        return "proceed_to_evaluate"

    def _evaluate_val(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Evaluate candidate prompt on validation split."""
        if state.get("zero_delta_iteration"):
            val_history = state.get("val_f1_history", [])
            prev_f1 = val_history[-1] if val_history else float(state.get("best_f1", 0.0))
            logger.info(
                f"Iter {state['current_iteration']}: Zero-Delta Fast-Forward — "
                f"reusing previous validation F1 ({prev_f1:.4f}) without re-running extraction."
            )
            return {
                "val_f1_history": val_history + [prev_f1],
                "best_f1": state.get("best_f1", prev_f1),
                "best_prompt": state.get("best_prompt", ""),
                "best_iteration": state.get("best_iteration", 1),
            }

        ctx: RunContext = state["context"]
        raw_cfg = getattr(ctx, "config_path", None)
        cfg_path = raw_cfg if isinstance(raw_cfg, (str, Path)) else None
        generated_path = self._resolve_prompt_path(ctx)
        current_prompt = (
            generated_path.read_text(encoding="utf-8") if generated_path.exists() else ""
        )

        val_ids = state["val_doc_ids"]
        if not val_ids:
            logger.info("No validation document IDs provided. Skipping validation evaluation.")
            return {
                "val_f1_history": state["val_f1_history"] + [0.0],
                "best_prompt": current_prompt or state["best_prompt"],
                "best_iteration": state["current_iteration"],
            }

        f1_score = 0.0
        try:
            from ae.core.evaluation import TaskMetric
            from ae.core.schema import load_schema_complete

            eval_schema_path = ctx.schema_path
            gen_schema_candidate = getattr(ctx, "generated_schema_path", None)
            if gen_schema_candidate is not None and Path(gen_schema_candidate).exists():
                eval_schema_path = Path(gen_schema_candidate)

            enable_judge = getattr(ctx, "enable_semantic_judge", True) is not False
            selection_metric = getattr(ctx, "selection_metric", "micro_f1")
            if selection_metric not in ("micro_f1", "macro_f1"):
                selection_metric = "micro_f1"
            schema_text_for_key = (
                Path(eval_schema_path).read_text(encoding="utf-8")
                if Path(eval_schema_path).exists()
                else ""
            )
            doc_repo = DocumentRepository(ingestion_dir=ctx.ingestion_dir)
            gt_hash = (
                hashlib.sha256(Path(ctx.active_gt_path).read_bytes()).hexdigest()
                if Path(ctx.active_gt_path).exists()
                else ""
            )
            val_md_hashes = {
                normalize_document_key(v_id): hashlib.sha256(
                    (doc_repo.get_canonical_manifest_text(normalize_document_key(v_id)) or "").encode("utf-8")
                ).hexdigest()
                for v_id in sorted(val_ids)
            }
            ext_model_id = ""
            if hasattr(ctx, "re_client") and hasattr(ctx.re_client, "extraction_config"):
                ext_cfg = ctx.re_client.extraction_config
                if ext_cfg is not None:
                    ext_model_id = str(getattr(ext_cfg, "model", ""))

            val_cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "prompt": current_prompt,
                        "schema": schema_text_for_key,
                        "judge": enable_judge,
                        "selection_metric": selection_metric,
                        "val_ids": sorted(val_ids),
                        "gt_hash": gt_hash,
                        "val_md_hashes": val_md_hashes,
                        "ext_model": ext_model_id,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

            val_cache_path: Optional[Path] = None
            artifacts_mgr = getattr(ctx, "artifacts", None)
            if artifacts_mgr is not None and hasattr(artifacts_mgr, "get_path"):
                cand_cache_path = artifacts_mgr.get_path(
                    "", "", "val_metrics.json", iteration=state["current_iteration"]
                )
                if isinstance(cand_cache_path, Path):
                    val_cache_path = cand_cache_path

            used_cache = False
            if (
                getattr(ctx, "resume", False) is True
                and val_cache_path is not None
                and val_cache_path.exists()
            ):
                try:
                    cached_val = json.loads(val_cache_path.read_text(encoding="utf-8"))
                    if cached_val.get("cache_key") == val_cache_key:
                        f1_score = float(cached_val["f1_score"])
                        used_cache = True
                        logger.info(
                            f"Iter {state['current_iteration']}: Loaded cached validation F1={f1_score:.4f} from {val_cache_path}"
                        )
                except Exception as cache_exc:
                    logger.debug(f"Could not read cached validation metrics {val_cache_path}: {cache_exc}")

            if not used_cache:
                from ae.core.schema.loader import create_row_converter, load_schema_with_models

                extract_task = load_schema_complete(
                    yaml_path=eval_schema_path,
                    instruction_path=generated_path if generated_path.exists() else None,
                )
                if Path(eval_schema_path).resolve() != Path(ctx.schema_path).resolve():
                    gt_cfg, gt_rec_model, gt_output_model = load_schema_with_models(
                        yaml_path=ctx.schema_path,
                        instruction_path=None,
                    )
                    gt_row_converter = create_row_converter(gt_cfg, gt_rec_model)
                else:
                    gt_output_model = extract_task.output_model
                    gt_row_converter = getattr(extract_task, "row_converter", None)

                metric_config = getattr(extract_task.config, "metric_config", {})
                if not metric_config and hasattr(extract_task.config, "model_dump"):
                    metric_config = extract_task.config.model_dump()

                if "compare_fields" not in metric_config:
                    metric_config["compare_fields"] = list(extract_task.config.compare_fields)
                if not metric_config.get("name"):
                    metric_config["name"] = (
                        getattr(extract_task.config, "name", None)
                        or getattr(ctx, "task_name", None)
                        or "unknown"
                    )

                descriptions = extract_task.config.judge_field_descriptions or {
                    name: spec.description for name, spec in extract_task.config.fields.items()
                }

                settings = Settings.load(config_path=cfg_path, load_env_file=False)
                flt_tol = getattr(
                    settings.syntactic_optimization,
                    "float_tolerance",
                    extract_task.config.float_tolerance,
                )

                student_llm = None
                if (
                    enable_judge
                    and hasattr(ctx, "re_client")
                    and hasattr(ctx.re_client, "get_validation_client")
                ):
                    try:
                        val_client = ctx.re_client.get_validation_client()
                        if val_client is not None and hasattr(val_client, "get_dspy_adapter"):
                            student_llm = val_client.get_dspy_adapter()
                    except Exception as val_exc:
                        logger.warning(
                            f"Could not resolve validation student_llm for Semantic Judge: {val_exc}"
                        )

                if enable_judge and student_llm is not None:
                    logger.info(
                        f"Validation evaluation: Semantic Judge ENABLED "
                        f"(model={getattr(student_llm, 'model', type(student_llm).__name__)})"
                    )
                elif enable_judge:
                    logger.warning(
                        "Validation evaluation: enable_semantic_judge=True, but student_llm could not be resolved; "
                        "falling back to strict matching."
                    )
                else:
                    logger.info("Validation evaluation: Semantic Judge DISABLED (strict matching only).")

                raw_constraints = getattr(extract_task.config, "field_constraints", None)
                field_constraints = raw_constraints if isinstance(raw_constraints, dict) else None
                raw_validators = getattr(extract_task.config, "field_validators", None)
                field_validators = raw_validators if isinstance(raw_validators, dict) else None

                metric = TaskMetric(
                    task_config=metric_config,
                    float_tolerance=flt_tol,
                    student_llm=student_llm,
                    field_descriptions=descriptions,
                    enable_semantic_judge=enable_judge,
                    field_constraints=field_constraints,
                    field_validators=field_validators,
                )

                # Evaluate over val_ids using active_gt_path
                df_gt = pd.read_csv(ctx.active_gt_path)
                df_gt.columns = df_gt.columns.str.lower().str.strip()
                id_col = _get_id_column(df_gt)

                from ae.extraction.agent import UniversalExtractor

                if hasattr(ctx, "re_client") and hasattr(ctx.re_client, "create_validation_extractor"):
                    extractor = ctx.re_client.create_validation_extractor(extract_task.signature)
                else:
                    extractor = UniversalExtractor(extract_task.signature)

                scores = []
                doc_scores_detail: dict[str, dict[str, Any]] = {}
                total_weight = 0
                weighted_f1_sum = 0.0
                import dspy

                for v_id in val_ids:
                    norm_v_id = normalize_document_key(v_id)
                    v_text = doc_repo.get(norm_v_id)
                    if not v_text:
                        continue
                    # Ground truth records
                    gt_rows = df_gt[
                        df_gt[id_col].astype(str).apply(normalize_document_key) == norm_v_id
                    ]
                    gt_records = gt_rows.to_dict(orient="records")
                    if not gt_records:
                        continue

                    clean_gt_records = []
                    for rec in gt_records:
                        clean_rec = {
                            k: (None if pd.isna(v) else v)
                            for k, v in rec.items()
                        }
                        if callable(gt_row_converter):
                            converted = gt_row_converter(clean_rec)
                            if converted is not None:
                                clean_gt_records.append(converted)
                        else:
                            clean_gt_records.append(clean_rec)

                    doc_header = f"<!-- document_id: {norm_v_id} -->\n\n"
                    eval_text = v_text if norm_v_id in v_text else doc_header + v_text
                    pred = extractor.forward(document_text=eval_text)

                    example = dspy.Example(
                        extracted_data=gt_output_model(records=clean_gt_records)
                    )
                    score = float(metric(example, pred))
                    doc_scores_detail[norm_v_id] = {"f1": score, "gts": len(clean_gt_records)}
                    row_weight = max(1, len(clean_gt_records))

                    scores.append(score)
                    weighted_f1_sum += score * row_weight
                    total_weight += row_weight

                macro_f1 = (sum(scores) / len(scores)) if scores else 0.0
                micro_f1 = (weighted_f1_sum / total_weight) if total_weight > 0 else macro_f1
                f1_score = micro_f1 if selection_metric == "micro_f1" else macro_f1

                if val_cache_path is not None and scores:
                    try:
                        val_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        val_cache_path.write_text(
                            json.dumps(
                                {
                                    "iteration": state["current_iteration"],
                                    "cache_key": val_cache_key,
                                    "selection_metric": selection_metric,
                                    "f1_score": f1_score,
                                    "macro_f1": macro_f1,
                                    "micro_f1": micro_f1,
                                    "doc_scores": doc_scores_detail,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except Exception as write_exc:
                        logger.debug(f"Could not write validation cache {val_cache_path}: {write_exc}")

        except Exception as e:
            logger.warning(f"Error during validation evaluation: {e}")
            f1_score = 0.0

        new_history = state["val_f1_history"] + [f1_score]
        best_f1 = state["best_f1"]
        best_prompt = state["best_prompt"]
        best_iteration = state.get("best_iteration", 1)

        if f1_score > best_f1 or (
            f1_score == best_f1
            and current_prompt
            and (not best_prompt or state["current_iteration"] > best_iteration)
        ):
            logger.info(
                f"Updating best validation F1 score: {f1_score:.4f} (previous: {best_f1:.4f}) "
                f"at iteration {state['current_iteration']}"
            )
            best_f1 = f1_score
            best_prompt = current_prompt
            best_iteration = state["current_iteration"]

        return {
            "val_f1_history": new_history,
            "best_f1": best_f1,
            "best_prompt": best_prompt,
            "best_iteration": best_iteration,
        }

    def _should_continue(self, state: SemanticOptimizationState) -> str:
        """Route to early stop, next iteration, or finalize."""
        val_history = state.get("val_f1_history", [])
        current_f1 = val_history[-1] if val_history else 0.0

        if current_f1 >= state["f1_threshold"]:
            logger.info(
                f"Early stopping threshold reached: F1 {current_f1:.4f} >= {state['f1_threshold']}"
            )
            return "early_stop"

        if len(state.get("active_doc_ids", [])) < len(state["train_doc_ids"]):
            return "next_iteration"

        logger.info("All training documents processed. Finalizing.")
        return "finalize"

    def _next_iteration(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Advance iteration counter."""
        return {
            "current_iteration": state["current_iteration"] + 1,
        }

    def _finalize(self, state: SemanticOptimizationState) -> dict[str, Any]:
        """Save final prompt, best iteration artifacts, and metrics summary."""
        val_history = state.get("val_f1_history", [])
        current_f1 = val_history[-1] if val_history else 0.0
        early_stopped = current_f1 >= state["f1_threshold"]

        comp_dir = state["context"].artifacts.get_phase_dir("compilation")
        comp_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = comp_dir / "iteration_metrics.json"

        best_k = state.get("best_iteration", 1)
        summary_data = {
            "total_iterations": state["current_iteration"],
            "best_iteration": best_k,
            "early_stopped": early_stopped,
            "best_f1": state["best_f1"],
            "f1_history": state["val_f1_history"],
            "train_doc_ids": state["train_doc_ids"],
            "val_doc_ids": state["val_doc_ids"],
            "corrected_gt_file": str(state["context"].active_gt_path),
            "expert_rulings_count": len(state["context"].rulings_store.rulings),
        }
        metrics_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

        # Copy best iteration artifacts to root consolidation and generalization directories
        ctx: RunContext = state["context"]
        best_cons = ctx.artifacts.get_phase_dir("consolidation", iteration=best_k)
        root_cons = ctx.artifacts.get_phase_dir("consolidation", iteration=None)
        if best_cons.exists() and best_cons != root_cons:
            shutil.copytree(best_cons, root_cons, dirs_exist_ok=True)

        best_gen = ctx.artifacts.get_phase_dir("generalization", iteration=best_k)
        root_gen = ctx.artifacts.get_phase_dir("generalization", iteration=None)
        if best_gen.exists() and best_gen != root_gen:
            shutil.copytree(best_gen, root_gen, dirs_exist_ok=True)

        raw_cfg = getattr(ctx, "config_path", None)
        cfg_path = raw_cfg if isinstance(raw_cfg, (str, Path)) else None
        best_schema_candidate = best_gen / "candidate_schema.yaml"
        if best_schema_candidate.exists():
            if getattr(ctx, "generated_schema_path", None) is not None:
                schema_out = Path(ctx.generated_schema_path).resolve()
            else:
                settings = Settings.load(config_path=cfg_path, load_env_file=False)
                schema_out = Path(settings.paths.generated_schema_file).resolve()
            schema_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_schema_candidate, schema_out)
            logger.info(f"Final best schema from iteration {best_k} saved to {schema_out}")

        # Save best prompt
        if state["best_prompt"]:
            if getattr(ctx, "generated_prompt_path", None) is not None:
                prompt_file = Path(ctx.generated_prompt_path).resolve()
            else:
                settings = Settings.load(config_path=cfg_path, load_env_file=False)
                prompt_file = Path(settings.paths.generated_prompt_file).resolve()
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(state["best_prompt"], encoding="utf-8")
            logger.info(f"Final best prompt from iteration {best_k} saved to {prompt_file}")

        return {
            "early_stopped": early_stopped,
        }

    def run(self) -> SemanticOptimizationState:
        """Run the iterative RE graph to completion."""
        initial_state: SemanticOptimizationState = {
            "current_iteration": 1,
            "train_doc_ids": self.train_doc_ids,
            "val_doc_ids": self.val_doc_ids,
            "active_doc_ids": [],
            "new_doc_ids": [],
            "current_doc_id": "",
            "iteration_mode": "anchor",
            "fallback_doc_ids": [],
            "changed_scopes": None,
            "zero_delta_iteration": False,
            "context": self.context,
            "f1_threshold": self.f1_threshold,
            "max_tool_iterations": self.max_tool_iterations,
            "val_f1_history": [],
            "best_f1": 0.0,
            "best_prompt": "",
            "best_iteration": 1,
            "early_stopped": False,
            "step_artifacts": {},
        }

        final_state: SemanticOptimizationState = self.app.invoke(initial_state)
        return final_state
