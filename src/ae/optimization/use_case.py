"""Optimize agent use case.

This use case orchestrates the entire agent optimization workflow,
from dataset preparation through optimization to agent persistence.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import dspy

from ae.core.evaluation import TaskMetric
from ae.core.exceptions import UseCaseExecutionError
from ae.core.llm import LMProvider, DSPyLMAdapter
from ae.core.llm.history_logger import save_optimization_history
from ae.core.llm.provider import TeacherWrapper
from ae.core.storage import GroundTruthRepository, DataSplitRepository
from ae.core.schema import ExtractionBundle
from ae.extraction.agent import BaseAgent
from ae.extraction.manager import AgentManager
from ae.optimization.dataset_builder import DatasetBuilder
from ae.optimization.data_validator import DataValidator
from ae.optimization.tracking import ExperimentTracker
from ae.optimization.mipro import CheckpointingMIPROv2, GracefulExit

logger = logging.getLogger(__name__)


@dataclass
class OptimizeAgentRequest:
    """Request for agent optimization.

    Attributes:
        task: Task definition to optimize for (TaskBundle).
        signature_class: DSPy signature class for the task.
        gt_path: Path to ground truth CSV.
        split_path: Path to data splits JSON.
        train_split_name: Name of training split (default: "train").
        val_split_name: Name of validation split (default: "val").
        student_lm: Student language model for optimization (LMProvider).
        teacher_lm: Optional teacher model for demonstrations (LMProvider).
        num_trials: Number of optimization trials (required).
        train_limit: Optional limit on training examples.
        val_limit: Optional limit on validation examples.
        model_version: Version string for the model.
        description: Optional description for the agent.
        seed: Random seed for reproducibility (required).
        mlflow_experiment_name: Optional MLflow experiment name.
        run_name_prefix: Optional prefix for MLflow run name (e.g., "A1_high").
        num_candidates: Number of candidate prompts to generate (required).
        max_bootstrapped_demos: Maximum number of bootstrapped demonstrations (required).
        max_labeled_demos: Maximum number of labeled demonstrations (required).
        minibatch: Whether to use minibatch evaluation (required).
        minibatch_size: Size of minibatches for evaluation (required).
        view_data_batch_size: Number of data examples to view per batch (required).
        metric_threshold: Metric threshold for early stopping (required).
        init_temperature: Initial temperature for prompt generation (required).
        verbose: Whether to enable verbose logging (default: True).
        initial_instruction_file: Path to the initial instruction file (relative to config dir).
        instruction_hash: SHA256 hash (first 12 chars) of the initial instruction.
        max_errors: Maximum number of errors allowed before stopping optimization (required).
    """

    task: ExtractionBundle
    signature_class: Any
    gt_path: Path
    split_path: Path
    student_lm: LMProvider
    num_trials: int
    seed: int
    num_candidates: int
    max_bootstrapped_demos: int
    max_labeled_demos: int
    minibatch: bool
    minibatch_size: int
    view_data_batch_size: int
    metric_threshold: float
    init_temperature: float
    max_errors: int
    train_split_name: str = "train"
    val_split_name: str = "val"
    teacher_lm: Optional[LMProvider] = None
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    model_version: str = "unknown"
    description: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    run_name_prefix: Optional[str] = None
    verbose: bool = True
    initial_instruction_file: Optional[str] = None
    instruction_hash: Optional[str] = None
    cancel_event: Optional[Any] = None
    dry_run: bool = False


@dataclass
class OptimizeAgentResponse:
    """Response from agent optimization.

    Attributes:
        success: Whether optimization succeeded.
        agent_path: Path to saved agent (if successful).
        final_metrics: Final validation metrics.
        trial_count: Number of trials completed.
        error_message: Error message (if failed).
        optimization_config: Configuration used for optimization.
    """

    success: bool
    agent_path: Optional[Path] = None
    final_metrics: Optional[Dict[str, float]] = None
    trial_count: int = 0
    error_message: Optional[str] = None
    optimization_config: Optional[Dict[str, Any]] = None


class OptimizeAgentUseCase:
    """Use case for optimizing an agent.

    This use case handles the complete agent optimization workflow:
    1. Load and prepare datasets
    2. Configure optimization
    3. Run optimization (MIPROv2)
    4. Evaluate on validation set
    5. Save optimized agent with metadata
    6. Track experiment with MLflow (optional)
    """

    def __init__(
        self,
        dataset_builder: DatasetBuilder,
        agent_manager: AgentManager,
        gt_repo: Optional[GroundTruthRepository] = None,
        split_repo: Optional[DataSplitRepository] = None,
        tracker: Optional[ExperimentTracker] = None,
        validator: Optional[DataValidator] = None,
        enable_preflight_check: bool = True,
    ):
        """Initialize the use case.

        Args:
            dataset_builder: Service for building datasets.
            agent_manager: Service for managing agents.
            gt_repo: Repository for ground truth data.
            split_repo: Repository for split configurations.
            tracker: Optional experiment tracker.
            validator: Optional data validator for pre-flight checks.
            enable_preflight_check: Whether to run pre-flight validation (default: True).
        """
        from ae.core.storage import GroundTruthRepository, DataSplitRepository
        self.dataset_builder = dataset_builder
        self.agent_manager = agent_manager
        self.gt_repo = gt_repo or GroundTruthRepository()
        self.split_repo = split_repo or DataSplitRepository()
        self.tracker = tracker
        self.validator = validator
        self.enable_preflight_check = enable_preflight_check

        # Only create default validator if explicitly enabled
        if enable_preflight_check and validator is None:
            self.validator = DataValidator(gt_repo=gt_repo, split_repo=split_repo)

        logger.debug(
            f"Initialized OptimizeAgentUseCase "
            f"(preflight_check={enable_preflight_check}, validator={validator is not None})"
        )

    def execute(self, request: OptimizeAgentRequest) -> OptimizeAgentResponse:
        """Execute the agent optimization workflow.

        Args:
            request: Optimization request with all parameters.

        Returns:
            Response with optimization results.
        """
        try:
            # Get task name from config object (TaskConfig has .name attribute)
            task_name = request.task.config.name
            logger.info(
                f"Starting agent optimization for task '{task_name}' "
                f"with {request.num_trials} trials"
            )

            # Start MLflow run if tracker provided
            if self.tracker:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if request.run_name_prefix:
                    run_name = f"{request.run_name_prefix}_{timestamp}"
                else:
                    run_name = f"optimization_{timestamp}"
                self.tracker.start_run(run_name=run_name)

            # Step 1: Load ground truth (once, reuse for validation and dataset building)
            logger.info(f"Loading ground truth from {request.gt_path}")
            gt_data = self.gt_repo.load(request.gt_path, request.task.row_converter)

            # Step 1.5: Pre-flight validation (optional, enabled by default)
            if self.enable_preflight_check:
                self._run_preflight_check(request, gt_data)
            else:
                logger.debug("Pre-flight validation skipped (disabled)")

            # Step 2: Prepare datasets
            trainset, valset = self._prepare_datasets(request, gt_data)

            if not trainset:
                raise UseCaseExecutionError(
                    "OptimizeAgent",
                    "Training set is empty. Cannot optimize."
                )

            if not valset:
                raise UseCaseExecutionError(
                    "OptimizeAgent",
                    "Validation set is empty. Cannot evaluate."
                )

            logger.info(f"Dataset sizes: train={len(trainset)}, val={len(valset)}")

            if request.dry_run:
                train_val_res = self.validator.validate_dataset(trainset, request.train_split_name)
                val_val_res = self.validator.validate_dataset(valset, request.val_split_name)
                
                self.validator.log_validation_result(train_val_res, "Train Dataset Validation")
                self.validator.log_validation_result(val_val_res, "Validation Dataset Validation")
                
                if not train_val_res.success or not val_val_res.success:
                    errors = train_val_res.errors + val_val_res.errors
                    error_msg = f"Dataset dry run validation failed with {len(errors)} error(s):\n"
                    for i, error in enumerate(errors, 1):
                        error_msg += f"  {i}. {error}\n"
                    raise UseCaseExecutionError("OptimizeAgent (Dry Run)", error_msg)
                
                logger.info("✓ Dry run dataset checks completed successfully. All datasets are valid.")
                return OptimizeAgentResponse(
                    success=True,
                    agent_path=None,
                    final_metrics={"train_size": len(trainset), "val_size": len(valset)},
                    trial_count=0,
                    optimization_config={},
                )

            # Step 3: Create metric
            metric = self._create_metric(request)

            # Step 4: Create base agent
            base_agent = self._create_base_agent(request, request.signature_class)

            # Step 5: Configure optimization
            config = self._build_optimization_config(request, len(trainset), len(valset))

            # Log config to MLflow
            if self.tracker:
                self.tracker.log_params(config)

            # Step 6: Run optimization
            logger.info(
                f"Running MIPROv2 optimization: {request.num_trials} trials, "
                f"seed={request.seed}, auto=None (custom params)"
            )
            optimized_agent = self._run_optimization(
                base_agent=base_agent,
                trainset=trainset,
                valset=valset,
                metric=metric,
                request=request,
            )

            # Step 7: Evaluate on validation set
            final_metrics = self._evaluate_agent(optimized_agent, valset, metric)

            logger.info(f"Optimization complete. Final metrics: {final_metrics}")

            # Step 8: Save optimized agent
            agent_path = self._save_agent(
                agent=optimized_agent,
                request=request,
                metrics=final_metrics,
                config=config,
            )

            # Log to MLflow
            if self.tracker:
                self.tracker.log_optimization_results(
                    metrics=final_metrics,
                    config=config,
                    agent_path=agent_path,
                    task_name=request.task.config.name,
                    dspy_model=optimized_agent,
                )

            # Remove checkpoint after successful completion
            checkpoint_path = Path(f"data/processed/agents/{request.task.config.name}_checkpoint.json")
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    logger.info("Cleaned up optimization checkpoint.")
                except Exception as e:
                    logger.warning(f"Failed to clean up checkpoint: {e}")

            # Success!
            return OptimizeAgentResponse(
                success=True,
                agent_path=agent_path,
                final_metrics=final_metrics,
                trial_count=request.num_trials,
                optimization_config=config,
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)

            return OptimizeAgentResponse(
                success=False,
                error_message=str(e),
                trial_count=0,
            )

        finally:
            # End MLflow run
            if self.tracker:
                self.tracker.end_run()

    def _prepare_datasets(
        self, request: OptimizeAgentRequest, gt_data: Dict[str, Any]
    ) -> Tuple[List[dspy.Example], List[dspy.Example]]:
        """Prepare training and validation datasets."""
        logger.info("Preparing datasets...")

        # Build training set
        trainset = self.dataset_builder.build_from_split(
            task=request.task,
            gt_path=request.gt_path,
            split_path=request.split_path,
            split_name=request.train_split_name,
            gt_data=gt_data,
            limit=request.train_limit,
            seed=request.seed,
        )

        # Build validation set
        valset = self.dataset_builder.build_from_split(
            task=request.task,
            gt_path=request.gt_path,
            split_path=request.split_path,
            split_name=request.val_split_name,
            gt_data=gt_data,
            limit=request.val_limit,
            seed=request.seed,
        )

        logger.info(
            f"Datasets prepared: Train={len(trainset)}, Val={len(valset)}"
        )

        return trainset, valset

    def _create_metric(self, request: OptimizeAgentRequest) -> TaskMetric:
        """Create evaluation metric for the task."""
        task_config = request.task.config
        student_lm_adapter = DSPyLMAdapter(request.student_lm)
        return TaskMetric(
            task_config={
                "compare_fields": task_config.compare_fields,
                "name": task_config.name,
            },
            float_tolerance=task_config.float_tolerance,
            student_llm=student_lm_adapter,
            field_descriptions=task_config.field_descriptions,
            enable_semantic_judge=True,
        )

    def _create_base_agent(
        self, request: OptimizeAgentRequest, signature_class: Any
    ) -> BaseAgent:
        """Create the base agent to optimize."""
        from ae.extraction.agent import UniversalExtractor

        return UniversalExtractor(signature_class=signature_class)

    def _run_optimization(
        self,
        base_agent: BaseAgent,
        trainset: List[dspy.Example],
        valset: List[dspy.Example],
        metric: TaskMetric,
        request: OptimizeAgentRequest,
    ) -> BaseAgent:
        """Run the optimization process."""
        # Wrap LMs with DSPyLMAdapter for DSPy integration
        student_lm_adapter = DSPyLMAdapter(request.student_lm)
        teacher_lm_adapter = DSPyLMAdapter(request.teacher_lm) if request.teacher_lm else None

        # Create teacher module for bootstrapping
        teacher_module = TeacherWrapper(
            signature_class=request.signature_class,
            teacher_lm=teacher_lm_adapter
        )

        # Configure MIPROv2
        teleprompter = CheckpointingMIPROv2(
            metric=metric,
            prompt_model=teacher_lm_adapter,
            task_model=student_lm_adapter,
            teacher_settings={"lm": teacher_lm_adapter},
            auto=None,  # Intentionally None: allows custom values
            num_candidates=request.num_candidates,
            max_bootstrapped_demos=request.max_bootstrapped_demos,
            max_labeled_demos=request.max_labeled_demos,
            seed=request.seed,
            init_temperature=request.init_temperature,
            verbose=request.verbose,
            metric_threshold=request.metric_threshold,
            num_threads=1,
            max_errors=request.max_errors,
            task_name=request.task.config.name,
            cancel_event=request.cancel_event,
        )

        try:
            with dspy.context(
                lm=student_lm_adapter,
                provide_traceback=True,
                num_threads=1,
                async_max_workers=1,
            ):
                # Run optimization with explicit valset
                optimized_agent = teleprompter.compile(
                    base_agent,
                    trainset=trainset,
                    valset=valset,
                    teacher=teacher_module,
                    num_trials=request.num_trials,
                    max_bootstrapped_demos=request.max_bootstrapped_demos,
                    max_labeled_demos=request.max_labeled_demos,
                    seed=request.seed,
                    minibatch=request.minibatch,
                    minibatch_size=request.minibatch_size,
                    view_data_batch_size=request.view_data_batch_size,
                )
        except Exception as e:
            logger.warning(f"Teacher LLM is unavailable (error: {e}). Switching to degraded mode...")
            try:
                logger.info("Attempting degraded mode fallback: using student LLM as teacher...")
                degraded_teacher_module = TeacherWrapper(
                    signature_class=request.signature_class,
                    teacher_lm=student_lm_adapter
                )
                degraded_teleprompter = CheckpointingMIPROv2(
                    metric=metric,
                    prompt_model=student_lm_adapter,
                    task_model=student_lm_adapter,
                    teacher_settings={"lm": student_lm_adapter},
                    auto=None,
                    num_candidates=request.num_candidates,
                    max_bootstrapped_demos=request.max_bootstrapped_demos,
                    max_labeled_demos=request.max_labeled_demos,
                    seed=request.seed,
                    init_temperature=request.init_temperature,
                    verbose=request.verbose,
                    metric_threshold=request.metric_threshold,
                    num_threads=1,
                    max_errors=request.max_errors,
                    task_name=request.task.config.name,
                    cancel_event=request.cancel_event,
                )
                with dspy.context(
                    lm=student_lm_adapter,
                    provide_traceback=True,
                    num_threads=1,
                    async_max_workers=1,
                ):
                    optimized_agent = degraded_teleprompter.compile(
                        base_agent,
                        trainset=trainset,
                        valset=valset,
                        teacher=degraded_teacher_module,
                        num_trials=request.num_trials,
                        max_bootstrapped_demos=request.max_bootstrapped_demos,
                        max_labeled_demos=request.max_labeled_demos,
                        seed=request.seed,
                        minibatch=request.minibatch,
                        minibatch_size=request.minibatch_size,
                        view_data_batch_size=request.view_data_batch_size,
                    )
            except Exception as e_inner:
                logger.warning(
                    f"Degraded mode fallback with student LLM failed (error: {e_inner}). "
                    "Switching to zero-shot training / optimization without bootstrapping..."
                )
                zero_shot_teleprompter = CheckpointingMIPROv2(
                    metric=metric,
                    prompt_model=student_lm_adapter,
                    task_model=student_lm_adapter,
                    teacher_settings={"lm": student_lm_adapter},
                    auto=None,
                    num_candidates=request.num_candidates,
                    max_bootstrapped_demos=0,
                    max_labeled_demos=request.max_labeled_demos,
                    seed=request.seed,
                    init_temperature=request.init_temperature,
                    verbose=request.verbose,
                    metric_threshold=request.metric_threshold,
                    num_threads=1,
                    max_errors=request.max_errors,
                    task_name=request.task.config.name,
                    cancel_event=request.cancel_event,
                )
                with dspy.context(
                    lm=student_lm_adapter,
                    provide_traceback=True,
                    num_threads=1,
                    async_max_workers=1,
                ):
                    optimized_agent = zero_shot_teleprompter.compile(
                        base_agent,
                        trainset=trainset,
                        valset=valset,
                        teacher=None,
                        num_trials=request.num_trials,
                        max_bootstrapped_demos=0,
                        max_labeled_demos=request.max_labeled_demos,
                        seed=request.seed,
                        minibatch=request.minibatch,
                        minibatch_size=request.minibatch_size,
                        view_data_batch_size=request.view_data_batch_size,
                    )

        # Save LLM history after optimization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = Path("logs/llm_history")
        history_counts = save_optimization_history(
            student_lm=request.student_lm,
            teacher_lm=request.teacher_lm,
            output_dir=history_dir,
            timestamp=timestamp,
        )
        logger.info(f"LLM history saved: {history_counts}")

        return optimized_agent

    def _evaluate_agent(
        self, agent: Any, valset: List[dspy.Example], metric: TaskMetric
    ) -> Dict[str, float]:
        """Evaluate agent on validation set."""
        logger.info(f"Evaluating agent on {len(valset)} validation examples...")

        total_score = 0.0
        for example in valset:
            try:
                prediction = agent(document_text=example.document_text)
                score = metric(example, prediction)
                total_score += score
            except Exception as e:
                logger.warning(f"Evaluation error: {e}")
                continue

        avg_score = total_score / len(valset) if valset else 0.0

        return {
            "f1": avg_score,
            "validation_examples": len(valset),
        }

    def _save_agent(
        self,
        agent: Any,
        request: OptimizeAgentRequest,
        metrics: Dict[str, float],
        config: Dict[str, Any],
    ) -> Path:
        """Save the optimized agent."""
        logger.info("Saving optimized agent...")

        return self.agent_manager.save_agent(
            agent=agent,
            schema_config=request.task.config,
            metrics=metrics,
            config=config,
            model_version=request.model_version,
            description=request.description or f"Optimized with {request.num_trials} trials",
            initial_instruction_file=request.initial_instruction_file,
            instruction_hash=request.instruction_hash,
        )

    def _build_optimization_config(
        self, request: OptimizeAgentRequest, train_size: int, val_size: int
    ) -> Dict[str, Any]:
        """Build configuration dictionary for logging."""
        return {
            "task_name": request.task.config.name,
            "num_trials": request.num_trials,
            "train_size": train_size,
            "val_size": val_size,
            "train_split": request.train_split_name,
            "val_split": request.val_split_name,
            "model_version": request.model_version,
            "seed": request.seed,
            "num_candidates": request.num_candidates,
            "max_bootstrapped_demos": request.max_bootstrapped_demos,
            "max_labeled_demos": request.max_labeled_demos,
            "minibatch": request.minibatch,
            "minibatch_size": request.minibatch_size,
            "view_data_batch_size": request.view_data_batch_size,
            "metric_threshold": request.metric_threshold,
            "init_temperature": request.init_temperature,
            "verbose": request.verbose,
        }

    def _run_preflight_check(
        self,
        request: OptimizeAgentRequest,
        gt_data: Dict[str, Any],
    ) -> None:
        """Run pre-flight validation checks before optimization.

        Raises:
            UseCaseExecutionError: If critical validation checks fail.
        """
        logger.info("Running pre-flight validation checks...")

        # Validate splits
        assert self.validator is not None, "Validator must be initialized for pre-flight checks"
        split_result = self.validator.validate_splits(
            gt_path=request.gt_path,
            split_path=request.split_path,
            task=request.task,
            gt_data=gt_data,
            required_splits=[request.train_split_name, request.val_split_name],
        )

        # Log validation result
        self.validator.log_validation_result(split_result, "Data Splits Validation")

        # Validate ground truth
        gt_result = self.validator.validate_ground_truth(
            gt_path=request.gt_path,
            task=request.task,
            gt_data=gt_data,
        )

        if gt_result.warnings or gt_result.errors:
            self.validator.log_validation_result(gt_result, "Ground Truth Validation")

        # Check for critical errors
        all_errors = split_result.errors + gt_result.errors
        if all_errors:
            error_msg = f"Pre-flight validation failed with {len(all_errors)} error(s):\n"
            for i, error in enumerate(all_errors, 1):
                error_msg += f"  {i}. {error}\n"
            raise UseCaseExecutionError("OptimizeAgent", error_msg)

        # Log warnings
        all_warnings = split_result.warnings + gt_result.warnings
        if all_warnings:
            logger.warning(
                f"Pre-flight validation passed with {len(all_warnings)} warning(s). "
                f"Review logs for details."
            )

        logger.info("✓ Pre-flight validation passed")
