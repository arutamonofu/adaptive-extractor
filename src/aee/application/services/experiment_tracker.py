"""Experiment tracking service for MLflow integration.

This service provides a clean interface for experiment tracking,
abstracting away MLflow specifics and providing convenience methods.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Service for tracking experiments with MLflow.

    This service provides a simplified interface for MLflow experiment
    tracking, handling common operations like run creation, parameter
    logging, and metric tracking.

    Example:
        ```python
        tracker = ExperimentTracker(experiment_name="nanozyme_optimization")

        # Start a run
        with tracker.start_run(run_name="trial_1"):
            tracker.log_params({"num_trials": 10, "model": "claude-sonnet"})
            tracker.log_metrics({"f1": 0.85, "precision": 0.82})
            tracker.log_artifact(Path("agent.json"))
        ```
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        enabled: bool = True,
        enable_dspy_autolog: bool = True,
    ):
        """Initialize the experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: Optional MLflow tracking URI.
            enabled: Whether tracking is enabled (useful for testing).
            enable_dspy_autolog: Whether to enable DSPy autologging.
        """
        self.experiment_name = experiment_name
        self.enabled = enabled
        self._run_id: Optional[str] = None
        self._dspy_autolog_enabled = False
        self._enable_dspy_autolog = enable_dspy_autolog
        self.mlflow: Optional[Any] = None
        self.experiment_id: Optional[str] = None
        self._active_run: Any = None

        if enabled:
            try:
                import mlflow

                self.mlflow = mlflow

                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)

                # Create or get experiment
                experiment = mlflow.set_experiment(experiment_name)
                self.experiment_id = experiment.experiment_id

                logger.info(f"Initialized ExperimentTracker: {experiment_name}")

                # NOTE: DSPy autologging is enabled in start_run() after the run is created
                # to avoid NonRecordingSpan warnings during DSPy operations

            except ImportError:
                logger.warning(
                    "MLflow not installed. Experiment tracking disabled."
                )
                self.enabled = False
        else:
            logger.debug("ExperimentTracker initialized in disabled mode")

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
    ) -> "ExperimentTracker":
        """Start a new MLflow run.

        Can be used as a context manager.

        Args:
            run_name: Optional name for the run. If None, auto-generated with timestamp.
            nested: Whether this is a nested run.

        Returns:
            Self (for context manager usage).
        """
        if not self.enabled or not self.mlflow:
            return self

        try:
            if run_name is None:
                # Auto-generate unique run name with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"run_{timestamp}"

            self._active_run = self.mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested,
            )
            self._run_id = self._active_run.info.run_id

            logger.info(f"Started MLflow run: '{run_name}'")

            # Enable DSPy autologging immediately after run is started
            # This ensures tracing works correctly for DSPy operations
            if self._enable_dspy_autolog and not self._dspy_autolog_enabled:
                self.enable_dspy_autolog()

        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")

        return self

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.enabled or not self.mlflow:
            return

        try:
            self.mlflow.end_run()
            logger.debug(f"Ended MLflow run: {self._run_id}")
            self._run_id = None

        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run.

        Args:
            params: Dictionary of parameters to log.
        """
        if not self.enabled or not self.mlflow or not self._run_id:
            return

        try:
            # MLflow requires string values for params
            str_params = {k: str(v) for k, v in params.items()}
            self.mlflow.log_params(str_params)
            logger.debug(f"Logged {len(str_params)} parameters")

        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        self.log_params({key: value})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number.
        """
        if not self.enabled or not self.mlflow or not self._run_id:
            return

        try:
            self.mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics")

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, artifact_path: Path) -> None:
        """Log an artifact (file) to the current run.

        Args:
            artifact_path: Path to artifact file.
        """
        if not self.enabled or not self.mlflow or not self._run_id:
            return

        try:
            self.mlflow.log_artifact(str(artifact_path))
            logger.debug(f"Logged artifact: {artifact_path.name}")

        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log.
            artifact_file: Name of artifact file (e.g., "config.json").
        """
        if not self.enabled or not self.mlflow or not self._run_id:
            return

        try:
            self.mlflow.log_dict(dictionary, artifact_file)
            logger.debug(f"Logged dictionary as {artifact_file}")

        except Exception as e:
            logger.warning(f"Failed to log dictionary: {e}")

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag name.
            value: Tag value.
        """
        if not self.enabled or not self.mlflow or not self._run_id:
            return

        try:
            self.mlflow.set_tag(key, str(value))
            logger.debug(f"Set tag: {key}={value}")

        except Exception as e:
            logger.warning(f"Failed to set tag: {e}")

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set multiple tags on the current run.

        Args:
            tags: Dictionary of tags to set.
        """
        if not self.enabled or not self.mlflow or not self._run_id:
            return

        try:
            str_tags = {k: str(v) for k, v in tags.items()}
            self.mlflow.set_tags(str_tags)
            logger.debug(f"Set {len(tags)} tags")

        except Exception as e:
            logger.warning(f"Failed to set tags: {e}")

    def enable_dspy_autolog(self) -> None:
        """Enable DSPy autologging for automatic tracking of DSPy operations.

        This enables automatic logging of:
        - DSPy program calls and predictions (tracing)
        - Prompt templates and their evolution
        - Optimization steps and metrics
        - Evaluation results
        """
        if not self.enabled or not self.mlflow:
            return

        if not hasattr(self.mlflow, 'dspy'):
            logger.warning(
                "DSPy autologging not available in this MLflow version. "
                "Please upgrade to mlflow>=2.10.0"
            )
            return

        try:
            # Enable full tracing for DSPy operations
            self.mlflow.dspy.autolog(
                log_traces=True,            # Enable tracing for all DSPy calls
                log_traces_from_compile=True,  # Enable tracing during optimization
                log_traces_from_eval=True,     # Enable tracing during evaluation
                log_compiles=True,          # Log optimization process
                log_evals=True,             # Log evaluation results
                silent=False,               # Show messages
            )
            self._dspy_autolog_enabled = True
            logger.info("Enabled DSPy autologging with full tracing")

        except Exception as e:
            logger.warning(f"Failed to enable DSPy autologging: {e}")

    def disable_dspy_autolog(self) -> None:
        """Disable DSPy autologging."""
        if not self.enabled or not self.mlflow or not self._dspy_autolog_enabled:
            return

        if not hasattr(self.mlflow, 'dspy'):
            return

        try:
            self.mlflow.dspy.autolog(disable=True)
            self._dspy_autolog_enabled = False
            logger.info("Disabled DSPy autologging")

        except Exception as e:
            logger.warning(f"Failed to disable DSPy autologging: {e}")

    def log_dspy_model(
        self,
        dspy_model: Any,
        name: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Log a DSPy model with proper serialization.

        This uses MLflow's DSPy integration to properly save and version
        DSPy models, including their prompts, demonstrations, and state.

        Uses DSPy's native save method (use_dspy_model_save=True) to avoid
        pickle serialization issues with non-serializable objects like thread locks.
        Requires dspy>=3.1.0.

        Args:
            dspy_model: The DSPy model/program to log.
            name: Name for the model artifact (replaces deprecated 'artifact_path').
            signature: Optional model signature.
            input_example: Optional example input.
            **kwargs: Additional arguments for mlflow.dspy.log_model.
        """
        if not self.enabled or not self.mlflow or not self._run_id:
            return

        if not hasattr(self.mlflow, 'dspy'):
            logger.warning(
                "DSPy model logging not available in this MLflow version. "
                "Please upgrade to mlflow>=2.10.0. "
                "Falling back to artifact logging."
            )
            self._log_dspy_model_fallback(dspy_model)
            return

        try:
            # Use DSPy's native save method to avoid pickle issues with thread locks
            # This requires dspy>=3.1.0 (currently using dspy-ai 3.1.3)
            self.mlflow.dspy.log_model(
                dspy_model=dspy_model,
                name=name,
                signature=signature,
                input_example=input_example,
                use_dspy_model_save=True,
                **kwargs,
            )
            logger.info(f"Logged DSPy model to {name}")

        except Exception as e:
            error_msg = str(e)
            # Handle pickle-related errors gracefully
            if "pickle" in error_msg.lower() or "_thread.lock" in error_msg:
                logger.warning(
                    f"DSPy model contains non-serializable objects (e.g., thread locks). "
                    f"Falling back to JSON artifact logging. Error: {e}"
                )
            else:
                logger.warning(f"Failed to log DSPy model: {e}")
            self._log_dspy_model_fallback(dspy_model)

    def _log_dspy_model_fallback(self, dspy_model: Any) -> None:
        """Fallback: log DSPy model as regular artifact if DSPy integration unavailable.

        Args:
            dspy_model: The DSPy model/program to log.
        """
        if hasattr(dspy_model, "save"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                dspy_model.save(f.name)
                self.log_artifact(Path(f.name))

    def log_optimization_results(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        agent_path: Path,
        task_name: str,
        dspy_model: Optional[Any] = None,
    ) -> None:
        """Log complete optimization results.

        Convenience method for logging all optimization outputs.

        Args:
            metrics: Final metrics.
            config: Configuration used.
            agent_path: Path to saved agent.
            task_name: Name of the task.
            dspy_model: Optional DSPy model to log with proper serialization.
        """
        if not self.enabled:
            return

        try:
            # Log metrics
            self.log_metrics(metrics)

            # Log config
            self.log_params(config)

            # Log model with DSPy integration if provided
            if dspy_model is not None:
                self.log_dspy_model(dspy_model, name="optimized_agent")

            # Always log the JSON artifact as well for compatibility
            self.log_artifact(agent_path)

            # Set tags
            self.set_tags({
                "task": task_name,
                "agent_name": agent_path.name,
                "status": "completed",
            })

            logger.info("Logged optimization results to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log optimization results: {e}")

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.end_run()

    @property
    def is_active(self) -> bool:
        """Check if a run is currently active."""
        return self.enabled and self._run_id is not None

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._run_id
