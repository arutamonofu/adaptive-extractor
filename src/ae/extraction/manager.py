"""Agent lifecycle management.

This module provides services for saving, loading, comparing,
and versioning extraction agents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ae.core.exceptions import AgentNotFoundError, UseCaseExecutionError
from ae.core.storage import AgentMetadata, AgentRepository

from .agent import SaveableAgent, SerializableAgent, UniversalExtractor

logger = logging.getLogger(__name__)


class AgentManager:
    """Service for managing agent lifecycle.

    Handles training, saving, loading, and versioning of agents.
    """

    def __init__(self, agent_repo: AgentRepository):
        """Initialize the agent manager."""
        self.agent_repo = agent_repo
        logger.debug("Initialized AgentManager")

    def save_agent(
        self,
        agent: Union[SerializableAgent, SaveableAgent, Dict[str, Any]],
        task: Any,  # TaskConfig
        metrics: Dict[str, float],
        config: Dict[str, Any],
        model_version: str = "unknown",
        description: Optional[str] = None,
        git_commit: Optional[str] = None,
        initial_instruction_file: Optional[str] = None,
        instruction_hash: Optional[str] = None,
    ) -> Path:
        """Save a trained agent with metadata."""
        try:
            metadata = AgentMetadata(
                task_name=task.name,
                created_at=datetime.now().isoformat(),
                model_version=model_version,
                metrics=metrics,
                config_snapshot=config,
                git_commit=git_commit,
                description=description,
                initial_instruction_file=initial_instruction_file,
                instruction_hash=instruction_hash,
            )

            agent_dict = self._serialize_agent(agent)

            agent_path = self.agent_repo.save(
                agent=agent_dict,
                task_name=task.name,
                metadata=metadata,
            )

            logger.info(
                f"Saved agent for task '{task.name}' to {agent_path} "
                f"(metrics: {metrics})"
            )

            return agent_path

        except Exception as e:
            raise UseCaseExecutionError(
                "AgentManager.save_agent",
                f"Failed to save agent: {e}"
            ) from e

    def load_agent(self, agent_path: Path) -> Dict[str, Any]:
        """Load an agent from a file."""
        try:
            agent_dict, metadata = self.agent_repo.load(agent_path)

            logger.info(
                f"Loaded agent from {agent_path} "
                f"(task={metadata.task_name}, created={metadata.created_at})"
            )

            return agent_dict

        except AgentNotFoundError:
            raise
        except Exception as e:
            raise UseCaseExecutionError(
                "AgentManager.load_agent",
                f"Failed to load agent: {e}"
            ) from e

    def load_agent_as_object(
        self,
        agent_path: Path,
        task_dict: Dict[str, Any],
    ) -> Any:
        """Load an agent and reconstruct it as a callable object."""
        try:
            agent_dict, metadata = self.agent_repo.load(agent_path)

            signature_class = task_dict.get("signature")
            if signature_class is None:
                raise UseCaseExecutionError(
                    "AgentManager.load_agent_as_object",
                    "Task dict must contain 'signature' key for agent reconstruction"
                )

            reconstructed_agent = UniversalExtractor(signature_class)
            state_to_load = self._normalize_agent_state(agent_dict)
            reconstructed_agent.load_state(state_to_load)

            logger.info(
                f"Reconstructed agent from {agent_path} "
                f"(task={metadata.task_name})"
            )

            return reconstructed_agent

        except AgentNotFoundError:
            raise
        except Exception as e:
            raise UseCaseExecutionError(
                "AgentManager.load_agent_as_object",
                f"Failed to reconstruct agent: {e}"
            ) from e

    def _normalize_agent_state(
        self, agent_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize agent state to DSPy native format."""
        if "prog.predict" in agent_dict or any(
            key.startswith("prog.") for key in agent_dict.keys()
        ):
            return agent_dict

        if "prog" in agent_dict and isinstance(agent_dict["prog"], dict):
            prog_dict = agent_dict["prog"]
            if "predict" in prog_dict:
                predict_dict = prog_dict["predict"]

                if "signature" not in predict_dict:
                    predict_dict["signature"] = {
                        "instructions": "Given the fields `input`, produce the fields `output`.",
                        "fields": [
                            {"prefix": "Input:", "description": "${input}"},
                            {"prefix": "Reasoning: Let's think step by step in order to", "description": "${reasoning}"},
                            {"prefix": "Output:", "description": "${output}"},
                        ]
                    }

                if "traces" not in predict_dict:
                    predict_dict["traces"] = []
                if "train" not in predict_dict:
                    predict_dict["train"] = []
                if "demos" not in predict_dict:
                    predict_dict["demos"] = []
                if "lm" not in predict_dict:
                    predict_dict["lm"] = None

                native_state = {"prog.predict": predict_dict}
                for key, value in prog_dict.items():
                    if key != "predict":
                        native_state[f"prog.{key}"] = value
                return native_state

        if "lm" in agent_dict or "traces" in agent_dict or "settings" in agent_dict:
            traces = agent_dict.get("traces", [])
            lm_config = agent_dict.get("lm")
            settings = agent_dict.get("settings", {})

            native_state = {
                "prog.predict": {
                    "traces": traces if isinstance(traces, list) else [],
                    "train": [],
                    "demos": [],
                    "signature": {
                        "instructions": "Given the fields `input`, produce the fields `output`.",
                        "fields": [
                            {"prefix": "Input:", "description": "${input}"},
                            {"prefix": "Reasoning: Let's think step by step in order to", "description": "${reasoning}"},
                            {"prefix": "Output:", "description": "${output}"},
                        ]
                    },
                    "lm": lm_config if isinstance(lm_config, dict) else None,
                }
            }

            if settings:
                native_state["_settings"] = settings

            return native_state

        raise UseCaseExecutionError(
            "AgentManager._normalize_agent_state",
            "Agent state format not recognized."
        )

    def load_agent_with_metadata(
        self, agent_path: Path
    ) -> tuple[Dict[str, Any], AgentMetadata]:
        """Load an agent with its metadata."""
        try:
            return self.agent_repo.load(agent_path)
        except AgentNotFoundError:
            raise
        except Exception as e:
            raise UseCaseExecutionError(
                "AgentManager.load_agent_with_metadata",
                f"Failed to load agent with metadata: {e}"
            ) from e

    def load_latest_agent(self, task_name: str) -> Optional[Any]:
        """Load the most recent agent for a task."""
        try:
            latest_path = self.agent_repo.get_latest(task_name)

            if latest_path is None:
                logger.warning(f"No agents found for task '{task_name}'")
                return None

            return self.load_agent(latest_path)

        except Exception as e:
            raise UseCaseExecutionError(
                "AgentManager.load_latest_agent",
                f"Failed to load latest agent: {e}"
            ) from e

    def get_agent_history(
        self, task_name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get history of agents for a task."""
        try:
            agent_paths = self.agent_repo.list_agents(
                task_name=task_name, sort_by="created_at"
            )

            if limit:
                agent_paths = agent_paths[:limit]

            history = []
            for path in agent_paths:
                try:
                    info = self.agent_repo.get_agent_info(path)
                    history.append(info)
                except Exception as e:
                    logger.warning(f"Failed to get info for {path}: {e}")
                    continue

            logger.debug(
                f"Retrieved {len(history)} agents for task '{task_name}'"
            )

            return history

        except Exception as e:
            logger.error(f"Failed to get agent history: {e}")
            return []

    def compare_agents(
        self, agent_paths: List[Path]
    ) -> Dict[str, Any]:
        """Compare multiple agents."""
        comparisons = []

        for path in agent_paths:
            try:
                _, metadata = self.agent_repo.load(path)
                comparisons.append({
                    "path": str(path),
                    "task": metadata.task_name,
                    "created_at": metadata.created_at,
                    "model_version": metadata.model_version,
                    "metrics": metadata.metrics,
                    "description": metadata.description,
                })
            except Exception as e:
                logger.warning(f"Failed to load {path} for comparison: {e}")
                continue

        if comparisons:
            first_metrics = comparisons[0].get("metrics")
            if isinstance(first_metrics, dict) and "f1" in first_metrics:
                def get_f1(x: dict) -> float:
                    metrics = x.get("metrics")
                    if isinstance(metrics, dict):
                        f1_val = metrics.get("f1", 0)
                        return float(f1_val) if f1_val is not None else 0.0
                    return 0.0

                comparisons.sort(
                    key=get_f1,
                    reverse=True
                )

        return {
            "total_agents": len(comparisons),
            "agents": comparisons,
        }

    def delete_agent(self, agent_path: Path) -> None:
        """Delete an agent."""
        try:
            self.agent_repo.delete(agent_path)
            logger.info(f"Deleted agent: {agent_path}")

        except AgentNotFoundError:
            raise
        except Exception as e:
            raise UseCaseExecutionError(
                "AgentManager.delete_agent",
                f"Failed to delete agent: {e}"
            ) from e

    def _serialize_agent(
        self, agent: Union[SerializableAgent, SaveableAgent, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Serialize agent to dictionary format."""
        if isinstance(agent, SerializableAgent):
            return agent.dump_state()

        if isinstance(agent, dict):
            return agent

        if isinstance(agent, SaveableAgent):
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                temp_path = f.name
                agent.save(temp_path)

            try:
                with open(temp_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            finally:
                import os
                os.unlink(temp_path)

        if hasattr(agent, "dump_state") and callable(agent.dump_state):
            return agent.dump_state()

        raise UseCaseExecutionError(
            "AgentManager._serialize_agent",
            f"Agent of type {type(agent).__name__} cannot be serialized"
        )

    def create_agent_with_demos(
        self,
        signature_class: Type,  # type: ignore[type-arg]
        demos: List[Any],
    ) -> Any:
        """Create a fresh agent with few-shot demonstrations."""
        logger.info(f"Creating agent with {len(demos)} few-shot demonstrations")

        agent = UniversalExtractor(signature_class)

        if hasattr(agent.prog, "predict") and hasattr(agent.prog.predict, "demos"):
            agent.prog.predict.demos = demos
        else:
            logger.warning(
                "Agent prog.predict.demos not found, "
                "demos will not be used"
            )

        return agent

    def get_best_agent(
        self, task_name: str, metric: str = "f1"
    ) -> Optional[Path]:
        """Get the best performing agent for a task."""
        try:
            history = self.get_agent_history(task_name)

            if not history:
                return None

            agents_with_metric = [
                h for h in history
                if metric in h.get("metrics", {})
            ]

            if not agents_with_metric:
                logger.warning(
                    f"No agents found with metric '{metric}' for task '{task_name}'"
                )
                return None

            best = max(
                agents_with_metric,
                key=lambda x: x["metrics"][metric]
            )

            best_path = Path(best["path"])
            logger.info(
                f"Best agent for '{task_name}' by {metric}: "
                f"{best_path.name} ({metric}={best['metrics'][metric]:.3f})"
            )

            return best_path

        except Exception as e:
            logger.error(f"Failed to find best agent: {e}")
            return None
