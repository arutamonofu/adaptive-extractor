"""Agent manager service for agent lifecycle management.

This service handles training, saving, loading, and versioning of agents,
providing a high-level interface for agent operations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from aee.domain.tasks import TaskConfig
from aee.infrastructure.storage import AgentMetadata, AgentRepository
from aee.shared.exceptions import AgentNotFoundError, UseCaseExecutionError

logger = logging.getLogger(__name__)


@runtime_checkable
class SerializableAgent(Protocol):
    """Protocol for agents that can be serialized.

    Agents must implement either `dump_state()` or `save()` method.
    """

    def dump_state(self) -> Dict[str, Any]:
        """Dump agent state to a dictionary."""
        ...


@runtime_checkable
class SaveableAgent(Protocol):
    """Protocol for agents that can be saved to a file."""

    def save(self, path: str) -> None:
        """Save agent to a file."""
        ...


class AgentManager:
    """Service for managing agent lifecycle.

    This service provides high-level operations for agents including
    saving with metadata, loading, versioning, and querying.

    Example:
        ```python
        manager = AgentManager(agent_repo=AgentRepository(agents_dir))

        # Save a trained agent
        agent_path = manager.save_agent(
            agent=compiled_agent,
            task=nanozyme_task,
            metrics={"f1": 0.85},
            config=config_dict,
            description="Optimized with MIPROv2"
        )

        # Load the latest agent
        agent = manager.load_latest_agent(task_name="nanozymes")

        # Get agent history
        history = manager.get_agent_history(task_name="nanozymes")
        ```
    """

    def __init__(self, agent_repo: AgentRepository):
        """Initialize the agent manager.

        Args:
            agent_repo: Repository for agent storage.
        """
        self.agent_repo = agent_repo
        logger.debug("Initialized AgentManager")

    def save_agent(
        self,
        agent: Union[SerializableAgent, SaveableAgent, Dict[str, Any]],
        task: TaskConfig,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        model_version: str = "unknown",
        description: Optional[str] = None,
        git_commit: Optional[str] = None,
        initial_instruction_file: Optional[str] = None,
        instruction_hash: Optional[str] = None,
    ) -> Path:
        """Save a trained agent with metadata.

        Args:
            agent: The compiled agent to save. Must implement dump_state() or save() method.
            task: Task config the agent was trained for.
            metrics: Performance metrics (e.g., {"f1": 0.85}).
            config: Configuration used for training.
            model_version: LLM model version used.
            description: Optional description of this agent.
            git_commit: Optional git commit hash.
            initial_instruction_file: Path to the initial instruction file used for optimization.
            instruction_hash: SHA256 hash (first 12 chars) of the initial instruction.

        Returns:
            Path to saved agent file.

        Raises:
            UseCaseExecutionError: If save fails.
        """
        try:
            # Create metadata
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

            # Convert agent to serializable format
            agent_dict = self._serialize_agent(agent)

            # Save via repository
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
        """Load an agent from a file.

        Args:
            agent_path: Path to agent file.

        Returns:
            Loaded agent (deserialized).

        Raises:
            AgentNotFoundError: If agent not found.
            UseCaseExecutionError: If load fails.
        """
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

    def load_agent_with_metadata(
        self, agent_path: Path
    ) -> tuple[Dict[str, Any], AgentMetadata]:
        """Load an agent with its metadata.

        Args:
            agent_path: Path to agent file.

        Returns:
            Tuple of (agent, metadata).

        Raises:
            AgentNotFoundError: If agent not found.
        """
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
        """Load the most recent agent for a task.

        Args:
            task_name: Name of the task.

        Returns:
            Latest agent, or None if no agents found.

        Raises:
            UseCaseExecutionError: If load fails.
        """
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
        """Get history of agents for a task.

        Args:
            task_name: Name of the task.
            limit: Optional limit on number of agents to return.

        Returns:
            List of agent info dictionaries, sorted by creation time (newest first).
        """
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
        """Compare multiple agents.

        Args:
            agent_paths: List of agent file paths.

        Returns:
            Dictionary with comparison data.
        """
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

        # Sort by F1 score if available
        if comparisons and "f1" in comparisons[0].get("metrics", {}):
            comparisons.sort(
                key=lambda x: x.get("metrics", {}).get("f1", 0),
                reverse=True
            )

        return {
            "total_agents": len(comparisons),
            "agents": comparisons,
        }

    def delete_agent(self, agent_path: Path) -> None:
        """Delete an agent.

        Args:
            agent_path: Path to agent file.

        Raises:
            AgentNotFoundError: If agent not found.
            UseCaseExecutionError: If delete fails.
        """
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
        """Serialize agent to dictionary format.

        Args:
            agent: Agent to serialize. Must implement dump_state() or save() method.

        Returns:
            Serialized agent dictionary.

        Raises:
            UseCaseExecutionError: If agent cannot be serialized.
        """
        # For DSPy agents with dump_state method
        if isinstance(agent, SerializableAgent):
            return agent.dump_state()

        # For dict agents (already serialized)
        if isinstance(agent, dict):
            return agent

        # For agents with save() method - save to temp file and load
        if isinstance(agent, SaveableAgent):
            import tempfile
            import json

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

        # Cannot serialize
        raise UseCaseExecutionError(
            "AgentManager._serialize_agent",
            f"Agent of type {type(agent).__name__} does not implement "
            f"dump_state() or save() method and cannot be serialized"
        )

    def get_best_agent(
        self, task_name: str, metric: str = "f1"
    ) -> Optional[Path]:
        """Get the best performing agent for a task.

        Args:
            task_name: Name of the task.
            metric: Metric to use for comparison (default: "f1").

        Returns:
            Path to best agent, or None if no agents found.
        """
        try:
            history = self.get_agent_history(task_name)

            if not history:
                return None

            # Filter agents with the metric
            agents_with_metric = [
                h for h in history
                if metric in h.get("metrics", {})
            ]

            if not agents_with_metric:
                logger.warning(
                    f"No agents found with metric '{metric}' for task '{task_name}'"
                )
                return None

            # Find best
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
