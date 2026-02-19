"""Functional API for managing trained agents.

This module provides simple functions for saving and loading agents,
replacing the class-based AgentRepository for simpler use cases.

Example:
    ```python
    from aee.infrastructure.storage.agents import save_agent, load_agent

    # Save an agent
    agent_path = save_agent(
        agent=agent_dict,
        task_name="nanozymes",
        agents_dir=Path("data/agents"),
        metrics={"f1": 0.85},
        config_snapshot={"num_trials": 10},
    )

    # Load an agent
    agent, metadata = load_agent(agent_path)
    ```
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aee.shared.exceptions import AgentNotFoundError, InvalidAgentError, RepositoryError

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    """Metadata associated with a trained agent.

    Attributes:
        task_name: Name of the task this agent was trained for.
        created_at: Timestamp when the agent was created.
        model_version: Version of the LLM model used.
        metrics: Performance metrics (e.g., f1, precision, recall).
        config_snapshot: Configuration used during training.
        git_commit: Git commit hash at time of creation (optional).
        description: Human-readable description (optional).
        initial_instruction_file: Path to the initial instruction file used (optional).
        instruction_hash: SHA256 hash (first 12 chars) of the initial instruction (optional).
    """

    task_name: str
    created_at: str
    model_version: str
    metrics: Dict[str, float]
    config_snapshot: Dict[str, Any]
    git_commit: Optional[str] = None
    description: Optional[str] = None
    initial_instruction_file: Optional[str] = None
    instruction_hash: Optional[str] = None


def _generate_filename(
    task_name: str,
    timestamp: Optional[str] = None,
    agents_dir: Optional[Path] = None,
) -> str:
    """Generate a unique filename for an agent.

    Format: {task}_v{version}_{timestamp}.json

    Args:
        task_name: Name of the task.
        timestamp: Optional timestamp (ISO format). If None, uses current time.
        agents_dir: Optional directory to count existing versions.

    Returns:
        Generated filename.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    # Convert ISO timestamp to filename-safe format
    timestamp_safe = timestamp.replace(":", "-").replace(".", "-")[:19]

    # Count existing versions for this task
    version = 1
    if agents_dir:
        existing = [
            f for f in agents_dir.glob(f"{task_name}_v*.json")
            if not f.name.endswith(".meta.json")
        ]
        version = len(existing) + 1

    return f"{task_name}_v{version}_{timestamp_safe}.json"


def _create_default_metadata(agent_path: Path) -> AgentMetadata:
    """Create default metadata for an agent without metadata file.

    Args:
        agent_path: Path to the agent file.

    Returns:
        Default metadata.
    """
    # Try to extract task name from filename
    task_name = agent_path.stem.split("_")[0] if "_" in agent_path.stem else "unknown"

    return AgentMetadata(
        task_name=task_name,
        created_at=datetime.fromtimestamp(agent_path.stat().st_mtime).isoformat(),
        model_version="unknown",
        metrics={},
        config_snapshot={},
    )


def save_agent(
    agent: Dict[str, Any],
    task_name: str,
    agents_dir: Path,
    metadata: Optional[AgentMetadata] = None,
    filename: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    model_version: str = "unknown",
    description: Optional[str] = None,
    initial_instruction_file: Optional[str] = None,
    instruction_hash: Optional[str] = None,
) -> Path:
    """Save an agent with metadata.

    Args:
        agent: The trained agent object (must be JSON-serializable).
        task_name: Name of the task.
        agents_dir: Directory where agents are stored.
        metadata: Optional AgentMetadata instance. If not provided, creates default.
        filename: Optional custom filename. If None, generates automatically.
        metrics: Performance metrics (used if metadata not provided).
        config_snapshot: Configuration snapshot (used if metadata not provided).
        model_version: LLM model version (used if metadata not provided).
        description: Human-readable description.
        initial_instruction_file: Path to instruction file.
        instruction_hash: Hash of instruction.

    Returns:
        Path to the saved agent file.

    Raises:
        RepositoryError: If save operation fails.
    """
    try:
        agents_dir = Path(agents_dir)
        agents_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata if not provided
        if metadata is None:
            timestamp = datetime.now().isoformat()
            metadata = AgentMetadata(
                task_name=task_name,
                created_at=timestamp,
                model_version=model_version,
                metrics=metrics or {},
                config_snapshot=config_snapshot or {},
                description=description,
                initial_instruction_file=initial_instruction_file,
                instruction_hash=instruction_hash,
            )

        # Generate filename if not provided
        if filename is None:
            filename = _generate_filename(task_name, metadata.created_at, agents_dir)

        agent_path = agents_dir / filename
        metadata_path = agent_path.with_suffix(".meta.json")

        # Save agent
        with open(agent_path, "w", encoding="utf-8") as f:
            json.dump(agent, f, indent=2, ensure_ascii=False)

        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved agent to {agent_path}")
        logger.debug(f"Saved metadata to {metadata_path}")

        return agent_path

    except Exception as e:
        raise RepositoryError(
            "save_agent", "save", f"Failed to save agent: {e}"
        ) from e


def load_agent(
    agent_path: Path,
) -> Tuple[Dict[str, Any], AgentMetadata]:
    """Load an agent with metadata.

    Args:
        agent_path: Path to the agent file.

    Returns:
        Tuple of (agent, metadata).

    Raises:
        AgentNotFoundError: If agent file not found.
        InvalidAgentError: If agent or metadata is invalid/corrupted.
    """
    agent_path = Path(agent_path)

    if not agent_path.exists():
        raise AgentNotFoundError(str(agent_path))

    metadata_path = agent_path.with_suffix(".meta.json")

    try:
        # Load agent
        with open(agent_path, "r", encoding="utf-8") as f:
            agent = json.load(f)

        # Load metadata (optional - create default if missing)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            metadata = AgentMetadata(**metadata_dict)
        else:
            logger.warning(f"Metadata not found for {agent_path}, creating default")
            metadata = _create_default_metadata(agent_path)

        logger.debug(f"Loaded agent from {agent_path}")
        return agent, metadata

    except json.JSONDecodeError as e:
        raise InvalidAgentError(
            str(agent_path), f"Invalid JSON format: {e}"
        ) from e
    except Exception as e:
        raise InvalidAgentError(
            str(agent_path), f"Failed to load agent: {e}"
        ) from e


def list_agents(
    agents_dir: Path,
    task_name: Optional[str] = None,
    sort_by: str = "created_at",
) -> List[Path]:
    """List all agent files, optionally filtered by task.

    Args:
        agents_dir: Directory where agents are stored.
        task_name: Optional task name filter.
        sort_by: Sort criterion ("created_at" or "name").

    Returns:
        List of agent file paths.
    """
    agents_dir = Path(agents_dir)
    pattern = f"{task_name}_*.json" if task_name else "*.json"
    agents = [
        p for p in agents_dir.glob(pattern)
        if not p.name.endswith(".meta.json")
    ]

    if sort_by == "created_at":
        agents.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif sort_by == "name":
        agents.sort()

    return agents


def get_latest_agent(
    agents_dir: Path,
    task_name: str,
) -> Optional[Path]:
    """Get the most recently created agent for a task.

    Args:
        agents_dir: Directory where agents are stored.
        task_name: Name of the task.

    Returns:
        Path to the latest agent, or None if no agents found.
    """
    agents = list_agents(agents_dir, task_name=task_name, sort_by="created_at")
    return agents[0] if agents else None


def delete_agent(
    agent_path: Path,
) -> None:
    """Delete an agent and its metadata.

    Args:
        agent_path: Path to the agent file.

    Raises:
        AgentNotFoundError: If agent file not found.
    """
    agent_path = Path(agent_path)

    if not agent_path.exists():
        raise AgentNotFoundError(str(agent_path))

    metadata_path = agent_path.with_suffix(".meta.json")

    # Delete agent file
    agent_path.unlink()
    logger.info(f"Deleted agent {agent_path}")

    # Delete metadata file if it exists
    if metadata_path.exists():
        metadata_path.unlink()
        logger.debug(f"Deleted metadata {metadata_path}")


def get_agent_info(
    agent_path: Path,
) -> Dict[str, Any]:
    """Get information about an agent without loading it fully.

    Args:
        agent_path: Path to the agent file.

    Returns:
        Dictionary with agent information.
    """
    _, metadata = load_agent(agent_path)

    return {
        "path": str(agent_path),
        "task_name": metadata.task_name,
        "created_at": metadata.created_at,
        "model_version": metadata.model_version,
        "metrics": metadata.metrics,
        "description": metadata.description,
    }


# Keep AgentRepository for backward compatibility
# It now delegates to the functional API
class AgentRepository:
    """Repository for managing trained agents with metadata.

    This repository handles saving/loading agents along with their metadata,
    providing version tracking and agent management capabilities.

    Note: This class is maintained for backward compatibility.
    New code should use the functional API (save_agent, load_agent, etc.).

    Example:
        ```python
        repo = AgentRepository(agents_dir=Path("data/agents"))

        # Save an agent with metadata
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="claude-sonnet-3.5",
            metrics={"f1": 0.85, "precision": 0.82, "recall": 0.88},
            config_snapshot={"num_trials": 10},
        )
        agent_path = repo.save(agent, "nanozymes", metadata)

        # Load an agent
        agent, metadata = repo.load(agent_path)
        ```
    """

    def __init__(self, agents_dir: Path):
        """Initialize the agent repository.

        Args:
            agents_dir: Directory where agents are stored.
        """
        self.agents_dir = Path(agents_dir)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized AgentRepository at {self.agents_dir}")

    def save(
        self,
        agent: Dict[str, Any],
        task_name: str,
        metadata: AgentMetadata,
        filename: Optional[str] = None,
    ) -> Path:
        """Save an agent with its metadata.

        Args:
            agent: The trained agent object.
            task_name: Name of the task.
            metadata: Agent metadata.
            filename: Optional custom filename.

        Returns:
            Path to the saved agent file.
        """
        return save_agent(
            agent=agent,
            task_name=task_name,
            agents_dir=self.agents_dir,
            metadata=metadata,
            filename=filename,
        )

    def load(
        self,
        agent_path: Path,
    ) -> Tuple[Dict[str, Any], AgentMetadata]:
        """Load an agent with its metadata.

        Args:
            agent_path: Path to the agent file.

        Returns:
            Tuple of (agent, metadata).
        """
        return load_agent(agent_path)

    def list_agents(
        self,
        task_name: Optional[str] = None,
        sort_by: str = "created_at",
    ) -> List[Path]:
        """List all agent files, optionally filtered by task.

        Args:
            task_name: Optional task name filter.
            sort_by: Sort criterion.

        Returns:
            List of agent file paths.
        """
        return list_agents(self.agents_dir, task_name=task_name, sort_by=sort_by)

    def get_latest(self, task_name: str) -> Optional[Path]:
        """Get the most recently created agent for a task.

        Args:
            task_name: Name of the task.

        Returns:
            Path to the latest agent, or None.
        """
        return get_latest_agent(self.agents_dir, task_name)

    def delete(self, agent_path: Path) -> None:
        """Delete an agent and its metadata.

        Args:
            agent_path: Path to the agent file.
        """
        delete_agent(agent_path)

    def get_agent_info(self, agent_path: Path) -> Dict[str, Any]:
        """Get information about an agent.

        Args:
            agent_path: Path to the agent file.

        Returns:
            Dictionary with agent information.
        """
        return get_agent_info(agent_path)
