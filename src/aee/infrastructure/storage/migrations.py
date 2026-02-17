"""Migration utilities for agent and data format evolution.

This module provides migration functions for handling backward compatibility
when agent or data formats change between versions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aee.shared.exceptions import InvalidDataFormatError

logger = logging.getLogger(__name__)


class AgentMigrator:
    """Handles migration of agent files between format versions.

    Example:
        ```python
        migrator = AgentMigrator()

        # Migrate an agent file
        if migrator.needs_migration(agent_path):
            migrator.migrate(agent_path)

        # Or migrate in-memory data
        data = migrator.migrate_data(agent_dict)
        ```
    """

    # Current agent format version
    CURRENT_VERSION = "1.0"

    def __init__(self) -> None:
        """Initialize the agent migrator."""
        self._migrations: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            # Add migration functions here when format changes
            # "0.9": self._migrate_v09_to_v10,
        }

    def detect_version(self, agent_data: Dict[str, Any]) -> Optional[str]:
        """Detect the format version of agent data.

        Args:
            agent_data: Agent data dictionary.

        Returns:
            Version string or None if undetectable.
        """
        # Check for explicit version field
        if "version" in agent_data:
            return agent_data["version"]

        # Infer version from structure
        # Current format (v1.0) has "dspy" key at root level
        if "dspy" in agent_data:
            return "1.0"

        # No version detected - needs migration
        return None

    def needs_migration(self, agent_path: Path) -> bool:
        """Check if an agent file needs migration.

        Args:
            agent_path: Path to agent JSON file.

        Returns:
            True if migration is needed, False otherwise.
        """
        try:
            with open(agent_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            version = self.detect_version(data)

            if version is None:
                logger.warning(f"Cannot detect version for {agent_path}")
                return True

            return version != self.CURRENT_VERSION

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Cannot read agent file {agent_path}: {e}")
            return True

    def migrate(self, agent_path: Path, backup: bool = True) -> Path:
        """Migrate an agent file to the current format.

        Args:
            agent_path: Path to agent JSON file.
            backup: Whether to create a backup before migration.

        Returns:
            Path to migrated agent file.

        Raises:
            InvalidDataFormatError: If migration fails.
        """
        # Load agent data
        with open(agent_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Migrate data
        migrated_data = self.migrate_data(data)

        # Create backup if requested
        if backup:
            backup_path = agent_path.with_suffix(agent_path.suffix + ".bak")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Created backup: {backup_path}")

        # Write migrated data
        with open(agent_path, "w", encoding="utf-8") as f:
            json.dump(migrated_data, f, indent=2)

        logger.info(f"Migrated agent {agent_path} to version {self.CURRENT_VERSION}")

        return agent_path

    def migrate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate agent data to the current format.

        Args:
            data: Agent data dictionary.

        Returns:
            Migrated data dictionary.

        Raises:
            InvalidDataFormatError: If migration fails.
        """
        version = self.detect_version(data)

        if version is None:
            # Try to handle as current format
            logger.warning("Undetectable version, attempting to use as-is")
            return data

        if version == self.CURRENT_VERSION:
            return data

        # Apply migrations sequentially
        current_version = version
        migrated_data = data.copy()

        while current_version != self.CURRENT_VERSION:
            if current_version not in self._migrations:
                logger.warning(
                    f"No migration path from {current_version} to {self.CURRENT_VERSION}. "
                    f"Using data as-is."
                )
                return migrated_data

            migrate_func = self._migrations[current_version]
            migrated_data = migrate_func(migrated_data)
            current_version = self.detect_version(migrated_data)

            if current_version is None:
                # Migration function should set version
                migrated_data["version"] = self.CURRENT_VERSION
                break

        return migrated_data

    # Migration functions (to be added when format changes)

    def _migrate_v09_to_v10(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Example migration function from v0.9 to v1.0.

        Args:
            data: Agent data in v0.9 format.

        Returns:
            Agent data in v1.0 format.
        """
        # This is a placeholder - implement when format changes
        data["version"] = "1.0"
        return data


class GroundTruthMigrator:
    """Handles migration of ground truth CSV files.

    This migrator handles changes in ground truth format, such as:
    - New required columns
    - Renamed columns
    - Deprecated columns
    """

    # Current ground truth format
    CURRENT_COLUMNS = {
        "nanozymes": [
            "filename",
            "formula",
            "activity",
            "syngony",
            "surface",
            "length",
            "width",
            "depth",
            "reaction_type",
            "km_value",
            "km_unit",
            "vmax_value",
            "vmax_unit",
            "ph",
            "temperature",
            "c_min",
            "c_max",
            "c_const",
            "c_const_unit",
            "ccat_value",
            "ccat_unit",
        ]
    }

    def __init__(self) -> None:
        """Initialize the ground truth migrator."""
        self._column_renames: Dict[str, Dict[str, str]] = {
            # Example: "nanozymes": {"old_name": "new_name"}
        }

    def detect_format(self, csv_path: Path) -> Optional[str]:
        """Detect the format version of a ground truth CSV.

        Args:
            csv_path: Path to CSV file.

        Returns:
            Task name or None if undetectable.
        """
        import pandas as pd

        try:
            df = pd.read_csv(csv_path, nrows=1)
            columns = set(df.columns)

            # Try to match against known formats
            for task_name, expected_columns in self.CURRENT_COLUMNS.items():
                if set(expected_columns).issubset(columns):
                    return task_name

            # Check for partial match
            for task_name, expected_columns in self.CURRENT_COLUMNS.items():
                overlap = len(columns.intersection(expected_columns))
                if overlap > len(expected_columns) * 0.5:
                    logger.warning(
                        f"Partial match for {task_name}: {overlap}/{len(expected_columns)} columns"
                    )
                    return task_name

            return None

        except Exception as e:
            logger.error(f"Cannot detect format for {csv_path}: {e}")
            return None

    def migrate(
        self,
        csv_path: Path,
        task_name: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Migrate a ground truth CSV to the current format.

        Args:
            csv_path: Path to input CSV file.
            task_name: Name of the task.
            output_path: Optional output path (default: overwrite input).

        Returns:
            Path to migrated CSV file.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Apply column renames
        if task_name in self._column_renames:
            df = df.rename(columns=self._column_renames[task_name])

        # Add missing columns with None
        expected_columns = self.CURRENT_COLUMNS.get(task_name, [])
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        # Reorder columns
        existing_columns = [c for c in expected_columns if c in df.columns]
        df = df[existing_columns]

        # Save
        target_path = output_path or csv_path
        df.to_csv(target_path, index=False)

        logger.info(f"Migrated ground truth {csv_path} for task {task_name}")

        return target_path


def migrate_all_agents(agents_dir: Path, dry_run: bool = True) -> List[Dict[str, Any]]:
    """Migrate all agents in a directory.

    Args:
        agents_dir: Directory containing agent JSON files.
        dry_run: If True, only report what would be migrated.

    Returns:
        List of migration reports.
    """
    migrator = AgentMigrator()
    reports = []

    for agent_file in agents_dir.glob("*.json"):
        try:
            needs_migration = migrator.needs_migration(agent_file)

            if needs_migration:
                report = {
                    "path": str(agent_file),
                    "action": "would_migrate" if dry_run else "migrated",
                    "status": "success",
                }

                if not dry_run:
                    try:
                        migrator.migrate(agent_file)
                    except Exception as e:
                        report["status"] = "failed"
                        report["error"] = str(e)

                reports.append(report)
            else:
                reports.append({
                    "path": str(agent_file),
                    "action": "skip",
                    "status": "up_to_date",
                })

        except Exception as e:
            reports.append({
                "path": str(agent_file),
                "action": "skip",
                "status": "error",
                "error": str(e),
            })

    return reports


def migrate_all_ground_truth(
    ground_truth_dir: Path,
    dry_run: bool = True,
) -> List[Dict[str, Any]]:
    """Migrate all ground truth files in a directory.

    Args:
        ground_truth_dir: Directory containing ground truth CSV files.
        dry_run: If True, only report what would be migrated.

    Returns:
        List of migration reports.
    """
    migrator = GroundTruthMigrator()
    reports = []

    for csv_file in ground_truth_dir.glob("*.csv"):
        try:
            task_name = migrator.detect_format(csv_file)

            if task_name:
                report = {
                    "path": str(csv_file),
                    "task": task_name,
                    "action": "would_migrate" if dry_run else "migrated",
                    "status": "success",
                }

                if not dry_run:
                    try:
                        migrator.migrate(csv_file, task_name)
                    except Exception as e:
                        report["status"] = "failed"
                        report["error"] = str(e)

                reports.append(report)
            else:
                reports.append({
                    "path": str(csv_file),
                    "action": "skip",
                    "status": "unknown_format",
                })

        except Exception as e:
            reports.append({
                "path": str(csv_file),
                "action": "skip",
                "status": "error",
                "error": str(e),
            })

    return reports
