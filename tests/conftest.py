"""Pytest fixtures and configuration for AutoEvoExtractor tests."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aee.domain.tasks import get_global_registry


# ============================================================================
# Test Data Paths
# ============================================================================

TEST_DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# Task Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def nanozyme_task():
    """Get nanozyme task components.

    Uses module scope to avoid reloading task for every test.

    Returns:
        Dictionary with task components (config, experiment_model, output_model, signature, row_converter).
    """
    from aee.domain.tasks import load_task_from_yaml, get_task, register_config

    # Load and register task if not already registered
    registry = get_global_registry()
    if not registry.has("nanozymes"):
        yaml_path = Path("config/tasks/nanozymes.yaml")
        # Load task from YAML (without instruction file)
        config = load_task_from_yaml(yaml_path)
        # Set instruction file from system config location
        config.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
        register_config(config)

    return get_task("nanozymes")


@pytest.fixture
def task_config_dict() -> Dict[str, Any]:
    """Sample task configuration dictionary.

    Returns:
        Dictionary with task configuration.
    """
    return {
        "name": "nanozymes",
        "compare_fields": [
            "formula",
            "activity",
            "length",
            "km_value",
            "vmax_value",
            "ph",
            "temperature",
        ],
        "float_tolerance": 0.10,
        "initial_instruction_file": "config/initial_instructions/nanozymes_sota.txt",
    }


# ============================================================================
# Ground Truth Data Fixtures
# ============================================================================

@pytest.fixture
def sample_gt_csv(tmp_path: Path) -> Path:
    """Create a sample ground truth CSV file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to created CSV file.
    """
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text(
        "filename,formula,activity,length,km_value,vmax_value,ph,temperature,surface\n"
        "paper1.pdf,Fe3O4,peroxidase,10,0.05,100,7.0,25.0,naked\n"
        "paper2.pdf,CuO,oxidase,20,0.08,150,7.5,30.0,PVP\n"
        "paper3.pdf,ZnO,catalase,15,0.06,120,6.8,28.0,PEG\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def sample_gt_dataframe(sample_gt_csv: Path) -> pd.DataFrame:
    """Load sample ground truth as DataFrame.

    Args:
        sample_gt_csv: Path to sample CSV.

    Returns:
        DataFrame with ground truth data.
    """
    return pd.read_csv(sample_gt_csv)


@pytest.fixture
def nanozyme_experiments(nanozyme_task) -> List:
    """Create a list of sample nanozyme experiments.

    Args:
        nanozyme_task: Task fixture providing experiment_model.

    Returns:
        List of experiment instances.
    """
    experiment_model = nanozyme_task["experiment_model"]
    return [
        experiment_model(
            formula="Fe3O4",
            activity="peroxidase",
            length=10.0,
            km_value=0.05,
            vmax_value=100.0,
            ph=7.0,
            temperature=25.0,
            surface="naked",
        ),
        experiment_model(
            formula="CuO",
            activity="oxidase",
            length=20.0,
            km_value=0.08,
            vmax_value=150.0,
            ph=7.5,
            temperature=30.0,
            surface="PVP",
        ),
    ]


@pytest.fixture
def nanozyme_output(nanozyme_task, nanozyme_experiments):
    """Create a sample extraction output.

    Args:
        nanozyme_task: Task fixture providing output_model.
        nanozyme_experiments: List of experiments.

    Returns:
        Extraction output instance.
    """
    output_model = nanozyme_task["output_model"]
    return output_model(experiments=nanozyme_experiments)


# ============================================================================
# Data Split Fixtures
# ============================================================================

@pytest.fixture
def sample_splits_json(tmp_path: Path) -> Path:
    """Create a sample splits JSON file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to created JSON file.
    """
    splits_path = tmp_path / "splits.json"
    splits_data = {
        "train": ["paper1", "paper2"],
        "val": ["paper3"],
        "test": ["paper4", "paper5"],
    }
    splits_path.write_text(json.dumps(splits_data), encoding="utf-8")
    return splits_path


@pytest.fixture
def sample_splits_dict() -> Dict[str, List[str]]:
    """Sample splits dictionary.

    Returns:
        Dictionary with train/val/test splits.
    """
    return {
        "train": ["paper1", "paper2", "paper3"],
        "val": ["paper4"],
        "test": ["paper5"],
    }


# ============================================================================
# Agent Storage Fixtures
# ============================================================================

@pytest.fixture
def tmp_agents_dir(tmp_path: Path) -> Path:
    """Create a temporary agents directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to temporary directory.
    """
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    return agents_dir


@pytest.fixture
def sample_agent_dict() -> Dict[str, Any]:
    """Create a sample agent dictionary.

    Returns:
        Dictionary representing a mock agent.
    """
    return {
        "lm": {
            "model": "test-model",
            "type": "mock",
        },
        "traces": [],
        "settings": {
            "num_trials": 5,
            "metric": "f1",
        },
    }


@pytest.fixture
def sample_agent_metadata() -> Dict[str, Any]:
    """Create sample agent metadata.

    Returns:
        Dictionary with agent metadata.
    """
    from datetime import datetime

    return {
        "task_name": "nanozymes",
        "created_at": datetime.now().isoformat(),
        "model_version": "test-model-v1",
        "metrics": {"f1": 0.85, "precision": 0.82, "recall": 0.88},
        "config_snapshot": {"num_trials": 5},
    }


# ============================================================================
# Document Fixtures
# ============================================================================

@pytest.fixture
def sample_parsed_document(tmp_path: Path) -> Path:
    """Create a sample parsed document Markdown file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to created .md file.
    """
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    doc_path = parsed_dir / "paper1.md"
    doc_path.write_text("Sample parsed content from paper1.pdf.", encoding="utf-8")
    return doc_path


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response.

    Returns:
        Dictionary with mock response data.
    """
    return {
        "extracted_data": {
            "experiments": [
                {
                    "formula": "Fe3O4",
                    "activity": "peroxidase",
                    "length": 10.0,
                    "km_value": 0.05,
                    "vmax_value": 100.0,
                    "ph": 7.0,
                    "temperature": 25.0,
                }
            ]
        }
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_task_registry():
    """Automatically reset task registry after each test.

    This fixture ensures clean state between tests by clearing
    the global task registry before and after each test.
    """
    from aee.domain.tasks import get_global_registry

    registry = get_global_registry()
    # Clear before test
    registry.clear()
    yield
    # Clear after test
    registry.clear()


@pytest.fixture
def cleanup_temp_files():
    """Fixture to clean up temporary files after tests.

    Yields:
        Cleanup function.
    """
    temp_files = []

    def register_cleanup(path: Path):
        temp_files.append(path)
        return path

    yield register_cleanup

    # Cleanup
    for path in temp_files:
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


# ============================================================================
# Comparison Data Fixtures
# ============================================================================

@pytest.fixture
def matched_pairs_data(nanozyme_task):
    """Create sample data for testing matcher alignment.

    Args:
        nanozyme_task: Task fixture providing experiment_model.

    Returns:
        Dictionary with test data for matching.
    """
    experiment_model = nanozyme_task["experiment_model"]
    return {
        "preds": [
            experiment_model(formula="Fe3O4", activity="peroxidase", length=10.0),
            experiment_model(formula="CuO", activity="oxidase", length=20.0),
        ],
        "gts": [
            experiment_model(formula="Fe3O4", activity="peroxidase", length=10.0),
            experiment_model(formula="ZnO", activity="catalase", length=15.0),
        ],
    }


@pytest.fixture
def experiment_model(nanozyme_task):
    """Get experiment model from nanozyme task.

    Args:
        nanozyme_task: Task fixture.

    Returns:
        Experiment model class.
    """
    return nanozyme_task["experiment_model"]


@pytest.fixture
def output_model(nanozyme_task):
    """Get output model from nanozyme task.

    Args:
        nanozyme_task: Task fixture.

    Returns:
        Output model class.
    """
    return nanozyme_task["output_model"]


@pytest.fixture
def row_converter(nanozyme_task):
    """Get row converter from nanozyme task.

    Args:
        nanozyme_task: Task fixture.

    Returns:
        Row converter function.
    """
    return nanozyme_task["row_converter"]


# ============================================================================
# Fixtures for generate_manual_agent.py tests
# ============================================================================

@pytest.fixture
def mock_settings_for_manual_agent():
    """Create mock Settings object for manual agent generation testing.

    Returns:
        MagicMock with settings for manual agent generation.
    """
    mock = MagicMock()
    mock.task.name = "nanozymes"
    mock.task.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
    mock.paths.parsed_dir = Path("data/parsed")
    mock.paths.agents_dir = Path("data/agents")
    mock.paths.splits_file = Path("data/splits.json")
    mock.paths.ground_truth_dir = Path("data/ground_truth")
    return mock


@pytest.fixture
def splits_with_train_manual(tmp_path: Path) -> Path:
    """Create a sample splits JSON file with train_manual split.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to created JSON file.
    """
    splits_path = tmp_path / "splits.json"
    splits_data = {
        "train": ["paper1", "paper2", "paper3"],
        "train_manual": ["paper1", "paper2"],
        "val": ["paper4"],
        "test": ["paper5"],
    }
    splits_path.write_text(json.dumps(splits_data), encoding="utf-8")
    return splits_path


@pytest.fixture
def splits_without_train_manual(tmp_path: Path) -> Path:
    """Create a sample splits JSON file without train_manual split.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to created JSON file.
    """
    splits_path = tmp_path / "splits.json"
    splits_data = {
        "train": ["paper1", "paper2", "paper3"],
        "val": ["paper4"],
        "test": ["paper5"],
    }
    splits_path.write_text(json.dumps(splits_data), encoding="utf-8")
    return splits_path


@pytest.fixture
def gt_csv_for_manual(tmp_path: Path) -> Path:
    """Create a sample ground truth CSV file for manual agent testing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to created CSV file.
    """
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text(
        "filename,formula,activity,length,km_value,vmax_value,ph,temperature,surface\n"
        "paper1,Fe3O4,peroxidase,10,0.05,100,7.0,25.0,naked\n"
        "paper2,CuO,oxidase,20,0.08,150,7.5,30.0,PVP\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def parsed_docs_for_manual(tmp_path: Path) -> Path:
    """Create sample parsed markdown documents for manual agent testing.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to parsed directory.
    """
    parsed_dir = tmp_path / "parsed" / "train" / "manual"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # Create sample markdown files
    (parsed_dir / "paper1.md").write_text(
        "# Sample Document 1\n\nThis is a test document with Fe3O4 nanozymes.",
        encoding="utf-8",
    )
    (parsed_dir / "paper2.md").write_text(
        "# Sample Document 2\n\nThis is a test document with CuO nanozymes.",
        encoding="utf-8",
    )

    return parsed_dir.parent.parent  # Return parsed_dir root


# ============================================================================
# Configuration File Fixtures
# ============================================================================

CONFIG_DATA_DIR = TEST_DATA_DIR / "configs"


@pytest.fixture
def minimal_config_path() -> Path:
    """Get path to minimal valid configuration file.

    Returns:
        Path to minimal_config.yaml
    """
    return CONFIG_DATA_DIR / "minimal_config.yaml"


@pytest.fixture
def llm_config_template_path() -> Path:
    """Get path to LLM configuration template file.

    Returns:
        Path to llm_config_template.yaml
    """
    return CONFIG_DATA_DIR / "llm_config_template.yaml"


@pytest.fixture
def minimal_config_with_custom_paths(tmp_path: Path, minimal_config_path: Path) -> Path:
    """Create a copy of minimal config with custom paths for isolated testing.

    Args:
        tmp_path: Pytest temporary directory.
        minimal_config_path: Path to minimal config template.

    Returns:
        Path to copied config file.
    """
    import shutil
    config_copy = tmp_path / "config.yaml"
    shutil.copy(minimal_config_path, config_copy)
    return config_copy


@pytest.fixture
def config_with_instruction_file(tmp_path: Path, llm_config_template_path: Path) -> Path:
    """Create config with resolved instruction file path.

    Args:
        tmp_path: Pytest temporary directory.
        llm_config_template_path: Path to LLM config template.

    Returns:
        Path to config file with resolved instruction path.
    """

    # Create instruction file
    instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
    instruction_file.parent.mkdir(parents=True, exist_ok=True)
    instruction_file.write_text("Test instruction")

    # Load template
    config_content = llm_config_template_path.read_text(encoding="utf-8")
    config_content = config_content.replace("${INSTRUCTION_FILE_PATH}", str(instruction_file))

    # Save resolved config
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content, encoding="utf-8")

    return config_file
