"""Integration tests for generate_manual_agent.py flow.

Tests cover:
- Manual agent generation with valid data
- Empty train_manual split handling
- Missing parsed files handling

Note: These tests use mock data to avoid actual LLM calls.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestManualAgentGeneration:
    """Integration tests for manual agent generation flow."""

    @pytest.fixture
    def manual_agent_setup(self, tmp_path: Path):
        """Setup test environment for manual agent generation tests."""
        # Create directories
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()

        # Create parsed documents
        for i in range(1, 4):
            doc_path = parsed_dir / f"paper{i}.md"
            doc_path.write_text(
                f"Sample scientific content about nanozymes from paper {i}.",
                encoding="utf-8",
            )

        # Create ground truth CSV
        gt_path = ground_truth_dir / "nanozymes.csv"
        header = (
            "filename,formula,activity,length,km,vmax,ph,temperature,"
            "substrate,cofactor,method,selectivity,stability,reference\n"
        )
        gt_path.write_text(
            header
            + "paper1,Fe3O4,peroxidase,10.0,0.05,100.0,7.0,25.0,TMB,None,UV-Vis,high,stable,Ref1\n"
            + "paper2,CuO,oxidase,20.0,0.08,150.0,7.5,30.0,ABTS,None,UV-Vis,medium,stable,Ref2\n"
            + "paper3,Au,catalase,15.0,0.06,120.0,7.2,28.0,H2O2,None,UV-Vis,high,unstable,Ref3\n",
            encoding="utf-8",
        )

        # Create data splits JSON with train_manual
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(
            '{"train_manual": ["paper1", "paper2"], "val": ["paper3"]}',
            encoding="utf-8",
        )

        return {
            "tmp_path": tmp_path,
            "parsed_dir": parsed_dir,
            "agents_dir": agents_dir,
            "ground_truth_dir": ground_truth_dir,
            "gt_path": gt_path,
            "splits_path": splits_path,
        }

    def test_manual_agent_generation_success(
        self, tmp_path: Path, manual_agent_setup
    ):
        """Test manual agent generation with valid data."""

        # Create config file
        config_file = manual_agent_setup["tmp_path"] / "config.yaml"
        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: {manual_agent_setup['parsed_dir']}
  ground_truth_dir: {manual_agent_setup['ground_truth_dir']}
  splits_file: {manual_agent_setup['splits_path']}
  agents_dir: {manual_agent_setup['agents_dir']}
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: config/initial_instructions/nanozymes_sota.txt
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    use_ollama: true
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
  teacher:
    use_ollama: true
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
parsing:
  parser: docling
  overwrite: false
  docling:
    device: cpu
    num_threads: 2
    do_ocr: false
    do_table_structure: false
    ocr_backend: onnxruntime
  marker:
    device: cpu
optimization:
  num_trials: 1
  train_split: 5
  total_load: 10
  random_seed: 42
  num_candidates: 2
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  verbose: false
  use_cache: false
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""", encoding="utf-8")

        # Mock LLM setup to avoid actual API calls
        with patch("scripts.generate_manual_agent.setup_student"):
            # Import and run main function
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

            from generate_manual_agent import main  # type: ignore[import-not-found]

            # Set sys.argv to simulate command line
            original_argv = sys.argv
            try:
                sys.argv = ["generate_manual_agent.py", "--config", str(config_file)]
                result = main()
            finally:
                sys.argv = original_argv

            # Should succeed
            assert result == 0

            # Check agent was created
            agents = list(manual_agent_setup["agents_dir"].glob("manual_*.json"))
            assert len(agents) > 0
            assert agents[0].exists()

    def test_empty_train_manual_split_returns_error(
        self, tmp_path: Path, manual_agent_setup
    ):
        """Test manual agent generation fails with empty train_manual split."""

        # Create splits with empty train_manual
        splits_path = manual_agent_setup["splits_path"]
        splits_path.write_text(
            '{"train_manual": [], "val": ["paper3"]}',
            encoding="utf-8",
        )

        # Create config file
        config_file = manual_agent_setup["tmp_path"] / "config.yaml"
        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: {manual_agent_setup['parsed_dir']}
  ground_truth_dir: {manual_agent_setup['ground_truth_dir']}
  splits_file: {splits_path}
  agents_dir: {manual_agent_setup['agents_dir']}
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: config/initial_instructions/nanozymes_sota.txt
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    use_ollama: true
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
  teacher:
    use_ollama: true
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
parsing:
  parser: docling
  overwrite: false
  docling:
    device: cpu
    num_threads: 2
    do_ocr: false
    do_table_structure: false
    ocr_backend: onnxruntime
  marker:
    device: cpu
optimization:
  num_trials: 1
  train_split: 5
  total_load: 10
  random_seed: 42
  num_candidates: 2
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  verbose: false
  use_cache: false
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""", encoding="utf-8")

        # Mock LLM setup
        with patch("scripts.generate_manual_agent.setup_student"):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

            from generate_manual_agent import main

            original_argv = sys.argv
            try:
                sys.argv = ["generate_manual_agent.py", "--config", str(config_file)]
                result = main()
            finally:
                sys.argv = original_argv

            # Should fail with exit code 1
            assert result == 1

            # No agent should be created
            agents = list(manual_agent_setup["agents_dir"].glob("manual_*.json"))
            assert len(agents) == 0

    def test_missing_parsed_files_partial_success(
        self, tmp_path: Path, manual_agent_setup
    ):
        """Test manual agent generation with some missing parsed files."""

        # Remove one parsed file to simulate missing data
        missing_file = manual_agent_setup["parsed_dir"] / "paper2.md"
        if missing_file.exists():
            missing_file.unlink()

        # Create config file
        config_file = manual_agent_setup["tmp_path"] / "config.yaml"
        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: {manual_agent_setup['parsed_dir']}
  ground_truth_dir: {manual_agent_setup['ground_truth_dir']}
  splits_file: {manual_agent_setup['splits_path']}
  agents_dir: {manual_agent_setup['agents_dir']}
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: config/initial_instructions/nanozymes_sota.txt
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    use_ollama: true
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
  teacher:
    use_ollama: true
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
parsing:
  parser: docling
  overwrite: false
  docling:
    device: cpu
    num_threads: 2
    do_ocr: false
    do_table_structure: false
    ocr_backend: onnxruntime
  marker:
    device: cpu
optimization:
  num_trials: 1
  train_split: 5
  total_load: 10
  random_seed: 42
  num_candidates: 2
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  verbose: false
  use_cache: false
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""", encoding="utf-8")

        # Mock LLM setup
        with patch("scripts.generate_manual_agent.setup_student"):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

            from generate_manual_agent import main

            original_argv = sys.argv
            try:
                sys.argv = ["generate_manual_agent.py", "--config", str(config_file)]
                result = main()
            finally:
                sys.argv = original_argv

            # Should succeed with partial data (at least one valid demo)
            assert result == 0

            # Agent should be created
            agents = list(manual_agent_setup["agents_dir"].glob("manual_*.json"))
            assert len(agents) > 0
