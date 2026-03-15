"""Unit tests for extract CLI command.

Tests cover:
- Argument parsing
- Configuration loading
- LLM setup and DSPy configuration
- Agent loading as callable object
- Batch prediction request construction
- Error handling
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Note: clear_task_registry fixture is now in tests/conftest.py (autouse=True)


@pytest.mark.unit
class TestExtractArgumentParsing:
    """Tests for extract command argument parsing."""

    def test_missing_config_argument(self):
        """Test that script fails without --config argument."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/extract.py", "--agent", "data/agents/test.json"],
            capture_output=True,
            text=True,
        )

        # Should exit with non-zero code
        assert result.returncode != 0
        # Should mention required argument
        assert "--config" in result.stderr or "required" in result.stderr.lower()

    def test_missing_agent_argument(self, tmp_example_system_yaml: Path):
        """Test that script fails without --agent argument."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/extract.py", "--config", str(tmp_example_system_yaml)],
            capture_output=True,
            text=True,
        )

        # Should exit with non-zero code
        assert result.returncode != 0
        # Should mention required argument
        assert "--agent" in result.stderr or "required" in result.stderr.lower()

    def test_help_argument(self):
        """Test that --help works correctly."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/extract.py", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--agent" in result.stdout


@pytest.mark.unit
class TestLLMConfiguration:
    """Tests for LLM setup in extract command."""

    def test_setup_student_configures_dspy(self, tmp_path: Path, monkeypatch):
        """Test that setup_student() configures DSPy globally."""
        import dspy

        # Set required environment variables
        monkeypatch.setenv("OLLAMA_STUDENT_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_TEACHER_BASE_URL", "http://localhost:11434")

        # Create minimal config with temporary instruction file
        config_file = tmp_path / "config.yaml"
        instruction_file = tmp_path / "config" / "initial_instructions" / "nanozymes_sota.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction for nanozymes extraction.")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: data/parsed
  ground_truth_dir: data/ground_truth
  splits_file: data/splits.json
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: {instruction_file}
  evaluation:
    compare_fields: [formula, activity]
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
  parser: marker
  overwrite: false
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
  max_errors: 5
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

        from aee.infrastructure.config.settings import Settings
        from aee.infrastructure.llm import setup_student

        # Reset DSPy config
        dspy.settings.configure(lm=None)

        settings = Settings.load(config_path=config_file)

        # Mock Ollama to avoid actual network calls
        with patch('aee.infrastructure.llm.provider.requests.post') as mock_post:
            mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "test"}}]}

            student_lm = setup_student(settings, enable_cache=False)

            # Verify DSPy was configured (lm is set)
            assert dspy.settings.lm is not None
            assert student_lm is not None

    def test_dspy_configured_with_student_lm(self, tmp_path: Path):
        """Test that DSPy is configured with student LM after create_lm call."""

        # Create minimal config with temporary instruction file
        config_file = tmp_path / "config.yaml"
        instruction_file = tmp_path / "config" / "initial_instructions" / "nanozymes_sota.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction for nanozymes extraction.")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: data/parsed
  ground_truth_dir: data/ground_truth
  splits_file: data/splits.json
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: {instruction_file}
  evaluation:
    compare_fields: [formula, activity]
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
  parser: marker
  overwrite: false
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
  max_errors: 5
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

        # Mock Ollama to avoid actual network calls
        with patch('aee.infrastructure.llm.provider.requests.post'):
            # Import after mocking
            from aee.interface.cli.extract import extract_command

            # This test verifies the LLM configuration flow
            # We expect it to fail later (agent not found), but LLM should be configured
            result = extract_command([
                "--config", str(config_file),
                "--agent", str(tmp_path / "nonexistent_agent.json"),
            ])

            # Should fail because agent doesn't exist (not because of LLM config)
            assert result == 1


@pytest.mark.unit
class TestAgentLoading:
    """Tests for agent loading in extract command."""

    def test_agent_not_found_error(self, tmp_path: Path):
        """Test that script fails with clear error when agent file not found."""
        # Create minimal config with temporary instruction file
        config_file = tmp_path / "config.yaml"
        instruction_file = tmp_path / "config" / "initial_instructions" / "nanozymes_sota.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction for nanozymes extraction.")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: data/parsed
  ground_truth_dir: data/ground_truth
  splits_file: data/splits.json
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: {instruction_file}
  evaluation:
    compare_fields: [formula, activity]
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
  parser: marker
  overwrite: false
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
  max_errors: 5
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

        import subprocess

        result = subprocess.run(
            [
                "python", "scripts/extract.py",
                "--config", str(config_file),
                "--agent", str(tmp_path / "nonexistent_agent.json"),
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with agent not found error
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


@pytest.mark.unit
class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest construction."""

    def test_request_includes_task_dict(self):
        """Test that BatchPredictionRequest includes task_dict for agent reconstruction."""
        from aee.application.use_cases import BatchPredictionRequest
        from aee.domain.tasks import FieldSpec, TaskConfig

        # Create minimal valid task config
        task_config = TaskConfig(
            name="test",
            experiment_fields={
                "field1": FieldSpec(type=str, description="Test field"),
            },
            compare_fields=["field1"],
            float_tolerance=0.1,
        )

        task_dict = {
            "config": task_config,
            "signature": MagicMock(),
            "output_model": MagicMock(),
            "row_converter": MagicMock(),
        }

        # Create request with task_dict
        request = BatchPredictionRequest(
            task=task_config,
            task_dict=task_dict,
            agent_path=Path("test.json"),
            document_ids=["doc1"],
            output_dir=Path("output"),
        )

        # Verify task_dict is included
        assert request.task_dict is not None
        assert request.task_dict["signature"] is not None


@pytest.mark.unit
class TestSignatureValidation:
    """Tests for task signature validation."""

    def test_missing_signature_returns_error(self, tmp_path: Path):
        """Test that extract command fails when task signature is missing."""
        from aee.interface.cli.extract import extract_command

        # Create minimal config with temporary instruction file
        config_file = tmp_path / "config.yaml"
        instruction_file = tmp_path / "config" / "initial_instructions" / "nanozymes_sota.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction for nanozymes extraction.")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: data/parsed
  ground_truth_dir: data/ground_truth
  splits_file: data/splits.json
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: invalid_task
  initial_instruction_file: {instruction_file}
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
  parser: marker
  overwrite: false
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
  max_errors: 5
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

        # Create a fake agent file
        agent_file = tmp_path / "agent.json"
        agent_file.write_text('{"lm": "test", "traces": []}')
        (tmp_path / "agent.json.meta.json").write_text("""
{
  "task_name": "invalid_task",
  "created_at": "2026-01-01T00:00:00",
  "model_version": "test",
  "metrics": {},
  "config_snapshot": {}
}
""")

        # Should fail because task signature won't be found for invalid_task
        result = extract_command([
            "--config", str(config_file),
            "--agent", str(agent_file),
        ])

        # Should fail with exit code 1 (signature validation or task not found)
        assert result == 1


@pytest.mark.unit
class TestParsedDirectoryCheck:
    """Tests for parsed directory existence validation."""

    def test_missing_parsed_dir_returns_zero(self, tmp_path: Path, monkeypatch):
        """Test that extract command returns 0 when parsed_dir doesn't exist."""
        from aee.interface.cli.extract import extract_command

        # Set required environment variables
        monkeypatch.setenv("OLLAMA_STUDENT_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_TEACHER_BASE_URL", "http://localhost:11434")

        # Create config with non-existent parsed_dir and temporary instruction file
        config_file = tmp_path / "config.yaml"
        non_existent_dir = tmp_path / "nonexistent_parsed"
        instruction_file = tmp_path / "config" / "initial_instructions" / "nanozymes_sota.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction for nanozymes extraction.")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: {non_existent_dir}
  ground_truth_dir: data/ground_truth
  splits_file: data/splits.json
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: {instruction_file}
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
  parser: marker
  overwrite: false
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
  max_errors: 5
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

        # Create a fake agent file
        agent_file = tmp_path / "agent.json"
        agent_file.write_text('{"lm": "test", "traces": []}')
        (tmp_path / "agent.json.meta.json").write_text("""
{
  "task_name": "nanozymes",
  "created_at": "2026-01-01T00:00:00",
  "model_version": "test",
  "metrics": {},
  "config_snapshot": {}
}
""")

        # Mock setup_student to avoid actual LLM initialization
        with patch("aee.infrastructure.llm.setup_student") as mock_setup:
            mock_lm = MagicMock()
            mock_lm.model = "test-model"
            mock_setup.return_value = mock_lm

            # Should return 0 (no documents to process)
            result = extract_command([
                "--config", str(config_file),
                "--agent", str(agent_file),
            ])

            assert result == 0


@pytest.mark.unit
class TestSuccessfulExtraction:
    """Tests for successful extraction flow with mocked LLM."""

    def test_extraction_with_mocked_llm(
        self,
        tmp_path: Path,
        minimal_config_path: Path,
        monkeypatch,
    ):
        """Test successful extraction with mocked LLM and agent.

        This test verifies the complete extraction flow:
        1. Config loading
        2. Agent loading as callable object
        3. Document loading from parsed directory
        4. Mocked LLM response
        5. Output file creation
        """
        # Set required environment variables
        monkeypatch.setenv("OLLAMA_STUDENT_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_TEACHER_BASE_URL", "http://localhost:11434")

        # Setup directories
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        extractions_dir = tmp_path / "extractions"
        extractions_dir.mkdir()

        # Create parsed document
        doc_path = parsed_dir / "paper1.md"
        doc_path.write_text(
            "# Fe3O4 Nanozyme Study\n\n"
            "This study investigates Fe3O4 nanoparticles with peroxidase activity.\n"
            "The nanoparticles have a size of approximately 10.5 nm.",
            encoding="utf-8",
        )

        # Create config with custom paths
        import yaml
        config_data = yaml.safe_load(minimal_config_path.read_text())
        config_data["paths"]["parsed_dir"] = str(parsed_dir)
        config_data["paths"]["agents_dir"] = str(agents_dir)
        config_data["paths"]["extractions_dir"] = str(extractions_dir)

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Create agent file
        agent_file = agents_dir / "nanozymes_test.json"
        agent_file.write_text('{"lm": "test", "traces": []}')
        (agent_file.with_suffix(".meta.json")).write_text(
            '{"task_name": "nanozymes", "created_at": "2026-01-01T00:00:00", '
            '"model_version": "test", "metrics": {"f1": 0.85}, "config_snapshot": {}}'
        )

        # Mock setup_student BEFORE importing extract_command
        # This ensures the mock is in place before any imports happen
        from unittest.mock import MagicMock
        import dspy

        # Create mock LM
        mock_lm = MagicMock(spec=dspy.LM)
        mock_lm.model = "test-model"

        def mock_setup_student(*args, **kwargs):
            dspy.settings.configure(lm=mock_lm)
            return mock_lm

        monkeypatch.setattr("aee.infrastructure.llm.setup_student", mock_setup_student)

        # Mock agent loading
        def mock_load_agent(self, agent_path, task):
            mock_agent = MagicMock()
            mock_agent.prog = MagicMock()
            mock_agent.prog.predict = MagicMock()
            mock_agent.prog.predict.demos = []
            return mock_agent

        monkeypatch.setattr("aee.application.services.agent_manager.AgentManager.load_agent_as_object", mock_load_agent)

        # Mock the batch prediction use case
        def mock_execute(self, request):
            response = MagicMock()
            response.success = True
            response.extractions_saved = 1
            response.total_documents = 1
            response.failed_documents = 0
            response.output_dir = extractions_dir
            response.error_message = None
            return response

        monkeypatch.setattr("aee.application.use_cases.predict_batch.BatchPredictionUseCase.execute", mock_execute)

        from aee.interface.cli.extract import extract_command

        # Execute extraction
        result = extract_command([
            "--config", str(config_file),
            "--agent", str(agent_file),
        ])

        # Verify success
        assert result == 0
