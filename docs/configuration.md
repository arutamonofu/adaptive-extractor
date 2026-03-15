# Configuration Reference

Complete reference for AutoEvoExtractor configuration.

## Configuration File Structure

Configuration files are organized into the following structure:

```
config/
├── systems/                    # System configurations (experiments)
│   ├── example.yaml            # Example environment
│   ├── exp_high_trials.yaml    # Experiment with high num_trials
│   └── exp_low_temp.yaml       # Experiment with low temperature
│
├── initial_instructions/       # Initial instructions for DSPy optimization
│   ├── nanozymes_sota.txt
│   ├── nanozymes_base.txt
│   └── proteins_v1.txt
│
└── tasks/                      # Task definitions (what to extract)
    ├── nanozymes.yaml         # Nanozymes extraction task
    ├── proteins.yaml          # Proteins extraction task
    └── ...
```

**Key points:**
- **System configs** (`config/systems/*.yaml`): Define experiment parameters (LLM settings, optimization parameters, paths, and initial instruction)
- **Task configs** (`config/tasks/*.yaml`): Define extraction fields, validation rules, and CSV mapping
- **Initial instructions** (`config/initial_instructions/`): Starting prompts for DSPy optimization (part of experiment configuration)

## Configuration Loading

**YAML configuration file is REQUIRED.** There is no fallback to internal defaults.

Configuration is loaded from the following sources:

1. **YAML file** — specified via `--config` CLI argument or `AEE_ENV` environment variable
2. **Environment variables** (`.env`) — secrets (API keys) and infrastructure URLs only

CLI arguments like `--overwrite` are passed directly to use cases and do not override YAML values.

> ⚠️ **Configuration file is mandatory.** The application will fail with an error if no config file is provided.

---

## YAML Configuration

### Complete Example

```yaml
# config/systems/example.yaml

project:
  log_level: "INFO"

llm:
  student:
    use_ollama: true
    model: "mistral-small3.1-24b-128k:latest"
    temperature: 0.0
    timeout: 600
    max_retries: 5
    rate_limit_delay: 10.0
    top_p: 0.1
    repeat_penalty: 1.2
    repeat_last_n: 2048
    enable_cache: true
    ollama:
      num_ctx: 64000
      num_predict: 2048
      repeat_penalty: 1.2
      repeat_last_n: 2048
      stream: false
    non_ollama:
      max_tokens: 4096

  teacher:
    use_ollama: true
    model: "gpt-oss:120b"
    temperature: 0.5
    timeout: 600
    max_retries: 2
    rate_limit_delay: 10.0
    top_p: 0.9
    repeat_penalty: 1.1
    repeat_last_n: 512
    enable_cache: true
    ollama:
      num_ctx: 64000
      num_predict: 2048
      repeat_penalty: 1.1
      repeat_last_n: 512
      stream: false
    non_ollama:
      max_tokens: 8192

paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits/nanozymes.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"

parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"

optimization:
  total_load: 20
  train_split: 20
  num_candidates: 10
  num_trials: 70
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 10
  view_data_batch_size: 3
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: true

task:
  name: "nanozymes"
  initial_instruction_file: "config/initial_instructions/nanozymes_sota.txt"
  evaluation:
    compare_fields:
      - formula
      - activity
      - reaction_type
      - ph
      - temperature
      - surface
      - syngony
      - length
      - width
      - depth
      - km_value
      - vmax_value
      - c_min
      - c_max
      - c_const
      - ccat_value
      - km_unit
      - vmax_unit
      - c_const_unit
      - ccat_unit
    float_tolerance: 0.05

extraction:
  enable_cache: false

cache:
  disk_size_limit_bytes: 30000000000
  memory_max_entries: 1000000

circuit_breaker:
  failure_threshold: 8
  reset_timeout: 30.0
  half_open_max_calls: 1
```

### Using Configuration Files

**Via CLI:**
```bash
# Specify config file explicitly
python -m aee.interface.cli.parse --config my_config.yaml
python -m aee.interface.cli.extract --config my_config.yaml --agent my_agent.json
python -m aee.interface.cli.optimize --config my_config.yaml
```

**Via environment variable:**
```bash
# Set environment to use config/{env}.yaml
export AEE_ENV="example"
python -m aee.interface.cli.parse  # Uses config/example.yaml
```

---

## Configuration Sections

### Project Configuration

```yaml
project:
  log_level: "INFO"               # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### LLM Configuration

```yaml
llm:
  student:
    use_ollama: true              # Use Ollama (true) or API (false)
    model: "mistral-small3.1-24b-128k:latest"
    temperature: 0.0              # 0.0 for deterministic output
    timeout: 600                  # Request timeout (seconds)
    max_retries: 5                # Maximum retry attempts
    rate_limit_delay: 10.0        # Delay between API calls (seconds)
    top_p: 0.1                    # Nucleus sampling top-p parameter
    repeat_penalty: 1.2           # Penalty for repeated tokens
    repeat_last_n: 2048           # Tokens to consider for repeat penalty
    enable_cache: true            # Cache LLM responses

    # Ollama-specific settings (URL is set via OLLAMA_STUDENT_BASE_URL env var)
    ollama:
      num_ctx: 64000              # Context window size
      num_predict: 2048           # Max tokens to generate
      repeat_penalty: 1.2         # Penalty for repeated tokens
      repeat_last_n: 2048         # Tokens to consider for repeat penalty
      stream: false               # Enable streaming responses

    # Non-Ollama settings (API key is set via *_API_KEY env var)
    non_ollama:
      max_tokens: 4096            # Max tokens for API providers

  teacher:
    use_ollama: true              # Use Ollama (true) or API (false)
    model: "gpt-oss:120b"
    temperature: 0.5              # Higher temperature for evaluation diversity
    timeout: 600                  # Request timeout (seconds)
    max_retries: 2                # Maximum retry attempts
    rate_limit_delay: 10.0        # Delay between API calls (seconds)
    top_p: 0.9                    # Nucleus sampling top-p parameter
    repeat_penalty: 1.1           # Penalty for repeated tokens
    repeat_last_n: 512            # Tokens to consider for repeat penalty
    enable_cache: true            # Cache LLM responses

    # Ollama-specific settings (URL is set via OLLAMA_TEACHER_BASE_URL env var)
    ollama:
      num_ctx: 64000              # Context window size
      num_predict: 2048           # Max tokens to generate
      repeat_penalty: 1.1         # Penalty for repeated tokens
      repeat_last_n: 512          # Tokens to consider for repeat penalty
      stream: false               # Enable streaming responses

    # Non-Ollama settings (API key is set via *_API_KEY env var)
    non_ollama:
      max_tokens: 8192            # Max tokens for API providers
```

### Optimization Configuration

```yaml
optimization:
  total_load: 20                # Total number of samples to load for optimization
  train_split: 20               # Number of samples for training split
  num_candidates: 10            # Candidates per trial
  num_trials: 70                # Number of MIPROv2 trials
  max_bootstrapped_demos: 1     # Max bootstrapped examples
  max_labeled_demos: 1          # Max labeled examples
  minibatch: false              # Use minibatch evaluation during optimization
  minibatch_size: 10            # Size of minibatch for evaluation
  view_data_batch_size: 3       # Batch size for viewing data samples
  metric_threshold: 1.0         # Threshold metric value for optimization stopping
  init_temperature: 0.5         # Initial temperature for candidate generation
  random_seed: 42               # Random seed for reproducibility
  use_cache: true               # Cache during optimization
  verbose: true                 # Enable verbose logging during optimization
```

> **Note:** All fields in the `optimization` section are **required**.

### Paths Configuration

```yaml
paths:
  pdf_dir: "data/pdf"                         # Input PDFs
  parsed_dir: "data/parsed"                   # Parsed JSON output
  ground_truth_dir: "data/ground_truth"       # Training CSV
  splits_file: "data/splits/nanozymes.json"   # Data splits (REQUIRED)
  agents_dir: "data/agents"                   # Trained agents
  extractions_dir: "data/extractions"         # Extraction results
```

### Task Configuration

Task configuration in YAML uses a nested structure under `task.evaluation.*`:

```yaml
# config/systems/example.yaml
task:
  name: "nanozymes"
  initial_instruction_file: "config/initial_instructions/nanozymes_sota.txt"  # REQUIRED (relative to project root)
  evaluation:
    compare_fields:            # Fields for evaluation - REQUIRED
      - formula
      - activity
      - syngony
      - surface
      - length
      - width
      - depth
      - reaction_type
      - km_value
      - km_unit
      - vmax_value
      - vmax_unit
      - ph
      - temperature
      - c_min
      - c_max
      - c_const
      - c_const_unit
      - ccat_value
      - ccat_unit
    float_tolerance: 0.05      # 5% tolerance for floats - REQUIRED
```

**Fields:**
- `name` — Task identifier (must match task config in `config/tasks/{task_name}.yaml`)
- `initial_instruction_file` — Path to initial instruction file for DSPy optimization, relative to project root (**required**)
- `evaluation.compare_fields` — List of field names used for evaluation during optimization (**required**)
- `evaluation.float_tolerance` — Tolerance for floating-point comparisons (0.0 to 1.0) (**required**)

### Parsing Configuration

```yaml
parsing:
  parser: "marker"                # Required: "marker" or "gemini"
  overwrite: false                # Optional: Overwrite existing parsed files (default: false)

  # Marker settings (required if parser: marker)
  marker:
    device: "cpu"                 # "cpu" or "cuda"

  # Gemini settings (required if parser: gemini)
  gemini:
    model_name: "gemini-3-flash-preview"  # Gemini model for PDF-to-Markdown
    upload_timeout: 300           # Timeout for file upload in seconds
    safety_settings: true         # Enable safety settings for Gemini API
```

> **Note:** All fields in `parsing.marker` are **required** when using Marker. The `parsing.gemini` section is **required** when using Gemini parser. The `overwrite` field is optional (default: `false`).

> **Environment Variable:** For Gemini parser, set `GEMINI_API_KEY` in `.env` file.

### Extraction Configuration

```yaml
extraction:
  enable_cache: false             # Enable LLM response caching during extraction
```

> **Note:** The `extraction` section has a single optional field (`enable_cache`).

### Cache Configuration

```yaml
cache:
  disk_size_limit_bytes: 30000000000  # Maximum disk cache size in bytes (30 GB)
  memory_max_entries: 1000000         # Maximum number of entries in memory cache
```

> **Note:** All fields in the `cache` section are **required**.

### Circuit Breaker Configuration

```yaml
circuit_breaker:
  failure_threshold: 8            # Number of failures before opening circuit
  reset_timeout: 30.0             # Seconds to wait before attempting reset (half-open state)
  half_open_max_calls: 1          # Maximum test calls allowed in half-open state
```

> **Note:** All fields in the `circuit_breaker` section are **required**.

---

## Environment Variables

The following environment variables are supported (set in `.env`):

### Required Variables

**For Ollama users:**
```bash
export OLLAMA_STUDENT_BASE_URL="http://localhost:11434"
export OLLAMA_TEACHER_BASE_URL="http://localhost:11434"
```

**For non-Ollama API users:**
```bash
export OPENAI_API_KEY="sk-..."
# OR
export ANTHROPIC_API_KEY="sk-ant-..."
# OR
export GEMINI_API_KEY="..."  # For LLM or Gemini parser
```

### Gemini Parser

**For Gemini PDF parser users:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

> **Note:** This is required when `parsing.parser: gemini` in your YAML config.

### Infrastructure

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"    # MLflow tracking URI
export DSPY_CACHE_DIR="${HOME}/.cache/dspy"         # DSPy cache directory
```

### Environment Selection

```bash
# Select environment-specific config (e.g., config/example.yaml, config/experiment.yaml)
export AEE_ENV="example"
```

> **Note:** All other configuration (LLM models, optimization parameters, paths, etc.) must be set in YAML configuration files.
