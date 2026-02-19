# Configuration Reference

Complete reference for all configuration options in AutoEvoExtractor.

## Configuration File Structure

Configuration is stored in YAML files under `config/`. The system uses a hierarchical structure organized into sections.

## Configuration Priority

Settings are loaded in the following order (highest to lowest priority):

1. **Environment variables** (`.env` file via pydantic-settings, `AEE__*` overrides) — highest priority
2. **CLI arguments** (`--config`, `--overwrite`, `--agent`, etc.)
3. **YAML configuration files** (`config/default.yaml`, `config/<env>.yaml` via `AEE_ENV`)
4. **Internal defaults** (fallback values in code)

Environment-specific configs can be loaded via `AEE_ENV` variable (`dev`, `test`, `prod`), or explicitly via `--config` CLI argument.

> ⚠️ **Important:** 
> - **API keys** (OpenAI, Anthropic, Gemini) MUST be set via environment variables only — never in YAML files
> - **Infrastructure URLs** (Ollama, MLflow) should be in `.env` for environment portability
> - Most configuration fields are **required** and must be explicitly set in your YAML config. Only infrastructure paths have safe defaults.

## Required vs. Optional Fields

| Category | Required Fields | Optional Fields |
|----------|-----------------|-----------------|
| **LLM** | `use_ollama`, `model`, `temperature`, `timeout`, `max_retries`, `rate_limit_delay`, `top_p`, `repeat_penalty`, `repeat_last_n`, `enable_cache`, `ollama.*`, `non_ollama.max_tokens` | `non_ollama.api_key` (use env var instead) |
| **Optimization** | `total_load`, `train_split`, `num_candidates`, `num_trials`, `max_bootstrapped_demos`, `max_labeled_demos`, `minibatch`, `minibatch_size`, `view_data_batch_size`, `metric_threshold`, `init_temperature`, `random_seed`, `use_cache`, `verbose` | — |
| **Task** | `name`, `evaluation.float_tolerance`, `evaluation.compare_fields` | `initial_instruction_file` (or use YAML task config) |
| **Parsing** | `parser`, `overwrite` | — |
| **Paths** | `splits_file` | `pdf_dir`, `parsed_dir`, `ground_truth_dir`, `agents_dir`, `extractions_dir`, `logs_dir` |
| **Extraction** | `enable_cache` | — |
| **Cache** | `disk_size_limit_bytes`, `memory_max_entries` | — |
| **Circuit Breaker** | `failure_threshold`, `reset_timeout`, `half_open_max_calls` | — |
| **Project** | — | `name`, `log_level` |

## Complete Configuration Schema

| Section | Description |
|---------|-------------|
| [`project`](#project-settings) | Project name and logging level |
| [`paths`](#paths-configuration) | File system paths for data directories |
| [`llm`](#llm-configuration) | Student and teacher LLM settings |
| [`parsing`](#parsing-configuration) | Document parser settings |
| [`optimization`](#optimization-configuration) | DSPy MIPROv2 optimization parameters |
| [`task`](#task-configuration) | Task definition and evaluation settings |
| [`extraction`](#extraction-configuration) | Batch extraction behavior |
| [`cache`](#cache-configuration) | DSPy response cache settings |
| [`circuit_breaker`](#circuit-breaker-configuration) | LLM circuit breaker protection |
| [`mlflow_tracking_uri`](#environment-variables) | MLflow tracking URI (env var only) |
| [`dspy_cache_dir`](#environment-variables) | DSPy cache directory (env var only) |

### Project Settings

```yaml
project:
  name: "autoevoextractor"      # Project name for logging/tracking
  log_level: "INFO"              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Environment Variables:**
- `LOG_LEVEL` - Override logging level (higher priority than YAML)
- `PROJECT__NAME` - Override project name
- `PROJECT__LOG_LEVEL` - Override logging level

**Notes:**
- `log_level` can be set via `LOG_LEVEL` env var in `.env` (recommended for environment-specific settings)
- Default: `"INFO"`

---

### Paths Configuration

```yaml
paths:
  pdf_dir: "data/pdf"                    # Directory containing input PDFs
  parsed_dir: "data/parsed"              # Directory for parsed document JSON files
  ground_truth_dir: "data/ground_truth"  # Directory containing CSV ground truth files
  splits_file: "data/splits/nanozymes.json"  # Path to JSON file with data splits (REQUIRED)
  agents_dir: "data/agents"              # Directory for optimized agent files
  extractions_dir: "data/extractions"    # Directory for extraction output
  logs_dir: "logs"                       # Directory for log files
```

**Environment Variables:**
- `PATHS__PDF_DIR` - Override PDF input directory
- `PATHS__PARSED_DIR` - Override parsed output directory
- `PATHS__GROUND_TRUTH_DIR` - Override ground truth directory
- `PATHS__SPLITS_FILE` - Override splits file path
- `PATHS__AGENTS_DIR` - Override agents directory
- `PATHS__EXTRACTIONS_DIR` - Override extractions directory
- `PATHS__LOGS_DIR` - Override logs directory

**Notes:**
- `splits_file` is **required** — all other paths have sensible defaults
- Paths are resolved relative to the project root (where `pyproject.toml` is located)
- Absolute paths are also supported

---

### LLM Configuration

The system uses two LLMs: a **student** (for extraction) and a **teacher** (for optimization).

> ⚠️ **Security Note:** API keys for non-Ollama providers (`api_key`) MUST be set via environment variables only. Never store them in YAML configuration files.

#### Student LLM

```yaml
llm:
  student:
    use_ollama: true               # Use Ollama (true) or API provider (false)
    model: "mistral-small3.1-24b-128k:latest"  # Model name (Ollama) or identifier (API)
    temperature: 0.0               # Sampling temperature (0.0 = deterministic)
    timeout: 600                   # Request timeout in seconds
    max_retries: 5                 # Maximum retry attempts on failure
    rate_limit_delay: 10.0         # Delay between rate-limited requests (seconds)
    top_p: 0.1                     # Nucleus sampling parameter
    repeat_penalty: 1.2            # Repetition penalty
    repeat_last_n: 2048            # Tokens to consider for repetition penalty
    enable_cache: true             # Enable LLM response caching (default: true for optimization)

    # Ollama-specific settings (when use_ollama: true)
    ollama:
      ollama_base_url: "http://localhost:11434"  # Ollama server URL (fallback)
      num_ctx: 64000               # Context window size
      num_predict: 2048            # Maximum tokens to generate
      repeat_penalty: 1.2          # Repetition penalty (Ollama parameter)
      repeat_last_n: 2048          # Tokens for repetition penalty
      stream: false                # Stream responses

    # Non-Ollama settings (when use_ollama: false)
    non_ollama:
      # api_key is NOT set here - use environment variables instead
      max_tokens: 4096             # Maximum tokens to generate
```

**Environment Variables:**
- `LLM__STUDENT__USE_OLLAMA` - Use Ollama or API provider
- `LLM__STUDENT__MODEL` - Model name
- `LLM__STUDENT__TEMPERATURE` - Sampling temperature
- `LLM__STUDENT__TIMEOUT` - Request timeout
- `LLM__STUDENT__MAX_RETRIES` - Retry attempts
- `LLM__STUDENT__ENABLE_CACHE` - Enable LLM response caching (true/false)
- `OLLAMA_STUDENT_BASE_URL` - Ollama server URL for student (overrides YAML `ollama_base_url`)
- `LLM__STUDENT__OLLAMA__NUM_CTX` - Context window
- `LLM__STUDENT__OLLAMA__NUM_PREDICT` - Max generation tokens
- `LLM__STUDENT__NON_OLLAMA__MAX_TOKENS` - Max tokens (API providers)

**API Keys for Non-Ollama Providers (set in `.env` only):**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GEMINI_API_KEY` - Google Gemini API key

#### Teacher LLM

```yaml
llm:
  teacher:
    use_ollama: true               # Use Ollama (true) or API provider (false)
    model: "gpt-oss:120b"          # Model name
    temperature: 0.5               # Higher temperature for diversity
    timeout: 600
    max_retries: 2
    rate_limit_delay: 10.0
    top_p: 0.9
    repeat_penalty: 1.1
    repeat_last_n: 512
    enable_cache: true             # Enable LLM response caching (default: true for optimization)

    ollama:
      ollama_base_url: "http://localhost:11434"  # Fallback URL
      num_ctx: 64000
      num_predict: 2048
      repeat_penalty: 1.1
      repeat_last_n: 512
      stream: false

    non_ollama:
      # api_key is NOT set here - use environment variables instead
      max_tokens: 8192
```

**Environment Variables:** Same pattern as student, but use `LLM__TEACHER__*` prefix.
- `LLM__TEACHER__USE_OLLAMA` - Use Ollama or API provider
- `LLM__TEACHER__MODEL` - Model name
- `LLM__TEACHER__TEMPERATURE` - Sampling temperature
- `LLM__TEACHER__TIMEOUT` - Request timeout
- `LLM__TEACHER__MAX_RETRIES` - Retry attempts
- `LLM__TEACHER__ENABLE_CACHE` - Enable LLM response caching (true/false)
- `OLLAMA_TEACHER_BASE_URL` - Ollama server URL for teacher (overrides YAML)
- `LLM__TEACHER__NON_OLLAMA__MAX_TOKENS` - Max tokens (API providers)

**API Keys for Non-Ollama Providers (set in `.env` only):**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GEMINI_API_KEY` - Google Gemini API key

**Notes:**
- `OLLAMA_STUDENT_BASE_URL` and `OLLAMA_TEACHER_BASE_URL` in `.env` take precedence over YAML `ollama_base_url`
- If specific URLs are not set, falls back to `OLLAMA_BASE_URL` or `http://localhost:11434`

---

### Optimization Configuration

Settings for DSPy MIPROv2 optimization.

```yaml
optimization:
  total_load: 20                    # Total examples to load
  train_split: 20                   # Number of training examples
  num_candidates: 10                # Candidate prompts per trial
  num_trials: 70                    # Optimization trials to run
  max_bootstrapped_demos: 2         # Max bootstrapped demonstrations
  max_labeled_demos: 2              # Max labeled demonstrations
  minibatch: false                  # Use minibatch optimization
  minibatch_size: 10                # Minibatch size (if enabled)
  view_data_batch_size: 3           # Batch size for data viewing
  metric_threshold: 1.0             # Target metric threshold
  init_temperature: 0.5             # Initial temperature for optimization
  random_seed: 42                   # Random seed for reproducibility
  use_cache: true                   # Cache LLM responses
  verbose: true                     # Verbose logging during optimization
```

**Environment Variables:**
- `OPTIMIZATION__TOTAL_LOAD` - Total examples to load
- `OPTIMIZATION__TRAIN_SPLIT` - Training examples count
- `OPTIMIZATION__NUM_CANDIDATES` - Candidate prompts per trial
- `OPTIMIZATION__NUM_TRIALS` - Number of trials
- `OPTIMIZATION__MAX_BOOTSTRAPPED_DEMOS` - Max bootstrapped demos
- `OPTIMIZATION__MAX_LABELED_DEMOS` - Max labeled demos
- `OPTIMIZATION__MINIBATCH` - Enable minibatch (true/false)
- `OPTIMIZATION__MINIBATCH_SIZE` - Minibatch size
- `OPTIMIZATION__METRIC_THRESHOLD` - Target threshold
- `OPTIMIZATION__INIT_TEMPERATURE` - Initial temperature
- `OPTIMIZATION__RANDOM_SEED` - Random seed
- `OPTIMIZATION__USE_CACHE` - Enable caching (true/false)
- `OPTIMIZATION__VERBOSE` - Verbose mode (true/false)

---

### Parsing Configuration

Settings for document parsing.

```yaml
parsing:
  parser: "docling"                 # Parser to use: "docling" or "marker"
  overwrite: false                  # Overwrite existing parsed files

  # Docling-specific settings
  docling:
    device: "cpu"                   # Device: "cpu", "cuda", or "mps"
    num_threads: 4                  # Number of CPU threads
    do_ocr: true                    # Perform OCR on images
    do_table_structure: true        # Extract table structure
    ocr_backend: "onnxruntime"      # OCR backend: "onnxruntime", "torch", "openvino", "paddlepaddle"

  # Marker-specific settings
  marker:
    device: "cpu"                   # Device: "cpu" or "cuda"
```

**Environment Variables:**
- `PARSING__PARSER` - Parser type (docling/marker)
- `PARSING__OVERWRITE` - Overwrite existing files (true/false)
- `PARSING__DOCLING__DEVICE` - Docling device
- `PARSING__DOCLING__NUM_THREADS` - CPU threads
- `PARSING__DOCLING__DO_OCR` - Enable OCR (true/false)
- `PARSING__DOCLING__DO_TABLE_STRUCTURE` - Extract tables (true/false)
- `PARSING__DOCLING__OCR_BACKEND` - OCR backend for docling
- `PARSING__MARKER__DEVICE` - Marker device

**Notes:**
- Docling is the default parser
- `ocr_backend` options: `onnxruntime` (CPU), `torch` (GPU), `openvino` (Intel), `paddlepaddle`

---

### Task Configuration

Task-specific settings for evaluation.

```yaml
task:
  name: "nanozymes"                          # Task name (must match registered task)

  evaluation:
    float_tolerance: 0.05                    # Tolerance for float comparisons
    compare_fields:                          # Fields to compare for matching
      - "formula"
      - "activity"
      - "length"
      - "km_value"
      - "vmax_value"
```

**Environment Variables:**
- `TASK__NAME` - Task name
- `TASK__EVALUATION__FLOAT_TOLERANCE` - Float comparison tolerance

**Notes:**
- For YAML-based tasks, the instruction is defined in `src/aee/domain/tasks/<task_name>/task.yaml`
- See [YAML Task Configuration](#yaml-task-configuration) and [Initial Instructions](#initial-instructions) for details.

---

### Extraction Configuration

Settings for batch extraction behavior.

```yaml
extraction:
  enable_cache: false                        # Enable LLM response caching for extractions
```

**Environment Variables:**
- `EXTRACTION__ENABLE_CACHE` - Enable caching (true/false)

**Notes:**
- Caching is disabled by default for extractions to ensure fresh results
- Enable caching to speed up repeated extractions on the same documents
- Cache is stored in the LLM infrastructure layer

---

### Cache Configuration

Settings for DSPy LLM response caching (disk and memory).

```yaml
cache:
  disk_size_limit_bytes: 30000000000         # Maximum disk cache size (30 GB)
  memory_max_entries: 1000000                # Maximum in-memory cache entries
```

**Environment Variables:**
- `CACHE__DISK_SIZE_LIMIT_BYTES` - Maximum disk cache size in bytes
- `CACHE__MEMORY_MAX_ENTRIES` - Maximum number of in-memory cache entries

**Notes:**
- Disk cache persists between program runs
- Memory cache is faster but cleared on exit
- Both caches can be enabled simultaneously
- Cache is automatically used by all DSPy LLM calls
- Default cache directory: `~/.dspy_cache` (override via `DSPY_CACHE_DIR` env var)

---

### Circuit Breaker Configuration

Settings for circuit breaker protection against LLM API failures.

```yaml
circuit_breaker:
  failure_threshold: 8                       # Failures before opening circuit
  reset_timeout: 30.0                        # Seconds before attempting reset
  half_open_max_calls: 1                     # Max test calls in half-open state
```

**Environment Variables:**
- `CIRCUIT_BREAKER__FAILURE_THRESHOLD` - Number of failures before opening
- `CIRCUIT_BREAKER__RESET_TIMEOUT` - Reset timeout in seconds
- `CIRCUIT_BREAKER__HALF_OPEN_MAX_CALLS` - Max test calls in half-open state

**Notes:**
- Circuit breaker prevents cascade failures during LLM API outages
- States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
- When OPEN, requests fail immediately without calling the API
- After `reset_timeout`, transitions to HALF_OPEN for a test request
- Success in HALF_OPEN resets to CLOSED; failure returns to OPEN
- Default values: `failure_threshold=8`, `reset_timeout=30.0`, `half_open_max_calls=1`

---

### Environment Variables (Infrastructure)

These settings are **only** configured via environment variables (not in YAML):

```bash
# .env file
AEE_ENV=dev                              # Environment: dev, test, prod

# API Keys (required for non-Ollama providers)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ollama URLs (override YAML ollama_base_url)
OLLAMA_STUDENT_BASE_URL=https://localhost:11434
OLLAMA_TEACHER_BASE_URL=https://localhost:11434
OLLAMA_BASE_URL=https://localhost:11434  # Fallback if specific URLs not set

# MLflow Tracking
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# DSPy Cache (absolute path recommended)
DSPY_CACHE_DIR=${HOME}/.dspy_cache

# Logging (overrides YAML project.log_level)
LOG_LEVEL=INFO
```

**Notes:**
- `AEE_ENV` controls which environment-specific config to load (`config/<env>.yaml`)
- `OLLAMA_STUDENT_BASE_URL` and `OLLAMA_TEACHER_BASE_URL` take precedence over YAML values
- API keys are read directly from environment and never stored in YAML

---

### YAML Task Configuration

Tasks can be defined declaratively using YAML files. For a complete guide with step-by-step instructions and examples, see [Adding New Extraction Tasks](adding_tasks.md).

**Quick Reference:**

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| `name` | str | Task identifier (must match directory name) | Yes |
| `description` | str | Human-readable task description | Yes |
| `version` | str | Task version (e.g., "1.0.0") | No |
| `instruction_file` | str | Path to initial instruction file | Yes* |
| `initial_instruction` | str | Inline instruction (alternative to file) | Yes* |
| `fields` | object | Field definitions (see below) | Yes |
| `compare_fields` | list[str] | Fields for experiment matching | Yes |
| `float_tolerance` | float | Tolerance for float comparisons (e.g., 0.05) | Yes |
| `row_converter` | object | CSV column name mappings | No |

**Field Definition Properties:**

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| `type` | str | Python type: `str`, `int`, `float`, `bool` | Yes |
| `description` | str | Human-readable description for LLM | Yes |
| `required` | bool | Whether field is mandatory | No (default: true) |
| `default` | any | Default value for optional fields | No |
| `choices` | list[str] | Valid choices (creates Literal type) | No |
| `min_value` | float | Minimum for numeric fields | No |
| `max_value` | float | Maximum for numeric fields | No |
| `pattern` | str | Regex pattern for string fields | No |
| `alt_names` | list[str] | Alternative CSV column names | No |

**Location:** `src/aee/domain/tasks/<task_name>/task.yaml`

> **Note:** Either `instruction_file` or `initial_instruction` must be provided. See [Initial Instructions](#initial-instructions) below.

---

### Initial Instructions

The system uses **initial instructions** as a starting point for prompt optimization.

**Location:** `config/initial_instructions/` (plain `.txt` files)

**Prompts vs. Instructions:**
- **Instruction**: Base guidance you provide (what to extract, how to format)
- **Prompt**: Instruction + examples (DSPy MIPROv2 generates during optimization)

**Instruction File Format:**

Plain `.txt` files (not YAML) to avoid escaping issues:

```txt
You are a helpful assistant specializing in [domain]. Your task is to analyze scientific articles
and extract detailed information about [entity] experiments.

For each experiment mentioned in the text, extract:
- [Field 1] (required): Description
- [Field 2] (required): Description
- [Field 3] (optional): Description

IMPORTANT: Extract each experiment separately. Be precise with numerical values and units.
```

**Creating Custom Instructions:**

```bash
# 1. Copy existing instruction
cp config/initial_instructions/nanozymes_sota.txt config/initial_instructions/nanozymes_v2.txt

# 2. Edit the instruction text

# 3. Update config
# task:
#   initial_instruction_file: "config/initial_instructions/nanozymes_v2.txt"

# 4. Run optimization
python scripts/optimize.py --task nanozymes --config default.yaml
```

**Instruction Metadata (Reproducibility):**

Each optimized agent stores:
- `initial_instruction_file`: Path to the instruction file used
- `instruction_hash`: SHA256 hash (first 12 chars) of the instruction content

See [Adding New Extraction Tasks](adding_tasks.md#step-2-create-initial-instruction) for detailed guidance on writing effective instructions.

---

## Configuration Files

### Available Config Files

1. **`config/default.yaml`** - Production configuration
   - Reasonable defaults for production use
   - Balanced between quality and speed
   - Suitable for final experiments

2. **`config/<env>.yaml`** - Environment-specific configuration
   - `config/dev.yaml` for development
   - `config/test.yaml` for testing
   - `config/prod.yaml` for production
   - Loaded automatically based on `AEE_ENV` environment variable

3. **Custom config files** - Your own configuration
   - Use with `--config path/to/config.yaml` CLI argument
   - Highest priority (overrides environment config)

### Using Custom Config Files

```bash
# Use default config (or AEE_ENV-specific)
python scripts/optimize.py --task nanozymes

# Use specific environment (dev, test, prod)
export AEE_ENV=dev
python scripts/optimize.py --task nanozymes

# Use custom config file (highest priority)
python scripts/optimize.py --task nanozymes --config my_config.yaml
```

---

## CLI Commands

### optimize

Run agent optimization using DSPy MIPROv2.

```bash
python scripts/optimize.py --task <task_name> [options]
```

**Options:**
- `--config PATH` - Path to configuration YAML file (optional)
- `--run-name NAME` - Prefix for MLflow run name (e.g., "A1_high")
- `--no-mlflow` - Disable MLflow tracking

**Example:**
```bash
python scripts/optimize.py --task nanozymes --config default.yaml --run-name A1_test
```

### extract

Run batch extraction on documents using a trained agent.

```bash
python scripts/extract.py --agent <agent_path> [options]
```

**Options:**
- `--config PATH` - Path to configuration YAML file (optional)
- `--agent PATH` - Path to trained agent JSON file (required)

**Example:**
```bash
python scripts/extract.py --agent nanozymes_v1.json --config default.yaml
```

**Notes:**
- Agent path can be relative to `data/agents/` directory
- Processes all documents in `data/parsed/` directory
- Output saved to `data/extractions/` directory

---

## Configuration Best Practices

### Development
- Use `AEE_ENV=dev` for development-specific settings
- Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs
- Enable `OPTIMIZATION__VERBOSE=true` to see optimization progress
- Use small `OPTIMIZATION__NUM_TRIALS` (3-5) for testing
- Create `config/dev.yaml` with reduced trial counts

### Production
- Use `AEE_ENV=prod` or `config/prod.yaml` for production
- Set `OPTIMIZATION__NUM_TRIALS` to 20+ for best results
- Disable `OPTIMIZATION__VERBOSE` to reduce log clutter
- Set `LOG_LEVEL=INFO` or `WARNING` in `.env`
- Enable `OPTIMIZATION__USE_CACHE=true` to save costs
- Enable `EXTRACTION__ENABLE_CACHE=true` for repeated extractions

### GPU Acceleration
```yaml
parsing:
  docling:
    device: "cuda"  # Use GPU for parsing
    ocr_backend: "torch"  # Use torch OCR backend for GPU
  marker:
    device: "cuda"
```

### Using Custom Ollama Server
```bash
# Set in .env file (recommended)
OLLAMA_STUDENT_BASE_URL="https://my-ollama-server.com"
OLLAMA_TEACHER_BASE_URL="https://my-ollama-server.com"

# Or use fallback URL
OLLAMA_BASE_URL="https://my-ollama-server.com"
```

### Using OpenAI or Anthropic
```bash
# Set API keys in .env (NEVER in YAML)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export GEMINI_API_KEY="..."

# Configure in YAML
# config/default.yaml:
llm:
  student:
    use_ollama: false
    model: "gpt-4"
    non_ollama:
      max_tokens: 4096
  
  teacher:
    use_ollama: false
    model: "gpt-4"
    non_ollama:
      max_tokens: 8192
```

---

## Troubleshooting Configuration Issues

### Configuration Not Loading

**Problem:** Changes to config file not taking effect

**Solutions:**
1. Verify config priority: CLI `--config` > `AEE_ENV` > `default.yaml`
2. Check for syntax errors in YAML (indentation matters!)
3. Verify environment variables aren't overriding your settings
4. Check file permissions
5. Ensure `--config` path is correct (relative to `config/` or absolute)

### Environment Variables Not Working

**Problem:** Environment variables not overriding config

**Solutions:**
1. Use double underscore notation: `LLM__STUDENT__MODEL`
2. Export variables before running: `export VAR=value`
3. Check variable names match config structure exactly
4. Boolean values should be lowercase: `true`/`false`
5. For `.env` file: ensure it's in project root
6. Restart Python process after changing `.env`

### API Keys Not Working

**Problem:** Non-Ollama providers report authentication errors

**Solutions:**
1. **Never** set `api_key` in YAML — use environment variables only
2. Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` in `.env`
3. Verify API key is valid: `echo $OPENAI_API_KEY`
4. Check `use_ollama: false` in YAML config
5. Ensure `non_ollama.max_tokens` is set

### Ollama Connection Issues

**Problem:** Cannot connect to Ollama server

**Solutions:**
1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check URL in `.env`: `OLLAMA_STUDENT_BASE_URL`, `OLLAMA_TEACHER_BASE_URL`
3. Verify firewall settings allow connections
4. Check Ollama logs for errors
5. Ensure model exists: `ollama list`

### Model Not Found

**Problem:** Specified model not available

**Solutions:**
1. List available models: `ollama list`
2. Pull model: `ollama pull <model_name>`
3. Verify model name spelling in config
4. Check Ollama server has the model
5. For long context models, verify `num_ctx` fits in VRAM

### Circuit Breaker Open

**Problem:** Requests fail immediately with "Circuit breaker is OPEN"

**Solutions:**
1. Check LLM server is running and responsive
2. Wait for `reset_timeout` seconds (default: 30s)
3. Reduce `failure_threshold` if transient errors are common
4. Increase `timeout` if requests are timing out
5. Check network connectivity to LLM server

### Task YAML Not Loading

**Problem:** Custom task YAML not being used

**Solutions:**
1. Ensure file is at `src/aee/domain/tasks/<task_name>/task.yaml`
2. Verify `task.name` in config matches directory name
3. Check YAML syntax (use `yamllint` or online validator)
4. Validate all required fields are present
5. Ensure instruction is provided (inline or via `instruction_file`)
