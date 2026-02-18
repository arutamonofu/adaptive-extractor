# Configuration Reference

Complete reference for all configuration options in AutoEvoExtractor.

## Configuration File Structure

Configuration is stored in YAML files under `config/`. The system uses a hierarchical structure organized into sections.

## Configuration Priority

Settings are loaded in the following order (highest to lowest priority):

1. **Environment variables** (`.env` file or shell exports)
2. **YAML configuration files** (`config/default.yaml`, `config/*.yaml`)
3. **Internal defaults** (fallback values in code)

> ⚠️ **Important:** Most configuration fields are **required** and must be explicitly set in your YAML config or via environment variables. Only infrastructure paths have safe defaults.

## Required vs. Optional Fields

| Category | Required Fields | Optional Fields |
|----------|-----------------|-----------------|
| **LLM** | `use_ollama`, `model`, `temperature`, `timeout`, `max_retries`, `rate_limit_delay`, `top_p`, `repeat_penalty`, `repeat_last_n`, `enable_cache`, `ollama.*`, `non_ollama.max_tokens` | — |
| **Optimization** | `total_load`, `train_split`, `num_candidates`, `num_trials`, `max_bootstrapped_demos`, `max_labeled_demos`, `minibatch`, `minibatch_size`, `view_data_batch_size`, `metric_threshold`, `init_temperature`, `random_seed`, `use_cache`, `verbose` | — |
| **Task** | `name`, `evaluation.float_tolerance`, `evaluation.compare_fields` | `initial_instruction_file` |
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

### Project Settings

```yaml
project:
  name: "autoevoextractor"      # Project name for logging/tracking
  log_level: "INFO"              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Environment Variables:**
- `PROJECT__NAME` - Override project name
- `PROJECT__LOG_LEVEL` - Override logging level

---

### Paths Configuration

```yaml
paths:
  pdf_dir: "data/pdf"                    # Directory containing input PDFs
  parsed_dir: "data/parsed"              # Directory for parsed document JSON files
  ground_truth_dir: "data/ground_truth"  # Directory containing CSV ground truth files
  splits_dir: "data/splits"              # Directory containing task-specific split files
  agents_dir: "data/agents"              # Directory for optimized agent files
  extractions_dir: "data/extractions"    # Directory for extraction output
  logs_dir: "logs"                       # Directory for log files
```

**Environment Variables:**
- `PATHS__PDF_DIR` - Override PDF input directory
- `PATHS__PARSED_DIR` - Override parsed output directory
- `PATHS__GROUND_TRUTH_DIR` - Override ground truth directory
- `PATHS__SPLITS_DIR` - Override splits directory
- `PATHS__AGENTS_DIR` - Override agents directory
- `PATHS__EXTRACTIONS_DIR` - Override extractions directory
- `PATHS__LOGS_DIR` - Override logs directory

---

### LLM Configuration

The system uses two LLMs: a **student** (for extraction) and a **teacher** (for optimization).

#### Student LLM

```yaml
llm:
  student:
    use_ollama: true               # Use Ollama (true) or API provider (false)
    model: "llama3.2:3b"          # Model name (Ollama) or identifier (API)
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
      ollama_base_url: "http://localhost:11434"  # Ollama server URL
      num_ctx: 128000              # Context window size
      num_predict: 4096            # Maximum tokens to generate
      repeat_penalty: 1.2          # Repetition penalty (Ollama parameter)
      stream: true                 # Stream responses

    # Non-Ollama settings (when use_ollama: false)
    non_ollama:
      api_key: null                # API key (should use env var)
      max_tokens: 4096             # Maximum tokens to generate
```

**Environment Variables:**
- `LLM__STUDENT__USE_OLLAMA` - Use Ollama or API provider
- `LLM__STUDENT__MODEL` - Model name
- `LLM__STUDENT__TEMPERATURE` - Sampling temperature
- `LLM__STUDENT__TIMEOUT` - Request timeout
- `LLM__STUDENT__MAX_RETRIES` - Retry attempts
- `LLM__STUDENT__ENABLE_CACHE` - Enable LLM response caching (true/false)
- `LLM__STUDENT__OLLAMA__OLLAMA_BASE_URL` - Ollama server URL
- `LLM__STUDENT__OLLAMA__NUM_CTX` - Context window
- `LLM__STUDENT__OLLAMA__NUM_PREDICT` - Max generation tokens
- `LLM__STUDENT__NON_OLLAMA__API_KEY` - API key for non-Ollama providers
- `LLM__STUDENT__NON_OLLAMA__MAX_TOKENS` - Max tokens (API providers)

#### Teacher LLM

```yaml
llm:
  teacher:
    use_ollama: true               # Use Ollama (true) or API provider (false)
    model: "llama3.2:3b"          # Model name
    temperature: 0.5               # Higher temperature for diversity
    timeout: 600
    max_retries: 2
    rate_limit_delay: 10.0
    top_p: 0.9
    repeat_penalty: 1.1
    repeat_last_n: 512
    enable_cache: true             # Enable LLM response caching (default: true for optimization)

    ollama:
      ollama_base_url: "http://localhost:11434"
      num_ctx: 100000
      num_predict: 8192
      repeat_penalty: 1.1
      stream: true

    non_ollama:
      api_key: null
      max_tokens: 8192
```

**Environment Variables:** Same pattern as student, but use `LLM__TEACHER__*` prefix.
- `LLM__TEACHER__ENABLE_CACHE` - Enable LLM response caching (true/false)

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
- `PARSING__MARKER__DEVICE` - Marker device

---

### Task Configuration

Task-specific settings including the initial instruction for prompt optimization.

```yaml
task:
  name: "nanozymes"                          # Task name (must match registered task)
  initial_instruction_file: "config/initial_instructions/nanozymes_sota.txt"  # Initial instruction file

  evaluation:
    float_tolerance: 0.05                    # Tolerance for float comparisons
    compare_fields:                          # Fields to compare for matching
      - "formula"
      - "activity"
      - "length"
      - "km_value"
      - "vmax_value"
      # ... more fields ...
```

**Environment Variables:**
- `TASK__NAME` - Task name
- `TASK__INITIAL_INSTRUCTION_FILE` - Path to initial instruction file (relative to project root)
- `TASK__EVALUATION__FLOAT_TOLERANCE` - Float comparison tolerance

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
  failure_threshold: 5                       # Failures before opening circuit
  reset_timeout: 60.0                        # Seconds before attempting reset
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

---

### Initial Instructions

The system uses **initial instructions** as a starting point for prompt optimization. Instructions are stored as plain text files in `config/initial_instructions/`.

**Prompts vs. Instructions:**
- **Instruction**: The base guidance you provide (what to extract, how to format). This is what you configure.
- **Prompt**: Instruction + examples (what DSPy MIPROv2 generates during optimization). This is what gets optimized.

The system optimizes both the instruction and examples to create effective prompts for your task.

**Instruction File Format:**

Instructions are stored as plain `.txt` files (not YAML) to avoid escaping issues:

```txt
You are a helpful assistant specializing in [domain]. Your task is to analyze scientific articles 
and extract detailed information about [entity] experiments.

For each experiment mentioned in the text, extract:
- [Field 1] (required): Description
- [Field 2] (required): Description  
- [Field 3] (optional): Description

IMPORTANT: Extract each experiment separately. Be precise with numerical values and units.
```

**Available Instructions:**

| File | Description |
|------|-------------|
| `initial_instructions/nanozymes_sota.txt` | Default nanozyme extraction instruction |

**Creating Custom Instructions:**

1. Create a new file in `config/initial_instructions/`:
   ```bash
   cp config/initial_instructions/nanozymes_sota.txt config/initial_instructions/nanozymes_v2.txt
   ```

2. Edit the instruction text (no YAML escaping needed!)

3. Update your config:
   ```yaml
   task:
     initial_instruction_file: "config/initial_instructions/nanozymes_v2.txt"
   ```

4. Run optimization:
   ```bash
   python scripts/optimize.py --task nanozymes --config default.yaml
   ```

**Instruction Metadata:**

Each optimized agent stores instruction metadata for reproducibility:
- `initial_instruction_file`: Path to the instruction file used
- `instruction_hash`: SHA256 hash (first 12 chars) of the instruction content

This allows you to track exactly which instruction was used for each agent.

---

## Configuration Files

### Available Config Files

1. **`config/default.yaml`** - Production configuration
   - Reasonable defaults for production use
   - Balanced between quality and speed
   - Suitable for final experiments

2. **`config/default_fast.yaml`** - Fast optimization
   - Fewer trials (3 vs 20)
   - Smaller datasets (3 examples)
   - Quick testing and development
   - Use with `--config default_fast.yaml`

### Using Custom Config Files

```bash
# Use default config
python scripts/optimize.py --task nanozymes

# Use fast config for testing
python scripts/optimize.py --task nanozymes --config default_fast.yaml

# Use custom config
python scripts/optimize.py --task nanozymes --config my_config.yaml
```

---

## Configuration Best Practices

### Development
- Use `default_fast.yaml` for quick iterations
- Set `PROJECT__LOG_LEVEL=DEBUG` for detailed logs
- Enable `OPTIMIZATION__VERBOSE=true` to see optimization progress
- Use small `OPTIMIZATION__NUM_TRIALS` (3-5) for testing

### Production
- Use `default.yaml` or custom production config
- Set `OPTIMIZATION__NUM_TRIALS` to 20+ for best results
- Disable `OPTIMIZATION__VERBOSE` to reduce log clutter
- Set `PROJECT__LOG_LEVEL=INFO` or `WARNING`
- Enable `OPTIMIZATION__USE_CACHE=true` to save costs

### GPU Acceleration
```yaml
parsing:
  docling:
    device: "cuda"  # Use GPU for parsing
  marker:
    device: "cuda"
```

### Using Custom Ollama Server
```bash
export LLM__STUDENT__OLLAMA__OLLAMA_BASE_URL="https://my-ollama-server.com"
export LLM__TEACHER__OLLAMA__OLLAMA_BASE_URL="https://my-ollama-server.com"
```

### Using OpenAI or Anthropic
```bash
export LLM__STUDENT__USE_OLLAMA=false
export LLM__STUDENT__MODEL="gpt-4"
export LLM__STUDENT__NON_OLLAMA__API_KEY="sk-..."

export LLM__TEACHER__USE_OLLAMA=false
export LLM__TEACHER__MODEL="gpt-4"
export LLM__TEACHER__NON_OLLAMA__API_KEY="sk-..."
```

---

## Troubleshooting Configuration Issues

### Configuration Not Loading

**Problem:** Changes to config file not taking effect

**Solutions:**
1. Verify you're specifying the config: `--config path/to/config.yaml`
2. Check for syntax errors in YAML (indentation matters!)
3. Verify environment variables aren't overriding your settings
4. Check file permissions

### Environment Variables Not Working

**Problem:** Environment variables not overriding config

**Solutions:**
1. Use double underscore notation: `LLM__STUDENT__MODEL`
2. Export variables before running: `export VAR=value`
3. Check variable names match config structure exactly
4. Boolean values should be lowercase: `true`/`false`

### Ollama Connection Issues

**Problem:** Cannot connect to Ollama server

**Solutions:**
1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check URL: `LLM__STUDENT__OLLAMA__OLLAMA_BASE_URL`
3. Verify firewall settings
4. Check Ollama logs

### Model Not Found

**Problem:** Specified model not available

**Solutions:**
1. List available models: `ollama list`
2. Pull model: `ollama pull llama3.2:3b`
3. Verify model name spelling in config
4. Check Ollama server has the model
