# CLI Reference

Complete reference for all AutoEvoExtractor command-line commands.

## Overview

The system provides 4 main scripts:

| Script | Purpose |
|--------|---------|
| [`parse.py`](#parsepy) | Parse PDFs into structured JSON |
| [`optimize.py`](#optimizepy) | Optimize extraction agent via MIPROv2 |
| [`extract.py`](#extractpy) | Extract data from documents using trained agent |
| [`generate_manual_agent.py`](#generate_manual_agentpy) | Create manual agent from train_manual split examples |

---

## Configuration

All scripts load configuration following the priority defined in [Configuration Guide](configuration.md#configuration-priority):

1. **Environment variables** (`.env` file via pydantic-settings, `AEE_ENV`, `AEE__*` overrides)
2. **CLI arguments** (`--config`, `--overwrite`, etc.)
3. **YAML configuration files** (`config/default.yaml`, `config/<env>.yaml`)
4. **Internal defaults**

> **For complete configuration reference**, see [Configuration Guide](configuration.md).

---

## `parse.py`

**Purpose:** Parse PDF documents into structured JSON format.

**Path:** `scripts/parse.py`

### Syntax

```bash
python scripts/parse.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | `Path` | `None` | Path to YAML configuration file. If not set, uses `AEE_ENV` env var or `default.yaml` |
| `--overwrite` | `flag` | `false` | Overwrite existing parsed files |

### Examples

**Parse all PDFs from configured directory (uses AEE_ENV or default.yaml):**
```bash
python scripts/parse.py
```

**Parse with explicit config:**
```bash
python scripts/parse.py --config config/default.yaml
```

**Parse with overwrite:**
```bash
python scripts/parse.py --config config/default.yaml --overwrite
```

**Parse with custom environment:**
```bash
AEE_ENV=config/fast.yaml python scripts/parse.py
```

### Notes

- **PDF directory:** Configured via `paths.pdf_dir` in YAML config (default: `data/pdf`)
- **Parser selection:** Configured via `parsing.parser` in YAML config (e.g., `docling` or `marker`)
- **Output directory:** Configured via `paths.parsed_dir` in YAML config (default: `data/parsed`)
- **Recursive search:** Automatically finds all `.pdf` and `.PDF` files in the configured directory

> **For path configuration reference**, see [Configuration Guide](configuration.md#paths-configuration).

### Output

- **Success:** JSON files in `data/parsed/` (or specified directory)
- **Exit codes:**
  - `0` — All documents parsed successfully
  - `1` — Error during parsing
  - `2` — Partial success (some documents failed)
  - `130` — Interrupted by user (Ctrl+C)

---

## `optimize.py`

**Purpose:** Optimize extraction agent using DSPy MIPROv2.

**Path:** `scripts/optimize.py`

### Syntax

```bash
python scripts/optimize.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | `Path` | `None` | Path to YAML configuration file. If not set, uses `AEE_ENV` env var or `default.yaml` |
| `--run-name` | `str` | `None` | Prefix for MLflow run name (e.g., `A1_high`, `A2_temp1.0`). Timestamp added automatically |
| `--no-mlflow` | `flag` | `false` | Disable MLflow tracking |

### Examples

**Basic optimization (uses AEE_ENV or default.yaml):**
```bash
python scripts/optimize.py
```

**Optimization with explicit config:**
```bash
python scripts/optimize.py --config config/default.yaml
```

**With MLflow run naming:**
```bash
python scripts/optimize.py --config config/default.yaml --run-name "A1_temp0.5"
```

**Without MLflow:**
```bash
python scripts/optimize.py --config config/default.yaml --no-mlflow
```

**With custom environment:**
```bash
AEE_ENV=config/fast.yaml python scripts/optimize.py
```

### Prerequisites

Before running optimization, prepare:

1. **Ground truth data:** `data/ground_truth/{task}.csv`
2. **Data splits:** `data/splits/{task}.json` with train/test/val splits
3. **Parsed documents:** `data/parsed/` must contain JSON for all documents in splits
4. **Initial instruction:** Configured via `task.initial_instruction_file` in YAML or loaded from `config/domain/tasks/{task}/task.yaml`

> **For MIPROv2 parameters reference**, see [Configuration Guide](configuration.md#optimization-configuration).

### Output

- **Success:** Optimized agent in `data/agents/{task}_{timestamp}.json`
- **Exit codes:**
  - `0` — Optimization successful
  - `1` — Optimization error
  - `130` — Interrupted by user

---

## `extract.py`

**Purpose:** Extract data from documents using a trained agent.

**Path:** `scripts/extract.py`

### Syntax

```bash
python scripts/extract.py [OPTIONS] --agent AGENT_PATH
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | `Path` | `None` | Path to YAML configuration file. If not set, uses `AEE_ENV` env var or `default.yaml` |
| `--agent` | `Path` | **Required** | Path to trained agent JSON file (relative to `data/agents/` or absolute) |

### Examples

**Extract from all documents (uses AEE_ENV or default.yaml):**
```bash
python scripts/extract.py --agent nanozymes_latest.json
```

**Extract with explicit config:**
```bash
python scripts/extract.py \
    --config config/default.yaml \
    --agent nanozymes_latest.json
```

**Extract with custom config:**
```bash
python scripts/extract.py \
    --config config/fast.yaml \
    --agent nanozymes_latest.json
```

**With custom environment:**
```bash
AEE_ENV=config/prod.yaml python scripts/extract.py --agent nanozymes_latest.json
```

> **For extraction and circuit breaker configuration**, see [Configuration Guide](configuration.md#extraction-configuration) and [Configuration Guide](configuration.md#circuit-breaker-configuration).

### Output

- **Success:** JSON files with extracted data in `data/extractions/` (directory configured via `paths.extractions_dir`)
- **Exit codes:**
  - `0` — All documents processed successfully
  - `1` — Extraction error
  - `2` — Partial success (some documents failed)
  - `130` — Interrupted by user

---

## `generate_manual_agent.py`

**Purpose:** Create a manual agent from examples in `train_manual` split.

**Path:** `scripts/generate_manual_agent.py`

### Syntax

```bash
python scripts/generate_manual_agent.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | `str` | `data/agents/manual_{task}.json` | Output path for agent (overrides default) |

### Examples

**Create manual agent (uses config from AEE_ENV or default.yaml):**
```bash
python scripts/generate_manual_agent.py
```

**Create with custom path:**
```bash
python scripts/generate_manual_agent.py --output data/agents/my_manual_agent.json
```

### Prerequisites

1. **Splits file** with `train_manual` split: `data/splits/{task}.json`
2. **Parsed documents** in `data/parsed/train/manual/`
3. **Ground truth data:** `data/ground_truth/{task}.csv`
4. **Task definition:** Configured via `task.name` in YAML or `config/domain/tasks/{task}/task.yaml`

### Output

- **Success:** Manual agent in `data/agents/manual_{task}.json`
- **Exit codes:**
  - `0` — Agent created successfully
  - `1` — Error during creation

---

## Arguments Summary Table

| Argument | parse.py | optimize.py | extract.py | generate_manual_agent.py |
|----------|:--------:|:-----------:|:----------:|:------------------------:|
| `--config` | ✅ (optional) | ✅ (optional) | ✅ (optional) | ❌ |
| `--overwrite` | ✅ | ❌ | ❌ | ❌ |
| `--run-name` | ❌ | ✅ | ❌ | ❌ |
| `--no-mlflow` | ❌ | ✅ | ❌ | ❌ |
| `--agent` | ❌ | ❌ | ✅ (required) | ❌ |
| `--output` | ❌ | ❌ | ❌ | ✅ (optional) |

---

## Exit Codes

All scripts use standard exit codes:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error |
| `2` | Partial success (some data processed) |
| `130` | Interrupted by user (Ctrl+C) |

---

## Logging

**Log level** is configured via `project.log_level` in YAML. See [Configuration Guide](configuration.md#project-settings) for details.

---

## Environment Variables

### Configuration Loading

The primary environment variable for configuration is `AEE_ENV`:

```bash
# Use specific config file via environment
export AEE_ENV=config/dev.yaml
python scripts/parse.py

# Or inline
AEE_ENV=config/prod.yaml python scripts/optimize.py
```

### Override Individual Settings

Any YAML setting can be overridden using double-underscore notation with `AEE__` prefix. See [Configuration Guide](configuration.md#environment-variables) for the complete reference.

**Common examples:**
```bash
# Paths
export AEE__PATHS__PDF_DIR="data/my_pdfs"

# LLM
export AEE__LLM__STUDENT__MODEL="llama3.2:3b"

# Optimization
export AEE__OPTIMIZATION__NUM_TRIALS="50"

# Logging
export AEE__PROJECT__LOG_LEVEL="DEBUG"
```
