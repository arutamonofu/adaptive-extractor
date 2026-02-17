# CLI Reference

Complete reference for all AutoEvoExtractor command-line commands.

## Overview

The system provides 4 main scripts:

| Script | Purpose |
|--------|---------|
| [`parse.py`](#parsepy) | Parse PDFs into structured JSON |
| [`optimize.py`](#optimizepy) | Optimize extraction agent via MIPROv2 |
| [`extract.py`](#extractpy) | Extract data from documents using trained agent |
| [`generate_manual_agent.py`](#generate_manual_agentpy) | Create manual agent from examples |

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
| `--overwrite` | `flag` | `false` | Overwrite existing parsed files |
| `--config` | `Path` | **Required** | Path to YAML configuration file |

### Examples

**Parse all PDFs from configured directory:**
```bash
python scripts/parse.py --config default.yaml
```

**Parse with overwrite:**
```bash
python scripts/parse.py --config default.yaml --overwrite
```

**Parse with custom config:**
```bash
python scripts/parse.py --config default_fast.yaml
```

### Notes

- **PDF directory:** Configured via `paths.pdf_dir` in YAML config (default: `data/pdf`)
- **Parser selection:** Configured via `parser.name` in YAML config (e.g., `docling` or `marker`)
- **Output directory:** Configured via `paths.parsed_dir` in YAML config

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
| `--config` | `Path` | **Required** | Path to YAML configuration file |
| `--run-name` | `str` | `None` | Prefix for MLflow run name (e.g., `A1_high`, `A2_temp1.0`). Timestamp added automatically |
| `--no-mlflow` | `flag` | `false` | Disable MLflow tracking |

### Examples

**Basic optimization:**
```bash
python scripts/optimize.py --config default.yaml
```

**Fast test optimization:**
```bash
python scripts/optimize.py --config default_fast.yaml
```

**With MLflow run naming:**
```bash
python scripts/optimize.py --config default.yaml --run-name "A1_temp0.5"
```

**Without MLflow:**
```bash
python scripts/optimize.py --config default.yaml --no-mlflow
```

### Prerequisites

Before running optimization, prepare:

1. **Ground truth data:** `data/ground_truth/{task}.csv`
2. **Data splits:** `data/splits/{task}.json` with train/test/val split
3. **Parsed documents:** `data/parsed/` must contain JSON for all documents in splits
4. **Initial instruction:** Configured via `task.initial_instruction_file` in YAML (e.g., `instructions/v1_standard.md`)

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
| `--agent` | `Path` | **Required** | Path to trained agent JSON file |
| `--config` | `Path` | **Required** | Path to YAML configuration file |

### Examples

**Extract from all documents:**
```bash
python scripts/extract.py \
    --config default.yaml \
    --agent nanozymes_latest.json
```

**Extract with custom config:**
```bash
python scripts/extract.py \
    --config default_fast.yaml \
    --agent nanozymes_latest.json
```

### Configuration

**LLM caching** is controlled via `extraction.enable_cache` in YAML config:
```yaml
extraction:
  enable_cache: false  # Set to true to enable caching
```

**Task name** is specified in YAML config:
```yaml
task:
  name: "nanozymes"
```

### Output

- **Success:** JSON files with extracted data in `data/extractions/` (directory configured in YAML)
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
| `--output` | `str` | `data/agents/manual_{task}.json` | Output path for agent |

### Examples

**Create manual agent:**
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

### Output

- **Success:** Manual agent in `data/agents/manual_{task}.json`
- **Exit codes:**
  - `0` — Agent created successfully
  - `1` — Error during creation

---

## Arguments Summary Table

| Argument | parse.py | optimize.py | extract.py | generate_manual_agent.py |
|----------|:--------:|:-----------:|:----------:|:------------------------:|
| `--config` | ✅ | ✅ | ✅ | ❌ |
| `--overwrite` | ✅ | ❌ | ❌ | ❌ |
| `--run-name` | ❌ | ✅ | ❌ | ❌ |
| `--no-mlflow` | ❌ | ✅ | ❌ | ❌ |
| `--task` | ❌ | ❌ | ❌ | ❌ |
| `--agent` | ❌ | ❌ | ✅ | ❌ |
| `--output` | ❌ | ❌ | ❌ | ✅ |

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

All commands output logs to console and (optionally) to log files.

**Log level** is configured in YAML:
```yaml
project:
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Override via environment variable:**
```bash
export PROJECT__LOG_LEVEL="DEBUG"
```

---

## Environment Variables

Any setting can be overridden via environment variables:

```bash
# Configuration
export PATHS__PDF_DIR="data/my_pdfs"
export PATHS__PARSED_DIR="data/my_parsed"
export PATHS__AGENTS_DIR="data/my_agents"

# LLM
export LLM__STUDENT__MODEL="llama3.2:3b"
export LLM__STUDENT__OLLAMA__OLLAMA_BASE_URL="http://localhost:11434"

# Optimization
export OPTIMIZATION__NUM_TRIALS="50"
export OPTIMIZATION__USE_CACHE="true"

# Logging
export PROJECT__LOG_LEVEL="DEBUG"
```
