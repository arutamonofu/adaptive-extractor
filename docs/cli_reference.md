# CLI Reference

Command-line interface reference for AutoEvoExtractor.

## Overview

| Script | Purpose |
|--------|---------|
| [`parse.py`](#parsepy) | Parse PDFs into structured JSON |
| [`optimize.py`](#optimizepy) | Optimize extraction agent via MIPROv2 |
| [`extract.py`](#extractpy) | Extract data using trained agent |
| [`generate_manual_agent.py`](#generate_manual_agentpy) | Create manual agent from examples |

> **Configuration:** All scripts follow [Configuration Priority](configuration.md#configuration-priority).

---

## `parse.py`

**Purpose:** Parse PDF documents into structured JSON.

### Syntax

```bash
python scripts/parse.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | `Path` | `None` | YAML config file |
| `--overwrite` | `flag` | `false` | Overwrite existing parsed files |

### Examples

```bash
# Parse all PDFs (uses AEE_ENV or default.yaml)
python scripts/parse.py

# Parse with explicit config
python scripts/parse.py --config config/default.yaml

# Parse with overwrite
python scripts/parse.py --config config/default.yaml --overwrite
```

### Output

- **Success:** JSON files in `data/parsed/`
- **Exit codes:** `0` (success), `1` (error), `2` (partial), `130` (interrupted)

> **Paths:** Configured via `paths.pdf_dir` and `paths.parsed_dir` in YAML config.

---

## `optimize.py`

**Purpose:** Optimize extraction agent using DSPy MIPROv2.

### Syntax

```bash
python scripts/optimize.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | `Path` | `None` | YAML config file |
| `--run-name` | `str` | `None` | MLflow run name prefix (timestamp auto-generated) |
| `--no-mlflow` | `flag` | `false` | Disable MLflow tracking |

### Examples

```bash
# Optimize with default config
python scripts/optimize.py

# Optimize with custom run name
python scripts/optimize.py --run-name "A1_high"

# Optimize without MLflow
python scripts/optimize.py --no-mlflow
```

### Output

- **Success:** Agent JSON in `data/agents/`
- **Exit codes:** `0` (success), `1` (error), `130` (interrupted)

> **Requirements:** Ground truth CSV and splits JSON must exist.

---

## `extract.py`

**Purpose:** Extract data from documents using trained agent.

### Syntax

```bash
python scripts/extract.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | `Path` | `None` | YAML config file |
| `--agent` | `Path` | **Required** | Path to trained agent JSON |

### Examples

```bash
# Extract with latest agent
python scripts/extract.py --agent data/agents/nanozymes_latest.json

# Extract with specific agent
python scripts/extract.py --agent data/agents/nanozymes_v1_20260218.json
```

### Output

- **Success:** JSON files in `data/extractions/`
- **Exit codes:** `0` (success), `1` (error), `2` (partial), `130` (interrupted)

---

## `generate_manual_agent.py`

**Purpose:** Create manual agent from train_manual split examples.

### Syntax

```bash
python scripts/generate_manual_agent.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | `Path` | `None` | Override output path for agent JSON |

### Examples

```bash
# Generate manual agent with default output path
python scripts/generate_manual_agent.py

# Generate manual agent with custom output path
python scripts/generate_manual_agent.py --output data/agents/manual_custom.json
```

### Output

- **Success:** Agent JSON in `data/agents/`
  - Default path: `data/agents/manual_{task_name}.json`
  - Override with `--output` flag

> **Note:** If `train_manual` split is missing or empty, the script will exit with a warning and no agent will be saved.

---

> **Configuration:** For environment variables and settings, see [Configuration Reference](configuration.md).
