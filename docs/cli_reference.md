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

**Purpose:** Parse PDF documents into structured Markdown.

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
# Parse all PDFs with default config (uses AEE_ENV or marker parser)
python scripts/parse.py

# Parse with explicit config
python scripts/parse.py --config config/systems/example.yaml

# Parse with Gemini API parser
python scripts/parse.py --config config/systems/gemini.yaml

# Parse with overwrite
python scripts/parse.py --config config/systems/example.yaml --overwrite
```

### Output

- **Success:** Markdown files in `paths.parsed_dir` (configured in YAML)
- **Exit codes:** `0` (success), `1` (error), `2` (partial), `130` (interrupted)

### Parsers

| Parser | Description | Configuration |
|--------|-------------|---------------|
| `marker` | Local PDF parsing using Marker library | `parsing.marker.device` |
| `gemini` | Cloud PDF parsing using Google Gemini API | `parsing.gemini.*` |

> **Note:** For Gemini parser, set `GEMINI_API_KEY` in `.env` file.

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

### Requirements

1. **Parsed documents** — Must exist in `paths.parsed_dir` from config
2. **Ground truth CSV** — Must exist at `paths.ground_truth_dir/<task_name>.csv`
3. **Data splits JSON** — Must exist at `paths.splits_file` with train/val splits
4. **Task configuration** — Task YAML must exist in `config/tasks/<task_name>.yaml`
5. **Initial instruction** — Must exist at `task.initial_instruction_file` from config

### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Configuration file not found` | Config YAML doesn't exist | Check path or run from project root |
| `Task signature not found` | Task config missing or invalid | Verify `config/tasks/<task_name>.yaml` exists |
| `Parsed directory not found` | Parsed docs directory missing | Run `parse.py` first or update config |
| `Pre-flight validation failed` | Data validation errors | Review logs for specific errors |
| `Optimization failed` | MIPROv2 error | Check LLM config and data quality |

> **See also:** `docs/configuration.md` for detailed troubleshooting guide.

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
# Extract with specific agent
python scripts/extract.py --agent data/agents/nanozymes_v1_20260218.json

# Extract with custom config
python scripts/extract.py --config config/systems/dev.yaml --agent data/agents/nanozymes_latest.json
```

### Output

- **Success:** JSON files in `data/extractions/`
- **Exit codes:** `0` (success), `1` (error), `2` (partial), `130` (interrupted)

### Requirements

1. **Trained agent JSON** — Must exist at specified path
   - Created by `optimize.py` or `generate_manual_agent.py`
   - Contains DSPy agent state and metadata
2. **Parsed documents** — Must exist in `data/parsed/`
3. **Task configuration** — Must match the task the agent was trained on

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
| `--config` | `Path` | - | Path to configuration file (**required**) |
| `--output` | `Path` | `None` | Override output path for agent JSON |

### Examples

```bash
# Generate manual agent with default output path
python scripts/generate_manual_agent.py --config config/default.yaml

# Generate manual agent with custom output path
python scripts/generate_manual_agent.py --config config/default.yaml --output data/agents/manual_custom.json
```

### Output

- **Success:** Agent JSON in `data/agents/`
  - Default path: `data/agents/manual_{task_name}.json`
  - Override with `--output` flag
- **Exit codes:** `0` (success), `1` (error), `130` (interrupted)

### Requirements

1. **Parsed documents** — Must exist in `paths.parsed_dir` from config
2. **Ground truth CSV** — Must exist at `paths.ground_truth_dir/<task_name>.csv`
3. **Data splits JSON** — Must exist at `paths.splits_file` with `train_manual` split
4. **Task configuration** — Task YAML must exist in `config/tasks/<task_name>.yaml`
5. **Initial instruction** — Must exist at `task.initial_instruction_file` from config

---

> **Configuration:** For environment variables and settings, see [Configuration Reference](configuration.md).
