# Data Artifacts Guide

Complete guide to all files and directories in AutoEvoExtractor.

## Data Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Source PDFs   │ ──► │    Parsed       │     │  Ground Truth   │
│   (input)       │     │  (intermediate) │     │   (input)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     data/splits/[task].json                     │
│                    (data split configuration)                   │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   optimize.py           │
                    │   (agent optimization)  │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   data/agents/          │
                    │   (trained agents)      │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   extract.py            │
                    │   (data extraction)     │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   data/extractions/     │
                    │   (results)             │
                    └─────────────────────────┘
```

---

## Directory Structure

```
autoevoextractor/
├── data/
│   ├── pdf/                    # Source PDF files
│   ├── parsed/                 # Parsed JSON documents
│   ├── ground_truth/           # CSV files with annotations
│   ├── splits/                 # Task-specific split files
│   │   └── nanozymes.json      # Train/test/val split for nanozymes task
│   ├── agents/                 # Trained agents
│   └── extractions/            # Extraction results
├── config/
│   ├── *.yaml                  # YAML configurations
│   └── initial_instructions/   # Instructions for optimization
└── src/aee/domain/tasks/
    └── [task_name]/
        └── task.yaml           # Task definition (YAML-backed tasks, v2.0+)
```

---

## Detailed Artifact Description

### 1. `data/pdf/`

**Type:** Input data
**Created by:** User
**Format:** PDF files

**Description:** Directory for source PDF documents. Place scientific articles here for processing.

**Example structure:**
```
data/pdf/
├── paper1.pdf
├── paper2.pdf
├── article_2024.pdf
└── ...
```

**Configuration:**
```yaml
paths:
  pdf_dir: "data/pdf"
```

> **For environment variables reference**, see [Configuration Guide](configuration.md#environment-variables).

---

### 2. `data/parsed/`

**Type:** Intermediate data  
**Created by:** `scripts/parse.py`  
**Format:** JSON

**Description:** Parsed documents in structured format. Each PDF corresponds to one JSON file.

**JSON file structure:**
```json
{
  "doc_id": "paper1",
  "text_content": "Full extracted text...",
  "metadata": {
    "source_file": "paper1.pdf",
    "parsed_at": "2026-02-17T10:30:00",
    "parser": "docling",
    "pages": 12
  },
  "sections": [
    {
      "title": "Abstract",
      "content": "..."
    }
  ]
}
```

**Directory structure:**
```
data/parsed/
├── paper1.json
├── paper2.json
└── ...
```

**Configuration:**
```yaml
paths:
  parsed_dir: "data/parsed"
```

---

### 3. `data/ground_truth/`

**Type:** Input data (training)
**Created by:** User
**Format:** CSV

**Description:** CSV files with annotated data for agent training and validation.

**CSV structure:**
```csv
filename,formula,activity,length,km_value,vmax_value
paper1.pdf,Cu-TEMPO,oxidation,10,0.05,100
paper1.pdf,Fe-TEMPO,oxidation,12,0.08,150
paper2.pdf,Mn-Salen,epoxidation,8,0.12,80
```

**Required columns:**
- `filename` — Document identifier. Supported formats:
  - With extension: `paper1.pdf`
  - Without extension: `paper1`
  - Alternative column names: `pdf`, `source`, `doi`, `document`
- Other columns depend on task (defined in task definition)

> **Note:** The system normalizes document identifiers by removing file extensions and converting to lowercase. Both formats (with/without `.pdf`) are supported.

**Directory structure:**
```
data/ground_truth/
├── nanozymes.csv
├── proteins.csv
└── ...
```

**Configuration:**
```yaml
paths:
  ground_truth_dir: "data/ground_truth"
```

---

### 4. `data/splits/[task].json`

**Type:** Intermediate data (configuration)
**Created by:** User (manually or via script)
**Format:** JSON

**Description:** File with document splits for training, validation, and testing. Each task has its own split file in the `data/splits/` directory. The path to the splits file is specified in the configuration (`paths.splits_file`).

**⚠️ Important:** Creating this file requires careful attention, as the quality of the split affects optimization results. Do not automate this process without understanding your data.

**JSON structure:**
```json
{
  "train": ["paper1", "paper2", "paper3", "paper4", "paper5"],
  "val": ["paper6", "paper7"],
  "test": ["paper8", "paper9", "paper10"]
}
```

**Split names:**
- `train` — training set (required)
- `val` or `validation` — validation set (optional)
- `test` — test set (optional)
- `train_manual` — manual examples for `generate_manual_agent.py` (optional)

**How to create:**

Method 1: Manually (recommended for small datasets)
```json
{
  "train": ["doc_001", "doc_002", "doc_003"],
  "val": ["doc_004"],
  "test": ["doc_005", "doc_006"]
}
```

Method 2: Via Python API
```python
from pathlib import Path
from aee.infrastructure.storage import DataSplitRepository
import pandas as pd

# Load ground truth
gt_path = Path("data/ground_truth/nanozymes.csv")
df = pd.read_csv(gt_path)

# Get unique document IDs
doc_ids = df["filename"].str.replace(".pdf", "").unique().tolist()

# Create split
repo = DataSplitRepository()
splits = repo.create_random_split(
    documents=doc_ids,
    train_ratio=0.8,
    seed=42
)

# Save to task-specific file
output_path = Path("data/splits/nanozymes.json")
repo.save_splits(splits, output_path)
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")
```

**Configuration:**
```yaml
paths:
  splits_file: "data/splits/nanozymes.json"
```

> **For environment variables reference**, see [Configuration Guide](configuration.md#environment-variables).

---

### 5. `data/agents/`

**Type:** Output data (models)
**Created by:** `scripts/optimize.py`
**Format:** JSON + metadata

**Description:** Optimized agents for data extraction. Each agent contains trained prompts and examples.

**Directory structure:**
```
data/agents/
├── nanozymes_v1_2026-02-17T15-30-00.json
├── nanozymes_v1_2026-02-17T15-30-00.meta.json
├── nanozymes_v2_2026-02-18T10-00-00.json
└── manual_nanozymes.json
```

**Files:**
- `*_v{version}_{timestamp}.json` — the agent itself (DSPy module)
- `*_v{version}_{timestamp}.meta.json` — metadata (metrics, date, model version)

**Metadata structure:**
```json
{
  "task_name": "nanozymes",
  "created_at": "2026-02-17T15:30:00",
  "model_version": "mistral-small3.1-24b-128k:latest",
  "metrics": {
    "f1": 0.85,
    "precision": 0.87,
    "recall": 0.83
  },
  "config_snapshot": {
    "num_trials": 70,
    "train_split": 20
  },
  "instruction_hash": "a1b2c3d4e5f6",
  "initial_instruction_file": "config/initial_instructions/nanozymes_sota.txt",
  "description": "Optimized with 70 trials"
}
```

**New in v2.0:** Agents track instruction provenance via `instruction_hash` (SHA256, first 12 chars) and `initial_instruction_file` for reproducibility.

**Configuration:**
```yaml
paths:
  agents_dir: "data/agents"
```

---

### 6. `data/extractions/`

**Type:** Output data (results)
**Created by:** `scripts/extract.py`
**Format:** JSON

**Description:** Results of data extraction from documents.

**Directory structure:**
```
data/extractions/
├── nanozymes_extractions.json
├── proteins_extractions.json
└── ...
```

**JSON structure:**
```json
{
  "extraction": {
    "experiments": [
      {
        "formula": "Cu-TEMPO",
        "activity": "oxidation",
        "km_value": 0.05,
        "vmax_value": 100
      }
    ]
  },
  "metadata": {
    "agent_path": "data/agents/nanozymes_latest.json",
    "task": "nanozymes",
    "processed_at": "2026-02-17T16:00:00"
  }
}
```

**Configuration:**
```yaml
paths:
  extractions_dir: "data/extractions"
```

> **For environment variables reference**, see [Configuration Guide](configuration.md#environment-variables).

---

### 7. `logs/`

**Type:** Logs
**Created by:** Automatically during script execution
**Format:** Text files

**Description:** Execution logs are configured via `project.log_level` in YAML config. Log output destination depends on your logging configuration (console, file, or both).

> **For logging configuration reference**, see [Configuration Guide](configuration.md#project-settings).

---

## Task Configuration (YAML-backed Tasks)

**Type:** Task definition
**Format:** YAML

**Description:** New in v2.0, tasks can be defined via YAML configuration files instead of Python code. This allows adding new extraction tasks without code changes.

**Location:**
```
src/aee/domain/tasks/
└── [task_name]/
    └── task.yaml
```

> **For complete task configuration guide**, including field specifications, examples, and step-by-step instructions, see [Adding New Extraction Tasks](adding_tasks.md).

---

## Data Lifecycle

### Stage 1: Preparation

| Artifact | Action |
|----------|--------|
| `data/pdf/` | Place PDF files |
| `data/ground_truth/` | Create CSV with annotations |
| `data/splits/[task].json` | Create data split configuration |

### Stage 2: Parsing

```bash
python scripts/parse.py --config default.yaml
```

> **Note:** PDF directory, parser selection (`docling` or `marker`), and output directory are configured via YAML config file.

| Input | Output |
|-------|--------|
| `data/pdf/*.pdf` | `data/parsed/*.json` |

### Stage 3: Data Splitting

| Artifact | Action |
|----------|--------|
| `data/splits/[task].json` | Create manually or via script |

**Example splits file (`data/splits/nanozymes.json`):**
```json
{
  "train": ["paper1", "paper2", "paper3"],
  "val": ["paper4"],
  "test": ["paper5", "paper6"]
}
```

### Stage 4: Agent Optimization

```bash
python scripts/optimize.py --config default.yaml
```

| Input | Output |
|-------|--------|
| `data/ground_truth/nanozymes.csv` | `data/agents/nanozymes_v*.json` |
| `data/splits/nanozymes.json` | |
| `data/parsed/*.json` | |
| `config/initial_instructions/*.txt` | |

### Stage 5: Extraction

```bash
python scripts/extract.py --config default.yaml --agent data/agents/nanozymes_latest.json
```

| Input | Output |
|-------|--------|
| `data/agents/*.json` | `data/extractions/*.json` |
| `data/parsed/*.json` | |

### Stage 6: Evaluation (optional)

Evaluation is performed manually or with custom scripts.

---

## Data Integrity Checks

### Before Optimization

Ensure all documents in splits file exist:

```python
import json
from pathlib import Path

# Load splits
splits_path = Path("data/splits/nanozymes.json")
with open(splits_path) as f:
    splits = json.load(f)

# Load ground truth
import pandas as pd
df = pd.read_csv("data/ground_truth/nanozymes.csv")
gt_docs = df["filename"].str.replace(".pdf", "").unique()

# Check all documents exist
for split_name, doc_ids in splits.items():
    missing = set(doc_ids) - set(gt_docs)
    if missing:
        print(f"⚠ {split_name}: missing {missing}")
```

### Check Parsed Files

```python
from pathlib import Path
import json

parsed_dir = Path("data/parsed")
splits_path = Path("data/splits/nanozymes.json")
with open(splits_path) as f:
    splits = json.load(f)

all_docs = []
for docs in splits.values():
    all_docs.extend(docs)

for doc_id in all_docs:
    json_path = parsed_dir / f"{doc_id}.json"
    if not json_path.exists():
        print(f"⚠ Missing parsed file: {json_path}")
```

---

## Environment Variables for Paths

All paths can be overridden via environment variables. See [Configuration Guide](configuration.md#environment-variables) for the complete reference.

**Common path overrides:**
```bash
# Input data
export PATHS__PDF_DIR="data/my_pdfs"
export PATHS__GROUND_TRUTH_DIR="data/my_gt"

# Output data
export PATHS__AGENTS_DIR="data/my_agents"
export PATHS__EXTRACTIONS_DIR="data/my_extractions"
```
