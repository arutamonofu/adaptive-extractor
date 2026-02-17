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
                    │   data/agents/          │
                    │   (trained agents)      │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   predict.py            │
                    │   (data extraction)     │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   data/predictions/     │
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
│   ├── predictions/            # Extraction results
├── config/
│   ├── *.yaml                  # YAML configurations
│   └── initial_instructions/   # Instructions for optimization
├── logs/                       # Execution logs
└── mlruns/                     # MLflow artifacts (optional)
```

---

## Detailed Artifact Description

### 1. `data/pdf/` or `data/pdfs/`

**Type:** Input data  
**Created by:** User  
**Format:** PDF files

**Description:** Directory for source PDF documents. Place scientific articles here for processing.

**Example structure:**
```
data/pdfs/
├── paper1.pdf
├── paper2.pdf
├── article_2024.pdf
└── ...
```

**Configuration:**
```yaml
paths:
  pdf_dir: "data/pdfs"
```

**Environment variable:**
```bash
export PATHS__PDF_DIR="data/my_pdfs"
```

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
- `filename` — PDF filename (with `.pdf` extension)
- Other columns depend on task (defined in task definition)

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

---

### 5. `data/agents/`

**Type:** Output data (models)  
**Created by:** `scripts/optimize.py`  
**Format:** JSON + metadata

**Description:** Optimized agents for data extraction. Each agent contains trained prompts and examples.

**Directory structure:**
```
data/agents/
├── nanozymes_v1_2026-02-17.json
├── nanozymes_v1_2026-02-17.meta.json
├── nanozymes_latest.json
└── manual_nanozymes.json
```

**Files:**
- `*.json` — the agent itself (DSPy module)
- `*.meta.json` — metadata (metrics, date, model version)

**Metadata structure:**
```json
{
  "task_name": "nanozymes",
  "created_at": "2026-02-17T15:30:00",
  "model_version": "llama3.2:3b",
  "metrics": {
    "f1": 0.85,
    "precision": 0.87,
    "recall": 0.83
  },
  "config_snapshot": {
    "num_trials": 20,
    "train_split": 15
  },
  "instruction_hash": "a1b2c3d4e5f6",
  "description": "Optimized with 20 trials"
}
```

**Configuration:**
```yaml
paths:
  agents_dir: "data/agents"
```

---

### 6. `data/predictions/`

**Type:** Output data (results)  
**Created by:** `scripts/predict.py`  
**Format:** JSON

**Description:** Results of data extraction from documents.

**Directory structure:**
```
data/predictions/
├── nanozymes_predictions.json
├── proteins_predictions.json
└── ...
```

**JSON structure:**
```json
{
  "predictions": {
    "paper1": [
      {
        "formula": "Cu-TEMPO",
        "activity": "oxidation",
        "km_value": 0.05,
        "vmax_value": 100
      }
    ],
    "paper2": [...]
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
  predictions_dir: "data/predictions"
```

---

### 7. `logs/`

**Type:** Logs  
**Created by:** Automatically during script execution  
**Format:** Text files

**Description:** Execution logs.

**Structure:**
```
logs/
├── parse_2026-02-17_10-30-00.log
├── optimize_2026-02-17_15-00-00.log
└── predict_2026-02-17_16-00-00.log
```

**Configuration:**
```yaml
paths:
  logs_dir: "logs"
project:
  log_level: "INFO"
```

---

### 9. `mlruns/`

**Type:** MLflow artifacts (optional)  
**Created by:** `scripts/optimize.py` (when MLflow enabled)  
**Format:** MLflow binary files

**Description:** Experiment tracking via MLflow.

**Configuration:**
```bash
# Disable MLflow
python scripts/optimize.py --no-mlflow
```

---

## Data Lifecycle

### Stage 1: Preparation

| Artifact | Action |
|----------|--------|
| `data/pdfs/` | Place PDF files |
| `data/ground_truth/` | Create CSV with annotations |

### Stage 2: Parsing

```bash
python scripts/parse.py data/pdfs/
```

> **Note:** Parser selection (`docling` or `marker`) and output directory are configured via YAML config file.

| Input | Output |
|-------|--------|
| `data/pdfs/*.pdf` | `data/parsed/*.json` |

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
python scripts/optimize.py --task nanozymes
```

| Input | Output |
|-------|--------|
| `data/ground_truth/nanozymes.csv` | `data/agents/nanozymes_*.json` |
| `data/splits/nanozymes.json` | |
| `data/parsed/*.json` | |

### Stage 5: Prediction

```bash
python scripts/predict.py --agent data/agents/nanozymes_latest.json
```

| Input | Output |
|-------|--------|
| `data/agents/*.json` | `data/predictions/*.json` |
| `data/parsed/*.json` | |

### Stage 6: Evaluation (optional)

```bash
python scripts/evaluate.py \
    --predictions data/predictions/output.json \
    --ground-truth data/ground_truth/nanozymes.csv
```

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

All paths can be overridden via environment variables:

```bash
# Input data
export PATHS__PDF_DIR="data/my_pdfs"
export PATHS__GROUND_TRUTH_DIR="data/my_gt"

# Intermediate data
export PATHS__PARSED_DIR="data/my_parsed"
export PATHS__SPLITS_FILE="data/my_splits.json"

# Output data
export PATHS__AGENTS_DIR="data/my_agents"
export PATHS__PREDICTIONS_DIR="data/my_predictions"
```
