# Data Artifacts Guide

Guide to all files and directories in AutoEvoExtractor.

## Data Pipeline

```
PDFs → Parsed MD ─┬─→ Optimize → Agent
                  │
Ground Truth CSV ─┘
                  │
Splits JSON ──────┘
                  │
                  └─→ Extract → Results
```

---

## Directory Structure

```
data/
├── pdf/              # Source PDF files (user-provided)
├── parsed/           # Parsed Markdown (created by parse.py)
├── ground_truth/     # Training CSV (user-provided)
├── splits/           # Data splits JSON (user-provided)
├── agents/           # Trained agents (created by optimize.py)
└── extractions/      # Results (created by extract.py)
```

---

## Input Files

### 1. PDF Files (`data/pdf/`)

**Format:** PDF documents  
**Created by:** User

Place scientific articles here for processing.

```
data/pdf/
├── paper1.pdf
├── paper2.pdf
└── ...
```

> **Config:** `paths.pdf_dir`

---

### 2. Ground Truth CSV (`data/ground_truth/{task}.csv`)

**Format:** CSV  
**Created by:** User

Training data for optimization.

```csv
filename,formula,activity,length,km_value,km_unit
paper1.pdf,Fe3O4,peroxidase,10,0.05,mM
paper2.pdf,CuO,oxidase,20,0.08,mM
```

**Required columns:**
- `filename` — PDF filename (must match file in `data/pdf/`)
- Task-specific fields (defined in `task.yaml`)

> **Config:** `paths.ground_truth_dir`  
> **Guide:** [Adding Tasks](adding_tasks.md)

---

### 3. Data Splits JSON (`data/splits/{task}.json`)

**Format:** JSON  
**Created by:** User

Defines train/validation/test splits.

```json
{
  "train": ["paper1", "paper2", "paper3"],
  "val": ["paper4"],
  "test": ["paper5", "paper6"]
}
```

> ⚠️ **Important:** Document IDs must match `filename` in ground truth CSV (without `.pdf` extension).

> **Config:** `paths.splits_file`

---

## Generated Files

### 4. Parsed Markdown (`data/parsed/`)

**Format:** Markdown (`.md`)
**Created by:** `parse.py`

Plain text document content in Markdown format.

```
data/parsed/
├── paper1.md
├── paper2.md
└── ...
```

**Content:** Hybrid markdown text with:
- Section headers (`#`, `##`, ...)
- Paragraphs and lists
- Tables in HTML format (for structured data)
- Scientific notations and formulas

> **Config:** `paths.parsed_dir`

---

### 5. Trained Agents (`data/agents/`)

**Format:** JSON + metadata JSON
**Created by:** `optimize.py`

```
data/agents/
├── nanozymes_v1_20260218.json       # Agent state
└── nanozymes_v1_20260218.meta.json  # Metadata
```

**Metadata fields:** `task_name`, `created_at`, `model_version`, `metrics`, `config_snapshot`, `git_commit` (optional), `description` (optional), `initial_instruction_file` (optional), `instruction_hash` (optional)

> **Config:** `paths.agents_dir`
> **Source:** `aee.infrastructure.storage.agents_fn.AgentMetadata`

---

### 6. Extraction Results (`data/extractions/`)

**Format:** JSON
**Created by:** `extract.py`

```json
{
  "extraction": {
    "experiments": [
      {
        "formula": "Fe3O4",
        "activity": "peroxidase",
        "length": 10.0
      }
    ]
  },
  "source_metadata": {
    "filename": "paper1.pdf",
    "document_id": "paper1"
  }
}
```

> **Note:** Supports multiple formats for compatibility: `{"extraction": {"experiments": [...]}}`, `{"experiments": [...]}`, `{"extracted_data": {"experiments": [...]}}`, or direct list.

> **Config:** `paths.extractions_dir`
> **Source:** `aee.infrastructure.storage.extractions.ExtractionRepository.save()`

---

## Instruction Files

### Initial Instructions (`config/initial_instructions/`)

**Format:** TXT  
**Created by:** User

Base instructions for DSPy optimization.

```
config/initial_instructions/
└── nanozymes_sota.txt
```

> **Referenced in:** `task.yaml` → `instruction_file`

---

## Quick Reference

| File | Location | Created By | Format |
|------|----------|------------|--------|
| PDFs | `data/pdf/` | User | `.pdf` |
| Ground Truth | `data/ground_truth/` | User | `.csv` |
| Splits | `data/splits/` | User | `.json` |
| Parsed MD | `data/parsed/` | `parse.py` | `.md` |
| Agent | `data/agents/` | `optimize.py` | `.json` |
| Extractions | `data/extractions/` | `extract.py` | `.json` |
| Task YAML | `config/tasks/` | User | `.yaml` |
| Instructions | `config/initial_instructions/` | User | `.txt` |
