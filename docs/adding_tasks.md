# Adding New Extraction Tasks

Guide for adding new extraction tasks to AutoEvoExtractor.

## Overview

AutoEvoExtractor uses **YAML-based configuration** for defining extraction tasks.

The system automatically generates from YAML:
- Pydantic models for experiments
- DSPy signatures for LLM extraction
- Row converters for CSV loading

**YAML approach supports:**
- Standard field types (str, int, float, bool)
- Basic validation (required/optional, choices, min/max)
- Regex patterns for string fields
- CSV column mapping via alt_names

---

## Step-by-Step Guide

### Step 1: Create Task Directory

```bash
mkdir -p src/aee/domain/tasks/{task_name}
```

### Step 2: Create Task YAML

Create `src/aee/domain/tasks/{task_name}/task.yaml`:

```yaml
name: proteins
description: Extract protein structure experiments

version: 1.0.0
tags:
  - biology
  - proteins

# Evaluation settings (all required)
compare_fields:
  - protein_name
  - structure_method
  - resolution

float_tolerance: 0.05

# Instruction file for DSPy (required)
instruction_file: config/initial_instructions/proteins_v1.txt

# Fields to extract
fields:
  protein_name:
    type: str
    description: "Protein name (e.g., 'Cytochrome C')"
    required: true
    alt_names:
      - protein
      - name

  structure_method:
    type: str
    description: "Method: X-ray, NMR, cryo-EM"
    required: true
    choices:
      - X-ray crystallography
      - NMR
      - cryo-EM

  resolution:
    type: float
    description: "Resolution in Angstroms"
    required: false
    min_value: 0

  pdb_id:
    type: str
    description: "PDB ID (e.g., '1A2B')"
    required: false
    pattern: "^[0-9][A-Za-z0-9]{3}$"

# CSV column mapping
row_converter:
  protein_name:
    - protein_name
    - protein
    - name
  structure_method:
    - structure_method
    - method
  resolution:
    - resolution
    - resolution_angstrom
```

### Step 3: Create Initial Instruction

Create `config/initial_instructions/proteins_v1.txt`:

```
You are a helpful assistant specializing in protein structures.

For each experiment, extract:
- protein_name (required): Protein name
- structure_method (required): X-ray, NMR, or cryo-EM
- resolution (optional): Resolution in Angstroms
- pdb_id (optional): PDB identifier

GUIDELINES:
1. Extract each experiment separately
2. Be precise with numerical values
3. Use null for missing information
```

### Step 4: Prepare Ground Truth

Create `data/ground_truth/proteins.csv`:

```csv
filename,protein_name,structure_method,resolution,pdb_id
paper1.pdf,Cytochrome C,X-ray crystallography,1.9,1A2B
paper2.pdf,Hemoglobin,X-ray crystallography,2.1,1A3N
paper3.pdf,GFP,cryo-EM,,
```

### Step 5: Create Data Splits

Create `data/splits/proteins.json`:

```json
{
  "train": ["paper1", "paper2"],
  "val": ["paper3"]
}
```

### Step 6: Test Task

```python
from aee.domain.tasks import get_task

task = get_task("proteins")
task["config"].validate()
print(f"✓ Task '{task['config'].name}' is valid")
print(f"  Fields: {len(task['config'].experiment_fields)}")
```

### Step 7: Run Optimization

```bash
python scripts/optimize.py --config default.yaml
```

---

## Field Specification Reference

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| `type` | str | Python type: `str`, `int`, `float`, `bool`, or `Literal` (via `choices`) | Yes |
| `description` | str | Human-readable description | Yes |
| `required` | bool | Whether field is required | No (default: true) |
| `default` | any | Default value | No |
| `choices` | list[str] | Valid choices (creates `Literal[...]` type) | No |
| `min_value` | float | Minimum for numeric | No |
| `max_value` | float | Maximum for numeric | No |
| `pattern` | str | Regex pattern for strings | No |
| `alt_names` | list[str] | Alternative CSV column names | No |

**Type examples:**
- `type: str` — string value
- `type: int` — integer value
- `type: float` — floating-point number
- `type: bool` — boolean value
- `choices: ["A", "B", "C"]` — creates `Literal["A", "B", "C"]`

---

## Tips

### Design Tips

- Use clear, descriptive field names (snake_case)
- Add detailed descriptions (helps LLM understand)
- Mark fields as `required: false` if optional
- Use `alt_names` for CSV flexibility
- Use `choices` for categorical fields

### Instruction Tips

- Be explicit about required vs optional
- Provide concrete examples
- Include guidance on handling missing data
- Keep it concise (DSPy adds examples during optimization)

### Ground Truth Tips

- Use `filename` column (with or without `.pdf`)
- Include both required and optional fields
- Leave optional fields empty (not "N/A")
- Start with 10-20 examples, expand to 50+

---

## Example Tasks

See existing tasks for reference:
- `src/aee/domain/tasks/nanozymes/task.yaml` — YAML-based task
