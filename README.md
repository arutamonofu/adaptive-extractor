# AutoEvoExtractor

AutoEvoExtractor (AEE) is a system designed for structured information extraction from scientific literature. It utilizes DSPy and MIPROv2 to automatically optimize prompts and extraction strategies, adapting to complex domains like Nanochemistry (Nanozymes).

## Installation

The project uses Conda for environment management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AutoEvoExtractor
    ```

2.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    conda activate aee
    ```

## Configuration

The system is configured via environment variables and the `src/aee/core/config.py` file.

### 1. Environment Setup (.env)
Create a `.env` file in the root directory.

**For Cloud Mode (Gemini):**
```ini
GEMINI_API_KEY=your_api_key_here
```

**For Local Mode (Ollama):**
No API key is required, but ensure your Ollama instance is reachable.

### 2. Application Settings
Settings are managed in `src/aee/core/config.py`. Key defaults:

*   `use_local_llm`: Set to `True` for Ollama, `False` for Gemini.
*   `ollama_base_url`: Default is `https://aicltr.itmo.ru/ollama`.
*   `local_student_model`: Default `mistral-small3.1-24b-128k:latest`.
*   `local_teacher_model`: Default `gpt-oss:120b`.

## Usage: Command Line

The workflow consists of sequential stages. All scripts are located in the `scripts/` directory.

### 1. Download Data
Downloads the ground truth dataset (e.g., Nanozymes) from Hugging Face.

```bash
python scripts/download_data.py --task nanozymes --output data/ground_truth
```

### 2. Ingestion (PDF Parsing)
Converts raw PDF files into structured JSON documents.
*   **Feature:** Automatically groups Main articles and Supplementary files based on naming conventions.
*   **Note:** We output to `data/parsed` because the optimization script expects data there.

```bash
python scripts/ingest.py \
  --input data/raw \
  --output data/parsed \
  --parser docling \
  --force
```

### 3. Evolutionary Optimization
Optimizes the extraction agent using DSPy MIPROv2.
**Prerequisite:** A `data/splits.json` file is required to define `train_auto` and `val` sets to prevent data leakage.

```bash
# Available modes: 'test' (quick debug) or 'production' (full optimization)
python scripts/optimize.py \
  --task nanozymes \
  --mode production \
  --output data/agents/optimized_agent.json
```

### 4. Inference
Run the agent on documents. You can run in Zero-Shot mode or load an Optimized Agent.

**Option A: Optimized Inference**
```bash
python scripts/predict.py \
  --task nanozymes \
  --input data/parsed \
  --output data/predictions \
  --agent_path data/agents/optimized_agent.json
```

**Option B: Static Few-Shot Inference**
Uses a fixed set of examples (defined in the script) without optimization.
```bash
python scripts/predict_fewshot.py \
  --task nanozymes \
  --shots 2
```

### 5. Evaluation (Benchmark)
Calculates Precision, Recall, and F1-Score (AEE Strict Metric) and Levenshtein distance (Legacy Metric).

```bash
python scripts/benchmark.py \
  --task nanozymes \
  --gt data/ground_truth/nanozymes.csv \
  --results data/predictions/nanozymes \
  --split_file data/splits.json \
  --output benchmark_report.csv
```

## Project Structure

```text
.
├── data/                   # Data storage
│   ├── ground_truth/       # CSVs from HuggingFace
│   ├── raw/                # Input PDFs
│   ├── parsed/             # Processed JSONs (after ingest.py)
│   ├── predictions/        # Agent outputs
│   └── agents/             # Optimized DSPy programs
├── scripts/                # CLI Pipelines
│   ├── benchmark.py        # Evaluation logic
│   ├── download_data.py    # HF Dataset downloader
│   ├── ingest.py           # PDF -> JSON (Docling/Marker)
│   ├── optimize.py         # DSPy MIPROv2 Loop
│   ├── predict.py          # Batch Inference
│   └── predict_fewshot.py  # Static Few-Shot Inference
├── src/
│   └── aee/                # Core Package
│       ├── agents/         # UniversalExtractor & DSPy modules
│       ├── core/           # Config (Settings), Logging, Types
│       ├── eval/           # Matcher (Hungarian Algo) & Metrics
│       ├── ingestion/      # Parsers (Docling, PyMuPDF, etc.)
│       ├── llm.py          # LLM Factory (Ollama/Gemini wrapper)
│       ├── tasks/          # Task Schemas (Nanozymes)
│       └── utils/          # I/O and Dataset helpers
└── README.md               # Project documentation
```