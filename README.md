# AutoEvoExtractor

AutoEvoExtractor (AEE) is an evolutionary multi-agent system designed for structured information extraction from scientific literature. It utilizes DSPy and MIPROv2 to automatically optimize prompts and extraction strategies, adapting to complex domains like Nanochemistry.

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

Create a `.env` file in the root directory to configure the LLM provider (Google Gemini is the default).

```ini
GEMINI_API_KEY=your_api_key_here
```

You can customize model parameters in `src/aee/core/config.py` or via environment variables (e.g., `STUDENT_MODEL`, `TEACHER_MODEL`).

## USAGE PIPELINE

The workflow consists of five sequential stages. All scripts are located in the `scripts/` directory.

### 1. Download Data

Downloads the ground truth dataset (ChemX/Nanozymes) from Hugging Face.

```bash
python scripts/download_data.py --task nanozymes
```

### 2. Data Splitting

Generates a splits.json file to strictly separate training and testing data. This ensures reproducible experiments and prevents data leakage during optimization.

```bash    
python scripts/create_splits.py --gt data/ground_truth/nanozymes.csv
```

### 3. Ingestion (PDF Parsing)

Converts raw PDF files into structured JSON documents containing Markdown text and metadata.

```bash 
python scripts/ingest.py --input data/raw --output data/processed --parser docling
```

### 4. Evolutionary Optimization

Optimizes the extraction agent using DSPy.
**Note:** A splits.json file defining train and test sets is required to prevent data leakage.

```bash    
python scripts/optimize.py \
  --task nanozymes \
  --train_size 20 \
  --split_file data/splits.json \
  --output data/artifacts/optimized_agent.json
```

### 5. Inference

Runs the optimized agent (or a zero-shot baseline) on the dataset.

```bash
python scripts/predict.py \
  --task nanozymes \
  --agent_path data/artifacts/optimized_agent.json
```

### 6. Evaluation

Calculates Precision, Recall, and F1-Score by comparing predictions against the Ground Truth using the Hungarian Algorithm for entity alignment.

```bash
python scripts/benchmark.py \
  --task nanozymes \
  --split_file data/splits.json
```

## Project Structure

```text
.
├── data/                   # Data storage (gitignored)
├── scripts/                # Execution scripts (ETL, Inference, Eval)
├── src/
│   └── aee/
│       ├── agents/         # DSPy modules and logic
│       ├── core/           # Config, logging, types
│       ├── eval/           # Metrics and matching logic
│       ├── ingestion/      # PDF parsers (Docling, Marker, etc.)
│       └── tasks/          # Task definitions (Nanozymes, etc.)
├── environment.yml         # Conda environment definition
└── README.md               # Project documentation
```