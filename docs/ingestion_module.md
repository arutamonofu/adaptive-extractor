# Ingestion Module Architecture & Workings

## 1. Overview & Core Features

The **Ingestion** module converts raw chemistry PDF papers into structured Markdown. This is the first step in the Adaptive Extractor pipeline, preceding prompt optimization and final data extraction.

### Key Capabilities:
*   **MinerU PDF Parsing**: Standard PDF parsing using the MinerU API to extract Markdown (`full.md`), list of content blocks with layout bounding boxes and labels (`content_list.json`), and cropped images (`images/`).
*   **Visual Reverse Engineering**: Automatic identification of chart blocks (`"type": "chart"`), running multimodal VLM (e.g. Gemini 3.5 Flash) on the cropped chart images, and converting them into structured Markdown tables.
*   **Table Insertion**: Substituting Markdown image links (`![](images/<hash>.jpg)`) in the text with the rendered Markdown tables.

---

## 2. Command Line Interface & Usage

*   **CLI Command**: `ae-parse` (defined in [cli.py](../src/ae/ingestion/cli.py)).
*   **CLI Usage**:
    ```bash
    ae-parse [--config CONFIG_DIR] [--overwrite]
    ```
*   **Arguments & Flags**:

| Flag / Option | Argument Type | Description |
| :--- | :--- | :--- |
| `--config` | Path | Path to configuration directory (defaults to root `config/` directory). |
| `--overwrite` | flag | If present, overwrites existing parsed Markdown files in the output directory. |

---

## 3. Architecture & Key Code Components

The Ingestion module implements the Strategy Pattern to dynamically switch between standard and visually enriched parsing engines.

*   **Key Code Components**:

| File Link | Class / Function | Role / Description |
| :--- | :--- | :--- |
| [pipeline.py](../src/ae/ingestion/pipeline.py) | [ParseDocumentsUseCase](../src/ae/ingestion/pipeline.py#L59) | Orchestrates the end-to-end PDF-to-Markdown batch parsing pipeline. |
| [base_parser.py](../src/ae/ingestion/base_parser.py) | [BaseParser](../src/ae/ingestion/base_parser.py#L9) | Abstract strategy interface defining the `.parse()` contract. |
| [parsers.py](../src/ae/ingestion/parsers.py) | [get_parser](../src/ae/ingestion/parsers.py#L14) | Factory function that instantiates the parser based on selected configurations. |
| [mineru_client.py](../src/ae/ingestion/mineru_client.py) | [MinerUClient](../src/ae/ingestion/mineru_client.py#L12) | Client wrapper around MinerU Web API with HTTP retries. |
| [mineru_parser.py](../src/ae/ingestion/mineru_parser.py) | [MinerUParser](../src/ae/ingestion/mineru_parser.py#L46) | Ingestion parser integrating MinerU client and visual reverse engineering. |
| [extract_chart_tables.py](../src/ae/ingestion/visual_pipeline/stages/extract_chart_tables.py) | [extract_single_chart](../src/ae/ingestion/visual_pipeline/stages/extract_chart_tables.py#L14) | Multimodal VLM extraction helper that parses cropped chart images into tabular JSON. |
| [insert_visual_tables.py](../src/ae/ingestion/visual_pipeline/stages/insert_visual_tables.py) | [replace_image_tags](../src/ae/ingestion/visual_pipeline/stages/insert_visual_tables.py#L58) | Replaces Markdown image tags with structured Markdown tables. |

---

## 4. Configuration & Parameter Mapping

Configuration parameters are loaded from `config/core.yaml` and `config/ingestion.yaml`:

| YAML Path | Variable Mapping | Type | Description |
| :--- | :--- | :--- | :--- |
| `paths.pdf_dir` | `custom_settings.paths.pdf_dir` | Path | Directory containing raw input PDFs (Default: `data/raw/pdf`). |
| `paths.ingestion_dir` | `custom_settings.paths.ingestion_dir` | Path | Directory where parsed Markdown results are stored (Default: `data/interim/ingestion`). |
| `parsing.mineru.api_url` | `custom_settings.parsing.mineru.api_url` | str | MinerU API base URL (Default: `https://mineru.net/api/v4`). |
| `parsing.mineru.model_version` | `custom_settings.parsing.mineru.model_version` | str | MinerU model version (Default: `vlm`). |
| `parsing.mineru.poll_interval` | `custom_settings.parsing.mineru.poll_interval` | int | Interval in seconds between status polls (Default: `3`). |
| `parsing.mineru.poll_timeout` | `custom_settings.parsing.mineru.poll_timeout` | int | Timeout in seconds for parsing (Default: `600`). |
| `parsing.chart_extraction.model` | `custom_settings.parsing.chart_extraction.model` | str | Gemini/VLM model used to extract chart tables (Default: `gemini-3.5-flash`). |

> [!NOTE]
> Authorization token for MinerU Web API is not stored in YAML configs; it is read dynamically from the `MINERU_API_TOKEN` environment variable (in `.env`).

---

## 5. Module Workings & Data Flow

```mermaid
flowchart TD
    PDF[Raw PDF Document] -->|Upload & Poll| MU[MinerU Web API]
    MU -->|Extract Zip| Workspace[MinerU Intermediate Directory]
    Workspace -->|Read| MD[full.md]
    Workspace -->|Read| JSON[content_list.json]
    
    JSON -->|Filter type: 'chart'| Filter[Chart Images List]
    Filter -->|For each chart| VLM["VLM (Gemini-3.5-Flash)
    Extract Table JSON"]
    VLM -->|Render| MarkdownTables[Markdown Tables]
    
    MarkdownTables -->|Replace image tags
    '![]\\(images/img_name.jpg\\)'| MD
    
    MD -->|Save final text| FinalMD[article.md]
    Workspace -->|Save for debug| DebugFolder[data/interim/ingestion/mineru_artifacts/pdf_stem]
```

### Detailed Phases:
1.  **MinerU Ingestion**: `MinerUClient` uploads the PDF, polls the API for completion, downloads the zipped extraction package, and unpacks it.
2.  **Chart Filtering**: The parser scans the content list (`*_content_list.json`) for blocks tagged with `"type": "chart"` to extract their local image paths and captions.
3.  **VLM Extraction**: For each chart, `extract_single_chart` is invoked with the cropped image and task extraction instructions, query-handling the multimodal model (Gemini 3.5 Flash) to generate formatted JSON tables.
4.  **Markdown Ingestion**: `replace_image_tags` locates the corresponding image tag `![](images/...)` inside the Markdown document and replaces it with the generated Markdown tables.
5.  **Audit & Debug**: Intermediate files (raw Markdown, content lists, cropped images, raw responses) are saved to `data/interim/ingestion/mineru_artifacts/<pdf_stem>`.

---

## 6. Input/Output Data Formats

### Workspace Directory Layout:
```text
├── data/
│   └── raw/
│       └── pdf/
│           └── article.pdf                   # Raw input PDF document
├── data/interim/
│   └── ingestion/
│       ├── article.md                        # Final enriched output Markdown
│       └── mineru_artifacts/
│           └── article/
│               ├── full.md               # Raw MinerU Markdown
│               ├── content_list.json     # MinerU layout block content list
│               ├── final_enriched.md     # Enriched Markdown
│               ├── enrichment_summary.json # Summary of visual enrichment results
│               └── images/
│                   └── <hash>.jpg        # Cropped chart images
```

---

## 7. Error Handling & Resiliency

*   **HTTP/Network Retries**: The parser uses simple exponential backoff for requests made by the MinerU client on 5xx status codes or connection errors.
*   **VLM Recovery**: JSON parsing failures on VLM outputs are automatically retried by increasing model output token allowances or requesting formatting adjustments.
