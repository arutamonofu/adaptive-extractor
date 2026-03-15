# Gemini PDF Parser Guide

**Google Gemini API-based PDF-to-Markdown conversion for chemistry papers.**

## Overview

The Gemini parser uses Google's Gemini API to convert PDF documents into Markdown format. It is specifically optimized for chemistry papers with support for:

- Scientific characters (Greek letters, mathematical symbols)
- Chemical and mathematical notation (subscripts, superscripts, LaTeX)
- Tables (converted to HTML)
- Figure captions and references
- Two-column layout handling

## Comparison: Gemini vs Marker

| Feature | Gemini Parser | Marker Parser |
|---------|---------------|---------------|
| **Processing** | Cloud API (Google servers) | Local (CPU/GPU) |
| **Accuracy** | High (LLM-based understanding) | Good (ML model-based) |
| **Speed** | Depends on API latency + upload time | Depends on local hardware |
| **Cost** | Pay per API call | Free (local compute only) |
| **Setup** | Requires API key | Requires model download |
| **Best for** | Complex layouts, high accuracy | Batch processing, offline use |

## Setup

### 1. Install Dependencies

```bash
pip install google-genai
```

Or update your environment:

```bash
pip install -e .
```

### 2. Configure API Key

Add your Gemini API key to `.env`:

```bash
GEMINI_API_KEY=your_api_key_here
```

> **Get API Key:** Visit [Google AI Studio](https://aistudio.google.com/apikey) to obtain an API key.

### 3. Configure YAML

Create or update your system configuration:

```yaml
# config/systems/gemini.yaml
parsing:
  parser: "gemini"
  overwrite: false
  
  gemini:
    model_name: "gemini-3-flash-preview"
    upload_timeout: 300
    safety_settings: true

paths:
  parsed_dir: "data/parsed/gemini_ocr"  # Separate from Marker output
```

### 4. Run Parsing

```bash
python scripts/parse.py --config config/systems/gemini.yaml
```

## Configuration Options

### `parsing.gemini.model_name`

**Type:** `str`  
**Default:** `"gemini-3-flash-preview"`  
**Description:** Gemini model to use for PDF-to-Markdown conversion.

Available models (check [Google AI documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for latest):
- `gemini-3-flash-preview` - Fast, cost-effective
- `gemini-2.0-flash` - Previous generation
- `gemini-1.5-pro` - Higher accuracy, slower

### `parsing.gemini.upload_timeout`

**Type:** `int`  
**Default:** `300` (5 minutes)  
**Description:** Timeout in seconds for file upload to Google servers.

Increase for large PDF files or slow network connections.

### `parsing.gemini.safety_settings`

**Type:** `bool`  
**Default:** `true`  
**Description:** Enable safety settings for Gemini API.

When enabled, sets all harm categories to `BLOCK_NONE` to allow processing of scientific content that might trigger false positives.

## Usage Examples

### Basic Usage

```bash
# Parse with default Gemini config
python scripts/parse.py --config config/systems/gemini.yaml
```

### Parse with Overwrite

```bash
# Re-parse existing files
python scripts/parse.py --config config/systems/gemini.yaml --overwrite
```

### Custom Configuration

```yaml
# config/systems/gemini_custom.yaml
parsing:
  parser: "gemini"
  overwrite: false
  
  gemini:
    model_name: "gemini-1.5-pro"  # Higher accuracy model
    upload_timeout: 600            # 10 minutes for large files
    safety_settings: true

paths:
  parsed_dir: "data/parsed/gemini_pro"
```

## Cost Estimation

Gemini API pricing (check [official pricing](https://ai.google.dev/pricing) for current rates):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Gemini 3 Flash | ~$0.075 | ~$0.30 |
| Gemini 1.5 Pro | ~$1.25 | ~$5.00 |

**Estimated tokens per paper:**
- 10-page paper: ~5,000-10,000 tokens
- 30-page paper: ~15,000-30,000 tokens

**Example cost for 100 papers (10 pages each) with Gemini 3 Flash:**
- Total tokens: ~750,000
- Estimated cost: ~$0.06 (input) + ~$0.23 (output) = **~$0.29**

> **Note:** Actual costs vary based on paper length and content density.

## Troubleshooting

### Error: `GEMINI_API_KEY environment variable must be set`

**Cause:** API key not configured.

**Solution:**
```bash
# Add to .env file
GEMINI_API_KEY=your_api_key_here
```

### Error: `Failed to process file on Google server`

**Cause:** File upload failed or server-side processing error.

**Solutions:**
1. Check file is a valid PDF
2. Increase `upload_timeout` in config
3. Check network connectivity
4. Verify API key has sufficient quota

### Error: `Empty response from Gemini`

**Cause:** Model returned no content.

**Solutions:**
1. Check PDF is not corrupted
2. Verify PDF contains text (not scanned images only)
3. Try a different model (e.g., `gemini-1.5-pro`)

### Error: Rate limit exceeded

**Cause:** Too many requests in short time.

**Solutions:**
1. Add delays between requests in batch processing
2. Request higher quota from Google
3. Use exponential backoff in custom scripts

### High API costs

**Solutions:**
1. Use `gemini-3-flash-preview` for cost-effective processing
2. Pre-filter PDFs to remove unnecessary files
3. Use local Marker parser for batch processing
4. Cache results to avoid re-processing

## Best Practices

### 1. Use Separate Output Directories

Keep Gemini and Marker outputs separate for comparison:

```yaml
# Gemini
parsed_dir: "data/parsed/gemini_ocr"

# Marker
parsed_dir: "data/parsed/marker_ocr"
```

### 2. Validate Results

Sample parsed outputs to verify quality:

```bash
# Check first few files
head -n 50 data/parsed/gemini_ocr/paper1.md
```

### 3. Monitor API Usage

Track your API usage in [Google Cloud Console](https://console.cloud.google.com/).

### 4. Handle Large Batches

For large batches, consider:
- Processing in chunks
- Adding retry logic
- Monitoring costs

### 5. Fallback Strategy

Use Gemini for difficult PDFs, Marker for standard ones:

```bash
# Parse with Marker first
python scripts/parse.py --config config/systems/example.yaml

# Then parse problematic files with Gemini
python scripts/parse.py --config config/systems/gemini.yaml --overwrite
```

## Prompt Engineering

The Gemini parser uses a specialized prompt for chemistry papers. Key features:

- **Mechanical conversion** - Copy text faithfully, avoid summarization
- **Complete coverage** - Process entire document including appendices
- **Two-column handling** - Read left column first, then right
- **Scientific notation** - Preserve Greek letters, mathematical symbols
- **Tables** - Convert to HTML format
- **Figures** - Include captions, skip images

To customize the prompt, modify `GEMINI_PDF_TO_MD_PROMPT` in `src/aee/infrastructure/parsers/parsers.py`.

## API Reference

### `GeminiParser`

```python
from aee.infrastructure.parsers import GeminiParser
from aee.infrastructure.config.settings import GeminiParserConfig

config = GeminiParserConfig(
    model_name="gemini-3-flash-preview",
    upload_timeout=300,
    safety_settings=True,
)

parser = GeminiParser(config)
markdown = parser.parse(Path("paper.pdf"))
```

### `get_parser()` Factory

```python
from aee.infrastructure.parsers import get_parser
from aee.infrastructure.config.settings import GeminiParserConfig

config = GeminiParserConfig()
parser = get_parser("gemini", config)
```

## Related Documents

- [CLI Reference](cli_reference.md) - Command-line usage
- [Configuration Reference](configuration.md) - YAML configuration
- [Architecture](architecture.md) - System design

## External Resources

- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Google AI Studio](https://aistudio.google.com/)
- [Gemini API Pricing](https://ai.google.dev/pricing)
