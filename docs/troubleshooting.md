# Troubleshooting Guide

Common issues and their solutions when using AutoEvoExtractor.

## Installation Issues

### Conda Environment Creation Fails

**Symptom:**
```bash
conda env create -f environment.yml
# Error: Package conflicts or missing dependencies
```

**Solutions:**
1. Update conda: `conda update -n base -c defaults conda`
2. Clear conda cache: `conda clean --all`
3. Try creating with specific Python version:
   ```bash
   conda create -n aee python=3.10
   conda activate aee
   pip install -r requirements.txt  # If available
   ```
4. Install dependencies manually:
   ```bash
   conda create -n aee python=3.10
   conda activate aee
   pip install dspy-ai pydantic pandas numpy
   pip install docling marker-pdf  # parsers
   pip install mlflow  # tracking
   ```

### Import Errors After Installation

**Symptom:**
```python
ImportError: cannot import name 'TaskDefinition'
ModuleNotFoundError: No module named 'aee'
```

**Solutions:**
1. Ensure you installed in editable mode:
   ```bash
   cd /path/to/autoevoextractor
   pip install -e .
   ```
2. Verify you're in the correct conda environment:
   ```bash
   conda activate aee
   which python  # Should point to conda env
   ```
3. Check PYTHONPATH:
   ```bash
   echo $PYTHONPATH
   export PYTHONPATH="/path/to/autoevoextractor/src:$PYTHONPATH"
   ```

---

## Document Parsing Issues

### Parser Not Found

**Symptom:**
```
ParserNotFoundError: Parser 'docling' not found
```

**Solutions:**
1. Install the parser:
   ```bash
   pip install docling  # For Docling
   pip install marker-pdf  # For Marker
   ```
2. Check spelling in config:
   ```yaml
   parsing:
     parser: "docling"  # Must be exactly "docling" or "marker"
   ```

### Parsing Takes Too Long

**Symptom:** PDF parsing is extremely slow (>2 minutes per page)

**Solutions:**
1. Disable OCR if not needed:
   ```yaml
   parsing:
     docling:
       do_ocr: false
   ```
2. Use GPU acceleration:
   ```yaml
   parsing:
     docling:
       device: "cuda"  # or "mps" for Mac
   ```
3. Reduce thread count if CPU is overloaded:
   ```yaml
   parsing:
     docling:
       num_threads: 2
   ```
4. Try Marker parser (sometimes faster):
   ```yaml
   parsing:
     parser: "marker"
   ```

### Out of Memory During Parsing

**Symptom:**
```
CUDA out of memory
RuntimeError: out of memory
```

**Solutions:**
1. Switch to CPU:
   ```yaml
   parsing:
     docling:
       device: "cpu"
   ```
2. Process PDFs one at a time instead of batch
3. Reduce num_threads:
   ```yaml
   parsing:
     docling:
       num_threads: 1
   ```

### Parsed Text Quality Is Poor

**Symptom:** Missing text, garbled characters, incorrect ordering

**Solutions:**
1. Enable OCR:
   ```yaml
   parsing:
     docling:
       do_ocr: true
   ```
2. Try different parser:
   - Docling is better for complex layouts
   - Marker is faster and works well for simple papers
3. Check PDF quality:
   - Scanned PDFs need OCR
   - Some PDFs have non-standard encodings
4. Manually inspect parsed JSON and verify quality

---

## LLM Connection Issues

### Ollama Connection Failed

**Symptom:**
```
ConnectionError: Could not connect to Ollama at http://localhost:11434
requests.exceptions.ConnectionError
```

**Solutions:**
1. Start Ollama:
   ```bash
   ollama serve
   ```
2. Verify Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```
3. Check correct URL in config:
   ```yaml
   llm:
     student:
       ollama:
         ollama_base_url: "http://localhost:11434"
   ```
4. If using remote Ollama:
   ```bash
   export LLM__STUDENT__OLLAMA__OLLAMA_BASE_URL="http://remote-host:11434"
   ```

### Model Not Found

**Symptom:**
```
ModelNotFoundError: Model 'llama3.2:3b' not found
```

**Solutions:**
1. List available models:
   ```bash
   ollama list
   ```
2. Pull the model:
   ```bash
   ollama pull llama3.2:3b
   ```
3. Use a different model:
   ```yaml
   llm:
     student:
       model: "llama3.2"  # Try without tag
   ```
4. Check model name spelling

### LLM Requests Timeout

**Symptom:**
```
TimeoutError: Request timed out after 600 seconds
```

**Solutions:**
1. Increase timeout:
   ```yaml
   llm:
     student:
       timeout: 1200  # 20 minutes
   ```
2. Use a smaller, faster model:
   ```yaml
   llm:
     student:
       model: "llama3.2:3b"  # Instead of larger models
   ```
3. Check system resources (CPU/GPU usage)
4. Reduce context window:
   ```yaml
   llm:
     student:
       ollama:
         num_ctx: 32000  # Smaller context
   ```

### Rate Limiting Issues

**Symptom:**
```
RateLimitError: Too many requests
429 Rate limit exceeded
```

**Solutions:**
1. Increase delay between requests:
   ```yaml
   llm:
     student:
       rate_limit_delay: 20.0  # Wait 20 seconds
   ```
2. Reduce max_retries:
   ```yaml
   llm:
     student:
       max_retries: 3
   ```
3. Use API key with higher limits
4. Enable caching to reduce requests:
   ```yaml
   optimization:
     use_cache: true
   ```

### LLM Cache Issues

**Symptom:** LLM responses not cached between runs, cache not working

**Solutions:**
1. Check cache configuration in YAML:
   ```yaml
   llm:
     student:
       enable_cache: true  # Enable caching
   ```
2. Check environment variable:
   ```bash
   export LLM__STUDENT__ENABLE_CACHE="true"
   ```
3. Clear cache if corrupted:
   ```bash
   # DSPy cache is typically in .dspy_cache or system temp directory
   rm -rf .dspy_cache
   ```
4. Verify cache directory permissions:
   ```bash
   ls -la .dspy_cache
   chmod -R u+rw .dspy_cache
   ```
5. For extraction, cache is disabled by default. Enable explicitly:
   ```bash
   python scripts/extract.py --config default.yaml --agent agent.json --enable-cache
   ```
6. For optimization, cache is enabled by default. Disable if needed by clearing cache before running:
   ```bash
   rm -rf .dspy_cache
   python scripts/optimize.py --config default.yaml
   ```

**Symptom:** Cache from optimization affects extractions (stale responses)

**Solutions:**
1. Cache is global state - it persists across script runs
2. For extraction, cache is disabled by default - simply don't use `--enable-cache`
3. Clear cache between different experiment runs:
   ```bash
   rm -rf .dspy_cache
   ```
4. Log cache status to verify:
   ```
   INFO: LLM cache: ENABLED (--enable-cache: True)
   ```

---

## Optimization Issues

### Optimization Fails Immediately

**Symptom:**
```
ValueError: No training examples found
FileNotFoundError: Ground truth file not found
```

**Solutions:**
1. Verify ground truth file exists:
   ```bash
   ls -la data/ground_truth/nanozymes.csv
   ```

### Pre-flight Validation Failed

**Symptom:**
```
ERROR: OptimizeAgent failed: Pre-flight validation failed with 2 error(s):
  1. Split 'val' contains 2 document(s) not found in ground truth: ['doc_025', 'doc_026']
  2. Validation split is empty
```

**Solutions:**
1. Check that all document IDs in splits file exist in ground truth CSV:
   ```bash
   # Check splits file
   cat data/splits/nanozymes.json | jq .

   # Check ground truth filenames
   cut -d',' -f1 data/ground_truth/nanozymes.csv | sort -u
   ```
2. Regenerate splits file with correct document IDs:
   ```python
   from pathlib import Path
   from aee.infrastructure.storage import DataSplitRepository
   import pandas as pd

   df = pd.read_csv('data/ground_truth/nanozymes.csv')
   doc_ids = df['filename'].str.replace('.pdf', '').unique().tolist()

   repo = DataSplitRepository()
   splits = repo.create_random_split(documents=doc_ids, train_ratio=0.8, seed=42)
   repo.save_splits(splits, Path('data/splits/nanozymes.json'))
   ```
3. Verify no overlap between train and val splits:
   ```python
   import json
   with open('data/splits/nanozymes.json') as f:
       splits = json.load(f)
   overlap = set(splits['train']) & set(splits['val'])
   if overlap:
       print(f"Overlap found: {overlap}")
   ```
4. Check that splits file is valid JSON:
   ```bash
   python -m json.tool data/splits/nanozymes.json > /dev/null && echo "Valid JSON"
   ```
5. Ensure ground truth CSV has required columns:
   ```bash
   head -1 data/ground_truth/nanozymes.csv
   ```

---

### Low Optimization Scores

**Symptom:** Final F1 score < 0.5 after optimization

**Solutions:**
1. Increase trials:
   ```yaml
   optimization:
     num_trials: 50  # More trials = better results
   ```
2. Add more training examples:
   - Aim for 50+ ground truth examples
   - Ensure variety in examples
3. Improve DSPy signature instructions:
   - Be more specific about what to extract
   - Add examples of phrases to look for
   - Clarify edge cases
4. Use a more capable model:
   ```yaml
   llm:
     teacher:
       model: "llama3.1:70b"  # Larger model
   ```
5. Adjust compare_fields:
   - Remove fields that are hard to match
   - Focus on core identifying fields
6. Check ground truth quality:
   - Verify accuracy
   - Ensure consistency

### Optimization Hangs

**Symptom:** Optimization starts but never completes, no progress

**Solutions:**
1. Enable verbose mode:
   ```yaml
   optimization:
     verbose: true
   ```
2. Check logs in `logs/` directory
3. Reduce dataset size for testing:
   ```yaml
   optimization:
     total_load: 10
     train_split: 8
   ```
4. Verify LLM is responding:
   ```bash
   curl http://localhost:11434/api/generate \
     -d '{"model": "llama3.2:3b", "prompt": "test"}'
   ```

### Out of Memory During Optimization

**Symptom:**
```
MemoryError
CUDA out of memory
```

**Solutions:**
1. Enable minibatch:
   ```yaml
   optimization:
     minibatch: true
     minibatch_size: 5
   ```
2. Reduce total_load:
   ```yaml
   optimization:
     total_load: 20  # Smaller dataset
   ```
3. Reduce demonstrations:
   ```yaml
   optimization:
     max_bootstrapped_demos: 2
     max_labeled_demos: 8
   ```
4. Use smaller model
5. Increase system swap space

---

## Task Plugin Issues

### Task Not Found

**Symptom:**
```
TaskNotFoundError: Task 'proteins' not found
```

**Solutions:**
1. Ensure task module is imported somewhere:
   ```python
   # In your script or optimization command
   from aee.domain.tasks.proteins import ProteinTask
   ```
2. Verify task registration:
   ```python
   from aee.domain.tasks import get_global_registry
   registry = get_global_registry()
   tasks = registry.list_tasks()
   print([t.name for t in tasks])  # Should include your task
   ```
3. Check task plugin `__init__.py` has registration call:
   ```python
   from aee.domain.tasks import register_task
   register_task(ProteinTask(initial_instruction="..."))
   ```
4. Verify task name matches:
   ```python
   class ProteinTask(TaskDefinition):
       @property
       def name(self) -> str:
           return "proteins"  # Must match CLI argument
   ```
5. For optimization, ensure task is registered in `scripts/optimize.py`:
   ```python
   # The optimize script loads tasks dynamically based on config
   # Make sure your task's initial_instruction is loaded from config
   ```

### Task Validation Fails

**Symptom:**
```
ValidationError: Task validation failed
```

**Solutions:**
1. Run validation manually:
   ```python
   from aee.domain.tasks.proteins import ProteinTask
   task = ProteinTask()
   task.validate()  # See specific error
   ```
2. Check all required properties are implemented:
   - `name`
   - `signature`
   - `output_model`
   - `experiment_model`
   - `row_converter`
   - `compare_fields`
3. Verify signature is valid DSPy signature
4. Ensure models inherit from correct base classes

### Row Converter Returns None

**Symptom:** All experiments are None, empty extraction results

**Solutions:**
1. Check CSV column names:
   ```python
   import pandas as pd
   df = pd.read_csv("data/ground_truth/task.csv")
   print(df.columns)
   ```
2. Debug converter:
   ```python
   from aee.domain.tasks.proteins import row_to_protein
   row = df.iloc[0]
   exp = row_to_protein(row)
   print(exp)  # Should not be None
   ```
3. Handle multiple column name variants:
   ```python
   protein_name = (
       row.get("protein_name") or
       row.get("protein") or
       row.get("name")
   )
   ```
4. Check for NaN values:
   ```python
   if pd.isna(protein_name):
       return None
   ```

---

## Evaluation Issues

### Zero Matches Found

**Symptom:** Precision/Recall = 0.0, no experiments matched

**Solutions:**
1. Check compare_fields are present in both predicted and ground truth
2. Verify field names match exactly (case-sensitive)
3. Inspect an extraction vs ground truth pair:
   ```python
   print("Extracted:", extracted_experiments[0])
   print("Ground Truth:", ground_truth_experiments[0])
   ```
4. Adjust float_tolerance if comparing floats:
   ```yaml
   task:
     evaluation:
       float_tolerance: 0.1  # More lenient
   ```
5. Implement custom matcher if default doesn't work

### Metrics Don't Make Sense

**Symptom:** Precision = 1.0 but Recall = 0.0 (or vice versa)

**Solutions:**
1. Understand metric definitions:
   - Precision = TP / (TP + FP) - How many predicted are correct
   - Recall = TP / (TP + FN) - How many actual were found
   - F1 = harmonic mean of precision and recall
2. Check if over/under-extracting:
   - High precision, low recall = too conservative
   - Low precision, high recall = too permissive
3. Review compare_fields:
   - Too strict = no matches
   - Too loose = false matches
4. Manually review extractions:
   ```bash
   cat data/extractions/output.json | jq .
   ```

---

## File and Path Issues

### File Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/...'
```

**Solutions:**
1. Check working directory:
   ```bash
   pwd
   # Should be project root
   ```
2. Verify paths in config:
   ```yaml
   paths:
     parsed_dir: "data/parsed"  # Relative to project root
   ```
3. Create missing directories:
   ```bash
   mkdir -p data/parsed data/agents data/extractions
   ```
4. Use absolute paths if needed:
   ```yaml
   paths:
     parsed_dir: "/absolute/path/to/data/parsed"
   ```

### Permission Denied

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'data/agents/...'
```

**Solutions:**
1. Check file permissions:
   ```bash
   ls -la data/agents/
   ```
2. Fix permissions:
   ```bash
   chmod -R u+rw data/
   ```
3. Check disk space:
   ```bash
   df -h
   ```
4. Verify not running in protected directory

---

## Performance Issues

### Everything Is Very Slow

**Solutions:**
1. Use GPU if available:
   ```yaml
   parsing:
     docling:
       device: "cuda"
   ```
2. Enable LLM caching:
   ```yaml
   optimization:
     use_cache: true
   ```
3. Use faster model:
   ```yaml
   llm:
     student:
       model: "llama3.2:3b"  # Smaller/faster
   ```
4. Reduce trials for testing:
   ```yaml
   optimization:
     num_trials: 3
   ```
5. Process fewer documents initially
6. Check system resources (CPU, RAM, GPU)

### High Memory Usage

**Solutions:**
1. Enable minibatch:
   ```yaml
   optimization:
     minibatch: true
     minibatch_size: 5
   ```
2. Reduce context window:
   ```yaml
   llm:
     student:
       ollama:
         num_ctx: 32000
   ```
3. Process fewer documents at once
4. Reduce num_threads:
   ```yaml
   parsing:
     docling:
       num_threads: 2
   ```

---

## Getting More Help

If you're still experiencing issues:

1. **Check logs**: Look in `logs/` directory for detailed error messages
2. **Enable debug logging**:
   ```yaml
   project:
     log_level: "DEBUG"
   ```
3. **Search issues**: Check [GitHub Issues](https://github.com/ai-chem/AutoEvoExtractor/issues)
4. **Ask for help**: Open a new issue with:
   - Complete error message
   - Configuration file
   - Steps to reproduce
   - System information (OS, Python version, etc.)
