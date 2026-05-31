# DSPy MIPROv2 Optimization Patches

To run prompt optimization successfully under various configurations (such as zero-shot optimization or specific evaluation thresholds), DSPy MIPROv2 requires two critical patches.

---

## 1. Metric Threshold Bug Fix (`patch_dspy_mipro_threshold.py`)

### Problem
When optimizing with `metric_threshold = 1.0` (or other high limits), MIPROv2 was selecting examples with lower metric scores (e.g. `0.8`) as "full traces", even though they should have been filtered out.

### Root Cause
In `dspy/teleprompt/utils.py`, the function `create_n_fewshot_demo_sets()` was not passing the `metric_threshold` parameter to `BootstrapFewShot` for the unshuffled few-shot case (`seed == -1`).

### Solution
The patch adds the missing `metric_threshold` parameter to the `BootstrapFewShot` constructor inside `utils.py`.

---

## 2. Zero-Shot Optimization Bug Fix (`patch_dspy_mipro_zero_bootstrap.py`)

### Problem
When setting `max_bootstrapped_demos = 0` (focusing on pure instruction tuning without few-shot examples), the optimization process crashed with a random range error:
```text
ValueError: empty range for randrange() (1, 0)
```

### Root Cause
1.  **Randrange Error**: In `dspy/teleprompt/utils.py`, when `max_bootstrapped_demos=0`, the code executed `rng.randint(min_num_samples, max_bootstrapped_demos)` which translates to `randint(1, 0)`, raising a ValueError.
2.  **Constants Override**: In `mipro_optimizer_v2.py`, internal constants were overriding the zero values, enforcing 3 bootstrapped demos.
3.  **Demos Pollution**: Demos were not properly cleared from predictors after bootstrap compilation, polluting zero-shot runs.

### Solution
The patch modifies `utils.py` to skip bootstrapping entirely if `max_bootstrapped_demos <= 0` and corrects the constant check/demo clearing behaviors in `mipro_optimizer_v2.py`.

---

## 3. Applying the Patches

Both patches can be applied using the scripts in the `scripts/` directory:

```bash
# Apply metric threshold patch
python scripts/patch_dspy_mipro_threshold.py

# Apply zero-shot support patch
python scripts/patch_dspy_mipro_zero_bootstrap.py
```

### Verification

To verify if the patches are applied:
```bash
# Run the patch scripts directly. They check if the files are already patched.
python scripts/patch_dspy_mipro_threshold.py
# Output should be: "✓ File already patched!"

python scripts/patch_dspy_mipro_zero_bootstrap.py
# Output should be: "✓ utils.py already patched!" / "✓ mipro_optimizer_v2.py already patched!"
```

---

## 4. Impact

After applying these patches:
*   Zero-shot optimization works correctly with `max_bootstrapped_demos: 0`.
*   Only demonstration traces meeting the strict `metric_threshold` are used.
*   Prompt tuning becomes significantly more robust and cost-efficient.
