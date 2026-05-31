# DSPy MIPROv2 metric_threshold Bug Fix

## Problem

When optimizing with `metric_threshold = 1.0`, MIPROv2 was selecting examples with metric 0.8 as "full traces", even though they should have been filtered out.

### Error Message
```
Bootstrapped 2 full traces after 2 examples for up to 1 rounds, amounting to 2 attempts.
```

## Root Cause

In `dspy/teleprompt/utils.py`, the function `create_n_fewshot_demo_sets()` was not passing `metric_threshold` to `BootstrapFewShot` for the unshuffled few-shot case (`seed == -1`).

## Solution

Added `metric_threshold` parameter:

```python
# BEFORE (incorrect)
teleprompter = BootstrapFewShot(
    metric=metric,
    max_errors=max_errors,
    max_bootstrapped_demos=max_bootstrapped_demos,
    # metric_threshold was NOT passed!
)

# AFTER (correct)
teleprompter = BootstrapFewShot(
    metric=metric,
    max_errors=max_errors,
    metric_threshold=metric_threshold,  # ← ADDED
    max_bootstrapped_demos=max_bootstrapped_demos,
)
```

## Applying the Patch

```bash
python scripts/patch_dspy_mipro_threshold.py
```

## Verification

**Option 1: Check patched file directly**
```bash
grep -A 10 "elif seed == -1:" $(python -c "import dspy; print(dspy.__file__.replace('__init__.py', 'teleprompt/utils.py'))")
```

**Option 2: Verify via patch script** (recommended)
```bash
# The patch script checks if already patched
python scripts/patch_dspy_mipro_threshold.py
# Output: "✓ File already patched!" if successful
```

Expected output (Option 1):
```python
elif seed == -1:
    teleprompter = BootstrapFewShot(
        metric=metric,
        max_errors=max_errors,
        metric_threshold=metric_threshold,  # ← Should be present
        ...
```

## Impact

After patching:
- Only examples meeting `metric_threshold` are selected
- Optimization quality improves
- Fewer wasted trials on poor demonstrations
