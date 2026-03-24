#!/usr/bin/env python
"""Patch for DSPy MIPROv2 zero-shot optimization support.

This script patches the DSPy MIPROv2 optimizer to properly support zero-shot
optimization (max_bootstrapped_demos=0, max_labeled_demos=0).

Bugs fixed:
1. utils.py: randint(1, 0) error when max_bootstrapped_demos=0
2. mipro_optimizer_v2.py: Constants override zero values (3 instead of 0)
3. mipro_optimizer_v2.py: Demos not cleared from predictors after compilation

Issue: https://github.com/stanfordnlp/dspy/issues/9039
"""

import os
import re
from pathlib import Path


def patch_dspy_utils():
    """Patch utils.py to handle max_bootstrapped_demos=0 without randint error."""

    # Find DSPy installation
    import dspy
    dspy_dir = Path(dspy.__file__).parent
    utils_path = dspy_dir / "teleprompt" / "utils.py"

    print(f"Patching DSPy utils.py at: {utils_path}")

    # Read the original file
    with open(utils_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already patched
    if "# PATCHED: Handle zero-shot case (max_bootstrapped_demos=0)" in content:
        print("✓ utils.py already patched!")
        return True

    # Find and replace the shuffled few-shot case (seed >= 0)
    # Original code (lines ~395-410):
    # else:
    #     # shuffled few-shot
    #     rng.shuffle(trainset_copy)
    #     size = rng.randint(min_num_samples, max_bootstrapped_demos)
    #
    #     teleprompter = BootstrapFewShot(...)

    old_code = """        else:
            # shuffled few-shot
            rng.shuffle(trainset_copy)
            size = rng.randint(min_num_samples, max_bootstrapped_demos)

            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )

            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy,
            )"""

    new_code = """        else:
            # shuffled few-shot
            # PATCHED: Handle zero-shot case (max_bootstrapped_demos=0) to avoid randint(1, 0) error
            rng.shuffle(trainset_copy)

            if max_bootstrapped_demos <= 0:
                # Zero-shot: skip bootstrap, use empty demos
                program2 = student.reset_copy()
            else:
                size = rng.randint(min_num_samples, max_bootstrapped_demos)

                teleprompter = BootstrapFewShot(
                    metric=metric,
                    max_errors=max_errors,
                    metric_threshold=metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=max_labeled_demos,
                    teacher_settings=teacher_settings,
                    max_rounds=max_rounds,
                )

                program2 = teleprompter.compile(
                    student,
                    teacher=teacher,
                    trainset=trainset_copy,
                )"""

    if old_code not in content:
        print("✗ Could not find the code section to patch in utils.py!")
        print("The DSPy version may have changed.")
        return False

    # Apply the patch
    patched_content = content.replace(old_code, new_code)

    # Write the patched file
    with open(utils_path, "w", encoding="utf-8") as f:
        f.write(patched_content)

    print("✓ Successfully patched DSPy utils.py!")
    print("\nChanges made:")
    print("  - Added check for max_bootstrapped_demos <= 0")
    print("  - Skip BootstrapFewShot for zero-shot mode")
    print("  - Use student.reset_copy() for empty demos")

    return True


def patch_mipro_optimizer():
    """Patch mipro_optimizer_v2.py to:
    1. Clear demos from predictors for zero-shot mode
    2. Pass actual parameter values instead of constants
    """

    # Find DSPy installation
    import dspy
    dspy_dir = Path(dspy.__file__).parent
    mipro_path = dspy_dir / "teleprompt" / "mipro_optimizer_v2.py"

    print(f"\nPatching DSPy mipro_optimizer_v2.py at: {mipro_path}")

    # Read the original file
    with open(mipro_path, "r", encoding="utf-8") as f:
        content = f.read()

    patched_count = 0

    # Patch 1: Clear demos from predictors for zero-shot mode
    # Original code (lines ~220-222):
    # # If zero-shot, discard demos
    # if zeroshot_opt:
    #     demo_candidates = None

    old_code_1 = """        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None"""

    new_code_1 = """        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None
            # PATCHED: Clear demos from predictors to ensure true zero-shot mode
            for i, predictor in enumerate(program.predictors()):
                predictor.demos = []"""

    if old_code_1 not in content:
        print("✗ Could not find code section to patch (clearing demos)!")
    else:
        content = content.replace(old_code_1, new_code_1)
        print("  - Added clearing of predictor.demos for zero-shot mode")
        patched_count += 1

    # Patch 2: Pass actual parameter values instead of constants
    # Original code (lines ~424-427):
    # max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_labeled_demos),
    # max_bootstrapped_demos=(
    #     BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_bootstrapped_demos
    # ),

    old_code_2 = """            max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_labeled_demos),
            max_bootstrapped_demos=(
                BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_bootstrapped_demos
            ),"""

    new_code_2 = """        # PATCHED: Pass actual parameter values instead of constants for zero-shot support
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,"""

    if old_code_2 not in content:
        print("✗ Could not find code section to patch (constants override)!")
    else:
        content = content.replace(old_code_2, new_code_2)
        print("  - Pass actual max_bootstrapped_demos/max_labeled_demos values")
        patched_count += 1

    if patched_count == 0:
        print("✗ No patches applied to mipro_optimizer_v2.py!")
        print("The DSPy version may have changed.")
        return False

    # Write the patched file
    with open(mipro_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✓ Successfully patched DSPy mipro_optimizer_v2.py!")
    print(f"\nChanges made: {patched_count}/2")

    return True


def main():
    """Run all patches."""
    print("=" * 60)
    print("DSPy MIPROv2 Zero-Shot Optimization Patch")
    print("=" * 60)

    success = True

    # Patch utils.py
    if not patch_dspy_utils():
        success = False

    # Patch mipro_optimizer_v2.py
    if not patch_mipro_optimizer():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All patches applied successfully!")
        print("\nZero-shot optimization is now supported:")
        print("  - max_bootstrapped_demos: 0")
        print("  - max_labeled_demos: 0")
    else:
        print("✗ Some patches failed to apply.")
        print("Check the DSPy version and try again.")
    print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
