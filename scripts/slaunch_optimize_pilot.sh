#!/bin/sh

. ~/miniconda3/etc/profile.d/conda.sh
conda activate ae
cd /mnt/tank/scratch/aartamonov/Adaptive Extractor

export PYTHONUNBUFFERED=1

.venv/bin/ae-optimize-syntactic || python scripts/optimize.py