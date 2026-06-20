#!/bin/sh
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=36:00:00
#SBATCH --output=/mnt/tank/scratch/aartamonov/adaptive-extractor/logs/optimize_re.out
#SBATCH --error=/mnt/tank/scratch/aartamonov/adaptive-extractor/logs/optimize_re.err

. ~/miniconda3/etc/profile.d/conda.sh
conda activate ae

cd "/mnt/tank/scratch/aartamonov/adaptive-extractor"

export PYTHONUNBUFFERED=1

ae-optimize --run-name nanozymes_re