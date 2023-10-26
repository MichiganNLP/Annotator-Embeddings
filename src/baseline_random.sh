#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embedding/experiment-results/%x-%j.log
#SBATCH --job-name=random-annotation
#SBATCH --account=mihalcea0

python src/baseline_models/random_baseline.py
