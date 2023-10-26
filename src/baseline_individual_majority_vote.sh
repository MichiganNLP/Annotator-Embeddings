#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embeddings/experiment-results/%x-%j.log
#SBATCH --job-name=individual_majority_vote
#SBATCH --account=mihalcea0

python src/baseline_models/individual_majority_vote.py
