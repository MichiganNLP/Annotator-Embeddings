#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embeddings/experiment-results/%x-%j.log
#SBATCH --job-name=overall_majority_vote_annotation
#SBATCH --account=mihalcea0

python src/baseline_models/overall_majority_vote.py
