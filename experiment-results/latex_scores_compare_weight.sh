#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embedding/experiment-results/%x-%j.log
#SBATCH --job-name=latex_scores_compare_weight
#SBATCH --account=linmacse0


python latex_scores_compare_weight.py
