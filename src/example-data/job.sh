#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embeddings/src/example-data/%x-%j.log
#SBATCH --job-name=pejorative
#SBATCH --account=mihalcea0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dnaihao/anaconda3/envs/ann-embed/lib/
python try_loading.py
