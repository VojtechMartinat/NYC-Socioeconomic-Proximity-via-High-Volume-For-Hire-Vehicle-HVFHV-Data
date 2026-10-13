#!/bin/bash
#SBATCH --job-name=fare_prediction_train_classifier
#SBATCH --account=coms038604
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.out

# Submit this job from the project root (ADS-Coursework):
# sbatch src/fare_prediction/train_xgb_classifier.sh

set -eu

source ~/miniconda3/etc/profile.d/conda.sh
cd "$SLURM_SUBMIT_DIR"
conda activate py312

python -m src.fare_prediction.train_xgb_classifier