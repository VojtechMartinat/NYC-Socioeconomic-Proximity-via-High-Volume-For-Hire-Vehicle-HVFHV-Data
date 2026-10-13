#!/bin/bash
#SBATCH --job-name=tip_prediction_train
#SBATCH --account=coms037985
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.out

# Submit this job from the project root (ADS-Coursework):
# CPU: sbatch src/tip_prediction/train.sh
# GPU: sbatch --partition=<gpu_partition> --gres=gpu:1 --export=TRAIN_DEVICE=cuda src/tip_prediction/train.sh

set -eu

source ~/miniconda3/etc/profile.d/conda.sh

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
  RUN_ID="$SLURM_JOB_ID"
else
  RUN_ID="manual_$(date +%Y%m%d_%H%M%S)"
fi

TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"

cd "$PROJECT_ROOT"
conda activate py312

python -m src.tip_prediction.train_xgboost \
  --sample-rows 2000000 \
  --n-estimators 1500 \
  --learning-rate 0.05 \
  --max-depth 8 \
  --n-jobs 4 \
  --device "$TRAIN_DEVICE" \
  --output-dir "$PROJECT_ROOT/models/tip_prediction/xgboost/$RUN_ID"
