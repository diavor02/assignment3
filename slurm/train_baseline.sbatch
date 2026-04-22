#!/bin/bash
#SBATCH --job-name=a3-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/home/supadh03/cs137/assignment3/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/supadh03/cs137/assignment3/slurm_logs/%x_%j.err

set -euo pipefail

MODEL_KIND=${1:-transformer}
TRAIN_EPOCHS=${2:-20}
TRAIN_BATCH_SIZE=${3:-2}

module load class/default
module load cs137/2026spring

cd /cluster/home/supadh03/cs137/assignment3/evaluation

echo "Starting model=${MODEL_KIND} epochs=${TRAIN_EPOCHS} batch=${TRAIN_BATCH_SIZE} on host $(hostname)"

if [[ "${MODEL_KIND}" == "transformer" ]]; then
  cd me
  python train.py
elif [[ "${MODEL_KIND}" == "rnn" ]]; then
  cd me-part2
  python train.py
else
  echo "Unknown MODEL_KIND: ${MODEL_KIND}. Use transformer or rnn."
  exit 1
fi

echo "Done."
