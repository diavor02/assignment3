#!/bin/bash
#SBATCH --job-name=a3-train
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/cluster/home/supadh03/cs137/assignment3/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/supadh03/cs137/assignment3/slurm_logs/%x_%j.err

set -euo pipefail

MODEL_KIND=${1:-transformer}
TRAIN_EPOCHS=${2:-20}
TRAIN_BATCH_SIZE=${3:-2}

module load class/default
module load cs137/2026spring

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRAIN_EPOCHS
export TRAIN_BATCH_SIZE

cd /cluster/home/supadh03/cs137/assignment3/evaluation

echo "Starting model=${MODEL_KIND} epochs=${TRAIN_EPOCHS} batch=${TRAIN_BATCH_SIZE} on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python - <<'PY'
import torch

print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch.cuda.current_device =", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
PY

if [[ "${MODEL_KIND}" == "transformer" ]]; then
  cd me
  srun --ntasks=1 --gres=gpu:1 python train.py
elif [[ "${MODEL_KIND}" == "rnn" ]]; then
  cd me-part2
  srun --ntasks=1 --gres=gpu:1 python train.py
else
  echo "Unknown MODEL_KIND: ${MODEL_KIND}. Use transformer or rnn."
  exit 1
fi

echo "Done."
