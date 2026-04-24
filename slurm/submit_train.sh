#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'USAGE'
Usage:
  submit_train.sh <profile> <model_kind> <epochs> <batch_size> [resume_checkpoint] [run_dir]

Profiles:
  tier1      strongest single-node request
  tier2      high-capacity request
  tier3      balanced request
  tier4      lean request
  baseline   current baseline with a longer time limit
USAGE
  exit 1
fi

PROFILE=${1}
MODEL_KIND=${2}
TRAIN_EPOCHS=${3}
TRAIN_BATCH_SIZE=${4}
TRAIN_RESUME_CKPT=${5:-}
TRAIN_RUN_DIR=${6:-}

case "${PROFILE}" in
  tier1)
    SBATCH_FLAGS=(--partition=preempt --nodes=1 --ntasks=1 --cpus-per-task=12 --gres=gpu:1 --time=12:00:00 --mem=96G)
    ;;
  tier2)
    SBATCH_FLAGS=(--partition=preempt --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --time=10:00:00 --mem=80G)
    ;;
  tier3)
    SBATCH_FLAGS=(--partition=preempt --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=08:00:00 --mem=64G)
    ;;
  tier4)
    SBATCH_FLAGS=(--partition=preempt --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=06:00:00 --mem=48G)
    ;;
  baseline)
    SBATCH_FLAGS=(--partition=preempt --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=08:00:00 --mem=64G)
    ;;
  *)
    echo "Unknown profile: ${PROFILE}"
    exit 1
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_SCRIPT="${SCRIPT_DIR}/train_baseline.sbatch"

echo "Submitting ${MODEL_KIND} with profile=${PROFILE}"
echo "  epochs=${TRAIN_EPOCHS} batch_size=${TRAIN_BATCH_SIZE}"
echo "  resume=${TRAIN_RESUME_CKPT:-<auto>} run_dir=${TRAIN_RUN_DIR:-<auto>}"
sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" "${MODEL_KIND}" "${TRAIN_EPOCHS}" "${TRAIN_BATCH_SIZE}" "${TRAIN_RESUME_CKPT}" "${TRAIN_RUN_DIR}"