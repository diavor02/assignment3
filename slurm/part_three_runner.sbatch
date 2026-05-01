#!/usr/bin/env bash
#SBATCH --job-name=a3-part3
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --output=/cluster/home/supadh03/cs137/assignment3/slurm_logs/%x_%j.out
#SBATCH --error=/cluster/home/supadh03/cs137/assignment3/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/cluster/home/supadh03/cs137/assignment3
PART3_DIR="${PROJECT_ROOT}/part3"
RESULTS_DIR="${PART3_DIR}/results"
LOG_DIR="${RESULTS_DIR}/logs"

ACTIVE_RUN_DIR="${PROJECT_ROOT}/evaluation/subhanga-additions/checkpoints/v2/transformer/runs/job-483062"
ACTIVE_RUN_FILE="${PROJECT_ROOT}/evaluation/subhanga-additions/checkpoints/v2/transformer/active_run.txt"
CHECKPOINT_FILE="${ACTIVE_RUN_DIR}/best.pt"

module load class/default
module load cs137/2026spring

mkdir -p "${LOG_DIR}"
cd "${PART3_DIR}"

RUN_LOG="${LOG_DIR}/part3_runner_${SLURM_JOB_ID:-manual}.log"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "=== Part 3 Runner ==="
echo "Host                : $(hostname)"
echo "Project root        : ${PROJECT_ROOT}"
echo "Part 3 dir          : ${PART3_DIR}"
echo "Results dir         : ${RESULTS_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-manual}"
echo ""
echo "=== Checkpoint Selection ==="
echo "Active run file     : ${ACTIVE_RUN_FILE}"
echo "Expected run dir    : ${ACTIVE_RUN_DIR}"
echo "Expected checkpoint : ${CHECKPOINT_FILE}"

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: checkpoint not found at ${CHECKPOINT_FILE}"
  exit 1
fi

# Keep the transformer resolver pointed at the best v2 checkpoint.
printf '%s\n' "${ACTIVE_RUN_DIR}" > "${ACTIVE_RUN_FILE}"

echo "Updated active run file to: $(cat "${ACTIVE_RUN_FILE}")"
echo ""
echo "=== Step 4: Collect Predictions ==="
python -u eval/collect_predictions.py

echo ""
echo "=== Step 5: Stratified Evaluation ==="
python -u analysis/zone_horizon_breakdown.py

echo ""
echo "=== Step 6: Generate Figures ==="
python -u analysis/plot_results.py

echo ""
echo "=== Part 3 Runner Complete ==="
echo "Logs written to ${RUN_LOG}"