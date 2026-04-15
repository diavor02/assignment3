#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=4:00:00
#SBATCH --mem=32G
# %x captures the job name for the log file paths
# %j captures the job id to differentiate logs per job
#SBATCH --output=%x/test_run_%j.out
#SBATCH --error=%x/test_run_%j.err

# Load modules
module load class/default
module load cs137/2026spring

N_DAYS=${1:-2}

# Go to directory
# cd /cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation
echo "Running ... $SLURM_JOB_NAME for $N_DAYS days"
# Run code using the Slurm environment variable!
python evaluate.py ./me/ 100
