#!/usr/bin/env bash
#SBATCH --job-name=othello-probes
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/11.8.0

source "${SLURM_SUBMIT_DIR}/cluster/env_othello.sh"

"${OTHELLO_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/reproduce_othello_coordinates.py" \
  --reference-root "${OTHELLO_REFERENCE_ROOT}" \
  --output-dir "${OTHELLO_REFERENCE_ROOT}/probe_runs/job-${SLURM_JOB_ID}" \
  --train-size 10000 \
  --valid-size 1000 \
  --batch-size 64 \
  --epochs 1 \
  --lr 0.01 \
  --weight-decay 0.01 \
  --seed 42
