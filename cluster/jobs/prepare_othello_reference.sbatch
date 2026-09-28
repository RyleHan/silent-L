#!/usr/bin/env bash
#SBATCH --job-name=othello-data
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/cluster/env_othello.sh"

"${OTHELLO_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/prepare_othello_reference.py" \
  --output-dir "${OTHELLO_REFERENCE_ROOT}/data" \
  --train-size 20000 \
  --valid-size 2000 \
  --seed 42
