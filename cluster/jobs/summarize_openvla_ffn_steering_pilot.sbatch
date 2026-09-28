#!/usr/bin/env bash
#SBATCH --job-name=mech-up-summary
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/cluster/env_mechanistic.sh"

"${VLA_MECH_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/summarize_openvla_ffn_steering.py" \
  --input-root "${VLA_WORK_ROOT}/runs/mechanistic_steering_openvla/pilot" \
  --output-dir "${VLA_WORK_ROOT}/runs/mechanistic_steering_openvla/pilot-summary"
