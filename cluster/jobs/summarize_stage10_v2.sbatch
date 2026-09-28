#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-final-summary
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

"${VLA_ENV}/bin/python" -u "${VLA_PROJECT_ROOT}/scripts/summarize_stage10_v2.py" \
  --work-root "${VLA_WORK_ROOT}" \
  --output-dir "${VLA_WORK_ROOT}/runs/stage10_v2_summary"
