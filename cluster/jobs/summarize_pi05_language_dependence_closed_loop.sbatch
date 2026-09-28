#!/usr/bin/env bash
#SBATCH --job-name=pi05-null-summary
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge

source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

STAGE5_ROOT="${VLA_WORK_ROOT}/runs/pi05_paired_patching/closed_loop"
STAGE7_ROOT="${VLA_WORK_ROOT}/runs/pi05_language_dependence/closed_loop"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/summarize_pi05_language_dependence_closed_loop.py" \
  --stage5-root "${STAGE5_ROOT}" \
  --null-root "${STAGE7_ROOT}" \
  --output-dir "${STAGE7_ROOT}/summary" \
  --bootstrap-samples 20000 \
  --seed 20260810
