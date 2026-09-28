#!/usr/bin/env bash
#SBATCH --job-name=pi05-coast-fit
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

"${PI05_ENV}/bin/python" -m unittest discover -s tests -p test_conceptors.py
"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/fit_pi05_coast_conceptor.py" \
  --activation-root "${PI05_COAST_PUBLIC_ROOT}" \
  --task-name "${PI05_COAST_TASK_NAME}" \
  --output-dir "${PI05_COAST_RUN_ROOT}/fit" \
  --layer 5 \
  --aperture 0.5 \
  --beta 0.1 \
  --random-seed 20260812
