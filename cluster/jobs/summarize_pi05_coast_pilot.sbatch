#!/usr/bin/env bash
#SBATCH --job-name=pi05-coast-summary
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

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/summarize_pi05_coast_pilot.py" \
  --run-dir "${PI05_COAST_RUN_ROOT}/heldout" \
  --fit-dir "${PI05_COAST_RUN_ROOT}/fit" \
  --output-dir "${PI05_COAST_RUN_ROOT}/summary" \
  --bootstrap-samples 20000 \
  --seed 20260812
