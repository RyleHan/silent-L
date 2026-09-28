#!/usr/bin/env bash
#SBATCH --job-name=pi05-loop-analysis
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

CLOSED_LOOP_ROOT="${VLA_WORK_ROOT}/runs/pi05_paired_patching/closed_loop"
SUMMARY_ROOT="${CLOSED_LOOP_ROOT}/summary"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/summarize_pi05_closed_loop.py" \
  --task-0-dir "${CLOSED_LOOP_ROOT}/task-0" \
  --task-1-dir "${CLOSED_LOOP_ROOT}/task-1" \
  --output-dir "${SUMMARY_ROOT}" \
  --bootstrap-samples 20000 \
  --seed 20260810

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/plot_pi05_closed_loop.py" \
  --summary "${SUMMARY_ROOT}/summary.json" \
  --output-dir "${SUMMARY_ROOT}/figures"
