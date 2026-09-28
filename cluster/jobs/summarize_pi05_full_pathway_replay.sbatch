#!/usr/bin/env bash
#SBATCH --job-name=pi05-path-summary
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
  "${VLA_PROJECT_ROOT}/scripts/summarize_pi05_full_pathway_replay.py" \
  --stage5-root "${VLA_WORK_ROOT}/runs/pi05_paired_patching/closed_loop" \
  --replay-root "${VLA_WORK_ROOT}/runs/pi05_full_pathway_replay" \
  --output-dir "${VLA_WORK_ROOT}/runs/pi05_full_pathway_replay/summary" \
  --bootstrap-samples 20000 \
  --seed 20260812 \
  --exact-tolerance 1e-6
