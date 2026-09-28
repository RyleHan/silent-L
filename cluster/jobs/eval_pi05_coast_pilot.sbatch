#!/usr/bin/env bash
#SBATCH --job-name=pi05-coast-pilot
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/eval_pi05_coast_libero.py" \
  --output-dir "${PI05_COAST_RUN_ROOT}/heldout" \
  --checkpoint "${PI05_COAST_PYTORCH_CHECKPOINT}" \
  --conceptor "${PI05_COAST_RUN_ROOT}/fit/coast_conceptor.npz" \
  --task-id 2 \
  --num-trials 30 \
  --init-state-start 15 \
  --max-action-steps 520 \
  --replan-steps 5 \
  --num-denoise-steps 10 \
  --layer 5 \
  --beta 0.1 \
  --seed 20260812 \
  --verify-sampler
