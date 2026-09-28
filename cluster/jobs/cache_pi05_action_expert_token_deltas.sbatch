#!/usr/bin/env bash
#SBATCH --job-name=pi05-token-delta
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
PATCH_ROOT="${VLA_WORK_ROOT}/runs/pi05_paired_patching"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/cache_pi05_action_expert_token_deltas.py" \
  --input-h5 "${DATA_ROOT}/pi05_inputs.h5" \
  --output-h5 "${PATCH_ROOT}/block13_token_deltas.h5" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --layer 13 \
  --timestep 1.0 \
  --noise-seed 20260810 \
  --verify-determinism \
  --resume
