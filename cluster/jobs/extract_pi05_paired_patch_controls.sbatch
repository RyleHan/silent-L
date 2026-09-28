#!/usr/bin/env bash
#SBATCH --job-name=pi05-controls
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE
#SBATCH --array=0-1

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
PATCH_ROOT="${VLA_WORK_ROOT}/runs/pi05_paired_patching"
ATLAS_ROOT="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_pi05_paired_patch_controls.py" \
  --input-h5 "${DATA_ROOT}/pi05_inputs.h5" \
  --token-delta-h5 "${PATCH_ROOT}/block13_token_deltas.h5" \
  --reference-patch-h5 "${PATCH_ROOT}/base-${SLURM_ARRAY_TASK_ID}.h5" \
  --probe-checkpoint "${ATLAS_ROOT}/paired/expert_action_mean/layer_13.pt" \
  --output-h5 "${PATCH_ROOT}/controls-base-${SLURM_ARRAY_TASK_ID}.h5" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --layer 13 \
  --base-prompt-index "${SLURM_ARRAY_TASK_ID}" \
  --verify-audits \
  --resume
