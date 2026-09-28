#!/usr/bin/env bash
#SBATCH --job-name=pi05-patch-summary
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE
#SBATCH --array=0-1

set -euo pipefail

module purge

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
PATCH_ROOT="${VLA_WORK_ROOT}/runs/pi05_paired_patching"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/summarize_pi05_paired_patching.py" \
  --patch-h5 "${PATCH_ROOT}/base-${SLURM_ARRAY_TASK_ID}.h5" \
  --input-h5 "${DATA_ROOT}/pi05_inputs.h5" \
  --output-dir "${PATCH_ROOT}/summary/base-${SLURM_ARRAY_TASK_ID}" \
  --bootstrap-samples 10000 \
  --seed 20260810
