#!/usr/bin/env bash
#SBATCH --job-name=pi05-v2-prefix
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas_v2"
mkdir -p "${OUTPUT_ROOT}"

EXTRA_ARGS=()
if [[ -f "${OUTPUT_ROOT}/pi05_prefix_residuals.h5" ]]; then
  EXTRA_ARGS+=(--resume)
fi

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_pi05_prefix_residuals.py" \
  --input-h5 "${DATA_ROOT}/pi05_inputs.h5" \
  --output-h5 "${OUTPUT_ROOT}/pi05_prefix_residuals.h5" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  "${EXTRA_ARGS[@]}"
