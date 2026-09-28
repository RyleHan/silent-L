#!/usr/bin/env bash
#SBATCH --job-name=vla-traj-resid
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT_H5="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1/openvla_residuals.h5"
RESUME_ARGS=()
if [[ -f "${OUTPUT_H5}" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_openvla_coordinate_residuals.py" \
  --input-h5 "${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1/paired_trajectories.h5" \
  --output-h5 "${OUTPUT_H5}" \
  "${RESUME_ARGS[@]}"
