#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-resid-smoke
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
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

SOURCE_ROOT="${VLA_WORK_ROOT}/runs/libero_object_trajectories_v2_smoke_1776080"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/v2_pipeline_smoke-${SLURM_JOB_ID}/openvla"
mkdir -p "${OUTPUT_ROOT}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_openvla_coordinate_residuals.py" \
  --input-h5 "${SOURCE_ROOT}/paired_trajectories.h5" \
  --output-h5 "${OUTPUT_ROOT}/openvla_residuals.h5" \
  --max-images 8
