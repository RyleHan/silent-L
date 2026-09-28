#!/usr/bin/env bash
#SBATCH --job-name=vla-resid-smoke
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

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_openvla_coordinate_residuals.py" \
  --input-h5 "${VLA_WORK_ROOT}/datasets/libero_object_pairs_v1/paired_states.h5" \
  --output-h5 "${VLA_WORK_ROOT}/runs/openvla_residuals_smoke_${SLURM_JOB_ID}.h5" \
  --max-images 2
