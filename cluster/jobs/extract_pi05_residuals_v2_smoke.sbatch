#!/usr/bin/env bash
#SBATCH --job-name=pi05-v2-resid-smoke
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"
SOURCE_ROOT="${VLA_WORK_ROOT}/runs/libero_object_trajectories_v2_smoke_1776080"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/v2_pipeline_smoke-${SLURM_JOB_ID}/pi05"
mkdir -p "${OUTPUT_ROOT}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/prepare_pi05_libero_inputs.py" \
  --source-h5 "${SOURCE_ROOT}/paired_trajectories.h5" \
  --output-h5 "${OUTPUT_ROOT}/pi05_inputs.h5" \
  --summary-json "${OUTPUT_ROOT}/pi05_inputs_summary.json" \
  --max-samples 8 \
  --audit-samples-per-task 1

source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"
"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_pi05_prefix_residuals.py" \
  --input-h5 "${OUTPUT_ROOT}/pi05_inputs.h5" \
  --output-h5 "${OUTPUT_ROOT}/pi05_prefix_residuals.h5" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --max-images 8

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_pi05_action_expert_residuals.py" \
  --input-h5 "${OUTPUT_ROOT}/pi05_inputs.h5" \
  --output-h5 "${OUTPUT_ROOT}/pi05_action_expert_residuals.h5" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --timestep 1.0 \
  --noise-seed 20260810 \
  --max-images 8 \
  --verify-determinism
