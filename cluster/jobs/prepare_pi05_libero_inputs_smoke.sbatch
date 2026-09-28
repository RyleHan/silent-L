#!/usr/bin/env bash
#SBATCH --job-name=pi05-input-smoke
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas/smoke-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/prepare_pi05_libero_inputs.py" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-h5 "${OUTPUT_ROOT}/pi05_inputs.h5" \
  --summary-json "${OUTPUT_ROOT}/pi05_inputs_summary.json" \
  --max-samples 4 \
  --audit-samples-per-task 4

