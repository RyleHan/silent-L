#!/usr/bin/env bash
#SBATCH --job-name=vla-paired-smoke
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
ATLAS_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas"
OUTPUT_ROOT="${ATLAS_ROOT}/paired_smoke_${SLURM_JOB_ID}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_world_state_atlas.py" \
  --residual-h5 "${DATA_ROOT}/openvla_residuals.h5" \
  --labels-h5 "${ATLAS_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${OUTPUT_ROOT}/prompt_end" \
  --readout prompt_end \
  --prompt-mode paired \
  --layers 0,15,31 \
  --epochs 3 \
  --patience 2
