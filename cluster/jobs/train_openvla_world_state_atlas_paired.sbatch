#!/usr/bin/env bash
#SBATCH --job-name=vla-paired-atlas
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE
#SBATCH --array=0-2

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
ATLAS_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas"
READOUTS=(prompt_end action_boundary visual_mean)
READOUT="${READOUTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT="${ATLAS_ROOT}/paired/${READOUT}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_world_state_atlas.py" \
  --residual-h5 "${DATA_ROOT}/openvla_residuals.h5" \
  --labels-h5 "${ATLAS_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${OUTPUT_ROOT}" \
  --readout "${READOUT}" \
  --prompt-mode paired
