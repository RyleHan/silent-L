#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-lopo
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-23%2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

READOUTS=(prompt_end visual_mean)
READOUT_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
PAIR_ID=$(((SLURM_ARRAY_TASK_ID / 2) % 4))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
READOUT="${READOUTS[${READOUT_INDEX}]}"
SEED=$((42 + SEED_INDEX))
DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
ATLAS_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas_v2"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_world_state_atlas.py" \
  --residual-h5 "${DATA_ROOT}/openvla_residuals.h5" \
  --labels-h5 "${ATLAS_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${ATLAS_ROOT}/lopo/seed-${SEED}/pair-${PAIR_ID}/${READOUT}" \
  --readout "${READOUT}" \
  --prompt-mode paired \
  --holdout-pair "${PAIR_ID}" \
  --seed "${SEED}"
