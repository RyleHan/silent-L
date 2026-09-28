#!/usr/bin/env bash
#SBATCH --job-name=pi05-v2-expert-atlas
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-8%2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

READOUTS=(expert_action_mean expert_action_first visual_input)
READOUT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
READOUT="${READOUTS[${READOUT_INDEX}]}"
SEED=$((42 + SEED_INDEX))
DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
LABEL_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas_v2"
ATLAS_ROOT="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas_v2"
EXTRA_ARGS=()
if [[ "${READOUT}" == "visual_input" ]]; then
  EXTRA_ARGS+=(--layers 0)
fi

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_world_state_atlas.py" \
  --residual-h5 "${ATLAS_ROOT}/pi05_action_expert_residuals.h5" \
  --labels-h5 "${LABEL_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${ATLAS_ROOT}/pooled/seed-${SEED}/${READOUT}" \
  --readout "${READOUT}" \
  --prompt-mode paired \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
