#!/usr/bin/env bash
#SBATCH --job-name=pi05-s12-laso
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-11%4
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

# Stage 12 Amendment 1: leave-anchor-scene-out refits of the Stage 10 pooled probes.
ANCHOR_TASKS=(0 5)
READOUTS=(expert_action_mean prompt_end)
ANCHOR_TASK="${ANCHOR_TASKS[$((SLURM_ARRAY_TASK_ID % 2))]}"
READOUT="${READOUTS[$(((SLURM_ARRAY_TASK_ID / 2) % 2))]}"
SEED=$((42 + SLURM_ARRAY_TASK_ID / 4))
DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
LABEL_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas_v2"
if [[ "${READOUT}" == "prompt_end" ]]; then
  RESIDUAL_H5="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas_v2/pi05_prefix_residuals.h5"
else
  RESIDUAL_H5="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas_v2/pi05_action_expert_residuals.h5"
fi

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_world_state_atlas.py" \
  --residual-h5 "${RESIDUAL_H5}" \
  --labels-h5 "${LABEL_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${VLA_WORK_ROOT}/runs/pi05_stage12_laso_probes/anchor-task-${ANCHOR_TASK}/seed-${SEED}/${READOUT}" \
  --readout "${READOUT}" \
  --prompt-mode paired \
  --exclude-task-ids "${ANCHOR_TASK}" \
  --seed "${SEED}"
