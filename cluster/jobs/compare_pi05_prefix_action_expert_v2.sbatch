#!/usr/bin/env bash
#SBATCH --job-name=pi05-v2-prefix-expert
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-2%1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
module purge
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

SEED=$((42 + SLURM_ARRAY_TASK_ID))
DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
LABEL_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas_v2"
PREFIX_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas_v2"
EXPERT_ROOT="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas_v2"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/compare_pi05_prefix_action_expert.py" \
  --prefix-residual-h5 "${PREFIX_ROOT}/pi05_prefix_residuals.h5" \
  --prefix-paired-root "${PREFIX_ROOT}/pooled/seed-${SEED}" \
  --expert-residual-h5 "${EXPERT_ROOT}/pi05_action_expert_residuals.h5" \
  --expert-paired-root "${EXPERT_ROOT}/pooled/seed-${SEED}" \
  --labels-h5 "${LABEL_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-json "${EXPERT_ROOT}/pooled/seed-${SEED}/expert_minus_prefix_pair_macro.json" \
  --resamples 10000 \
  --seed "${SEED}"
