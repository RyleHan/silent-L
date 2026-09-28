#!/usr/bin/env bash
#SBATCH --job-name=pi05-v2-expert-boot
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-2%1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
LABEL_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas_v2"
ATLAS_ROOT="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas_v2"
SEED=$((42 + SLURM_ARRAY_TASK_ID))

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/bootstrap_openvla_world_state_atlas.py" \
  --residual-h5 "${ATLAS_ROOT}/pi05_action_expert_residuals.h5" \
  --labels-h5 "${LABEL_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --paired-root "${ATLAS_ROOT}/pooled/seed-${SEED}" \
  --output-json "${ATLAS_ROOT}/pooled/seed-${SEED}/pair_macro_episode_bootstrap.json" \
  --readouts expert_action_mean,expert_action_first,visual_input \
  --visual-readout visual_input \
  --resamples 10000 \
  --seed "${SEED}"
