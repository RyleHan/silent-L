#!/usr/bin/env bash
#SBATCH --job-name=pi05-prefix-expert
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
LABEL_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas"
PREFIX_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas"
EXPERT_ROOT="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/compare_pi05_prefix_action_expert.py" \
  --prefix-residual-h5 "${PREFIX_ROOT}/pi05_prefix_residuals.h5" \
  --prefix-paired-root "${PREFIX_ROOT}/paired" \
  --expert-residual-h5 "${EXPERT_ROOT}/pi05_action_expert_residuals.h5" \
  --expert-paired-root "${EXPERT_ROOT}/paired" \
  --labels-h5 "${LABEL_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-json "${EXPERT_ROOT}/paired/expert_minus_prefix_bootstrap.json" \
  --resamples 10000
