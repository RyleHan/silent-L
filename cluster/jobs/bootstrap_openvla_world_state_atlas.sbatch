#!/usr/bin/env bash
#SBATCH --job-name=vla-atlas-bootstrap
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

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/bootstrap_openvla_world_state_atlas.py" \
  --residual-h5 "${DATA_ROOT}/openvla_residuals.h5" \
  --labels-h5 "${ATLAS_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --paired-root "${ATLAS_ROOT}/paired" \
  --output-json "${ATLAS_ROOT}/paired_episode_bootstrap.json" \
  --resamples 10000 \
  --seed 42
