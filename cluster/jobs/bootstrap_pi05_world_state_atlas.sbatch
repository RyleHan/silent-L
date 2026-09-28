#!/usr/bin/env bash
#SBATCH --job-name=pi05-atlas-bootstrap
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
LABEL_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas"
ATLAS_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/bootstrap_openvla_world_state_atlas.py" \
  --residual-h5 "${ATLAS_ROOT}/pi05_prefix_residuals.h5" \
  --labels-h5 "${LABEL_ROOT}/world_state_labels.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --paired-root "${ATLAS_ROOT}/paired" \
  --output-json "${ATLAS_ROOT}/paired/episode_bootstrap.json" \
  --readouts prompt_end,visual_input \
  --visual-readout visual_input

