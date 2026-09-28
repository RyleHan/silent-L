#!/usr/bin/env bash
#SBATCH --job-name=pi05-atlas-plot
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

ATLAS_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/plot_pi05_world_state_atlas.py" \
  --atlas-root "${ATLAS_ROOT}" \
  --output-dir "${ATLAS_ROOT}/figures"

