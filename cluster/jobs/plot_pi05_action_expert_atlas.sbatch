#!/usr/bin/env bash
#SBATCH --job-name=pi05-expert-plots
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

PREFIX_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas"
EXPERT_ROOT="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/plot_pi05_action_expert_atlas.py" \
  --prefix-root "${PREFIX_ROOT}" \
  --expert-root "${EXPERT_ROOT}" \
  --output-dir "${EXPERT_ROOT}/figures"
