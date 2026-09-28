#!/usr/bin/env bash
#SBATCH --job-name=vla-trajectories
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/generate_libero_object_trajectories.py" \
  --output-dir "${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1" \
  --episodes-per-task 25 \
  --max-action-steps 180 \
  --sample-every 4 \
  --seed 42
