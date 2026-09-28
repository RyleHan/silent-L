#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-trajectories
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE
#SBATCH --array=0-7

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASK_IDS=(0 1 2 4 3 8 5 6)
TASK_ID="${TASK_IDS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2/shards/task-${TASK_ID}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/generate_libero_object_trajectories.py" \
  --output-dir "${OUTPUT_ROOT}" \
  --pair-config "${VLA_PROJECT_ROOT}/configs/libero_object_pairs_v2.json" \
  --task-ids "${TASK_ID}" \
  --episodes-per-task 50 \
  --max-action-steps 180 \
  --sample-every 4 \
  --seed 42
