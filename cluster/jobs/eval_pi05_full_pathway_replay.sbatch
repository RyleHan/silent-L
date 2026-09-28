#!/usr/bin/env bash
#SBATCH --job-name=pi05-path-replay
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE
#SBATCH --array=0-1

set -euo pipefail

module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/pi05_full_pathway_replay"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/eval_pi05_paired_patch_closed_loop.py" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${OUTPUT_ROOT}/task-${SLURM_ARRAY_TASK_ID}" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --conditions pathway \
  --num-trials 20 \
  --max-action-steps 280 \
  --replan-steps 5 \
  --num-denoise-steps 10 \
  --seed 20260810 \
  --verify-sampler
