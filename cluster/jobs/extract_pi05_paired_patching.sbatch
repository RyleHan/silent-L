#!/usr/bin/env bash
#SBATCH --job-name=pi05-patch
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
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
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/pi05_paired_patching"
mkdir -p "${OUTPUT_ROOT}"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_pi05_paired_patching.py" \
  --input-h5 "${DATA_ROOT}/pi05_inputs.h5" \
  --output-h5 "${OUTPUT_ROOT}/base-${SLURM_ARRAY_TASK_ID}.h5" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --layers 0,5,9,13,17 \
  --alphas 0,0.25,0.5,0.75,1 \
  --base-prompt-index "${SLURM_ARRAY_TASK_ID}" \
  --timestep 1.0 \
  --noise-seed 20260810 \
  --verify-audits \
  --resume
