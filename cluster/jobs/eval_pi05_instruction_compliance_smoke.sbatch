#!/usr/bin/env bash
#SBATCH --job-name=pi05-lang-smoke
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE
#SBATCH --array=0-3

set -euo pipefail

module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/pi05_instruction_compliance_smoke-${SLURM_ARRAY_JOB_ID}"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/eval_pi05_instruction_compliance.py" \
  --pair-config "${VLA_PROJECT_ROOT}/configs/libero_object_pairs_v2.json" \
  --pair-id "${SLURM_ARRAY_TASK_ID}" \
  --scene-side 0 \
  --output-dir "${OUTPUT_ROOT}/pair-${SLURM_ARRAY_TASK_ID}" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --num-trials 1 \
  --max-action-steps 280 \
  --replan-steps 5 \
  --num-denoise-steps 10 \
  --seed 20260902 \
  --save-videos
