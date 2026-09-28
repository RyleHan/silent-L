#!/usr/bin/env bash
#SBATCH --job-name=pi05-s13-read
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE
#SBATCH --array=0-5

set -euo pipefail

module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

# Stage 13: every Stage 11 pair-by-scene condition, replayed with read-only capture.
PAIRS=(0 1 2 3 0 3)
SIDES=(0 0 0 0 1 1)
PAIR_ID="${PAIRS[${SLURM_ARRAY_TASK_ID}]}"
SCENE_SIDE="${SIDES[${SLURM_ARRAY_TASK_ID}]}"
if [[ "${SCENE_SIDE}" == "0" ]]; then
  STAGE11_DIR="${VLA_WORK_ROOT}/runs/pi05_instruction_compliance_v1/pair-${PAIR_ID}"
else
  STAGE11_DIR="${VLA_WORK_ROOT}/runs/pi05_instruction_compliance_reciprocal_v1/pair-${PAIR_ID}"
fi

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/eval_pi05_stage12_probe_read.py" \
  --pair-config "${VLA_PROJECT_ROOT}/configs/libero_object_pairs_v2.json" \
  --pair-id "${PAIR_ID}" \
  --scene-side "${SCENE_SIDE}" \
  --stage11-dir "${STAGE11_DIR}" \
  --output-dir "${VLA_WORK_ROOT}/runs/pi05_stage13_language_shift_v1/pair-${PAIR_ID}-side-${SCENE_SIDE}" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --num-trials 50 \
  --max-action-steps 280 \
  --replan-steps 5 \
  --num-denoise-steps 10 \
  --seed 20260902 \
  --save-videos
