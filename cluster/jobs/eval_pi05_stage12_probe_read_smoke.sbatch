#!/usr/bin/env bash
#SBATCH --job-name=pi05-s12-smoke
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/eval_pi05_stage12_probe_read.py" \
  --pair-config "${VLA_PROJECT_ROOT}/configs/libero_object_pairs_v2.json" \
  --pair-id 0 \
  --scene-side 0 \
  --stage11-dir "${VLA_WORK_ROOT}/runs/pi05_instruction_compliance_v1/pair-0" \
  --output-dir "${VLA_WORK_ROOT}/runs/pi05_stage12_probe_read_smoke-${SLURM_JOB_ID}/pair-0" \
  --checkpoint "${PI05_PYTORCH_CHECKPOINT}" \
  --num-trials 1 \
  --max-action-steps 280 \
  --replan-steps 5 \
  --num-denoise-steps 10 \
  --seed 20260902 \
  --save-videos
