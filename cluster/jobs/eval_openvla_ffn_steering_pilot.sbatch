#!/usr/bin/env bash
#SBATCH --job-name=mech-up-pilot
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9%2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_mechanistic.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT_DIR="${VLA_WORK_ROOT}/runs/mechanistic_steering_openvla/pilot/task-${SLURM_ARRAY_TASK_ID}-state-0-job-${SLURM_JOB_ID}"

"${VLA_MECH_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/eval_openvla_ffn_steering.py" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --init-state-id 0 \
  --conditions baseline,up_10,random_10 \
  --coefficient 4 \
  --max-action-steps 520 \
  --output-dir "${OUTPUT_DIR}" \
  --save-video
