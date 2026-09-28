#!/usr/bin/env bash
#SBATCH --job-name=vla-object-rollout
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:45:00
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
  "${VLA_PROJECT_ROOT}/scripts/eval_openvla_object_one.py" \
  --checkpoint openvla/openvla-7b-finetuned-libero-object \
  --revision 287d6cfdf12d07b1449505f66d9bf3550257e9b3 \
  --task-id 0 \
  --init-state-id 0 \
  --max-action-steps 280 \
  --save-video
