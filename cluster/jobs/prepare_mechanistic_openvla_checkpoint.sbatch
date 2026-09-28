#!/usr/bin/env bash
#SBATCH --job-name=mech-vla-ckpt
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/cluster/env_mechanistic.sh"

"${VLA_MECH_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/prepare_hf_checkpoint.py" \
  --repo-id openvla/openvla-7b-finetuned-libero-10 \
  --revision 80970322773f81baa2e22fe495d0487b93a05cfa \
  --output "${VLA_WORK_ROOT}/environment-locks/mechanistic-openvla/checkpoint-manifest.json"
