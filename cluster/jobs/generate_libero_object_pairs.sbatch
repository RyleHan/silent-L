#!/usr/bin/env bash
#SBATCH --job-name=vla-object-pairs
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/generate_libero_object_pairs.py" \
  --output-dir "${VLA_WORK_ROOT}/datasets/libero_object_pairs_v1" \
  --samples-per-layout 200 \
  --wait-steps 10 \
  --seed 42
