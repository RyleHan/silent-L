#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-pair-audit
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/audit_libero_object_pair_config.py" \
  --pair-config "${VLA_PROJECT_ROOT}/configs/libero_object_pairs_v2.json" \
  --output "${VLA_WORK_ROOT}/runs/libero_object_pair_audit_v2.json"
