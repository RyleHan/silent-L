#!/usr/bin/env bash
#SBATCH --job-name=vla-probe-smoke
#SBATCH --partition=work
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_coordinate_probes.py" \
  --residual-h5 "${VLA_WORK_ROOT}/datasets/libero_object_pairs_v1/openvla_residuals.h5" \
  --source-h5 "${VLA_WORK_ROOT}/datasets/libero_object_pairs_v1/paired_states.h5" \
  --output-dir "${VLA_WORK_ROOT}/runs/openvla_probe_smoke_${SLURM_JOB_ID}" \
  --readout action_boundary \
  --layers 0,16,31 \
  --epochs 2 \
  --patience 2 \
  --batch-size 128
