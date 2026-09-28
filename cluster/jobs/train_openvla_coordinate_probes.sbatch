#!/usr/bin/env bash
#SBATCH --job-name=vla-coord-probes
#SBATCH --partition=work
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-2%1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

READOUTS=(prompt_end action_boundary visual_mean)
READOUT="${READOUTS[${SLURM_ARRAY_TASK_ID}]}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_coordinate_probes.py" \
  --residual-h5 "${VLA_WORK_ROOT}/datasets/libero_object_pairs_v1/openvla_residuals.h5" \
  --source-h5 "${VLA_WORK_ROOT}/datasets/libero_object_pairs_v1/paired_states.h5" \
  --output-dir "${VLA_WORK_ROOT}/runs/openvla_coordinate_probes/${READOUT}" \
  --readout "${READOUT}" \
  --layers all \
  --epochs 30 \
  --patience 8 \
  --batch-size 128 \
  --lr 0.003 \
  --weight-decay 0.01 \
  --seed 42
