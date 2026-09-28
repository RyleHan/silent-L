#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-coordinates
#SBATCH --partition=work
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-8%3
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
READOUT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
READOUT="${READOUTS[${READOUT_INDEX}]}"
SEED=$((42 + SEED_INDEX))
DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/openvla_trajectory_coordinate_probes_v2/pooled/seed-${SEED}/${READOUT}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_coordinate_probes.py" \
  --residual-h5 "${DATA_ROOT}/openvla_residuals.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${OUTPUT_ROOT}" \
  --readout "${READOUT}" \
  --layers all \
  --epochs 30 \
  --patience 8 \
  --batch-size 128 \
  --lr 0.003 \
  --weight-decay 0.01 \
  --seed "${SEED}"
