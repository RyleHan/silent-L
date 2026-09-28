#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-coordinate-boot
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --array=0-2%1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
SEED=$((42 + SLURM_ARRAY_TASK_ID))
PROBE_ROOT="${VLA_WORK_ROOT}/runs/openvla_trajectory_coordinate_probes_v2/pooled/seed-${SEED}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/bootstrap_openvla_coordinate_probes.py" \
  --probe-root "${PROBE_ROOT}" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output "${PROBE_ROOT}/pair_macro_episode_bootstrap.json" \
  --bootstrap-samples 10000 \
  --seed "${SEED}"
