#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-data-smoke
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
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

OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/libero_object_trajectories_v2_smoke_${SLURM_JOB_ID}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/generate_libero_object_trajectories.py" \
  --output-dir "${OUTPUT_ROOT}" \
  --pair-config "${VLA_PROJECT_ROOT}/configs/libero_object_pairs_v2.json" \
  --episodes-per-task 1 \
  --max-action-steps 4 \
  --sample-every 4 \
  --seed 42 \
  --allow-incomplete-splits

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/extract_libero_world_state_labels.py" \
  --source-h5 "${OUTPUT_ROOT}/paired_trajectories.h5" \
  --output-h5 "${OUTPUT_ROOT}/world_state_labels.h5" \
  --summary-json "${OUTPUT_ROOT}/world_state_labels_summary.json"
