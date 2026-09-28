#!/usr/bin/env bash
#SBATCH --job-name=pi05-lang-analysis
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge

source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v1"
LABEL_ROOT="${VLA_WORK_ROOT}/runs/openvla_world_state_atlas"
PREFIX_ROOT="${VLA_WORK_ROOT}/runs/pi05_world_state_atlas"
EXPERT_ROOT="${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/pi05_language_dependence/offline"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/analyze_pi05_language_dependence.py" \
  --input-h5 "${DATA_ROOT}/pi05_inputs.h5" \
  --labels-h5 "${LABEL_ROOT}/world_state_labels.h5" \
  --paired-prefix-h5 "${PREFIX_ROOT}/pi05_prefix_residuals.h5" \
  --paired-expert-h5 "${EXPERT_ROOT}/pi05_action_expert_residuals.h5" \
  --null-h5 "${VLA_WORK_ROOT}/runs/pi05_language_dependence/null_residuals.h5" \
  --prefix-probe-dir "${PREFIX_ROOT}/paired/prompt_end" \
  --expert-probe-dir "${EXPERT_ROOT}/paired/expert_action_mean" \
  --output-dir "${OUTPUT_ROOT}" \
  --bootstrap-samples 10000 \
  --seed 20260810
