#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-lopo-summary
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

"${VLA_ENV}/bin/python" -u "${VLA_PROJECT_ROOT}/scripts/summarize_lopo_atlas.py" \
  --root "${VLA_WORK_ROOT}/runs/openvla_world_state_atlas_v2/lopo" \
  --model openvla \
  --prompt-readout prompt_end \
  --visual-readout visual_mean \
  --output-dir "${VLA_WORK_ROOT}/runs/openvla_world_state_atlas_v2/lopo/summary"

"${VLA_ENV}/bin/python" -u "${VLA_PROJECT_ROOT}/scripts/summarize_lopo_atlas.py" \
  --root "${VLA_WORK_ROOT}/runs/pi05_world_state_atlas_v2/lopo" \
  --model pi05_prefix \
  --prompt-readout prompt_end \
  --visual-readout visual_input \
  --output-dir "${VLA_WORK_ROOT}/runs/pi05_world_state_atlas_v2/lopo/summary"

"${VLA_ENV}/bin/python" -u "${VLA_PROJECT_ROOT}/scripts/summarize_lopo_atlas.py" \
  --root "${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas_v2/lopo" \
  --model pi05_action_expert \
  --prompt-readout expert_action_mean \
  --visual-readout visual_input \
  --output-dir "${VLA_WORK_ROOT}/runs/pi05_action_expert_atlas_v2/lopo/summary"
