#!/usr/bin/env bash
set -euo pipefail

DATA_JOB_ID="${1:?Usage: cluster/submit_v2_analysis.sh DATA_ARRAY_JOB_ID [OPENVLA_SMOKE_JOB_ID]}"
OPENVLA_SMOKE_JOB_ID="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

submit() {
  local message
  message="$(sbatch.tinygpu "$@")"
  printf '%s\n' "${message}" >&2
  awk '{print $4}' <<<"${message}"
}

MERGE_JOB="$(submit --dependency="afterok:${DATA_JOB_ID}" \
  cluster/jobs/merge_libero_object_trajectories_v2.sbatch)"

LABEL_JOB="$(submit --dependency="afterok:${MERGE_JOB}" \
  cluster/jobs/extract_libero_world_state_labels_v2.sbatch)"
OPENVLA_RESID_DEPENDENCY="${MERGE_JOB}"
if [[ -n "${OPENVLA_SMOKE_JOB_ID}" ]]; then
  OPENVLA_RESID_DEPENDENCY="${OPENVLA_RESID_DEPENDENCY}:${OPENVLA_SMOKE_JOB_ID}"
fi
OPENVLA_RESID_JOB="$(submit --dependency="afterok:${OPENVLA_RESID_DEPENDENCY}" \
  cluster/jobs/extract_openvla_residuals_v2.sbatch)"
PI05_INPUT_JOB="$(submit --dependency="afterok:${MERGE_JOB}" \
  cluster/jobs/prepare_pi05_libero_inputs_v2.sbatch)"

PI05_PREFIX_RESID_JOB="$(submit --dependency="afterok:${PI05_INPUT_JOB}" \
  cluster/jobs/extract_pi05_prefix_residuals_v2.sbatch)"
PI05_EXPERT_RESID_JOB="$(submit --dependency="afterok:${PI05_INPUT_JOB}" \
  cluster/jobs/extract_pi05_action_expert_residuals_v2.sbatch)"

COORD_JOB="$(submit --dependency="afterok:${OPENVLA_RESID_JOB}" \
  cluster/jobs/train_openvla_coordinate_probes_v2.sbatch)"
OPENVLA_ATLAS_JOB="$(submit --dependency="afterok:${OPENVLA_RESID_JOB}:${LABEL_JOB}" \
  cluster/jobs/train_openvla_world_state_atlas_v2.sbatch)"
PI05_PREFIX_ATLAS_JOB="$(submit --dependency="afterok:${PI05_PREFIX_RESID_JOB}:${LABEL_JOB}" \
  cluster/jobs/train_pi05_world_state_atlas_v2.sbatch)"
PI05_EXPERT_ATLAS_JOB="$(submit --dependency="afterok:${PI05_EXPERT_RESID_JOB}:${LABEL_JOB}" \
  cluster/jobs/train_pi05_action_expert_atlas_v2.sbatch)"

COORD_BOOT_JOB="$(submit --dependency="afterok:${COORD_JOB}" \
  cluster/jobs/bootstrap_openvla_coordinate_probes_v2.sbatch)"
OPENVLA_BOOT_JOB="$(submit --dependency="afterok:${OPENVLA_ATLAS_JOB}" \
  cluster/jobs/bootstrap_openvla_world_state_atlas_v2.sbatch)"
PI05_PREFIX_BOOT_JOB="$(submit --dependency="afterok:${PI05_PREFIX_ATLAS_JOB}" \
  cluster/jobs/bootstrap_pi05_world_state_atlas_v2.sbatch)"
PI05_EXPERT_BOOT_JOB="$(submit --dependency="afterok:${PI05_EXPERT_ATLAS_JOB}" \
  cluster/jobs/bootstrap_pi05_action_expert_atlas_v2.sbatch)"
PI05_COMPARE_JOB="$(submit --dependency="afterok:${PI05_PREFIX_ATLAS_JOB}:${PI05_EXPERT_ATLAS_JOB}" \
  cluster/jobs/compare_pi05_prefix_action_expert_v2.sbatch)"

OPENVLA_LOPO_JOB="$(submit --dependency="afterok:${OPENVLA_RESID_JOB}:${LABEL_JOB}" \
  cluster/jobs/train_openvla_world_state_lopo_v2.sbatch)"
PI05_PREFIX_LOPO_JOB="$(submit --dependency="afterok:${PI05_PREFIX_RESID_JOB}:${LABEL_JOB}" \
  cluster/jobs/train_pi05_world_state_lopo_v2.sbatch)"
PI05_EXPERT_LOPO_JOB="$(submit --dependency="afterok:${PI05_EXPERT_RESID_JOB}:${LABEL_JOB}" \
  cluster/jobs/train_pi05_action_expert_lopo_v2.sbatch)"
LOPO_SUMMARY_JOB="$(submit --dependency="afterok:${OPENVLA_LOPO_JOB}:${PI05_PREFIX_LOPO_JOB}:${PI05_EXPERT_LOPO_JOB}" \
  cluster/jobs/summarize_lopo_v2.sbatch)"
FINAL_SUMMARY_JOB="$(submit --dependency="afterok:${COORD_BOOT_JOB}:${OPENVLA_BOOT_JOB}:${PI05_PREFIX_BOOT_JOB}:${PI05_EXPERT_BOOT_JOB}:${PI05_COMPARE_JOB}:${LOPO_SUMMARY_JOB}" \
  cluster/jobs/summarize_stage10_v2.sbatch)"

cat <<EOF
data=${DATA_JOB_ID}
openvla_smoke=${OPENVLA_SMOKE_JOB_ID:-not_provided}
merge=${MERGE_JOB}
labels=${LABEL_JOB}
openvla_residuals=${OPENVLA_RESID_JOB}
pi05_inputs=${PI05_INPUT_JOB}
pi05_prefix_residuals=${PI05_PREFIX_RESID_JOB}
pi05_expert_residuals=${PI05_EXPERT_RESID_JOB}
coordinate_probes=${COORD_JOB}
openvla_atlas=${OPENVLA_ATLAS_JOB}
pi05_prefix_atlas=${PI05_PREFIX_ATLAS_JOB}
pi05_expert_atlas=${PI05_EXPERT_ATLAS_JOB}
coordinate_bootstrap=${COORD_BOOT_JOB}
openvla_bootstrap=${OPENVLA_BOOT_JOB}
pi05_prefix_bootstrap=${PI05_PREFIX_BOOT_JOB}
pi05_expert_bootstrap=${PI05_EXPERT_BOOT_JOB}
pi05_prefix_expert_compare=${PI05_COMPARE_JOB}
openvla_lopo=${OPENVLA_LOPO_JOB}
pi05_prefix_lopo=${PI05_PREFIX_LOPO_JOB}
pi05_expert_lopo=${PI05_EXPERT_LOPO_JOB}
lopo_summary=${LOPO_SUMMARY_JOB}
final_summary=${FINAL_SUMMARY_JOB}
EOF
