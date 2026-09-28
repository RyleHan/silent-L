#!/usr/bin/env bash
#SBATCH --job-name=vla-v2-coordinate-filter
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_openvla.sh"

DATA_ROOT="${VLA_WORK_ROOT}/datasets/libero_object_trajectories_v2"
OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/openvla_trajectory_coordinate_probes_v2/filter-smoke-${SLURM_JOB_ID}"

"${VLA_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/train_openvla_coordinate_probes.py" \
  --residual-h5 "${DATA_ROOT}/openvla_residuals.h5" \
  --source-h5 "${DATA_ROOT}/paired_trajectories.h5" \
  --output-dir "${OUTPUT_ROOT}" \
  --readout prompt_end \
  --layers 0 \
  --epochs 2 \
  --patience 2 \
  --batch-size 128 \
  --lr 0.003 \
  --weight-decay 0.01 \
  --seed 42

"${VLA_ENV}/bin/python" - "${OUTPUT_ROOT}/metrics.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
stats = summary["label_stats"]
assert stats["num_images_total"] == 15275, stats
assert stats["num_images_excluded"] == 10, stats
assert stats["num_images_eligible"] == 15265, stats
assert sum(stats["excluded_by_split"].values()) == 10, stats
print(json.dumps(stats, indent=2))
PY
