#!/usr/bin/env bash
#SBATCH --job-name=pi05-s13-sum
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail
module purge
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

RUNS="${VLA_WORK_ROOT}/runs"
"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/summarize_pi05_stage13_language_shift.py" \
  --run-root "${RUNS}/pi05_stage13_language_shift_v1" \
  --laso-root "${RUNS}/pi05_stage12_laso_probes" \
  --lopo-root "${RUNS}/pi05_action_expert_atlas_v2/lopo" \
  --stage12-reads "${RUNS}/pi05_stage12_probe_read_v1/summary/reads.csv" \
  --output-dir "${RUNS}/pi05_stage13_language_shift_v1/summary"
