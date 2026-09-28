#!/usr/bin/env bash
#SBATCH --job-name=pi05-lang-summary
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"

OUTPUT_ROOT="${VLA_WORK_ROOT}/runs/pi05_instruction_compliance_v1"

"${PI05_ENV}/bin/python" -u \
  "${VLA_PROJECT_ROOT}/scripts/summarize_pi05_instruction_compliance.py" \
  --run-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --num-pairs 4 \
  --bootstrap-samples 20000 \
  --seed 20260902
