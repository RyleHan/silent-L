#!/usr/bin/env bash
#SBATCH --job-name=pi05-coast-assets
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1
source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"
export JAX_PLATFORMS=cpu

if [[ ! -f "${PI05_COAST_PUBLIC_ROOT}/download_manifest.json" ]]; then
  echo "Run the task-scoped public downloader on the login node first." >&2
  exit 1
fi
if [[ ! -d "${PI05_COAST_JAX_CHECKPOINT}/params" ]]; then
  echo "Missing public step-2000 checkpoint parameters." >&2
  exit 1
fi

if [[ ! -d "${PI05_COAST_PYTORCH_CHECKPOINT}" ]]; then
  PARTIAL_CHECKPOINT="${PI05_COAST_PYTORCH_CHECKPOINT}.partial-${SLURM_JOB_ID}"
  if [[ -e "${PARTIAL_CHECKPOINT}" ]]; then
    echo "Refusing to overwrite ${PARTIAL_CHECKPOINT}." >&2
    exit 1
  fi
  "${PI05_ENV}/bin/python" -u \
    "${VLA_VENDOR_ROOT}/openpi/examples/convert_jax_model_to_pytorch.py" \
    --checkpoint-dir "${PI05_COAST_JAX_CHECKPOINT}" \
    --config-name pi05_libero \
    --output-path "${PARTIAL_CHECKPOINT}" \
    --precision bfloat16
  if [[ ! -d "${PARTIAL_CHECKPOINT}/assets" ]]; then
    cp -R "${PI05_COAST_JAX_CHECKPOINT}/assets" "${PARTIAL_CHECKPOINT}/assets"
  fi
  mv "${PARTIAL_CHECKPOINT}" "${PI05_COAST_PYTORCH_CHECKPOINT}"
fi

sha256sum "${PI05_COAST_PYTORCH_CHECKPOINT}/model.safetensors" \
  > "${PI05_COAST_PYTORCH_CHECKPOINT}/model.safetensors.sha256"
