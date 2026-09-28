#!/usr/bin/env bash
#SBATCH --job-name=pi05-checkpoint
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

module purge
module load cuda/12.4.1

source "${SLURM_SUBMIT_DIR}/cluster/env_pi05.sh"
export JAX_PLATFORMS=cpu

JAX_CHECKPOINT="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero"
if [[ ! -d "${JAX_CHECKPOINT}" ]]; then
  "${PI05_ENV}/bin/python" -u \
    "${VLA_PROJECT_ROOT}/scripts/download_public_gcs_prefix.py" \
    --bucket openpi-assets \
    --prefix checkpoints/pi05_libero \
    --output-dir "${JAX_CHECKPOINT}" \
    --workers 4
fi

PARTIAL_CHECKPOINT="${PI05_PYTORCH_CHECKPOINT}.partial-${SLURM_JOB_ID}"
if [[ -e "${PI05_PYTORCH_CHECKPOINT}" || -e "${PARTIAL_CHECKPOINT}" ]]; then
  echo "Refusing to overwrite an existing checkpoint path." >&2
  exit 1
fi

"${PI05_ENV}/bin/python" -u \
  "${VLA_VENDOR_ROOT}/openpi/examples/convert_jax_model_to_pytorch.py" \
  --checkpoint-dir "${JAX_CHECKPOINT}" \
  --config-name pi05_libero \
  --output-path "${PARTIAL_CHECKPOINT}" \
  --precision bfloat16

if [[ ! -d "${PARTIAL_CHECKPOINT}/assets" ]]; then
  cp -R "${JAX_CHECKPOINT}/assets" \
    "${PARTIAL_CHECKPOINT}/assets"
fi

mv "${PARTIAL_CHECKPOINT}" "${PI05_PYTORCH_CHECKPOINT}"
sha256sum "${PI05_PYTORCH_CHECKPOINT}/model.safetensors" \
  > "${PI05_PYTORCH_CHECKPOINT}/model.safetensors.sha256"
