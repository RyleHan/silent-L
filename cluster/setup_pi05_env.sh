#!/usr/bin/env bash

set -euo pipefail

VLA_HOME_ROOT="${VLA_HOME_ROOT:-${HOME}}"
VLA_VENDOR_ROOT="${VLA_VENDOR_ROOT:-${VLA_HOME_ROOT}/code/vla_coordinates_vendor}"
VLA_WORK_ROOT="${VLA_WORK_ROOT:-${WORK:?Set WORK or VLA_WORK_ROOT}/vla_coordinates}"
PI05_ENV="${PI05_ENV:-${VLA_WORK_ROOT}/envs/pi05-atlas}"
UV_TOOL_ENV="${UV_TOOL_ENV:-${VLA_WORK_ROOT}/tools/uv-pi05}"
CONDA="${VLA_HOME_ROOT}/miniforge3/bin/conda"
OPENPI_ROOT="${VLA_VENDOR_ROOT}/openpi"

mkdir -p \
  "${VLA_WORK_ROOT}/cache/openpi" \
  "${VLA_WORK_ROOT}/cache/pi05-hf" \
  "${VLA_WORK_ROOT}/cache/uv-pi05" \
  "${VLA_WORK_ROOT}/cache/pip-pi05" \
  "${VLA_WORK_ROOT}/checkpoints" \
  "${VLA_WORK_ROOT}/environment-locks/pi05-atlas" \
  "${VLA_WORK_ROOT}/logs" \
  "${VLA_WORK_ROOT}/runs/pi05_world_state_atlas"

if [[ ! -x "${UV_TOOL_ENV}/bin/uv" ]]; then
  "${CONDA}" create --prefix "${UV_TOOL_ENV}" python=3.11 pip -y
  "${UV_TOOL_ENV}/bin/pip" install uv
fi

export UV_PROJECT_ENVIRONMENT="${PI05_ENV}"
export UV_CACHE_DIR="${VLA_WORK_ROOT}/cache/uv-pi05"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1

"${UV_TOOL_ENV}/bin/uv" sync \
  --project "${OPENPI_ROOT}" \
  --frozen \
  --no-install-project
"${UV_TOOL_ENV}/bin/uv" pip install \
  --python "${PI05_ENV}/bin/python" \
  --editable "${OPENPI_ROOT}"

SITE_PACKAGES=$("${PI05_ENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')
cp -R "${OPENPI_ROOT}/src/openpi/models_pytorch/transformers_replace/." \
  "${SITE_PACKAGES}/transformers/"

"${UV_TOOL_ENV}/bin/uv" pip freeze --python "${PI05_ENV}/bin/python" \
  > "${VLA_WORK_ROOT}/environment-locks/pi05-atlas/pip-freeze.txt"
"${UV_TOOL_ENV}/bin/uv" --version \
  > "${VLA_WORK_ROOT}/environment-locks/pi05-atlas/uv-version.txt"
git -C "${OPENPI_ROOT}" rev-parse HEAD \
  > "${VLA_WORK_ROOT}/environment-locks/pi05-atlas/openpi-revision.txt"

"${PI05_ENV}/bin/python" - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_build={torch.version.cuda}")
PY
