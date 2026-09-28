#!/usr/bin/env bash

set -euo pipefail

VLA_WORK_ROOT="${VLA_WORK_ROOT:-${WORK:?Set WORK or VLA_WORK_ROOT}/vla_coordinates}"
PI05_ENV="${PI05_ENV:-${VLA_WORK_ROOT}/envs/pi05-atlas}"
UV="${VLA_WORK_ROOT}/tools/uv-pi05/bin/uv"

export UV_CACHE_DIR="${VLA_WORK_ROOT}/cache/uv-pi05"
export UV_LINK_MODE=copy

"${UV}" pip install \
  --python "${PI05_ENV}/bin/python" \
  robosuite==1.4.1 \
  bddl==3.6.0 \
  easydict==1.13 \
  gym==0.26.2 \
  mujoco==2.3.7

"${PI05_ENV}/bin/python" -c \
  'import bddl, gym, mujoco, robosuite; print(robosuite.__version__, mujoco.__version__)'

"${UV}" pip freeze --python "${PI05_ENV}/bin/python" \
  > "${VLA_WORK_ROOT}/environment-locks/pi05-atlas/pip-freeze.txt"
