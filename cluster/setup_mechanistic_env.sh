#!/usr/bin/env bash
set -euo pipefail

HOME_ROOT="${VLA_HOME_ROOT:-${HOME}}"
WORK_ROOT="${VLA_WORK_ROOT:-${WORK:?Set WORK or VLA_WORK_ROOT}/vla_coordinates}"
SOURCE_ENV="${VLA_OPENVLA_SOURCE_ENV:-${WORK_ROOT}/envs/openvla-object}"
TARGET_ENV="${VLA_MECH_ENV:-${WORK_ROOT}/envs/mechanistic-openvla}"
MECH_REPO="${VLA_MECH_REPO:-${HOME_ROOT}/code/vla_coordinates_vendor/mechanistic-steering-vlas}"
LOCK_DIR="${WORK_ROOT}/environment-locks/mechanistic-openvla"

if [[ ! -x "${SOURCE_ENV}/bin/python" ]]; then
  echo "Missing verified source environment: ${SOURCE_ENV}" >&2
  exit 1
fi
if [[ ! -d "${MECH_REPO}/.git" ]]; then
  echo "Missing official Mechanistic Steering repository: ${MECH_REPO}" >&2
  exit 1
fi

if [[ ! -x "${TARGET_ENV}/bin/python" ]]; then
  conda create --prefix "${TARGET_ENV}" --clone "${SOURCE_ENV}" --yes
fi

"${TARGET_ENV}/bin/python" -m pip install --no-deps --editable "${MECH_REPO}"

mkdir -p "${LOCK_DIR}"
conda list --prefix "${TARGET_ENV}" --explicit > "${LOCK_DIR}/conda-explicit.txt"
"${TARGET_ENV}/bin/python" -m pip freeze > "${LOCK_DIR}/pip-freeze.txt"
git -C "${MECH_REPO}" rev-parse HEAD > "${LOCK_DIR}/mechanistic-steering-revision.txt"

"${TARGET_ENV}/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys

print(sys.version)
for package in ("torch", "transformers", "libero", "robosuite", "mujoco", "vla-mech-interp"):
    print(package, metadata.version(package))
PY
