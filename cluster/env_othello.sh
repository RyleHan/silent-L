#!/usr/bin/env bash

export VLA_HOME_ROOT="${VLA_HOME_ROOT:-${HOME}}"
export VLA_PROJECT_ROOT="${VLA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VLA_VENDOR_ROOT="${VLA_VENDOR_ROOT:-${VLA_HOME_ROOT}/code/vla_coordinates_vendor}"
export VLA_WORK_ROOT="${VLA_WORK_ROOT:-${WORK:?Set WORK or VLA_WORK_ROOT}/vla_coordinates}"

export OTHELLO_ENV="${OTHELLO_ENV:-${VLA_WORK_ROOT}/envs/othello-ref}"
export OTHELLO_VENDOR_ROOT="${OTHELLO_VENDOR_ROOT:-${VLA_VENDOR_ROOT}/mech_int_othelloGPT}"
export OTHELLO_REFERENCE_ROOT="${OTHELLO_REFERENCE_ROOT:-${VLA_WORK_ROOT}/reference/othello}"

export PATH="${OTHELLO_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${VLA_PROJECT_ROOT}:${OTHELLO_VENDOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PIP_CACHE_DIR="${VLA_WORK_ROOT}/cache/pip"

