#!/usr/bin/env bash

# Isolated runtime for the Mechanistic Steering positive-control replication.
export VLA_HOME_ROOT="${VLA_HOME_ROOT:-${HOME}}"
export VLA_PROJECT_ROOT="${VLA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VLA_VENDOR_ROOT="${VLA_VENDOR_ROOT:-${VLA_HOME_ROOT}/code/vla_coordinates_vendor}"
export VLA_WORK_ROOT="${VLA_WORK_ROOT:-${WORK:?Set WORK or VLA_WORK_ROOT}/vla_coordinates}"
export VLA_MECH_REPO="${VLA_MECH_REPO:-${VLA_VENDOR_ROOT}/mechanistic-steering-vlas}"
export VLA_MECH_ENV="${VLA_MECH_ENV:-${VLA_WORK_ROOT}/envs/mechanistic-openvla}"

export PATH="${VLA_MECH_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${VLA_PROJECT_ROOT}:${VLA_MECH_REPO}/src:${VLA_HOME_ROOT}/code/LIBERO:${VLA_VENDOR_ROOT}/openvla${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${VLA_WORK_ROOT}/config/libero"

export HF_HOME="${VLA_WORK_ROOT}/cache/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/transformers"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${VLA_WORK_ROOT}/cache/pip-mechanistic"
export HF_HUB_DISABLE_XET=1

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
