#!/usr/bin/env bash

# Isolated runtime for the pi0.5 transfer experiments.
export VLA_HOME_ROOT="${VLA_HOME_ROOT:-${HOME}}"
export VLA_PROJECT_ROOT="${VLA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VLA_VENDOR_ROOT="${VLA_VENDOR_ROOT:-${VLA_HOME_ROOT}/code/vla_coordinates_vendor}"
export VLA_WORK_ROOT="${VLA_WORK_ROOT:-${WORK:?Set WORK or VLA_WORK_ROOT}/vla_coordinates}"
export PI05_ENV="${PI05_ENV:-${VLA_WORK_ROOT}/envs/pi05-atlas}"
export PI05_CHECKPOINT_ROOT="${PI05_CHECKPOINT_ROOT:-${VLA_WORK_ROOT}/checkpoints/pi05_libero}"
export PI05_PYTORCH_CHECKPOINT="${PI05_PYTORCH_CHECKPOINT:-${VLA_WORK_ROOT}/checkpoints/pi05_libero_pytorch}"
export PI05_COAST_JAX_CHECKPOINT="${PI05_COAST_JAX_CHECKPOINT:-${VLA_WORK_ROOT}/checkpoints/pi05-openpi-libero-2000-jax}"
export PI05_COAST_PYTORCH_CHECKPOINT="${PI05_COAST_PYTORCH_CHECKPOINT:-${VLA_WORK_ROOT}/checkpoints/openpi-libero-2000-pytorch}"
export PI05_COAST_PUBLIC_ROOT="${PI05_COAST_PUBLIC_ROOT:-${VLA_WORK_ROOT}/datasets/coast_public}"
export PI05_COAST_RUN_ROOT="${PI05_COAST_RUN_ROOT:-${VLA_WORK_ROOT}/runs/pi05_coast_pilot}"
export PI05_COAST_TASK_NAME="${PI05_COAST_TASK_NAME:-KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it}"

export PATH="${PI05_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${VLA_PROJECT_ROOT}:${VLA_HOME_ROOT}/code/LIBERO:${VLA_VENDOR_ROOT}/openpi${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${VLA_WORK_ROOT}/config/libero"

export OPENPI_DATA_HOME="${VLA_WORK_ROOT}/cache/openpi"
export HF_HOME="${VLA_WORK_ROOT}/cache/pi05-hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export UV_CACHE_DIR="${VLA_WORK_ROOT}/cache/uv-pi05"
export PIP_CACHE_DIR="${VLA_WORK_ROOT}/cache/pip-pi05"
export UV_LINK_MODE=copy
export HF_HUB_DISABLE_XET=1

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
