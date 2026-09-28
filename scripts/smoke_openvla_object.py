#!/usr/bin/env python3
"""Run one real LIBERO-Object observation through the OpenVLA action head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from vla_coordinates.openvla_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REVISION,
    build_openvla_prompt,
    load_openvla,
    predict_openvla_action,
    prepare_openvla_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--wait-steps", type=int, default=10)
    return parser.parse_args()


def get_observation(task_id: int, init_state_id: int, wait_steps: int) -> tuple[np.ndarray, str]:
    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if not 0 <= init_state_id < len(initial_states):
        raise IndexError(f"init-state-id {init_state_id} is outside [0, {len(initial_states)})")

    env, task_description = get_libero_env(task, "openvla", resolution=256)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[init_state_id])
        for _ in range(wait_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        image = get_libero_image(obs, 224)
    finally:
        env.close()

    return image, task_description


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script must run inside a GPU allocation.")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(capability),
                "checkpoint": args.checkpoint,
                "revision": args.revision,
            },
            indent=2,
        )
    )

    image_array, task_description = get_observation(args.task_id, args.init_state_id, args.wait_steps)
    prompt = build_openvla_prompt(task_description)
    print(f"Task: {task_description}")
    print(f"Prompt: {prompt!r}")

    model, processor = load_openvla(
        args.checkpoint,
        args.revision,
        device,
        dtype=torch.float16,
    )

    if not hasattr(model, "norm_stats") or "libero_object" not in model.norm_stats:
        keys = sorted(getattr(model, "norm_stats", {}).keys())
        raise KeyError(f"Checkpoint has no libero_object normalization statistics; available keys: {keys}")

    inputs = prepare_openvla_inputs(processor, prompt, image_array, device, dtype=torch.float16)
    action = predict_openvla_action(model, inputs)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"Invalid action output: shape={action.shape}, values={action}")

    result = {
        "status": "ok",
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "task": task_description,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "action": action.tolist(),
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    print(json.dumps(result, indent=2))

    output_dir = Path(os.environ["VLA_WORK_ROOT"]) / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"openvla-object-smoke-{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
