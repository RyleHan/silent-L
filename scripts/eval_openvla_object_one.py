#!/usr/bin/env python3
"""Evaluate one OpenVLA episode in LIBERO-Object."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
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
    parser.add_argument("--max-action-steps", type=int, default=280)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script must run inside a GPU allocation.")

    device = torch.device("cuda:0")
    model, processor = load_openvla(
        args.checkpoint,
        args.revision,
        device,
        dtype=torch.float16,
    )

    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    task_description = task.language
    prompt = build_openvla_prompt(task_description)
    env, _ = get_libero_env(task, "openvla", resolution=256)

    frames: list[np.ndarray] = []
    inference_times: list[float] = []
    success = False
    action_steps = 0
    started_at = time.perf_counter()
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.init_state_id])
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

        for action_step in range(args.max_action_steps):
            image = get_libero_image(obs, 224)
            if args.save_video:
                frames.append(image)

            inputs = prepare_openvla_inputs(processor, prompt, image, device, dtype=torch.float16)
            inference_started_at = time.perf_counter()
            raw_action = predict_openvla_action(model, inputs)
            inference_times.append(time.perf_counter() - inference_started_at)

            action = normalize_gripper_action(raw_action.copy(), binarize=True)
            action = invert_gripper_action(action)
            obs, _, success, _ = env.step(action.tolist())
            action_steps = action_step + 1

            if action_steps == 1 or action_steps % 20 == 0 or success:
                print(
                    f"step={action_steps} success={bool(success)} "
                    f"inference_s={inference_times[-1]:.3f} action={raw_action.tolist()}",
                    flush=True,
                )
            if success:
                break
    finally:
        env.close()

    output_dir = Path(os.environ["VLA_WORK_ROOT"]) / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"openvla-object-rollout-{os.environ.get('SLURM_JOB_ID', 'local')}"
    result = {
        "status": "ok",
        "success": bool(success),
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "task": task_description,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "action_steps": action_steps,
        "elapsed_seconds": time.perf_counter() - started_at,
        "mean_inference_seconds": float(np.mean(inference_times)),
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    result_path = output_dir / f"{run_name}.json"

    if args.save_video and frames:
        video_path = output_dir / f"{run_name}.mp4"
        with imageio.get_writer(video_path, fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)
        result["video"] = str(video_path)

    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote: {result_path}", flush=True)


if __name__ == "__main__":
    main()
