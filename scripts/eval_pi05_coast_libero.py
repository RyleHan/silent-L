#!/usr/bin/env python3
"""Evaluate a fixed COAST conceptor pilot on held-out LIBERO initial states."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
from pathlib import Path

import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from vla_coordinates.pi05_runtime import (
    Pi05ActionExpertResidualExtractor,
    load_pi05_policy,
    prepare_pi05_observation_batch,
)


CONDITIONS = ("baseline", "coast", "random")
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conceptor", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--init-state-start", type=int, default=15)
    parser.add_argument("--max-action-steps", type=int, default=520)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--verify-sampler", action="store_true")
    return parser.parse_args()


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(
        quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator,
        dtype=np.float32,
    )


def resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return np.asarray(image, dtype=np.uint8)
    pil_image = Image.fromarray(image)
    current_width, current_height = pil_image.size
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    padded = Image.new(resized.mode, (width, height), 0)
    padded.paste(
        resized,
        (max(0, (width - resized_width) // 2), max(0, (height - resized_height) // 2)),
    )
    return np.asarray(padded, dtype=np.uint8)


def transformed_images(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    base = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    return resize_with_pad(base), resize_with_pad(wrist)


def robot_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def deterministic_noise(
    *,
    seed: int,
    task_id: int,
    init_state_id: int,
    query_index: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    sequence = np.random.SeedSequence([seed, task_id, init_state_id, query_index])
    rng = np.random.default_rng(sequence)
    values = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    return torch.from_numpy(values).to(device)


def unnormalize_actions(policy, observation, normalized: torch.Tensor) -> np.ndarray:
    outputs = {
        "state": observation.state[0].detach().cpu().numpy(),
        "actions": normalized[0].detach().cpu().numpy(),
    }
    return np.asarray(policy._output_transform(outputs)["actions"], dtype=np.float32)  # noqa: SLF001


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.num_trials <= 0 or args.init_state_start < 0:
        raise ValueError("Trial count must be positive and initial-state start nonnegative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    fit_summary_path = args.conceptor.parent / "summary.json"
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
    if int(fit_summary["layer"]) != args.layer or float(fit_summary["beta"]) != args.beta:
        raise ValueError("Evaluation layer/beta does not match the locked fitting artifact.")
    with np.load(args.conceptor) as payload:
        steering_conceptor = torch.from_numpy(
            np.asarray(payload["steering_conceptor"], dtype=np.float32)
        )
        random_conceptor = torch.from_numpy(
            np.asarray(payload["random_conceptor"], dtype=np.float32)
        )

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    sampler = Pi05ActionExpertResidualExtractor(model)
    suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
    task = suite.get_task(args.task_id)
    init_states_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    initial_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    init_state_ids = list(
        range(args.init_state_start, args.init_state_start + args.num_trials)
    )
    if init_state_ids[-1] >= len(initial_states):
        raise ValueError(
            f"Initial-state range ends at {init_state_ids[-1]}, but only "
            f"{len(initial_states)} states exist."
        )
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed + args.task_id)
    records = []
    sampler_audits: dict[str, float | int] = {}
    started_at = time.perf_counter()

    try:
        for init_state_id in tqdm(init_state_ids, desc="pi0.5 COAST KS3", unit="state"):
            for condition in args.conditions:
                observation = env.reset()
                observation = env.set_init_state(initial_states[init_state_id])
                for _ in range(args.wait_steps):
                    observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                action_plan: collections.deque[np.ndarray] = collections.deque()
                replay_images = []
                query_index = 0
                success = False
                executed_steps = 0
                hook_counts = []
                relative_changes = []

                for action_step in range(args.max_action_steps):
                    base_image, wrist_image = transformed_images(observation)
                    if args.save_videos:
                        replay_images.append(base_image)
                    if not action_plan:
                        policy_observation = prepare_pi05_observation_batch(
                            policy,
                            base_images=[base_image],
                            wrist_images=[wrist_image],
                            states=[robot_state(observation)],
                            prompts=[task.language],
                            device=device,
                        )
                        noise = deterministic_noise(
                            seed=args.seed,
                            task_id=args.task_id,
                            init_state_id=init_state_id,
                            query_index=query_index,
                            horizon=model.config.action_horizon,
                            action_dim=model.config.action_dim,
                            device=device,
                        )
                        official = None
                        if condition == "baseline" or (
                            args.verify_sampler and init_state_id == init_state_ids[0] and query_index == 0
                        ):
                            official = model.sample_actions(
                                device,
                                policy_observation,
                                noise=noise,
                                num_steps=args.num_denoise_steps,
                            ).float().cpu()
                        if condition == "baseline":
                            normalized = official
                        else:
                            conceptor = (
                                steering_conceptor if condition == "coast" else random_conceptor
                            )
                            steered = sampler.sample_actions_conceptor(
                                policy_observation,
                                noise=noise,
                                layer_index=args.layer,
                                conceptor=conceptor,
                                beta=args.beta,
                                num_steps=args.num_denoise_steps,
                            )
                            normalized = steered["actions"]
                            hook_counts.append(int(steered["hook_count"].item()))
                            relative_changes.extend(
                                steered["relative_residual_change"].tolist()
                            )
                            if official is not None:
                                sampler_audits[f"{condition}_first_action_max_abs_change"] = float(
                                    (normalized - official).abs().max().item()
                                )

                        if (
                            args.verify_sampler
                            and condition == "baseline"
                            and init_state_id == init_state_ids[0]
                            and query_index == 0
                        ):
                            noop = sampler.sample_actions_conceptor(
                                policy_observation,
                                noise=noise,
                                layer_index=args.layer,
                                conceptor=steering_conceptor,
                                beta=0.0,
                                num_steps=args.num_denoise_steps,
                            )
                            noop_error = float((noop["actions"] - official).abs().max().item())
                            sampler_audits["beta_zero_vs_official_max_abs_difference"] = noop_error
                            sampler_audits["beta_zero_hook_count"] = int(
                                noop["hook_count"].item()
                            )
                            if noop_error != 0.0:
                                raise RuntimeError(
                                    f"COAST beta=0 sampler differs from official: {noop_error}."
                                )

                        actions = unnormalize_actions(policy, policy_observation, normalized)
                        action_plan.extend(actions[: args.replan_steps])
                        query_index += 1

                    observation, _, done, _ = env.step(action_plan.popleft().tolist())
                    executed_steps = action_step + 1
                    if env.check_success():
                        success = True
                        break
                    if done:
                        break

                record = {
                    "task_id": args.task_id,
                    "task_language": task.language,
                    "init_state_id": init_state_id,
                    "condition": condition,
                    "success": success,
                    "executed_steps": executed_steps,
                    "num_policy_queries": query_index,
                    "hook_count_min": min(hook_counts) if hook_counts else 0,
                    "hook_count_max": max(hook_counts) if hook_counts else 0,
                    "relative_residual_change_mean": (
                        float(np.mean(relative_changes)) if relative_changes else 0.0
                    ),
                    "relative_residual_change_max": (
                        float(np.max(relative_changes)) if relative_changes else 0.0
                    ),
                }
                records.append(record)
                with (args.output_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if args.save_videos:
                    suffix = "success" if success else "failure"
                    imageio.mimwrite(
                        video_dir / f"init{init_state_id}_{condition}_{suffix}.mp4",
                        replay_images,
                        fps=10,
                    )
    finally:
        env.close()

    condition_summaries = {}
    for condition in args.conditions:
        selected = [record for record in records if record["condition"] == condition]
        condition_summaries[condition] = {
            "num_trials": len(selected),
            "num_successes": int(sum(record["success"] for record in selected)),
            "success_rate": float(np.mean([record["success"] for record in selected])),
            "executed_steps_mean": float(
                np.mean([record["executed_steps"] for record in selected])
            ),
            "relative_residual_change_mean": float(
                np.mean([record["relative_residual_change_mean"] for record in selected])
            ),
        }
    summary = {
        "protocol": "fixed COAST KS3 held-out pilot",
        "task_id": args.task_id,
        "task_language": task.language,
        "checkpoint": str(args.checkpoint),
        "conceptor": str(args.conceptor),
        "fit_task": fit_summary["task_name"],
        "fit_episode_counts": {
            "success": fit_summary["num_success_episodes"],
            "failure": fit_summary["num_failure_episodes"],
        },
        "layer": args.layer,
        "aperture": fit_summary["aperture"],
        "beta": args.beta,
        "conditions": list(args.conditions),
        "init_state_ids": init_state_ids,
        "fit_test_disjoint": min(init_state_ids) >= 15,
        "sampler_audits": sampler_audits,
        "condition_summaries": condition_summaries,
        "elapsed_seconds": time.perf_counter() - started_at,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
