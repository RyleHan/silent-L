#!/usr/bin/env python3
"""Evaluate pi0.5 object selection under two unmodified instructions."""

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

from vla_coordinates.libero_object_pairs import load_pair_config, prompts_for_pair
from vla_coordinates.pi05_runtime import load_pi05_policy, prepare_pi05_observation_batch


LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
FIRST_GRASP_NONE = -1
FIRST_GRASP_BOTH = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--scene-side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--max-action-steps", type=int, default=280)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--save-videos", action="store_true")
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
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    padded.paste(resized, (pad_width, pad_height))
    return np.asarray(padded, dtype=np.uint8)


def robot_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def transformed_images(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    base = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    return resize_with_pad(base), resize_with_pad(wrist)


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


def check_in(base_env: object, object_name: str, basket_region: str) -> bool:
    return bool(base_env._eval_predicate(["in", object_name, basket_region]))


def check_grasp(base_env: object, object_name: str) -> bool:
    obj = base_env.objects_dict[object_name]
    return bool(base_env._check_grasp(base_env.robots[0].gripper, obj.contact_geoms))


def first_grasp_side(grasped: tuple[bool, bool]) -> int:
    if grasped == (True, False):
        return 0
    if grasped == (False, True):
        return 1
    if grasped == (True, True):
        return FIRST_GRASP_BOTH
    return FIRST_GRASP_NONE


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
    if args.num_trials <= 0 or args.replan_steps <= 0 or args.num_denoise_steps <= 0:
        raise ValueError("Trial, replanning, and denoising counts must be positive.")

    config = load_pair_config(args.pair_config)
    pairs = config["pairs"]
    if args.pair_id < 0 or args.pair_id >= len(pairs):
        raise ValueError(f"pair-id must be in [0, {len(pairs) - 1}].")
    pair = pairs[args.pair_id]
    task_id = int(pair["task_ids"][args.scene_side])
    object_instances = tuple(pair["object_instances"])
    object_names = tuple(pair["object_names"])
    prompts = tuple(prompts_for_pair(pair, "plain"))
    basket_region = str(config["basket_region"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(task_id)
    init_states_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    # LIBERO init states are trusted local simulator arrays, not model weights.
    initial_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    if args.num_trials > len(initial_states):
        raise ValueError(
            f"Task {task_id} has {len(initial_states)} initial states; "
            f"requested {args.num_trials}."
        )
    rng = np.random.default_rng(args.seed + task_id)
    init_state_ids = rng.permutation(len(initial_states))[: args.num_trials].astype(int).tolist()

    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed + task_id)

    records: list[dict[str, object]] = []
    started_at = time.perf_counter()
    try:
        for init_state_id in tqdm(init_state_ids, desc=f"pair {args.pair_id}", unit="state"):
            for prompt_side in (0, 1):
                observation = env.reset()
                observation = env.set_init_state(initial_states[init_state_id])
                for _ in range(args.wait_steps):
                    observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                initial_positions = [
                    np.asarray(observation[f"{name}_pos"], dtype=np.float32).copy()
                    for name in object_instances
                ]
                min_eef_distance = [math.inf, math.inf]
                max_displacement = [0.0, 0.0]
                ever_grasped = [False, False]
                ever_in_basket = [False, False]
                selected_side = FIRST_GRASP_NONE
                selected_step = None
                action_plan: collections.deque[np.ndarray] = collections.deque()
                query_index = 0
                executed_steps = 0
                bddl_success = False
                replay_images: list[np.ndarray] = []

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
                            prompts=[prompts[prompt_side]],
                            device=device,
                        )
                        noise = deterministic_noise(
                            seed=args.seed,
                            task_id=task_id,
                            init_state_id=init_state_id,
                            query_index=query_index,
                            horizon=model.config.action_horizon,
                            action_dim=model.config.action_dim,
                            device=device,
                        )
                        normalized = model.sample_actions(
                            device,
                            policy_observation,
                            noise=noise,
                            num_steps=args.num_denoise_steps,
                        ).float().cpu()
                        actions = unnormalize_actions(policy, policy_observation, normalized)
                        if len(actions) < args.replan_steps:
                            raise RuntimeError("Action chunk is shorter than --replan-steps.")
                        action_plan.extend(actions[: args.replan_steps])
                        query_index += 1

                    action = action_plan.popleft()
                    observation, _, done, _ = env.step(action.tolist())
                    executed_steps = action_step + 1
                    eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
                    current_grasped = []
                    for side, object_name in enumerate(object_instances):
                        position = np.asarray(
                            observation[f"{object_name}_pos"], dtype=np.float32
                        )
                        min_eef_distance[side] = min(
                            min_eef_distance[side], float(np.linalg.norm(eef - position))
                        )
                        max_displacement[side] = max(
                            max_displacement[side],
                            float(np.linalg.norm(position - initial_positions[side])),
                        )
                        grasped = check_grasp(env.env, object_name)
                        in_basket = check_in(env.env, object_name, basket_region)
                        current_grasped.append(grasped)
                        ever_grasped[side] |= grasped
                        ever_in_basket[side] |= in_basket

                    if selected_side == FIRST_GRASP_NONE:
                        candidate = first_grasp_side(tuple(current_grasped))
                        if candidate != FIRST_GRASP_NONE:
                            selected_side = candidate
                            selected_step = executed_steps

                    bddl_success = bool(env.check_success())
                    if any(ever_in_basket) or done:
                        break

                record = {
                    "pair_id": args.pair_id,
                    "pair_name": pair["name"],
                    "scene_side": args.scene_side,
                    "task_id": task_id,
                    "task_language": task.language,
                    "init_state_id": init_state_id,
                    "prompt_side": prompt_side,
                    "prompt": prompts[prompt_side],
                    "prompt_object": object_names[prompt_side],
                    "scene_target_object": object_names[args.scene_side],
                    "condition": "aligned" if prompt_side == args.scene_side else "swapped",
                    "first_grasp_side": selected_side,
                    "first_grasp_object": (
                        object_names[selected_side]
                        if selected_side in (0, 1)
                        else "both" if selected_side == FIRST_GRASP_BOTH else "none"
                    ),
                    "first_grasp_step": selected_step,
                    "commanded_first_grasp": selected_side == prompt_side,
                    "scene_target_first_grasp": selected_side == args.scene_side,
                    "ever_grasped_a": ever_grasped[0],
                    "ever_grasped_b": ever_grasped[1],
                    "commanded_ever_grasped": ever_grasped[prompt_side],
                    "alternative_ever_grasped": ever_grasped[1 - prompt_side],
                    "ever_in_basket_a": ever_in_basket[0],
                    "ever_in_basket_b": ever_in_basket[1],
                    "commanded_in_basket": ever_in_basket[prompt_side],
                    "bddl_success": bddl_success,
                    "min_eef_distance_a": min_eef_distance[0],
                    "min_eef_distance_b": min_eef_distance[1],
                    "commanded_min_eef_distance": min_eef_distance[prompt_side],
                    "max_displacement_a": max_displacement[0],
                    "max_displacement_b": max_displacement[1],
                    "commanded_max_displacement": max_displacement[prompt_side],
                    "executed_steps": executed_steps,
                    "num_policy_queries": query_index,
                }
                records.append(record)
                with (args.output_dir / "episodes.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(record) + "\n")
                if args.save_videos:
                    imageio.mimwrite(
                        video_dir
                        / f"pair{args.pair_id}_state{init_state_id}_prompt{prompt_side}_{record['first_grasp_object']}.mp4",
                        replay_images,
                        fps=10,
                    )
    finally:
        env.close()

    condition_summaries = {}
    for prompt_side in (0, 1):
        selected = [record for record in records if record["prompt_side"] == prompt_side]
        condition_summaries[str(prompt_side)] = {
            "prompt_object": object_names[prompt_side],
            "condition": "aligned" if prompt_side == args.scene_side else "swapped",
            "num_trials": len(selected),
            "commanded_first_grasp_rate": float(
                np.mean([record["commanded_first_grasp"] for record in selected])
            ),
            "commanded_ever_grasp_rate": float(
                np.mean([record["commanded_ever_grasped"] for record in selected])
            ),
            "commanded_in_basket_rate": float(
                np.mean([record["commanded_in_basket"] for record in selected])
            ),
            "first_grasp_counts": dict(
                collections.Counter(record["first_grasp_object"] for record in selected)
            ),
        }

    summary = {
        "protocol": (
            "unmodified pi0.5; fixed anchor scene; same initial states under both pair prompts; "
            "deterministic query-indexed noise; first grasp is primary object-selection endpoint"
        ),
        "pair": pair,
        "scene_side": args.scene_side,
        "task_id": task_id,
        "task_language": task.language,
        "num_paired_initial_states": len(init_state_ids),
        "num_rollouts": len(records),
        "init_state_ids": init_state_ids,
        "conditions": condition_summaries,
        "patching": False,
        "hooks_registered": False,
        "sampler": "official pi0.5 model.sample_actions",
        "replan_steps": args.replan_steps,
        "num_denoise_steps": args.num_denoise_steps,
        "elapsed_seconds": time.perf_counter() - started_at,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
