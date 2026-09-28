#!/usr/bin/env python3
"""Evaluate fixed block-13 natural paired patching in closed-loop LIBERO-Object."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time
from pathlib import Path

import h5py
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


OBJECT_A = "alphabet_soup_1"
OBJECT_B = "cream_cheese_1"
BASKET_REGION = "basket_1_contain_region"
PROMPTS = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
)
NULL_PROMPT = "pick up an object and place it in the basket"
DEFAULT_CONDITIONS = ("correct", "mismatch", "paired", "random")
CONDITIONS = (*DEFAULT_CONDITIONS, "null", "pathway")
TARGET_PROMPT_INDEX = {0: 0, 1: 1}
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator, dtype=np.float32)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--max-action-steps", type=int, default=280)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(DEFAULT_CONDITIONS),
    )
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--verify-sampler", action="store_true")
    return parser.parse_args()


def check_in(base_env: object, object_name: str) -> bool:
    return bool(base_env._eval_predicate(["in", object_name, BASKET_REGION]))


def check_grasp(base_env: object, object_name: str) -> bool:
    obj = base_env.objects_dict[object_name]
    return bool(base_env._check_grasp(base_env.robots[0].gripper, obj.contact_geoms))


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
    batch_size: int = 2,
) -> torch.Tensor:
    sequence = np.random.SeedSequence([seed, task_id, init_state_id, query_index])
    rng = np.random.default_rng(sequence)
    single = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    return torch.from_numpy(np.repeat(single, batch_size, axis=0)).to(device)


def unused_init_state_ids(
    source_h5: Path, task_id: int, available: int, count: int, seed: int
) -> list[int]:
    with h5py.File(source_h5, "r") as source:
        task_ids = np.asarray(source["task_ids"][:], dtype=np.int64)
        used = set(
            np.asarray(source["init_state_ids"][:], dtype=np.int64)[task_ids == task_id].tolist()
        )
    candidates = np.asarray(sorted(set(range(available)) - used), dtype=np.int64)
    if len(candidates) < count:
        raise ValueError(
            f"Task {task_id} has only {len(candidates)} unused initial states; requested {count}."
        )
    rng = np.random.default_rng(seed + task_id)
    return rng.permutation(candidates)[:count].astype(int).tolist()


def condition_spec(condition: str, target_prompt_index: int) -> tuple[int, int, str]:
    alternative = 1 - target_prompt_index
    if condition == "correct":
        return target_prompt_index, alternative, "none"
    if condition == "mismatch":
        return alternative, target_prompt_index, "none"
    if condition == "paired":
        return alternative, target_prompt_index, "paired"
    if condition == "random":
        return alternative, target_prompt_index, "random"
    if condition == "pathway":
        return alternative, target_prompt_index, "pathway"
    raise ValueError(f"Unknown condition: {condition}")


def unnormalize_actions(policy, observation, normalized: torch.Tensor, index: int) -> np.ndarray:
    outputs = {
        "state": observation.state[index].detach().cpu().numpy(),
        "actions": normalized[index].detach().cpu().numpy(),
    }
    return np.asarray(policy._output_transform(outputs)["actions"], dtype=np.float32)  # noqa: SLF001


def episode_metrics(
    *,
    initial_positions: dict[str, np.ndarray],
    min_eef_distance: dict[str, float],
    max_displacement: dict[str, float],
    ever_grasped: dict[str, bool],
    ever_in_basket: dict[str, bool],
    task_id: int,
    success: bool,
) -> dict[str, object]:
    target = OBJECT_A if task_id == 0 else OBJECT_B
    alternative = OBJECT_B if task_id == 0 else OBJECT_A
    return {
        "success": success,
        "target_object": target,
        "alternative_object": alternative,
        "target_min_eef_distance": min_eef_distance[target],
        "alternative_min_eef_distance": min_eef_distance[alternative],
        "target_max_displacement": max_displacement[target],
        "alternative_max_displacement": max_displacement[alternative],
        "target_ever_grasped": ever_grasped[target],
        "alternative_ever_grasped": ever_grasped[alternative],
        "target_ever_in_basket": ever_in_basket[target],
        "alternative_ever_in_basket": ever_in_basket[alternative],
        "object_a_initial_xyz": initial_positions[OBJECT_A].tolist(),
        "object_b_initial_xyz": initial_positions[OBJECT_B].tolist(),
    }


def update_episode_tracking(
    *,
    env: OffScreenRenderEnv,
    observation: dict[str, np.ndarray],
    initial_positions: dict[str, np.ndarray],
    min_eef_distance: dict[str, float],
    max_displacement: dict[str, float],
    ever_grasped: dict[str, bool],
    ever_in_basket: dict[str, bool],
) -> None:
    eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    for object_name in (OBJECT_A, OBJECT_B):
        position = np.asarray(observation[f"{object_name}_pos"], dtype=np.float32)
        min_eef_distance[object_name] = min(
            min_eef_distance[object_name], float(np.linalg.norm(eef - position))
        )
        max_displacement[object_name] = max(
            max_displacement[object_name],
            float(np.linalg.norm(position - initial_positions[object_name])),
        )
        ever_grasped[object_name] |= check_grasp(env.env, object_name)
        ever_in_basket[object_name] |= check_in(env.env, object_name)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.num_trials <= 0 or args.replan_steps <= 0:
        raise ValueError("Trial and replanning counts must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    sampler = Pi05ActionExpertResidualExtractor(model)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(args.task_id)
    init_states_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    # LIBERO init states are trusted local simulator arrays, not a model checkpoint.
    initial_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    init_state_ids = unused_init_state_ids(
        args.source_h5,
        args.task_id,
        len(initial_states),
        args.num_trials,
        args.seed,
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
    target_prompt_index = TARGET_PROMPT_INDEX[args.task_id]
    records = []
    sampler_audits = {}
    started_at = time.perf_counter()

    try:
        for trial_index, init_state_id in enumerate(
            tqdm(init_state_ids, desc=f"pi05 closed loop task {args.task_id}", unit="state")
        ):
            for condition in args.conditions:
                observation = env.reset()
                observation = env.set_init_state(initial_states[init_state_id])
                for _ in range(args.wait_steps):
                    observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                if condition == "null":
                    base_prompt_index, source_prompt_index, mode = 0, None, "none"
                else:
                    base_prompt_index, source_prompt_index, mode = condition_spec(
                        condition, target_prompt_index
                    )
                initial_positions = {
                    OBJECT_A: np.asarray(observation[f"{OBJECT_A}_pos"], dtype=np.float32),
                    OBJECT_B: np.asarray(observation[f"{OBJECT_B}_pos"], dtype=np.float32),
                }
                min_eef_distance = {OBJECT_A: math.inf, OBJECT_B: math.inf}
                max_displacement = {OBJECT_A: 0.0, OBJECT_B: 0.0}
                ever_grasped = {OBJECT_A: False, OBJECT_B: False}
                ever_in_basket = {OBJECT_A: False, OBJECT_B: False}
                action_plan: collections.deque[np.ndarray] = collections.deque()
                replay_images = []
                query_index = 0
                success = False
                executed_steps = 0
                pathway_action_max_abs_difference = 0.0
                pathway_velocity_max_abs_difference = 0.0
                pathway_layer_hook_count_min = None
                pathway_layer_hook_count_max = None

                update_episode_tracking(
                    env=env,
                    observation=observation,
                    initial_positions=initial_positions,
                    min_eef_distance=min_eef_distance,
                    max_displacement=max_displacement,
                    ever_grasped=ever_grasped,
                    ever_in_basket=ever_in_basket,
                )

                for action_step in range(args.max_action_steps):
                    base_image, wrist_image = transformed_images(observation)
                    if args.save_videos:
                        replay_images.append(base_image)
                    if not action_plan:
                        if condition == "null":
                            policy_observation = prepare_pi05_observation_batch(
                                policy,
                                base_images=[base_image],
                                wrist_images=[wrist_image],
                                states=[robot_state(observation)],
                                prompts=[NULL_PROMPT],
                                device=device,
                            )
                            noise_batch_size = 1
                        else:
                            policy_observation = prepare_pi05_observation_batch(
                                policy,
                                base_images=[base_image, base_image],
                                wrist_images=[wrist_image, wrist_image],
                                states=[robot_state(observation), robot_state(observation)],
                                prompts=list(PROMPTS),
                                device=device,
                            )
                            noise_batch_size = 2
                        noise = deterministic_noise(
                            seed=args.seed,
                            task_id=args.task_id,
                            init_state_id=init_state_id,
                            query_index=query_index,
                            horizon=model.config.action_horizon,
                            action_dim=model.config.action_dim,
                            device=device,
                            batch_size=noise_batch_size,
                        )
                        if condition == "null":
                            normalized = model.sample_actions(
                                device,
                                policy_observation,
                                noise=noise,
                                num_steps=args.num_denoise_steps,
                            ).float().cpu()
                        elif condition == "pathway":
                            replay = sampler.sample_actions_full_expert_replay(
                                policy_observation,
                                noise=noise,
                                base_prompt_index=base_prompt_index,
                                source_prompt_index=source_prompt_index,
                                num_steps=args.num_denoise_steps,
                            )
                            normalized = replay["actions"]
                            query_action_error = float(
                                replay["action_max_abs_difference"].item()
                            )
                            query_velocity_error = float(
                                replay["velocity_max_abs_difference"].max().item()
                            )
                            query_hook_min = int(replay["layer_hook_counts"].min().item())
                            query_hook_max = int(replay["layer_hook_counts"].max().item())
                            if query_hook_min != args.num_denoise_steps or query_hook_max != args.num_denoise_steps:
                                raise RuntimeError(
                                    "Full-pathway replay hook count mismatch: "
                                    f"min={query_hook_min}, max={query_hook_max}."
                                )
                            pathway_action_max_abs_difference = max(
                                pathway_action_max_abs_difference, query_action_error
                            )
                            pathway_velocity_max_abs_difference = max(
                                pathway_velocity_max_abs_difference, query_velocity_error
                            )
                            pathway_layer_hook_count_min = (
                                query_hook_min
                                if pathway_layer_hook_count_min is None
                                else min(pathway_layer_hook_count_min, query_hook_min)
                            )
                            pathway_layer_hook_count_max = (
                                query_hook_max
                                if pathway_layer_hook_count_max is None
                                else max(pathway_layer_hook_count_max, query_hook_max)
                            )
                        else:
                            normalized = sampler.sample_actions_paired_patch(
                                policy_observation,
                                noise=noise,
                                layer_index=args.layer,
                                base_prompt_index=base_prompt_index,
                                source_prompt_index=source_prompt_index,
                                mode=mode,
                                alpha=args.alpha,
                                num_steps=args.num_denoise_steps,
                                random_seed=(
                                    args.seed
                                    + init_state_id * 10000
                                    + query_index * 20
                                    + base_prompt_index
                                ),
                            )
                        if args.verify_sampler and trial_index == 0 and query_index == 0:
                            official = model.sample_actions(
                                device,
                                policy_observation,
                                noise=noise,
                                num_steps=args.num_denoise_steps,
                            ).float().cpu()
                            if condition == "pathway":
                                source_error = float(
                                    (
                                        normalized[source_prompt_index]
                                        - official[source_prompt_index]
                                    )
                                    .abs()
                                    .max()
                                    .item()
                                )
                                replay_error = float(
                                    (
                                        normalized[base_prompt_index]
                                        - official[source_prompt_index]
                                    )
                                    .abs()
                                    .max()
                                    .item()
                                )
                                sampler_audits[condition] = {
                                    "source_vs_official_max_abs_difference": source_error,
                                    "replay_vs_official_source_max_abs_difference": replay_error,
                                    "num_expert_layers": len(sampler.layers),
                                    "hook_count_per_layer": args.num_denoise_steps,
                                }
                                if source_error != 0.0:
                                    raise RuntimeError(
                                        "Full-pathway replay changed the source branch: "
                                        f"{source_error}."
                                    )
                                audit_error = None
                            elif condition == "null" or mode == "none":
                                audit_error = float((normalized - official).abs().max().item())
                            else:
                                audit_error = float(
                                    (
                                        normalized[source_prompt_index]
                                        - official[source_prompt_index]
                                    )
                                    .abs()
                                    .max()
                                    .item()
                                )
                            if condition != "pathway":
                                sampler_audits[condition] = audit_error
                            if audit_error is not None and audit_error != 0.0:
                                raise RuntimeError(
                                    f"Closed-loop sampler audit failed for {condition}: {audit_error}."
                                )
                        actions = unnormalize_actions(
                            policy, policy_observation, normalized, base_prompt_index
                        )
                        if len(actions) < args.replan_steps:
                            raise RuntimeError("Action chunk is shorter than --replan-steps.")
                        action_plan.extend(actions[: args.replan_steps])
                        query_index += 1

                    action = action_plan.popleft()
                    observation, _, done, _ = env.step(action.tolist())
                    executed_steps = action_step + 1
                    update_episode_tracking(
                        env=env,
                        observation=observation,
                        initial_positions=initial_positions,
                        min_eef_distance=min_eef_distance,
                        max_displacement=max_displacement,
                        ever_grasped=ever_grasped,
                        ever_in_basket=ever_in_basket,
                    )
                    if env.check_success():
                        success = True
                        break
                    if done:
                        break

                metrics = episode_metrics(
                    initial_positions=initial_positions,
                    min_eef_distance=min_eef_distance,
                    max_displacement=max_displacement,
                    ever_grasped=ever_grasped,
                    ever_in_basket=ever_in_basket,
                    task_id=args.task_id,
                    success=success,
                )
                record = {
                    "task_id": args.task_id,
                    "task_language": task.language,
                    "trial_index": trial_index,
                    "init_state_id": init_state_id,
                    "condition": condition,
                    "base_prompt_index": base_prompt_index,
                    "source_prompt_index": source_prompt_index,
                    "patch_mode": mode,
                    "executed_steps": executed_steps,
                    "num_policy_queries": query_index,
                    "pathway_action_max_abs_difference": (
                        pathway_action_max_abs_difference if condition == "pathway" else None
                    ),
                    "pathway_velocity_max_abs_difference": (
                        pathway_velocity_max_abs_difference if condition == "pathway" else None
                    ),
                    "pathway_layer_hook_count_min": (
                        pathway_layer_hook_count_min if condition == "pathway" else None
                    ),
                    "pathway_layer_hook_count_max": (
                        pathway_layer_hook_count_max if condition == "pathway" else None
                    ),
                    **metrics,
                }
                records.append(record)
                with (args.output_dir / "episodes.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(record) + "\n")
                if args.save_videos:
                    suffix = "success" if success else "failure"
                    imageio.mimwrite(
                        video_dir
                        / f"task{args.task_id}_init{init_state_id}_{condition}_{suffix}.mp4",
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
            "success_rate": float(np.mean([record["success"] for record in selected])),
            "target_grasp_rate": float(
                np.mean([record["target_ever_grasped"] for record in selected])
            ),
            "alternative_grasp_rate": float(
                np.mean([record["alternative_ever_grasped"] for record in selected])
            ),
            "target_in_basket_rate": float(
                np.mean([record["target_ever_in_basket"] for record in selected])
            ),
            "target_min_eef_distance_mean": float(
                np.mean([record["target_min_eef_distance"] for record in selected])
            ),
            "target_max_displacement_mean": float(
                np.mean([record["target_max_displacement"] for record in selected])
            ),
        }
    summary = {
        "task_id": args.task_id,
        "task_language": task.language,
        "target_prompt_index": target_prompt_index,
        "unused_init_state_protocol": True,
        "init_state_ids": init_state_ids,
        "conditions": list(args.conditions),
        "null_prompt": NULL_PROMPT if "null" in args.conditions else None,
        "full_pathway_replay": (
            "all_18_action_expert_block_outputs_at_every_denoising_step"
            if "pathway" in args.conditions
            else None
        ),
        "layer": args.layer,
        "alpha": args.alpha,
        "replan_steps": args.replan_steps,
        "num_denoise_steps": args.num_denoise_steps,
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
