#!/usr/bin/env python3
"""Replay Stage 11 pi0.5 rollouts under the official sampler with read-only residual capture."""

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
STAGE11_MATCH_FIELDS = (
    "first_grasp_side",
    "first_grasp_step",
    "executed_steps",
    "num_policy_queries",
    "ever_grasped_a",
    "ever_grasped_b",
    "ever_in_basket_a",
    "ever_in_basket_b",
    "bddl_success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--scene-side", type=int, choices=(0, 1), default=0)
    parser.add_argument("--stage11-dir", type=Path, required=True)
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


def hidden_from_hook(output: object) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class ReadOnlyResidualCapture:
    """Record post-block residuals during one official ``sample_actions`` call.

    Hooks return ``None`` and therefore never modify the forward pass. The prefix
    stores the final prompt token after every PaliGemma block. The expert stores
    the ten-action-token mean after every Gemma-expert block at the first
    denoising evaluation (``t=1``), matching the atlas extraction convention.
    """

    def __init__(self, model: torch.nn.Module):
        self.prefix_layers = model.paligemma_with_expert.paligemma.language_model.layers
        self.expert_layers = model.paligemma_with_expert.gemma_expert.model.layers
        self.action_horizon = int(model.config.action_horizon)

    def run(self, model, device, observation, noise, num_steps):
        language_capacity = int(observation.tokenized_prompt.shape[1])
        prompt_length = int(observation.tokenized_prompt_mask[0].sum().item())
        prefix_end: list[torch.Tensor | None] = [None] * len(self.prefix_layers)
        expert_first: list[torch.Tensor | None] = [None] * len(self.expert_layers)
        prefix_calls = [0] * len(self.prefix_layers)
        expert_calls = [0] * len(self.expert_layers)

        def prefix_hook(_module, _inputs, output, index):
            hidden = hidden_from_hook(output)
            prompt_end = hidden.shape[1] - language_capacity + prompt_length - 1
            prefix_end[index] = hidden[0, prompt_end].detach().float().cpu()
            prefix_calls[index] += 1

        def expert_hook(_module, _inputs, output, index):
            hidden = hidden_from_hook(output)
            if hidden.shape[1] != self.action_horizon:
                raise RuntimeError(f"Unexpected expert token count {hidden.shape[1]}.")
            if expert_calls[index] == 0:
                expert_first[index] = hidden[0].detach().float().mean(dim=0).cpu()
            expert_calls[index] += 1

        handles = []
        for index, layer in enumerate(self.prefix_layers):
            handles.append(
                layer.register_forward_hook(
                    lambda module, inputs, output, index=index: prefix_hook(
                        module, inputs, output, index
                    )
                )
            )
        for index, layer in enumerate(self.expert_layers):
            handles.append(
                layer.register_forward_hook(
                    lambda module, inputs, output, index=index: expert_hook(
                        module, inputs, output, index
                    )
                )
            )
        try:
            normalized = model.sample_actions(
                device, observation, noise=noise, num_steps=num_steps
            )
        finally:
            for handle in handles:
                handle.remove()

        if any(count != 1 for count in prefix_calls):
            raise RuntimeError(f"Prefix hook counts {prefix_calls}; expected one per block.")
        if any(count != num_steps for count in expert_calls):
            raise RuntimeError(f"Expert hook counts {expert_calls}; expected {num_steps}.")
        return normalized, {
            "prefix_prompt_end": torch.stack(prefix_end).numpy(),
            "expert_action_mean_t1": torch.stack(expert_first).numpy(),
        }


def load_stage11_records(stage11_dir: Path) -> tuple[dict, dict[tuple[int, int], dict]]:
    summary = json.loads((stage11_dir / "summary.json").read_text(encoding="utf-8"))
    records = {}
    with (stage11_dir / "episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[(int(record["init_state_id"]), int(record["prompt_side"]))] = record
    return summary, records


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

    stage11_summary, stage11_records = load_stage11_records(args.stage11_dir)
    if int(stage11_summary["task_id"]) != task_id or int(stage11_summary["scene_side"]) != args.scene_side:
        raise ValueError("Stage 11 directory does not match the requested pair and scene side.")
    for key, expected in (
        ("replan_steps", args.replan_steps),
        ("num_denoise_steps", args.num_denoise_steps),
    ):
        if int(stage11_summary[key]) != expected:
            raise ValueError(f"Stage 11 used {key}={stage11_summary[key]}, not {expected}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual_dir = args.output_dir / "residuals"
    residual_dir.mkdir()
    video_dir = args.output_dir / "videos"
    if args.save_videos:
        video_dir.mkdir()

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    capture = ReadOnlyResidualCapture(model)

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
    if init_state_ids != stage11_summary["init_state_ids"][: args.num_trials]:
        raise ValueError("Initial-state order differs from the Stage 11 record.")

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
                initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float32).copy()
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
                query_prefix: list[np.ndarray] = []
                query_expert: list[np.ndarray] = []
                query_steps: list[int] = []
                query_object_xyz: list[np.ndarray] = []
                query_eef_xyz: list[np.ndarray] = []
                query_grasped: list[tuple[bool, bool]] = []
                first_query_unhooked_max_abs = None
                current_grasped: list[bool] = [False, False]

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
                        normalized, residuals = capture.run(
                            model, device, policy_observation, noise, args.num_denoise_steps
                        )
                        normalized = normalized.float().cpu()
                        if query_index == 0:
                            unhooked = model.sample_actions(
                                device,
                                policy_observation,
                                noise=noise,
                                num_steps=args.num_denoise_steps,
                            ).float().cpu()
                            first_query_unhooked_max_abs = float(
                                (unhooked - normalized).abs().max()
                            )
                        query_prefix.append(residuals["prefix_prompt_end"])
                        query_expert.append(residuals["expert_action_mean_t1"])
                        query_steps.append(action_step)
                        query_object_xyz.append(
                            np.stack(
                                [
                                    np.asarray(observation[f"{name}_pos"], dtype=np.float32)
                                    for name in object_instances
                                ]
                            )
                        )
                        query_eef_xyz.append(
                            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).copy()
                        )
                        query_grasped.append(tuple(current_grasped))
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

                residual_file = f"state{init_state_id:02d}_prompt{prompt_side}.npz"
                np.savez_compressed(
                    residual_dir / residual_file,
                    prefix_prompt_end=np.stack(query_prefix).astype(np.float16),
                    expert_action_mean_t1=np.stack(query_expert).astype(np.float16),
                    query_action_steps=np.asarray(query_steps, dtype=np.int16),
                    query_object_xyz=np.stack(query_object_xyz),
                    query_eef_xyz=np.stack(query_eef_xyz),
                    query_grasped=np.asarray(query_grasped, dtype=bool),
                )

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
                    "ever_grasped_a": ever_grasped[0],
                    "ever_grasped_b": ever_grasped[1],
                    "ever_in_basket_a": ever_in_basket[0],
                    "ever_in_basket_b": ever_in_basket[1],
                    "bddl_success": bddl_success,
                    "min_eef_distance_a": min_eef_distance[0],
                    "min_eef_distance_b": min_eef_distance[1],
                    "max_displacement_a": max_displacement[0],
                    "max_displacement_b": max_displacement[1],
                    "executed_steps": executed_steps,
                    "num_policy_queries": query_index,
                    "initial_object_a_xyz": initial_positions[0].tolist(),
                    "initial_object_b_xyz": initial_positions[1].tolist(),
                    "initial_eef_xyz": initial_eef.tolist(),
                    "residual_file": f"residuals/{residual_file}",
                    "first_query_hooked_vs_unhooked_max_abs": first_query_unhooked_max_abs,
                }
                reference = stage11_records.get((init_state_id, prompt_side))
                if reference is None:
                    raise KeyError(f"No Stage 11 record for state {init_state_id}, prompt {prompt_side}.")
                mismatches = {
                    field: {"stage11": reference[field], "stage12": record[field]}
                    for field in STAGE11_MATCH_FIELDS
                    if reference[field] != record[field]
                }
                record["stage11_match"] = not mismatches
                record["stage11_mismatches"] = mismatches
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

    unhooked_errors = [record["first_query_hooked_vs_unhooked_max_abs"] for record in records]
    summary = {
        "protocol": (
            "Stage 12 read-only replay of Stage 11: unmodified pi0.5, official sample_actions, "
            "forward hooks that only store residuals; same initial states, prompts, and "
            "query-indexed noise as Stage 11"
        ),
        "pair": pair,
        "scene_side": args.scene_side,
        "task_id": task_id,
        "task_language": task.language,
        "stage11_dir": str(args.stage11_dir),
        "num_paired_initial_states": len(init_state_ids),
        "num_rollouts": len(records),
        "init_state_ids": init_state_ids,
        "audits": {
            "patching": False,
            "hooks_registered": True,
            "hooks_return_none": True,
            "sampler": "official pi0.5 model.sample_actions",
            "prefix_hook_calls_per_query_per_block": 1,
            "expert_hook_calls_per_query_per_block": args.num_denoise_steps,
            "stage11_outcomes_reproduced": sum(record["stage11_match"] for record in records),
            "stage11_outcomes_total": len(records),
            "all_stage11_outcomes_reproduced": all(record["stage11_match"] for record in records),
            "max_first_query_hooked_vs_unhooked_action_error": max(unhooked_errors),
        },
        "residual_readouts": {
            "prefix_prompt_end": "final prompt token after each PaliGemma block, [queries, 18, 2048]",
            "expert_action_mean_t1": (
                "ten-action-token mean after each expert block at the first denoising "
                "evaluation t=1, [queries, 18, 1024]"
            ),
            "storage_dtype": "float16",
        },
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
