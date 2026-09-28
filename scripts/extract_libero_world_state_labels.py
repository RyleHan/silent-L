#!/usr/bin/env python3
"""Extract simulator-grounded world-state labels for the OpenVLA atlas."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv
from vla_coordinates.libero_object_pairs import (
    read_pair_ids,
    read_pairs,
    read_target_sides,
    task_lookup,
)


BASKET_REGION = "basket_1_contain_region"
PHASE_NAMES = ("approach", "pregrasp", "transport", "placing", "complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--samples-per-task", type=int)
    parser.add_argument("--near-target-m", type=float, default=0.08)
    parser.add_argument("--near-basket-m", type=float, default=0.12)
    parser.add_argument("--moved-m", type=float, default=0.03)
    parser.add_argument("--lifted-m", type=float, default=0.03)
    parser.add_argument("--max-action-steps", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def check_in(base_env: object, object_name: str) -> bool:
    return bool(base_env._eval_predicate(["in", object_name, BASKET_REGION]))


def check_grasp(base_env: object, object_name: str) -> bool:
    obj = base_env.objects_dict[object_name]
    return bool(base_env._check_grasp(base_env.robots[0].gripper, obj.contact_geoms))


def groupwise_kinematics(
    values: np.ndarray,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    action_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.empty_like(values)
    speed = np.zeros(len(values), dtype=np.float32)
    for task_id, episode_id in sorted(set(zip(task_ids.tolist(), episode_ids.tolist()))):
        indices = np.flatnonzero((task_ids == task_id) & (episode_ids == episode_id))
        order = indices[np.argsort(action_steps[indices])]
        displacement[order] = values[order] - values[order[0]]
        if len(order) > 1:
            deltas = np.linalg.norm(np.diff(values[order], axis=0), axis=1)
            step_deltas = np.diff(action_steps[order]).clip(min=1)
            speed[order[1:]] = deltas / step_deltas
    return displacement, speed


def numeric_summary(values: np.ndarray) -> dict[str, object]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "shape": list(values.shape),
        "finite": int(np.isfinite(flat).sum()),
        "min": float(np.nanmin(flat)),
        "median": float(np.nanmedian(flat)),
        "max": float(np.nanmax(flat)),
        "mean": float(np.nanmean(flat)),
        "std": float(np.nanstd(flat)),
    }


def main() -> None:
    args = parse_args()
    if args.output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_h5}. Use --overwrite to replace it.")
    started_at = time.perf_counter()

    with h5py.File(args.source_h5, "r") as source:
        available = source["task_ids"].shape[0]
        all_task_ids = np.asarray(source["task_ids"])
        pairs = read_pairs(source)
        tasks = task_lookup(pairs)
        all_pair_ids = read_pair_ids(source)
        all_target_sides = read_target_sides(source)
        if args.samples_per_task:
            selected_indices = np.concatenate(
                [
                    np.flatnonzero(all_task_ids == task_id)[: args.samples_per_task]
                    for task_id in sorted(np.unique(all_task_ids))
                ]
            )
        else:
            selected_indices = np.arange(min(available, args.max_samples or available))
        count = len(selected_indices)
        task_ids = np.asarray(source["task_ids"][selected_indices], dtype=np.int8)
        pair_ids = np.asarray(all_pair_ids[selected_indices], dtype=np.int8)
        target_sides = np.asarray(all_target_sides[selected_indices], dtype=np.int8)
        episode_ids = np.asarray(source["episode_ids"][selected_indices], dtype=np.int16)
        init_state_ids = np.asarray(source["init_state_ids"][selected_indices], dtype=np.int16)
        action_steps = np.asarray(source["action_steps"][selected_indices], dtype=np.int16)
        splits = decode_strings(source["layout_splits"][selected_indices])
        simulator_states = source["simulator_states"]

        arrays: dict[str, np.ndarray] = {
            "object_a_xyz": np.empty((count, 3), dtype=np.float32),
            "object_b_xyz": np.empty((count, 3), dtype=np.float32),
            "basket_xyz": np.empty((count, 3), dtype=np.float32),
            "eef_xyz": np.empty((count, 3), dtype=np.float32),
            "object_a_quat_xyzw": np.empty((count, 4), dtype=np.float32),
            "object_b_quat_xyzw": np.empty((count, 4), dtype=np.float32),
            "eef_quat_xyzw": np.empty((count, 4), dtype=np.float32),
            "object_a_to_eef_xyz": np.empty((count, 3), dtype=np.float32),
            "object_b_to_eef_xyz": np.empty((count, 3), dtype=np.float32),
            "gripper_qpos": np.empty((count, 2), dtype=np.float32),
            "robot_joint_pos": np.empty((count, 7), dtype=np.float32),
            "object_a_in_basket": np.empty(count, dtype=np.bool_),
            "object_b_in_basket": np.empty(count, dtype=np.bool_),
            "object_a_grasped": np.empty(count, dtype=np.bool_),
            "object_b_grasped": np.empty(count, dtype=np.bool_),
            "task_success": np.empty(count, dtype=np.bool_),
        }
        task_metadata = []
        suite = benchmark.get_benchmark_dict()["libero_object"]()

        for task_id in sorted(np.unique(task_ids).astype(int).tolist()):
            if task_id not in tasks:
                raise ValueError(f"No target mapping declared for task {task_id}.")
            pair, target_side = tasks[task_id]
            object_a, object_b = pair["object_instances"]
            if not np.all(target_sides[task_ids == task_id] == target_side):
                raise ValueError(f"Task {task_id} has inconsistent target_sides metadata.")
            task = suite.get_task(task_id)
            bddl_path = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            env = SegmentationRenderEnv(
                bddl_file_name=bddl_path,
                camera_names=["agentview"],
                camera_heights=256,
                camera_widths=256,
                camera_segmentations="instance",
            )
            try:
                env.reset()
                indices = np.flatnonzero(task_ids == task_id)
                progress = tqdm(indices, desc=f"semantic labels task {task_id}", unit="state")
                for index in progress:
                    observation = env.set_init_state(
                        np.asarray(simulator_states[int(selected_indices[index])])
                    )
                    base_env = env.env
                    basket_state = base_env.object_states_dict[BASKET_REGION].get_geom_state()

                    arrays["object_a_xyz"][index] = observation[f"{object_a}_pos"]
                    arrays["object_b_xyz"][index] = observation[f"{object_b}_pos"]
                    arrays["basket_xyz"][index] = basket_state["pos"]
                    arrays["eef_xyz"][index] = observation["robot0_eef_pos"]
                    arrays["object_a_quat_xyzw"][index] = observation[f"{object_a}_quat"]
                    arrays["object_b_quat_xyzw"][index] = observation[f"{object_b}_quat"]
                    arrays["eef_quat_xyzw"][index] = observation["robot0_eef_quat"]
                    arrays["object_a_to_eef_xyz"][index] = observation[
                        f"{object_a}_to_robot0_eef_pos"
                    ]
                    arrays["object_b_to_eef_xyz"][index] = observation[
                        f"{object_b}_to_robot0_eef_pos"
                    ]
                    arrays["gripper_qpos"][index] = observation["robot0_gripper_qpos"]
                    arrays["robot_joint_pos"][index] = observation["robot0_joint_pos"]
                    arrays["object_a_in_basket"][index] = check_in(base_env, object_a)
                    arrays["object_b_in_basket"][index] = check_in(base_env, object_b)
                    arrays["object_a_grasped"][index] = check_grasp(base_env, object_a)
                    arrays["object_b_grasped"][index] = check_grasp(base_env, object_b)
                    arrays["task_success"][index] = env.check_success()

                task_metadata.append(
                    {
                        "task_id": task_id,
                        "pair_id": int(pair["pair_id"]),
                        "target_side": int(target_side),
                        "language": task.language,
                        "bddl_path": bddl_path,
                        "goal_state": base_env.parsed_problem["goal_state"],
                        "target_object": pair["object_instances"][target_side],
                    }
                )
            finally:
                env.close()

    target_is_a = target_sides == 0
    target_xyz = np.where(target_is_a[:, None], arrays["object_a_xyz"], arrays["object_b_xyz"])
    alternative_xyz = np.where(target_is_a[:, None], arrays["object_b_xyz"], arrays["object_a_xyz"])
    target_to_eef_xyz = np.where(
        target_is_a[:, None], arrays["object_a_to_eef_xyz"], arrays["object_b_to_eef_xyz"]
    )
    alternative_to_eef_xyz = np.where(
        target_is_a[:, None], arrays["object_b_to_eef_xyz"], arrays["object_a_to_eef_xyz"]
    )
    target_grasped = np.where(
        target_is_a, arrays["object_a_grasped"], arrays["object_b_grasped"]
    )
    target_in_basket = np.where(
        target_is_a, arrays["object_a_in_basket"], arrays["object_b_in_basket"]
    )
    target_displacement, target_speed = groupwise_kinematics(
        target_xyz, task_ids, episode_ids, action_steps
    )
    eef_displacement, eef_speed = groupwise_kinematics(
        arrays["eef_xyz"], task_ids, episode_ids, action_steps
    )

    target_to_basket_xyz = arrays["basket_xyz"] - target_xyz
    eef_target_distance = np.linalg.norm(arrays["eef_xyz"] - target_xyz, axis=1)
    eef_alternative_distance = np.linalg.norm(arrays["eef_xyz"] - alternative_xyz, axis=1)
    target_basket_distance = np.linalg.norm(target_to_basket_xyz, axis=1)
    target_xy_displacement = np.linalg.norm(target_displacement[:, :2], axis=1)
    target_height_delta = target_displacement[:, 2]
    target_moved = target_xy_displacement >= args.moved_m
    target_lifted = target_height_delta >= args.lifted_m
    near_target = eef_target_distance <= args.near_target_m
    near_basket = target_basket_distance <= args.near_basket_m

    phase = np.zeros(count, dtype=np.int8)
    phase[near_target] = 1
    phase[target_grasped | target_lifted] = 2
    placing = near_basket & (target_moved | target_lifted) & ~target_grasped
    phase[placing] = 3
    phase[target_in_basket] = 4

    derived = {
        "target_xyz": target_xyz.astype(np.float32),
        "alternative_xyz": alternative_xyz.astype(np.float32),
        "target_to_eef_xyz": target_to_eef_xyz.astype(np.float32),
        "alternative_to_eef_xyz": alternative_to_eef_xyz.astype(np.float32),
        "target_to_basket_xyz": target_to_basket_xyz.astype(np.float32),
        "target_displacement_xyz": target_displacement.astype(np.float32),
        "eef_displacement_xyz": eef_displacement.astype(np.float32),
        "eef_target_distance": eef_target_distance.astype(np.float32),
        "eef_alternative_distance": eef_alternative_distance.astype(np.float32),
        "target_basket_distance": target_basket_distance.astype(np.float32),
        "target_xy_displacement": target_xy_displacement.astype(np.float32),
        "target_height_delta": target_height_delta.astype(np.float32),
        "target_speed_per_action_step": target_speed,
        "eef_speed_per_action_step": eef_speed,
        "gripper_width": (arrays["gripper_qpos"][:, 0] - arrays["gripper_qpos"][:, 1]).astype(
            np.float32
        ),
        "normalized_action_progress": (action_steps / args.max_action_steps).astype(np.float32),
        "target_grasped": target_grasped.astype(np.bool_),
        "target_in_basket": target_in_basket.astype(np.bool_),
        "target_moved": target_moved,
        "target_lifted": target_lifted,
        "near_target": near_target,
        "near_basket": near_basket,
        "phase_heuristic": phase,
    }

    definitions = {
        "raw": {
            "objects_are_pair_local": True,
            "pairs": pairs,
            "basket_region": BASKET_REGION,
            "in_basket": "LIBERO in predicate: contact with and contained by basket region",
            "grasped": "robosuite _check_grasp using gripper and object contact geoms",
        },
        "derived": {
            "target": "object selected by the actual rollout task instruction",
            "target_to_basket_xyz": "basket_xyz - target_xyz in world coordinates",
            "displacement": "current xyz - first cached xyz in the same task/episode",
            "speed_per_action_step": "Euclidean finite difference divided by action-step gap",
            "phase_heuristic": dict(enumerate(PHASE_NAMES)),
        },
        "thresholds_m": {
            "near_target": args.near_target_m,
            "near_basket": args.near_basket_m,
            "target_moved_xy": args.moved_m,
            "target_lifted_z": args.lifted_m,
        },
    }

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output_h5, "w") as output:
        for key, values in {
            "task_ids": task_ids,
            "pair_ids": pair_ids,
            "target_sides": target_sides,
            "episode_ids": episode_ids,
            "init_state_ids": init_state_ids,
            "action_steps": action_steps,
            "source_indices": selected_indices,
        }.items():
            output.create_dataset(key, data=values)
        output.create_dataset("layout_splits", data=splits.astype(object), dtype=string_dtype)
        raw_group = output.create_group("raw")
        for key, values in arrays.items():
            raw_group.create_dataset(key, data=values, compression="gzip")
        derived_group = output.create_group("derived")
        for key, values in derived.items():
            derived_group.create_dataset(key, data=values, compression="gzip")
        output.attrs["source_h5"] = str(args.source_h5)
        output.attrs["num_samples"] = count
        output.attrs["complete"] = True
        output.attrs["definitions_json"] = json.dumps(definitions)
        output.attrs["task_metadata_json"] = json.dumps(task_metadata)
        output.attrs["pair_metadata_json"] = json.dumps(pairs)
        output.attrs["pair_schema_version"] = 2

    summary = {
        "source_h5": str(args.source_h5),
        "output_h5": str(args.output_h5),
        "num_samples": count,
        "source_indices_contiguous": bool(np.array_equal(selected_indices, np.arange(count))),
        "split_counts": dict(Counter(splits.tolist())),
        "task_counts": {str(key): int((task_ids == key).sum()) for key in np.unique(task_ids)},
        "pair_counts": {str(key): int((pair_ids == key).sum()) for key in np.unique(pair_ids)},
        "pairs": pairs,
        "binary_positive_counts": {
            key: int(values.sum())
            for key, values in (arrays | derived).items()
            if values.dtype == np.bool_
        },
        "phase_counts": {
            PHASE_NAMES[index]: int((phase == index).sum()) for index in range(len(PHASE_NAMES))
        },
        "continuous_summaries": {
            key: numeric_summary(values)
            for key, values in derived.items()
            if np.issubdtype(values.dtype, np.floating)
        },
        "definitions": definitions,
        "tasks": task_metadata,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote: {args.output_h5}", flush=True)


if __name__ == "__main__":
    main()
