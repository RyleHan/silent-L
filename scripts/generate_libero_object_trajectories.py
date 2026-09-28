#!/usr/bin/env python3
"""Generate dynamic paired-prompt LIBERO-Object trajectory states."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from vla_coordinates.openvla_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REVISION,
    build_openvla_prompt,
    center_crop_openvla_image,
    load_openvla,
    predict_openvla_action,
    prepare_openvla_inputs,
)
from vla_coordinates.libero_object_pairs import legacy_pairs, load_pair_config, task_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--episodes-per-task", type=int, default=25)
    parser.add_argument("--max-action-steps", type=int, default=180)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-incomplete-splits",
        action="store_true",
        help="Permit fewer than six episodes per task for schema smoke tests only.",
    )
    parser.add_argument(
        "--pair-config",
        type=Path,
        help="JSON protocol defining one or more paired LIBERO-Object tasks.",
    )
    parser.add_argument(
        "--task-ids",
        help="Optional comma-separated task subset for parallel sharded generation.",
    )
    return parser.parse_args()


def transform_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask[::-1, ::-1].astype(np.float32)
    tensor = tf.convert_to_tensor(mask[..., None])
    tensor = tf.image.resize(tensor, (224, 224), method="nearest")
    crop_fraction = np.sqrt(0.9)
    offset = (1.0 - crop_fraction) / 2.0
    boxes = tf.constant(
        [[offset, offset, offset + crop_fraction, offset + crop_fraction]],
        dtype=tf.float32,
    )
    tensor = tf.image.crop_and_resize(
        tensor[None],
        boxes,
        box_indices=tf.constant([0]),
        crop_size=(224, 224),
        method="nearest",
    )[0, ..., 0]
    return (tensor.numpy() > 0.5).astype(np.uint8)


def segmentation_key(observation: dict[str, np.ndarray]) -> str:
    keys = [key for key in observation if key.startswith("agentview_segmentation")]
    if len(keys) != 1:
        raise KeyError(f"Expected one agentview segmentation observation, found: {keys}")
    return keys[0]


def split_for_episode(episode_index: int, num_episodes: int) -> str:
    train_end = max(1, round(0.7 * num_episodes))
    valid_end = max(train_end + 1, round(0.85 * num_episodes))
    if episode_index < train_end:
        return "train"
    if episode_index < valid_end:
        return "valid"
    return "test"


def create_output(path: Path, pairs: list[dict[str, object]]) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(
        "images",
        shape=(0, 224, 224, 3),
        maxshape=(None, 224, 224, 3),
        dtype=np.uint8,
        chunks=(1, 224, 224, 3),
        compression="gzip",
    )
    handle.create_dataset(
        "masks",
        shape=(0, 224, 224, 2),
        maxshape=(None, 224, 224, 2),
        dtype=np.uint8,
        chunks=(1, 224, 224, 2),
        compression="gzip",
    )
    handle.create_dataset(
        "simulator_states",
        shape=(0, 110),
        maxshape=(None, 110),
        dtype=np.float32,
        chunks=(32, 110),
        compression="gzip",
    )
    for key, dtype in (
        ("task_ids", np.int8),
        ("pair_ids", np.int8),
        ("target_sides", np.int8),
        ("episode_ids", np.int16),
        ("init_state_ids", np.int16),
        ("action_steps", np.int16),
    ):
        handle.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(256,))
    handle.create_dataset(
        "layout_splits",
        shape=(0,),
        maxshape=(None,),
        dtype=string_dtype,
        chunks=(256,),
    )
    for key in ("raw_actions", "executed_actions"):
        handle.create_dataset(
            key,
            shape=(0, 7),
            maxshape=(None, 7),
            dtype=np.float32,
            chunks=(256, 7),
        )
    handle.attrs["pair_metadata_json"] = json.dumps(pairs)
    handle.attrs["pair_schema_version"] = 2
    if len(pairs) == 1:
        handle.attrs["object_a_instance"] = pairs[0]["object_instances"][0]
        handle.attrs["object_b_instance"] = pairs[0]["object_instances"][1]
        handle.attrs["prompt_a"] = pairs[0]["prompts_openvla"][0]
        handle.attrs["prompt_b"] = pairs[0]["prompts_openvla"][1]
    handle.attrs["split_unit"] = "episode"
    return handle


def append_episode(handle: h5py.File, episode: dict[str, list[object]]) -> None:
    count = len(episode["images"])
    if count == 0:
        return
    start = handle["images"].shape[0]
    stop = start + count
    for key, values in episode.items():
        dataset = handle[key]
        dataset.resize((stop, *dataset.shape[1:]))
        dataset[start:stop] = np.asarray(values)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.episodes_per_task < 6 and not args.allow_incomplete_splits:
        raise ValueError("At least six episodes per task are required for nonempty episode-held-out splits.")
    config = load_pair_config(args.pair_config) if args.pair_config else {"pairs": legacy_pairs()}
    pairs = config["pairs"]
    tasks = task_lookup(pairs)
    selected_task_ids = sorted(tasks)
    if args.task_ids:
        selected_task_ids = sorted({int(value) for value in args.task_ids.split(",")})
        unknown = set(selected_task_ids) - set(tasks)
        if unknown:
            raise ValueError(f"Unknown configured task IDs: {sorted(unknown)}.")
    h5_path = args.output_dir / "paired_trajectories.h5"
    if h5_path.exists():
        raise FileExistsError(f"Output already exists: {h5_path}")

    device = torch.device("cuda:0")
    model, processor = load_openvla(
        args.checkpoint,
        args.revision,
        device,
        dtype=torch.float16,
        attention_implementation="eager",
        local_files_only=True,
    )
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    records: list[dict[str, object]] = []
    started_at = time.perf_counter()

    with create_output(h5_path, pairs) as output:
        output.attrs["checkpoint"] = args.checkpoint
        output.attrs["revision"] = args.revision
        output.attrs["seed"] = args.seed
        output.attrs["sample_every"] = args.sample_every

        for task_id in selected_task_ids:
            pair, target_side = tasks[task_id]
            pair_id = int(pair["pair_id"])
            object_a_instance, object_b_instance = pair["object_instances"]
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if args.episodes_per_task > len(initial_states):
                raise ValueError(
                    f"Task {task_id} has only {len(initial_states)} initial states, "
                    f"requested {args.episodes_per_task}."
                )
            rng = np.random.default_rng(args.seed + task_id)
            init_state_ids = rng.permutation(len(initial_states))[: args.episodes_per_task]
            bddl_path = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            env = SegmentationRenderEnv(
                bddl_file_name=bddl_path,
                camera_names=["agentview"],
                camera_heights=256,
                camera_widths=256,
                camera_segmentations="instance",
            )
            env.seed(args.seed + task_id)
            prompt = build_openvla_prompt(task.language)
            try:
                progress = tqdm(enumerate(init_state_ids), total=len(init_state_ids), desc=f"task {task_id}")
                for episode_index, init_state_id in progress:
                    split = split_for_episode(episode_index, args.episodes_per_task)
                    observation = env.reset()
                    observation = env.set_init_state(initial_states[int(init_state_id)])
                    for _ in range(args.wait_steps):
                        observation, _, _, _ = env.step(get_libero_dummy_action("openvla"))

                    missing = {object_a_instance, object_b_instance} - set(env.instance_to_id)
                    if missing:
                        raise KeyError(f"Task {task_id} lacks segmentation instances {sorted(missing)}")

                    episode: dict[str, list[object]] = {
                        key: []
                        for key in (
                            "images",
                            "masks",
                            "simulator_states",
                            "task_ids",
                            "pair_ids",
                            "target_sides",
                            "episode_ids",
                            "init_state_ids",
                            "action_steps",
                            "layout_splits",
                            "raw_actions",
                            "executed_actions",
                        )
                    }
                    success = False
                    action_steps = 0
                    skipped_empty_masks = 0
                    inference_times: list[float] = []

                    for action_step in range(args.max_action_steps):
                        raw_image = get_libero_image(observation, 224)
                        processed_image = center_crop_openvla_image(raw_image)
                        inputs = prepare_openvla_inputs(
                            processor,
                            prompt,
                            processed_image,
                            device,
                            dtype=torch.float16,
                            center_crop=False,
                        )
                        inference_started = time.perf_counter()
                        raw_action = predict_openvla_action(model, inputs)
                        inference_times.append(time.perf_counter() - inference_started)
                        executed_action = normalize_gripper_action(raw_action.copy(), binarize=True)
                        executed_action = invert_gripper_action(executed_action)

                        if action_step % args.sample_every == 0:
                            segmentation = np.asarray(observation[segmentation_key(observation)]).squeeze()
                            mask_a = transform_mask(segmentation == env.instance_to_id[object_a_instance])
                            mask_b = transform_mask(segmentation == env.instance_to_id[object_b_instance])
                            if mask_a.sum() == 0 or mask_b.sum() == 0:
                                skipped_empty_masks += 1
                            elif np.any(mask_a & mask_b):
                                raise RuntimeError("Object masks overlap after transformation.")
                            else:
                                episode["images"].append(np.asarray(processed_image, dtype=np.uint8))
                                episode["masks"].append(np.stack((mask_a, mask_b), axis=-1))
                                episode["simulator_states"].append(env.get_sim_state().astype(np.float32))
                                episode["task_ids"].append(task_id)
                                episode["pair_ids"].append(pair_id)
                                episode["target_sides"].append(target_side)
                                episode["episode_ids"].append(episode_index)
                                episode["init_state_ids"].append(int(init_state_id))
                                episode["action_steps"].append(action_step)
                                episode["layout_splits"].append(split)
                                episode["raw_actions"].append(raw_action.astype(np.float32))
                                episode["executed_actions"].append(executed_action.astype(np.float32))

                        observation, _, success, _ = env.step(executed_action.tolist())
                        action_steps = action_step + 1
                        if success:
                            break

                    append_episode(output, episode)
                    output.flush()
                    record = {
                        "task_id": task_id,
                        "pair_id": pair_id,
                        "target_side": target_side,
                        "task": task.language,
                        "episode_id": episode_index,
                        "init_state_id": int(init_state_id),
                        "split": split,
                        "success": bool(success),
                        "action_steps": action_steps,
                        "num_samples": len(episode["images"]),
                        "skipped_empty_masks": skipped_empty_masks,
                        "mean_inference_seconds": float(np.mean(inference_times)),
                    }
                    records.append(record)
                    progress.set_postfix(success=int(success), samples=len(episode["images"]))
            finally:
                env.close()

        output.attrs["complete"] = True
        output.attrs["num_samples"] = output["images"].shape[0]

    split_counts = {
        split: sum(record["num_samples"] for record in records if record["split"] == split)
        for split in ("train", "valid", "test")
    }
    metadata = {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "seed": args.seed,
        "task_ids": selected_task_ids,
        "pairs": pairs,
        "pair_config": str(args.pair_config) if args.pair_config else None,
        "episodes_per_task": args.episodes_per_task,
        "max_action_steps": args.max_action_steps,
        "sample_every": args.sample_every,
        "split_unit": "episode",
        "split_sample_counts": split_counts,
        "num_samples": sum(split_counts.values()),
        "successes": sum(record["success"] for record in records),
        "num_episodes": len(records),
        "elapsed_seconds": time.perf_counter() - started_at,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "h5_path": str(h5_path),
        "records": records,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, indent=2), flush=True)
    print(f"Wrote: {h5_path}", flush=True)


if __name__ == "__main__":
    main()
