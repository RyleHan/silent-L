#!/usr/bin/env python3
"""Replay cached LIBERO states into the official pi0.5-LIBERO input format."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_image
from vla_coordinates.libero_object_pairs import copy_pair_metadata, read_pair_ids, read_pairs
from vla_coordinates.openvla_runtime import center_crop_openvla_image


OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--audit-samples-per-task", type=int, default=8)
    return parser.parse_args()


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator, dtype=np.float32)


def resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """Exact copy of openpi_client.image_tools.resize_with_pad for one image."""
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


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def create_output(path: Path, count: int) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, "w")
    for key in ("base_images", "wrist_images"):
        output.create_dataset(
            key,
            shape=(count, 224, 224, 3),
            dtype=np.uint8,
            chunks=(1, 224, 224, 3),
            compression="gzip",
        )
    output.create_dataset(
        "robot_states",
        shape=(count, 8),
        dtype=np.float32,
        chunks=(min(256, count), 8),
        compression="gzip",
    )
    return output


def main() -> None:
    args = parse_args()
    if args.output_h5.exists():
        raise FileExistsError(f"Output already exists: {args.output_h5}")
    started_at = time.perf_counter()

    with h5py.File(args.source_h5, "r") as source:
        available = source["task_ids"].shape[0]
        count = min(available, args.max_samples or available)
        selected_indices = np.arange(count, dtype=np.int64)
        pairs = read_pairs(source)
        pair_ids = read_pair_ids(source, count)
        task_ids = np.asarray(source["task_ids"][:count], dtype=np.int8)
        episode_ids = np.asarray(source["episode_ids"][:count], dtype=np.int16)
        init_state_ids = np.asarray(source["init_state_ids"][:count], dtype=np.int16)
        action_steps = np.asarray(source["action_steps"][:count], dtype=np.int16)
        splits = decode_strings(source["layout_splits"][:count])

        args.output_h5.parent.mkdir(parents=True, exist_ok=True)
        audit_records: list[dict[str, float | int]] = []
        suite = benchmark.get_benchmark_dict()["libero_object"]()

        with create_output(args.output_h5, count) as output:
            output.create_dataset("source_indices", data=selected_indices)
            copy_pair_metadata(source, output, count)
            output.create_dataset("task_ids", data=task_ids)
            output.create_dataset("episode_ids", data=episode_ids)
            output.create_dataset("init_state_ids", data=init_state_ids)
            output.create_dataset("action_steps", data=action_steps)
            output.create_dataset(
                "layout_splits",
                data=splits.astype(object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            output.attrs["source_h5"] = str(args.source_h5)
            output.attrs["openpi_revision"] = OPENPI_REVISION
            output.attrs["image_preprocessing"] = (
                "rotate source camera 180 degrees, PIL bilinear resize_with_pad to 224x224"
            )
            output.attrs["state_definition"] = (
                "eef_xyz + eef_quaternion_axis_angle + two gripper qpos values"
            )

            for task_id in sorted(np.unique(task_ids).astype(int).tolist()):
                task = suite.get_task(task_id)
                bddl_path = os.path.join(
                    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
                )
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl_path,
                    camera_names=["agentview", "robot0_eye_in_hand"],
                    camera_heights=256,
                    camera_widths=256,
                )
                env.seed(42 + task_id)
                try:
                    env.reset()
                    task_indices = np.flatnonzero(task_ids == task_id)
                    audit_indices = set(task_indices[: args.audit_samples_per_task].tolist())
                    for index in tqdm(task_indices, desc=f"pi05 inputs task {task_id}", unit="state"):
                        observation = env.set_init_state(np.asarray(source["simulator_states"][index]))
                        base_image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                        wrist_image = np.ascontiguousarray(
                            observation["robot0_eye_in_hand_image"][::-1, ::-1]
                        )
                        output["base_images"][index] = resize_with_pad(base_image)
                        output["wrist_images"][index] = resize_with_pad(wrist_image)
                        output["robot_states"][index] = np.concatenate(
                            (
                                observation["robot0_eef_pos"],
                                quat_to_axisangle(observation["robot0_eef_quat"]),
                                observation["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32)

                        if int(index) in audit_indices:
                            openvla_image = center_crop_openvla_image(
                                get_libero_image(observation, 224)
                            )
                            difference = np.abs(
                                np.asarray(openvla_image, dtype=np.int16)
                                - np.asarray(source["images"][index], dtype=np.int16)
                            )
                            audit_records.append(
                                {
                                    "source_index": int(index),
                                    "task_id": task_id,
                                    "mean_abs_pixel_error": float(difference.mean()),
                                    "max_abs_pixel_error": int(difference.max()),
                                }
                            )
                finally:
                    env.close()

            output.attrs["complete"] = True
            output.attrs["num_samples"] = count

    states_finite = True
    with h5py.File(args.output_h5, "r") as output:
        states_finite = bool(np.isfinite(output["robot_states"][:]).all())

    summary = {
        "source_h5": str(args.source_h5),
        "output_h5": str(args.output_h5),
        "num_samples": count,
        "split_counts": {split: int((splits == split).sum()) for split in np.unique(splits)},
        "task_counts": {str(task): int((task_ids == task).sum()) for task in np.unique(task_ids)},
        "pair_counts": {str(pair): int((pair_ids == pair).sum()) for pair in np.unique(pair_ids)},
        "robot_states_finite": states_finite,
        "pairs": pairs,
        "pixel_replay_audit": {
            "num_samples": len(audit_records),
            "mean_abs_pixel_error": float(
                np.mean([record["mean_abs_pixel_error"] for record in audit_records])
            ),
            "max_abs_pixel_error": int(
                max(record["max_abs_pixel_error"] for record in audit_records)
            ),
            "records": audit_records,
        },
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote: {args.output_h5}", flush=True)


if __name__ == "__main__":
    main()
