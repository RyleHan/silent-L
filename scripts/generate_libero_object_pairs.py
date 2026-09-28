#!/usr/bin/env python3
"""Generate same-image, swapped-prompt LIBERO-Object probe data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image
from vla_coordinates.openvla_runtime import build_openvla_prompt, center_crop_openvla_image

OBJECT_A_INSTANCE = "alphabet_soup_1"
OBJECT_B_INSTANCE = "cream_cheese_1"
OBJECT_A_NAME = "alphabet soup"
OBJECT_B_NAME = "cream cheese"
LAYOUT_TASK_IDS = [0, 1, 2, 3, 4]
LAYOUT_SPLITS = {0: "train", 1: "valid", 2: "train", 3: "test", 4: "train"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-layout", type=int, default=200)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def transform_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask[::-1, ::-1].astype(np.float32)
    tensor = tf.convert_to_tensor(mask[..., None])
    tensor = tf.image.resize(tensor, (224, 224), method="nearest")
    crop_fraction = np.sqrt(0.9)
    offset = (1.0 - crop_fraction) / 2.0
    boxes = tf.constant([[offset, offset, offset + crop_fraction, offset + crop_fraction]], dtype=tf.float32)
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


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark_dict()["libero_object"]()

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    simulator_states: list[np.ndarray] = []
    task_ids: list[int] = []
    sample_ids: list[int] = []
    split_names: list[str] = []
    records: list[dict[str, object]] = []
    started_at = time.perf_counter()

    for task_id in LAYOUT_TASK_IDS:
        task = suite.get_task(task_id)
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = SegmentationRenderEnv(
            bddl_file_name=bddl_path,
            camera_names=["agentview"],
            camera_heights=256,
            camera_widths=256,
            camera_segmentations="instance",
        )
        env.seed(args.seed + task_id)
        try:
            progress = tqdm(range(args.samples_per_layout), desc=f"layout task {task_id}")
            for sample_id in progress:
                observation = env.reset()
                for _ in range(args.wait_steps):
                    observation, _, _, _ = env.step(get_libero_dummy_action("openvla"))

                missing = {
                    OBJECT_A_INSTANCE,
                    OBJECT_B_INSTANCE,
                } - set(env.instance_to_id)
                if missing:
                    raise KeyError(
                        f"Layout task {task_id} lacks required segmentation instances {sorted(missing)}; "
                        f"available={sorted(env.instance_to_id)}"
                    )

                image = get_libero_image(observation, 224)
                image = np.asarray(center_crop_openvla_image(image), dtype=np.uint8)
                segmentation = np.asarray(observation[segmentation_key(observation)]).squeeze()
                mask_a = transform_mask(segmentation == env.instance_to_id[OBJECT_A_INSTANCE])
                mask_b = transform_mask(segmentation == env.instance_to_id[OBJECT_B_INSTANCE])
                if mask_a.sum() == 0 or mask_b.sum() == 0:
                    raise RuntimeError(
                        f"Empty object mask at task={task_id}, sample={sample_id}: "
                        f"A={int(mask_a.sum())}, B={int(mask_b.sum())}"
                    )
                if np.any(mask_a & mask_b):
                    raise RuntimeError("Object masks overlap after transformation.")

                images.append(image)
                masks.append(np.stack((mask_a, mask_b), axis=-1))
                simulator_states.append(env.get_sim_state().astype(np.float32))
                task_ids.append(task_id)
                sample_ids.append(sample_id)
                split_names.append(LAYOUT_SPLITS[task_id])
                records.append(
                    {
                        "index": len(records),
                        "task_id": task_id,
                        "sample_id": sample_id,
                        "split": LAYOUT_SPLITS[task_id],
                        "mask_pixels_a": int(mask_a.sum()),
                        "mask_pixels_b": int(mask_b.sum()),
                    }
                )
        finally:
            env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.output_dir / "paired_states.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("images", data=np.stack(images), compression="gzip", chunks=(1, 224, 224, 3))
        handle.create_dataset("masks", data=np.stack(masks), compression="gzip", chunks=(1, 224, 224, 2))
        handle.create_dataset("simulator_states", data=np.stack(simulator_states), compression="gzip")
        handle.create_dataset("task_ids", data=np.asarray(task_ids, dtype=np.int8))
        handle.create_dataset("sample_ids", data=np.asarray(sample_ids, dtype=np.int16))
        handle.create_dataset("layout_splits", data=np.asarray(split_names, dtype=object), dtype=string_dtype)
        handle.attrs["object_a_instance"] = OBJECT_A_INSTANCE
        handle.attrs["object_b_instance"] = OBJECT_B_INSTANCE
        handle.attrs["prompt_a"] = build_openvla_prompt(f"pick up the {OBJECT_A_NAME} and place it in the basket")
        handle.attrs["prompt_b"] = build_openvla_prompt(f"pick up the {OBJECT_B_NAME} and place it in the basket")
        handle.attrs["seed"] = args.seed

    metadata = {
        "seed": args.seed,
        "samples_per_layout": args.samples_per_layout,
        "num_images": len(images),
        "num_prompt_samples": 2 * len(images),
        "layout_task_ids": LAYOUT_TASK_IDS,
        "layout_splits": LAYOUT_SPLITS,
        "objects": {
            "a": {"instance": OBJECT_A_INSTANCE, "name": OBJECT_A_NAME},
            "b": {"instance": OBJECT_B_INSTANCE, "name": OBJECT_B_NAME},
        },
        "label_systems": {
            "absolute": ["other", "object_a", "object_b"],
            "relative_prompt_a": ["other", "target_a", "alternative_b"],
            "relative_prompt_b": ["other", "target_b", "alternative_a"],
        },
        "image_shape": list(images[0].shape),
        "mask_shape": list(masks[0].shape),
        "simulator_state_shape": list(simulator_states[0].shape),
        "elapsed_seconds": time.perf_counter() - started_at,
        "h5_path": str(h5_path),
        "records": records,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, indent=2))
    print(f"Wrote: {h5_path}")


if __name__ == "__main__":
    main()

