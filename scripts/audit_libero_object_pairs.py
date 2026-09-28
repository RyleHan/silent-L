#!/usr/bin/env python3
"""Audit diversity and duplication in generated LIBERO coordinate data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def patch_labels(masks: np.ndarray, grid_size: int = 16, min_mask_pixels: int = 5) -> np.ndarray:
    patch_size = masks.shape[1] // grid_size
    pooled = masks.reshape(
        len(masks), grid_size, patch_size, grid_size, patch_size, 2
    ).sum(axis=(2, 4))
    labels = np.zeros(pooled.shape[:-1], dtype=np.uint8)
    foreground = pooled.max(axis=-1) >= min_mask_pixels
    labels[foreground] = pooled.argmax(axis=-1)[foreground] + 1
    return labels


def main() -> None:
    args = parse_args()
    result: dict[str, object] = {"input_h5": str(args.input_h5), "layouts": {}}
    with h5py.File(args.input_h5, "r") as handle:
        task_ids = handle["task_ids"][:]
        all_labels = patch_labels(handle["masks"][:])
        for task_id in sorted(np.unique(task_ids).tolist()):
            indices = np.flatnonzero(task_ids == task_id)
            layout: dict[str, object] = {"num_samples": len(indices)}
            for key in ("images", "masks", "simulator_states"):
                values = handle[key][indices]
                hashes = [array_hash(value) for value in values]
                reference = values[0].astype(np.float32)
                mean_absolute_difference = np.abs(values.astype(np.float32) - reference).reshape(len(values), -1).mean(axis=1)
                layout[key] = {
                    "num_unique": len(set(hashes)),
                    "reference_mean_absolute_difference_min_median_max": [
                        float(mean_absolute_difference.min()),
                        float(np.median(mean_absolute_difference)),
                        float(mean_absolute_difference.max()),
                    ],
                }
            labels = patch_labels(handle["masks"][indices])
            layout["patch_labels"] = {
                "num_unique": len({array_hash(value) for value in labels}),
                "object_a_union_cells": int(np.any(labels == 1, axis=0).sum()),
                "object_b_union_cells": int(np.any(labels == 2, axis=0).sum()),
                "object_a_always_positive_cells": int(np.all(labels == 1, axis=0).sum()),
                "object_b_always_positive_cells": int(np.all(labels == 2, axis=0).sum()),
            }
            result["layouts"][str(task_id)] = layout
        if "layout_splits" in handle:
            split_names = np.asarray(
                [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in handle["layout_splits"][:]
                ]
            )
            split_support = {}
            unions: dict[str, list[np.ndarray]] = {}
            for split in ("train", "valid", "test"):
                labels = all_labels[split_names == split]
                object_unions = [np.any(labels == object_class, axis=0) for object_class in (1, 2)]
                unions[split] = object_unions
                split_support[split] = {
                    "num_samples": len(labels),
                    "num_unique_patch_labels": len({array_hash(value) for value in labels}),
                    "object_a_union_cells": int(object_unions[0].sum()),
                    "object_b_union_cells": int(object_unions[1].sum()),
                }
            for split in ("valid", "test"):
                split_support[split]["object_a_cells_unseen_in_train"] = int(
                    np.logical_and(unions[split][0], ~unions["train"][0]).sum()
                )
                split_support[split]["object_b_cells_unseen_in_train"] = int(
                    np.logical_and(unions[split][1], ~unions["train"][1]).sum()
                )
            result["split_support"] = split_support
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
