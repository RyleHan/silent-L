#!/usr/bin/env python3
"""Episode-bootstrap matched OpenVLA coordinate-probe differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


READOUTS = ("prompt_end", "action_boundary", "visual_mean")
MODES = ("absolute", "target_alternative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def patch_labels(masks: np.ndarray, grid_size: int = 16, min_mask_pixels: int = 5) -> np.ndarray:
    patch_size = masks.shape[1] // grid_size
    pooled = masks.reshape(
        len(masks), grid_size, patch_size, grid_size, patch_size, 2
    ).sum(axis=(2, 4))
    labels = np.zeros(pooled.shape[:-1], dtype=np.uint8)
    foreground = pooled.max(axis=-1) >= min_mask_pixels
    labels[foreground] = pooled.argmax(axis=-1)[foreground] + 1
    return labels


def paired_labels(image_labels: np.ndarray) -> np.ndarray:
    absolute = np.repeat(image_labels[:, None], 2, axis=1)
    swapped = image_labels.copy()
    swapped[image_labels == 1] = 2
    swapped[image_labels == 2] = 1
    relative = np.stack((image_labels, swapped), axis=1)
    return np.stack((absolute, relative), axis=0).reshape(2, 2 * len(image_labels), 16, 16)


def confusion(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    encoded = labels.reshape(-1) * 3 + predictions.reshape(-1)
    return np.bincount(encoded, minlength=9).reshape(3, 3)


def foreground_macro_f1(confusions: np.ndarray) -> np.ndarray:
    confusions = confusions.astype(np.float64)
    true_positive = np.diagonal(confusions, axis1=-2, axis2=-1)
    precision = true_positive / np.maximum(confusions.sum(axis=-2), 1)
    recall = true_positive / np.maximum(confusions.sum(axis=-1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(np.float64).eps)
    return f1[..., 1:].mean(axis=-1)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.source_h5, "r") as source:
        image_labels = patch_labels(source["masks"][:])
        eligible_images = (
            (image_labels == 1).any(axis=(1, 2))
            & (image_labels == 2).any(axis=(1, 2))
        )
        source_image_indices = np.flatnonzero(eligible_images).astype(np.int64)
        labels = paired_labels(image_labels[source_image_indices])
        task_ids = source["task_ids"][:]
        pair_ids = source["pair_ids"][:] if "pair_ids" in source else np.zeros(len(task_ids))
        if "episode_ids" not in source:
            raise ValueError("Episode bootstrap requires an episode_ids dataset.")
        episode_ids = source["episode_ids"][:]

    result: dict[str, object] = {
        "selection_rule": "Layer maximizes validation target/alternative minus absolute foreground macro-F1.",
        "resampling_unit": "complete (task_id, episode_id) test episode",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "readouts": {},
    }
    for readout in READOUTS:
        metrics = json.loads((args.probe_root / readout / "metrics.json").read_text(encoding="utf-8"))
        layer_keys = sorted(metrics["layers"], key=int)
        validation_deltas = [
            metrics["layers"][layer]["metrics"]["valid"][MODES[1]]["foreground_macro_f1"]
            - metrics["layers"][layer]["metrics"]["valid"][MODES[0]]["foreground_macro_f1"]
            for layer in layer_keys
        ]
        selected_layer = int(layer_keys[int(np.argmax(validation_deltas))])
        predictions_file = np.load(
            args.probe_root / readout / "predictions" / f"layer_{selected_layer:02d}.npz"
        )
        prediction_source_indices = predictions_file["source_image_indices"]
        if not np.array_equal(prediction_source_indices, source_image_indices):
            raise ValueError(
                "Probe prediction eligibility mapping differs from bootstrap mapping."
            )
        test_indices = predictions_file["test_indices"]
        predictions = predictions_file["test_predictions"]
        test_labels = labels[:, test_indices]
        image_indices = source_image_indices[test_indices // 2]
        group_keys = np.stack(
            (pair_ids[image_indices], task_ids[image_indices], episode_ids[image_indices]), axis=1
        )
        unique_groups = np.unique(group_keys, axis=0)

        group_confusions = np.zeros((len(unique_groups), 2, 3, 3), dtype=np.int64)
        for group_index, group in enumerate(unique_groups):
            group_mask = np.all(group_keys == group, axis=1)
            for mode_index in range(2):
                group_confusions[group_index, mode_index] = confusion(
                    predictions[mode_index, group_mask],
                    test_labels[mode_index, group_mask],
                )

        unique_pairs = np.unique(unique_groups[:, 0]).astype(int)
        pair_group_indices = {
            pair_id: np.flatnonzero(unique_groups[:, 0] == pair_id) for pair_id in unique_pairs
        }
        per_pair_f1 = {
            pair_id: foreground_macro_f1(group_confusions[indices].sum(axis=0))
            for pair_id, indices in pair_group_indices.items()
        }
        point_f1 = np.mean(np.stack(list(per_pair_f1.values())), axis=0)
        deltas = np.empty(args.bootstrap_samples, dtype=np.float64)
        for bootstrap_index in range(args.bootstrap_samples):
            pair_f1 = []
            for pair_id, indices in pair_group_indices.items():
                draws = rng.choice(indices, size=len(indices), replace=True)
                pair_f1.append(foreground_macro_f1(group_confusions[draws].sum(axis=0)))
            macro_f1 = np.mean(np.stack(pair_f1), axis=0)
            deltas[bootstrap_index] = macro_f1[1] - macro_f1[0]
        point_delta = float(point_f1[1] - point_f1[0])
        result["readouts"][readout] = {
            "selected_layer": selected_layer,
            "num_test_episodes": len(unique_groups),
            "num_test_pairs": len(unique_pairs),
            "aggregation": "equal-weight macro average over object pairs",
            "test_foreground_macro_f1": {
                MODES[index]: float(point_f1[index]) for index in range(2)
            },
            "per_pair": {
                str(pair_id): {
                    "num_test_episodes": len(pair_group_indices[pair_id]),
                    "absolute": float(per_pair_f1[pair_id][0]),
                    "target_alternative": float(per_pair_f1[pair_id][1]),
                    "difference": float(per_pair_f1[pair_id][1] - per_pair_f1[pair_id][0]),
                }
                for pair_id in unique_pairs
            },
            "test_delta": point_delta,
            "bootstrap_delta_95_percent_ci": np.percentile(deltas, [2.5, 97.5]).tolist(),
            "bootstrap_probability_delta_positive": float(np.mean(deltas > 0)),
            "one_sided_p_delta_le_zero": float((np.count_nonzero(deltas <= 0) + 1) / (len(deltas) + 1)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote: {args.output}", flush=True)


if __name__ == "__main__":
    main()
