#!/usr/bin/env python3
"""Extract a compact, held-out latent-world trajectory for the paper Figure 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch


DEFAULT_WORK_ROOT = Path(os.environ.get("VLA_WORK_ROOT", Path(os.environ.get("WORK", ".")) / "vla_coordinates"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def choose_episode(source: h5py.File, labels: h5py.File) -> tuple[np.ndarray, list[dict[str, object]]]:
    task_ids = np.asarray(labels["task_ids"], dtype=np.int16)
    episode_ids = np.asarray(labels["episode_ids"], dtype=np.int16)
    action_steps = np.asarray(labels["action_steps"], dtype=np.int16)
    splits = decode_strings(labels["layout_splits"][:])
    phases = np.minimum(np.asarray(labels["derived/phase_heuristic"], dtype=np.int16), 2)
    grasped = np.asarray(labels["derived/target_grasped"], dtype=np.float32)
    displacement = np.asarray(labels["derived/target_displacement_xyz"], dtype=np.float32)

    candidates = []
    pairs = sorted(set(zip(task_ids[splits == "test"].tolist(), episode_ids[splits == "test"].tolist())))
    for task_id, episode_id in pairs:
        indices = np.flatnonzero(
            (task_ids == task_id) & (episode_ids == episode_id) & (splits == "test")
        )
        indices = indices[np.argsort(action_steps[indices])]
        masks = np.asarray(source["masks"][indices], dtype=np.uint8)
        areas = masks.reshape(len(indices), -1, 2).sum(axis=1)
        visible_fraction = float(np.mean(np.all(areas > 12, axis=1)))
        phase_coverage = int(len(np.unique(phases[indices])))
        movement = float(np.linalg.norm(displacement[indices, :2], axis=1).max())
        record = {
            "task_id": int(task_id),
            "episode_id": int(episode_id),
            "num_states": int(len(indices)),
            "phase_coverage": phase_coverage,
            "contains_grasp": bool(np.any(grasped[indices] > 0.5)),
            "both_objects_visible_fraction": visible_fraction,
            "max_target_xy_displacement": movement,
            "first_index": int(indices[0]),
        }
        record["selection_key"] = (
            phase_coverage,
            int(record["contains_grasp"]),
            round(visible_fraction, 6),
            round(movement, 6),
            len(indices),
            -int(task_id),
            -int(episode_id),
        )
        record["indices"] = indices
        candidates.append(record)

    candidates.sort(key=lambda item: item["selection_key"], reverse=True)
    chosen = np.asarray(candidates[0].pop("indices"), dtype=np.int64)
    for record in candidates[1:]:
        record.pop("indices")
    candidates[0].pop("selection_key")
    for record in candidates[1:]:
        record.pop("selection_key")
    return chosen, candidates


def read_feature_trajectory(
    residual_path: Path,
    dataset_name: str,
    indices: np.ndarray,
    *,
    position: int | None = None,
) -> np.ndarray:
    with h5py.File(residual_path, "r") as residuals:
        dataset = residuals[dataset_name]
        prompts = []
        for prompt_index in (0, 1):
            states = []
            for sample_index in indices:
                value = dataset[int(sample_index), prompt_index]
                if position is not None:
                    value = value[:, position, :]
                states.append(np.asarray(value, dtype=np.float32))
            prompts.append(np.stack(states, axis=1))
    return np.stack(prompts, axis=0)


def decode_probe_trajectory(features: np.ndarray, checkpoint_dir: Path) -> np.ndarray:
    num_prompts, num_layers, num_states, _ = features.shape
    predictions = np.empty((num_prompts, num_layers, num_states, 3), dtype=np.float32)
    for layer in range(num_layers):
        checkpoint = torch.load(
            checkpoint_dir / f"layer_{layer:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        start, stop = checkpoint["factor_slices"]["target_xyz"]
        weight = checkpoint["states"]["continuous"]["weight"].detach().cpu().numpy()
        bias = checkpoint["states"]["continuous"]["bias"].detach().cpu().numpy()
        feature_mean = checkpoint["feature_mean"].detach().cpu().numpy()
        feature_std = checkpoint["feature_std"].detach().cpu().numpy()
        continuous_mean = np.asarray(checkpoint["continuous_mean"], dtype=np.float32)
        continuous_std = np.asarray(checkpoint["continuous_std"], dtype=np.float32)
        normalized = (features[:, layer] - feature_mean) / feature_std
        decoded = normalized @ weight.T + bias
        decoded = decoded * continuous_std + continuous_mean
        predictions[:, layer] = decoded[:, :, start:stop]
    return predictions


def prompt_preference(predictions: np.ndarray, object_a: np.ndarray, object_b: np.ndarray) -> np.ndarray:
    distance_a = np.linalg.norm(predictions - object_a[None, None, :, :], axis=-1)
    distance_b = np.linalg.norm(predictions - object_b[None, None, :, :], axis=-1)
    return ((distance_b - distance_a) / (distance_a + distance_b + 1e-8)).astype(np.float32)


def select_frame_positions(phases: np.ndarray, count: int = 10) -> np.ndarray:
    positions = set(np.rint(np.linspace(0, len(phases) - 1, count)).astype(int).tolist())
    for phase in (1, 2):
        matches = np.flatnonzero(phases == phase)
        if len(matches):
            positions.add(int(matches[0]))
    positions = np.asarray(sorted(positions), dtype=np.int64)
    if len(positions) > count:
        keep = np.rint(np.linspace(0, len(positions) - 1, count)).astype(int)
        positions = positions[keep]
    return positions


def main() -> None:
    args = parse_args()
    work_root = args.work_root
    dataset_root = work_root / "datasets/libero_object_trajectories_v1"
    run_root = work_root / "runs"
    source_path = dataset_root / "paired_trajectories.h5"
    labels_path = run_root / "openvla_world_state_atlas/world_state_labels.h5"

    with h5py.File(source_path, "r") as source, h5py.File(labels_path, "r") as labels:
        indices, candidates = choose_episode(source, labels)
        phases = np.minimum(
            np.asarray(labels["derived/phase_heuristic"][indices], dtype=np.int16), 2
        )
        frame_positions = select_frame_positions(phases)
        images = np.asarray(source["images"][indices[frame_positions]], dtype=np.uint8)
        masks = np.asarray(source["masks"][indices[frame_positions]], dtype=np.uint8)
        object_a_xyz = np.asarray(labels["raw/object_a_xyz"][indices], dtype=np.float32)
        object_b_xyz = np.asarray(labels["raw/object_b_xyz"][indices], dtype=np.float32)
        eef_xyz = np.asarray(labels["raw/eef_xyz"][indices], dtype=np.float32)
        basket_xyz = np.asarray(labels["raw/basket_xyz"][indices], dtype=np.float32)
        action_steps = np.asarray(labels["action_steps"][indices], dtype=np.int16)
        progress = np.asarray(labels["derived/normalized_action_progress"][indices], dtype=np.float32)
        task_id = int(labels["task_ids"][indices[0]])
        episode_id = int(labels["episode_ids"][indices[0]])
        metadata = {
            "task_id": task_id,
            "episode_id": episode_id,
            "split": "test",
            "selection_rule": (
                "Lexicographic maximum of phase coverage, grasp occurrence, both-object mask "
                "visibility, target motion, and trajectory length; independent of probe output."
            ),
            "candidate_episodes": candidates,
            "prompt_a": str(source.attrs["prompt_a"]),
            "prompt_b": str(source.attrs["prompt_b"]),
            "object_a": str(source.attrs["object_a_instance"]),
            "object_b": str(source.attrs["object_b_instance"]),
        }

    model_specs = {
        "openvla": {
            "residual": dataset_root / "openvla_residuals.h5",
            "dataset": "prompt_residuals",
            "position": 0,
            "checkpoint": run_root / "openvla_world_state_atlas/paired/prompt_end",
        },
        "pi05_prefix": {
            "residual": run_root / "pi05_world_state_atlas/pi05_prefix_residuals.h5",
            "dataset": "prompt_end_residuals",
            "position": None,
            "checkpoint": run_root / "pi05_world_state_atlas/paired/prompt_end",
        },
        "pi05_expert": {
            "residual": run_root / "pi05_action_expert_atlas/pi05_action_expert_residuals.h5",
            "dataset": "expert_action_mean_residuals",
            "position": None,
            "checkpoint": run_root / "pi05_action_expert_atlas/paired/expert_action_mean",
        },
    }

    payload: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "source_indices": indices,
        "action_steps": action_steps,
        "progress": progress,
        "phases": phases,
        "frame_positions": frame_positions,
        "images": images,
        "masks": masks,
        "object_a_xyz": object_a_xyz,
        "object_b_xyz": object_b_xyz,
        "eef_xyz": eef_xyz,
        "basket_xyz": basket_xyz,
    }
    for name, specification in model_specs.items():
        print(f"Reading {name} trajectory features...", flush=True)
        features = read_feature_trajectory(
            specification["residual"],
            specification["dataset"],
            indices,
            position=specification["position"],
        )
        print(f"Decoding {name} target position...", flush=True)
        predictions = decode_probe_trajectory(features, specification["checkpoint"])
        payload[f"{name}_target_xyz"] = predictions
        payload[f"{name}_preference"] = prompt_preference(
            predictions, object_a_xyz, object_b_xyz
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
