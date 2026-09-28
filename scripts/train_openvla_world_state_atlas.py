#!/usr/bin/env python3
"""Train layerwise linear probes for an OpenVLA world-state atlas."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


READOUT_POSITION = {"prompt_end": 0, "action_boundary": 1}
EXPERT_READOUT_DATASET = {
    "expert_action_mean": "expert_action_mean_residuals",
    "expert_action_first": "expert_action_first_residuals",
}
PHASE_NAMES = ("approach", "pregrasp", "manipulation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-h5", type=Path, required=True)
    parser.add_argument("--labels-h5", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--readout",
        choices=[
            "prompt_end",
            "action_boundary",
            "visual_mean",
            "visual_input",
            "expert_action_mean",
            "expert_action_first",
        ],
        required=True,
    )
    parser.add_argument(
        "--prompt-mode", choices=["actual", "counterfactual", "paired"], default="actual"
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-pair",
        type=int,
        help=(
            "Leave one pair unseen during fitting: train/valid use other pairs and test uses "
            "only this pair's locked test episodes."
        ),
    )
    parser.add_argument(
        "--exclude-task-ids",
        help=(
            "Comma-separated task scenes left unseen during fitting: train/valid use all other "
            "tasks and test uses only these tasks' locked test episodes."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def decode_attr(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def parse_layers(specification: str, num_layers: int) -> list[int]:
    if specification == "all":
        return list(range(num_layers))
    layers = sorted({int(value) for value in specification.split(",")})
    if not layers or layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError(f"Layers must be in [0, {num_layers - 1}], got {layers}.")
    return layers


def groupwise_displacement(
    values: np.ndarray, task_ids: np.ndarray, episode_ids: np.ndarray, action_steps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.empty_like(values)
    speed = np.zeros(len(values), dtype=np.float32)
    for task_id, episode_id in sorted(set(zip(task_ids.tolist(), episode_ids.tolist()))):
        indices = np.flatnonzero((task_ids == task_id) & (episode_ids == episode_id))
        order = indices[np.argsort(action_steps[indices])]
        displacement[order] = values[order] - values[order[0]]
        if len(order) > 1:
            distances = np.linalg.norm(np.diff(values[order], axis=0), axis=1)
            gaps = np.diff(action_steps[order]).clip(min=1)
            speed[order[1:]] = distances / gaps
    return displacement, speed


def add_factor(
    factors: OrderedDict[str, np.ndarray], name: str, values: np.ndarray
) -> None:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Factor {name} must be rank 1 or 2, got {values.shape}.")
    factors[name] = values


def load_targets(
    labels: h5py.File,
    source: h5py.File,
    *,
    prompt_mode: str,
) -> tuple[
    np.ndarray,
    OrderedDict[str, np.ndarray],
    OrderedDict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
]:
    base_task_ids = np.asarray(labels["task_ids"], dtype=np.int8)
    base_pair_ids = (
        np.asarray(labels["pair_ids"], dtype=np.int8)
        if "pair_ids" in labels
        else np.zeros(len(base_task_ids), dtype=np.int8)
    )
    base_target_sides = (
        np.asarray(labels["target_sides"], dtype=np.int8)
        if "target_sides" in labels
        else base_task_ids.copy()
    )
    base_episode_ids = np.asarray(labels["episode_ids"], dtype=np.int16)
    base_action_steps = np.asarray(labels["action_steps"], dtype=np.int16)
    base_splits = decode_strings(labels["layout_splits"][:])
    source_indices = np.asarray(labels["source_indices"], dtype=np.int64)
    if not np.array_equal(source_indices, np.arange(len(source_indices))):
        raise ValueError("Atlas training requires the full contiguous label file.")
    if not np.array_equal(
        base_task_ids, np.asarray(source["task_ids"][: len(base_task_ids)])
    ):
        raise ValueError("Label and source task IDs are misaligned.")

    raw = labels["raw"]
    object_a_xyz = np.asarray(raw["object_a_xyz"], dtype=np.float32)
    object_b_xyz = np.asarray(raw["object_b_xyz"], dtype=np.float32)
    object_a_to_eef = np.asarray(raw["object_a_to_eef_xyz"], dtype=np.float32)
    object_b_to_eef = np.asarray(raw["object_b_to_eef_xyz"], dtype=np.float32)
    basket_xyz = np.asarray(raw["basket_xyz"], dtype=np.float32)
    eef_xyz = np.asarray(raw["eef_xyz"], dtype=np.float32)
    object_a_displacement, object_a_speed = groupwise_displacement(
        object_a_xyz, base_task_ids, base_episode_ids, base_action_steps
    )
    object_b_displacement, object_b_speed = groupwise_displacement(
        object_b_xyz, base_task_ids, base_episode_ids, base_action_steps
    )

    if prompt_mode == "paired":
        prompt_indices = np.tile(np.asarray([0, 1], dtype=np.int64), len(base_task_ids))

        def duplicate(values: np.ndarray) -> np.ndarray:
            return np.repeat(values, 2, axis=0)

        def prompt_pair(a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
            return np.stack((a_values, b_values), axis=1).reshape(
                2 * len(a_values), *a_values.shape[1:]
            )

        task_ids = duplicate(base_task_ids)
        pair_ids = duplicate(base_pair_ids)
        target_sides = duplicate(base_target_sides)
        episode_ids = duplicate(base_episode_ids)
        action_steps = duplicate(base_action_steps)
        splits = duplicate(base_splits)
        absolute_object_a_xyz = duplicate(object_a_xyz)
        absolute_object_b_xyz = duplicate(object_b_xyz)
        absolute_eef_xyz = duplicate(eef_xyz)
        target_xyz = prompt_pair(object_a_xyz, object_b_xyz)
        alternative_xyz = prompt_pair(object_b_xyz, object_a_xyz)
        target_to_eef = prompt_pair(object_a_to_eef, object_b_to_eef)
        alternative_to_eef = prompt_pair(object_b_to_eef, object_a_to_eef)
        target_displacement = prompt_pair(object_a_displacement, object_b_displacement)
        target_speed = prompt_pair(object_a_speed[:, None], object_b_speed[:, None])[:, 0]
        repeated_basket_xyz = duplicate(basket_xyz)
    else:
        prompt_indices = base_target_sides.astype(np.int64)
        if prompt_mode == "counterfactual":
            prompt_indices = 1 - prompt_indices
        task_ids = base_task_ids
        pair_ids = base_pair_ids
        target_sides = base_target_sides
        episode_ids = base_episode_ids
        action_steps = base_action_steps
        splits = base_splits
        absolute_object_a_xyz = object_a_xyz
        absolute_object_b_xyz = object_b_xyz
        absolute_eef_xyz = eef_xyz
        target_is_a = prompt_indices == 0
        target_xyz = np.where(target_is_a[:, None], object_a_xyz, object_b_xyz)
        alternative_xyz = np.where(target_is_a[:, None], object_b_xyz, object_a_xyz)
        target_to_eef = np.where(target_is_a[:, None], object_a_to_eef, object_b_to_eef)
        alternative_to_eef = np.where(target_is_a[:, None], object_b_to_eef, object_a_to_eef)
        target_displacement = np.where(
            target_is_a[:, None], object_a_displacement, object_b_displacement
        )
        target_speed = np.where(target_is_a, object_a_speed, object_b_speed)
        repeated_basket_xyz = basket_xyz

    target_to_basket = repeated_basket_xyz - target_xyz

    factors: OrderedDict[str, np.ndarray] = OrderedDict()
    add_factor(factors, "object_a_xyz", absolute_object_a_xyz)
    add_factor(factors, "object_b_xyz", absolute_object_b_xyz)
    add_factor(factors, "target_xyz", target_xyz)
    add_factor(factors, "alternative_xyz", alternative_xyz)
    add_factor(factors, "eef_xyz", absolute_eef_xyz)
    add_factor(
        factors,
        "robot_joint_pos",
        duplicate(raw["robot_joint_pos"][:])
        if prompt_mode == "paired"
        else raw["robot_joint_pos"][:],
    )
    add_factor(
        factors,
        "gripper_width",
        duplicate(labels["derived/gripper_width"][:])
        if prompt_mode == "paired"
        else labels["derived/gripper_width"][:],
    )
    add_factor(factors, "target_to_eef_xyz", target_to_eef)
    add_factor(factors, "alternative_to_eef_xyz", alternative_to_eef)
    add_factor(factors, "target_to_basket_xyz", target_to_basket)
    add_factor(
        factors, "eef_target_distance", np.linalg.norm(absolute_eef_xyz - target_xyz, axis=1)
    )
    add_factor(
        factors,
        "eef_alternative_distance",
        np.linalg.norm(absolute_eef_xyz - alternative_xyz, axis=1),
    )
    add_factor(factors, "target_basket_distance", np.linalg.norm(target_to_basket, axis=1))
    add_factor(factors, "target_displacement_xyz", target_displacement)
    add_factor(factors, "target_speed_per_action_step", target_speed)
    add_factor(
        factors,
        "normalized_action_progress",
        duplicate(labels["derived/normalized_action_progress"][:])
        if prompt_mode == "paired"
        else labels["derived/normalized_action_progress"][:],
    )
    if prompt_mode == "actual":
        add_factor(factors, "raw_action", source["raw_actions"][:])

    binary: OrderedDict[str, np.ndarray] = OrderedDict()
    if prompt_mode == "actual":
        for name in (
            "target_grasped",
            "target_moved",
            "target_lifted",
            "near_target",
            "near_basket",
        ):
            binary[name] = np.asarray(labels[f"derived/{name}"], dtype=np.float32)
        phase = np.minimum(np.asarray(labels["derived/phase_heuristic"], dtype=np.int64), 2)
    elif prompt_mode == "paired":
        definitions = json.loads(labels.attrs["definitions_json"])
        thresholds = definitions["thresholds_m"]
        object_a_grasped = np.asarray(raw["object_a_grasped"], dtype=np.float32)
        object_b_grasped = np.asarray(raw["object_b_grasped"], dtype=np.float32)
        target_grasped = prompt_pair(object_a_grasped[:, None], object_b_grasped[:, None])[:, 0]
        target_moved = (
            np.linalg.norm(target_displacement[:, :2], axis=1)
            >= float(thresholds["target_moved_xy"])
        )
        target_lifted = target_displacement[:, 2] >= float(thresholds["target_lifted_z"])
        near_target = (
            np.linalg.norm(absolute_eef_xyz - target_xyz, axis=1)
            <= float(thresholds["near_target"])
        )
        near_basket = (
            np.linalg.norm(target_to_basket, axis=1) <= float(thresholds["near_basket"])
        )
        for name, values in (
            ("target_grasped", target_grasped),
            ("target_moved", target_moved),
            ("target_lifted", target_lifted),
            ("near_target", near_target),
            ("near_basket", near_basket),
        ):
            binary[name] = np.asarray(values, dtype=np.float32)
        phase = np.zeros(len(target_xyz), dtype=np.int64)
        phase[near_target] = 1
        phase[(target_grasped > 0) | target_lifted] = 2
    else:
        phase = np.empty(0, dtype=np.int64)

    metadata = {
        "task_ids": task_ids,
        "pair_ids": pair_ids,
        "target_sides": target_sides,
        "episode_ids": episode_ids,
        "action_steps": action_steps,
        "splits": splits,
    }
    return prompt_indices, factors, binary, phase, metadata


def concatenate_factors(
    factors: OrderedDict[str, np.ndarray], train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, tuple[int, int]], np.ndarray, np.ndarray]:
    values = np.concatenate(list(factors.values()), axis=1).astype(np.float32)
    slices = {}
    start = 0
    for name, factor in factors.items():
        slices[name] = (start, start + factor.shape[1])
        start += factor.shape[1]
    mean = values[train_indices].mean(axis=0)
    std = values[train_indices].std(axis=0)
    active = std > 1e-7
    if not active.all():
        inactive = np.flatnonzero(~active).tolist()
        raise ValueError(f"Continuous target dimensions have no training variance: {inactive}.")
    standardized = (values - mean) / std
    return standardized, slices, mean, std


def load_feature_cache(
    residuals: h5py.File,
    readout: str,
    prompt_indices: np.ndarray,
    prompt_mode: str,
) -> np.ndarray:
    if readout == "visual_input":
        visual = np.asarray(residuals["visual_input_mean"], dtype=np.float16)
        if prompt_mode == "paired":
            visual = np.repeat(visual, 2, axis=0)
        num_layers = int(residuals.attrs["num_layers"])
        return np.repeat(visual[:, None, :], num_layers, axis=1)
    if readout == "visual_mean":
        if not np.all(prompt_indices >= 0):
            raise ValueError("Invalid prompt indices.")
        visual = np.asarray(residuals["visual_mean_residuals"], dtype=np.float16)
        return np.repeat(visual, 2, axis=0) if prompt_mode == "paired" else visual
    if readout in EXPERT_READOUT_DATASET:
        dataset = residuals[EXPERT_READOUT_DATASET[readout]]
        num_samples, _, num_layers, hidden_size = dataset.shape
        output_samples = 2 * num_samples if prompt_mode == "paired" else num_samples
        cache = np.empty((output_samples, num_layers, hidden_size), dtype=np.float16)
        for index in tqdm(range(num_samples), desc=f"cache {readout}", unit="state"):
            if prompt_mode == "paired":
                cache[2 * index] = dataset[index, 0]
                cache[2 * index + 1] = dataset[index, 1]
            else:
                cache[index] = dataset[index, int(prompt_indices[index])]
        return cache
    if "prompt_end_residuals" in residuals:
        if readout != "prompt_end":
            raise ValueError(f"pi0.5 prefix cache does not provide readout {readout}.")
        dataset = residuals["prompt_end_residuals"]
        num_samples, _, num_layers, hidden_size = dataset.shape
        output_samples = 2 * num_samples if prompt_mode == "paired" else num_samples
        cache = np.empty((output_samples, num_layers, hidden_size), dtype=np.float16)
        for index in tqdm(range(num_samples), desc="cache pi05 prompt_end", unit="state"):
            if prompt_mode == "paired":
                cache[2 * index] = dataset[index, 0]
                cache[2 * index + 1] = dataset[index, 1]
            else:
                cache[index] = dataset[index, int(prompt_indices[index])]
        return cache

    dataset = residuals["prompt_residuals"]
    num_samples, _, num_layers, _, hidden_size = dataset.shape
    output_samples = 2 * num_samples if prompt_mode == "paired" else num_samples
    cache = np.empty((output_samples, num_layers, hidden_size), dtype=np.float16)
    position = READOUT_POSITION[readout]
    for index in tqdm(range(num_samples), desc=f"cache {readout}", unit="state"):
        if prompt_mode == "paired":
            cache[2 * index] = dataset[index, 0, :, position, :]
            cache[2 * index + 1] = dataset[index, 1, :, position, :]
        else:
            cache[index] = dataset[index, int(prompt_indices[index]), :, position, :]
    return cache


class AtlasProbe(torch.nn.Module):
    def __init__(self, hidden_size: int, continuous_dim: int, binary_dim: int, phase_dim: int):
        super().__init__()
        self.continuous = torch.nn.Linear(hidden_size, continuous_dim)
        self.binary = torch.nn.Linear(hidden_size, binary_dim) if binary_dim else None
        self.phase = torch.nn.Linear(hidden_size, phase_dim) if phase_dim else None


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        predictions = probabilities >= threshold
        tp = np.logical_and(predictions, labels == 1).sum()
        fp = np.logical_and(predictions, labels == 0).sum()
        fn = np.logical_and(~predictions, labels == 1).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = np.logical_and(predictions, positives).sum()
    fp = np.logical_and(predictions, negatives).sum()
    fn = np.logical_and(~predictions, positives).sum()
    tn = np.logical_and(~predictions, negatives).sum()
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "auc": binary_auc(labels, probabilities),
        "accuracy": float((predictions == positives).mean()),
        "balanced_accuracy": float((recall + specificity) / 2),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "threshold": threshold,
        "positive_support": int(positives.sum()),
        "negative_support": int(negatives.sum()),
    }


def phase_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((len(PHASE_NAMES), len(PHASE_NAMES)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    f1 = []
    for index in range(len(PHASE_NAMES)):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        f1.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": dict(zip(PHASE_NAMES, f1)),
        "confusion": confusion.tolist(),
        "support": dict(zip(PHASE_NAMES, confusion.sum(axis=1).astype(int).tolist())),
    }


def regression_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    factor_slices: dict[str, tuple[int, int]],
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> dict[str, dict[str, object]]:
    result = {}
    for name, (start, stop) in factor_slices.items():
        truth = labels[:, start:stop]
        estimate = predictions[:, start:stop]
        errors = estimate - truth
        rmse_components = np.sqrt(np.mean(errors**2, axis=0))
        denominator = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
        r2_components = 1 - np.sum(errors**2, axis=0) / np.maximum(denominator, 1e-12)
        baseline_errors = train_mean[start:stop] - truth
        baseline_r2 = 1 - np.sum(baseline_errors**2, axis=0) / np.maximum(denominator, 1e-12)
        correlations = []
        for component in range(truth.shape[1]):
            if np.std(truth[:, component]) <= 1e-12 or np.std(estimate[:, component]) <= 1e-12:
                correlations.append(float("nan"))
            else:
                correlations.append(
                    float(np.corrcoef(truth[:, component], estimate[:, component])[0, 1])
                )
        result[name] = {
            "dimensions": stop - start,
            "mean_r2": float(np.mean(r2_components)),
            "r2_components": r2_components.tolist(),
            "mean_train_baseline_r2": float(np.mean(baseline_r2)),
            "rmse_components": rmse_components.tolist(),
            "mean_normalized_rmse": float(np.mean(rmse_components / train_std[start:stop])),
            "mean_pearson": float(np.nanmean(correlations)),
            "pearson_components": correlations,
        }
    return result


@torch.inference_mode()
def validation_losses(
    model: AtlasProbe,
    features: torch.Tensor,
    continuous: torch.Tensor,
    binary: torch.Tensor | None,
    phase: torch.Tensor | None,
    indices: torch.Tensor,
    pos_weight: torch.Tensor | None,
    phase_weight: torch.Tensor | None,
) -> dict[str, float]:
    result = {"continuous": F.mse_loss(model.continuous(features[indices]), continuous[indices]).item()}
    if model.binary is not None and binary is not None and pos_weight is not None:
        result["binary"] = F.binary_cross_entropy_with_logits(
            model.binary(features[indices]), binary[indices], pos_weight=pos_weight
        ).item()
    if model.phase is not None and phase is not None and phase_weight is not None:
        result["phase"] = F.cross_entropy(
            model.phase(features[indices]), phase[indices], weight=phase_weight
        ).item()
    return result


def train_layer(
    layer: int,
    feature_cache: np.ndarray,
    continuous_values: np.ndarray,
    binary_values: np.ndarray | None,
    phase_values: np.ndarray | None,
    split_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor] | None], list[dict[str, float]], dict[str, object]]:
    features_cpu = torch.from_numpy(np.asarray(feature_cache[:, layer], dtype=np.float32))
    train_cpu = torch.from_numpy(split_indices["train"])
    feature_mean = features_cpu[train_cpu].mean(dim=0)
    feature_std = features_cpu[train_cpu].std(dim=0).clamp_min(1e-6)
    features = ((features_cpu - feature_mean) / feature_std).to(device)
    continuous = torch.from_numpy(continuous_values).to(device)
    binary = torch.from_numpy(binary_values).to(device) if binary_values is not None else None
    phase = torch.from_numpy(phase_values).to(device) if phase_values is not None else None
    indices = {
        name: torch.from_numpy(values).to(device) for name, values in split_indices.items()
    }

    torch.manual_seed(args.seed + layer)
    model = AtlasProbe(
        features.shape[1],
        continuous.shape[1],
        0 if binary is None else binary.shape[1],
        0 if phase is None else len(PHASE_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if binary is not None:
        train_binary = binary[indices["train"]]
        positives = train_binary.sum(dim=0)
        pos_weight = (len(train_binary) - positives) / positives.clamp_min(1)
    else:
        pos_weight = None
    if phase is not None:
        counts = torch.bincount(phase[indices["train"]], minlength=len(PHASE_NAMES)).float()
        phase_weight = counts.sum() / (len(PHASE_NAMES) * counts.clamp_min(1))
    else:
        phase_weight = None

    head_names = ["continuous"]
    if model.binary is not None:
        head_names.append("binary")
    if model.phase is not None:
        head_names.append("phase")
    best_losses = {name: float("inf") for name in head_names}
    best_states: dict[str, dict[str, torch.Tensor] | None] = {name: None for name in head_names}
    best_epochs = {name: 0 for name in head_names}
    stale = {name: 0 for name in head_names}
    history = []
    generator = torch.Generator(device=device).manual_seed(args.seed + layer * 1000)

    for epoch in range(args.epochs):
        model.train()
        permutation = indices["train"][torch.randperm(len(indices["train"]), generator=generator, device=device)]
        running = 0.0
        batches = 0
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.continuous(features[batch]), continuous[batch])
            if model.binary is not None and binary is not None and pos_weight is not None:
                loss = loss + F.binary_cross_entropy_with_logits(
                    model.binary(features[batch]), binary[batch], pos_weight=pos_weight
                )
            if model.phase is not None and phase is not None and phase_weight is not None:
                loss = loss + F.cross_entropy(
                    model.phase(features[batch]), phase[batch], weight=phase_weight
                )
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1

        model.eval()
        losses = validation_losses(
            model,
            features,
            continuous,
            binary,
            phase,
            indices["valid"],
            pos_weight,
            phase_weight,
        )
        history.append({"epoch": epoch + 1, "train_loss": running / batches} | losses)
        for name in head_names:
            if losses[name] < best_losses[name]:
                best_losses[name] = losses[name]
                best_epochs[name] = epoch + 1
                module = getattr(model, name)
                best_states[name] = {
                    key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                }
                stale[name] = 0
            else:
                stale[name] += 1
        if all(stale[name] >= args.patience for name in head_names):
            break

    metadata = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "best_epochs": best_epochs,
        "best_validation_losses": best_losses,
    }
    return best_states, history, metadata


@torch.inference_mode()
def evaluate_layer(
    feature_cache: np.ndarray,
    layer: int,
    states: dict[str, dict[str, torch.Tensor] | None],
    training_metadata: dict[str, object],
    continuous_raw: np.ndarray,
    continuous_mean: np.ndarray,
    continuous_std: np.ndarray,
    factor_slices: dict[str, tuple[int, int]],
    binary_names: list[str],
    binary_values: np.ndarray | None,
    phase_values: np.ndarray | None,
    split_indices: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, object]:
    features = torch.from_numpy(np.asarray(feature_cache[:, layer], dtype=np.float32))
    features = (
        (features - training_metadata["feature_mean"]) / training_metadata["feature_std"]
    ).to(device)
    continuous_head = torch.nn.Linear(features.shape[1], continuous_raw.shape[1]).to(device)
    continuous_head.load_state_dict(states["continuous"])
    continuous_predictions = continuous_head(features).cpu().numpy() * continuous_std + continuous_mean

    result: dict[str, object] = {"splits": {}}
    validation_thresholds = {}
    binary_probabilities = None
    if binary_values is not None:
        binary_head = torch.nn.Linear(features.shape[1], binary_values.shape[1]).to(device)
        binary_head.load_state_dict(states["binary"])
        binary_probabilities = binary_head(features).sigmoid().cpu().numpy()
        valid = split_indices["valid"]
        for index, name in enumerate(binary_names):
            validation_thresholds[name] = best_f1_threshold(
                binary_values[valid, index].astype(int), binary_probabilities[valid, index]
            )
    phase_predictions = None
    if phase_values is not None:
        phase_head = torch.nn.Linear(features.shape[1], len(PHASE_NAMES)).to(device)
        phase_head.load_state_dict(states["phase"])
        phase_predictions = phase_head(features).argmax(dim=1).cpu().numpy()

    for split_name in ("valid", "test"):
        split = split_indices[split_name]
        split_result: dict[str, object] = {
            "regression": regression_metrics(
                continuous_raw[split],
                continuous_predictions[split],
                factor_slices,
                continuous_mean,
                continuous_std,
            )
        }
        if binary_values is not None and binary_probabilities is not None:
            split_result["binary"] = {
                name: binary_metrics(
                    binary_values[split, index].astype(int),
                    binary_probabilities[split, index],
                    validation_thresholds[name],
                )
                for index, name in enumerate(binary_names)
            }
        if phase_values is not None and phase_predictions is not None:
            split_result["phase_coarse"] = phase_metrics(
                phase_values[split], phase_predictions[split]
            )
        result["splits"][split_name] = split_result
    result["validation_thresholds"] = validation_thresholds
    result["best_epochs"] = training_metadata["best_epochs"]
    result["best_validation_losses"] = training_metadata["best_validation_losses"]
    return result


def select_best_layers(results: list[dict[str, object]], factor_names: list[str], binary_names: list[str]) -> dict[str, object]:
    selected: dict[str, object] = {"regression": {}, "binary": {}}
    for factor in factor_names:
        best = max(
            results,
            key=lambda item: item["splits"]["valid"]["regression"][factor]["mean_r2"],
        )
        selected["regression"][factor] = {
            "layer": best["layer"],
            "validation": best["splits"]["valid"]["regression"][factor],
            "test": best["splits"]["test"]["regression"][factor],
        }
    for label in binary_names:
        best = max(
            results,
            key=lambda item: item["splits"]["valid"]["binary"][label]["auc"],
        )
        selected["binary"][label] = {
            "layer": best["layer"],
            "validation": best["splits"]["valid"]["binary"][label],
            "test": best["splits"]["test"]["binary"][label],
        }
    if binary_names:
        best = max(
            results,
            key=lambda item: item["splits"]["valid"]["phase_coarse"]["macro_f1"],
        )
        selected["phase_coarse"] = {
            "layer": best["layer"],
            "validation": best["splits"]["valid"]["phase_coarse"],
            "test": best["splits"]["test"]["phase_coarse"],
        }
    return selected


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.readout in {"visual_mean", "visual_input"} and args.prompt_mode == "counterfactual":
        raise ValueError("Use paired mode for a prompt-invariant visual control.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    started_at = time.perf_counter()

    with h5py.File(args.labels_h5, "r") as labels, h5py.File(args.source_h5, "r") as source:
        prompt_indices, factors, binary, phase, metadata = load_targets(
            labels, source, prompt_mode=args.prompt_mode
        )
    exclude_task_ids = (
        sorted({int(value) for value in args.exclude_task_ids.split(",")})
        if args.exclude_task_ids
        else []
    )
    if exclude_task_ids and args.holdout_pair is not None:
        raise ValueError("Use either --holdout-pair or --exclude-task-ids, not both.")
    if exclude_task_ids:
        known_tasks = set(np.unique(metadata["task_ids"]).astype(int).tolist())
        if not set(exclude_task_ids) <= known_tasks:
            raise ValueError(
                f"--exclude-task-ids={exclude_task_ids} not all present; available tasks are {sorted(known_tasks)}."
            )
        is_excluded = np.isin(metadata["task_ids"], exclude_task_ids)
        split_indices = {
            "train": np.flatnonzero((metadata["splits"] == "train") & ~is_excluded).astype(np.int64),
            "valid": np.flatnonzero((metadata["splits"] == "valid") & ~is_excluded).astype(np.int64),
            "test": np.flatnonzero((metadata["splits"] == "test") & is_excluded).astype(np.int64),
        }
    elif args.holdout_pair is None:
        split_indices = {
            split: np.flatnonzero(metadata["splits"] == split).astype(np.int64)
            for split in ("train", "valid", "test")
        }
    else:
        known_pairs = set(np.unique(metadata["pair_ids"]).astype(int).tolist())
        if args.holdout_pair not in known_pairs:
            raise ValueError(
                f"--holdout-pair={args.holdout_pair} is absent; available pairs are {sorted(known_pairs)}."
            )
        is_holdout = metadata["pair_ids"] == args.holdout_pair
        split_indices = {
            "train": np.flatnonzero((metadata["splits"] == "train") & ~is_holdout).astype(np.int64),
            "valid": np.flatnonzero((metadata["splits"] == "valid") & ~is_holdout).astype(np.int64),
            "test": np.flatnonzero((metadata["splits"] == "test") & is_holdout).astype(np.int64),
        }
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise ValueError(f"Empty split after applying holdout protocol: {split_indices}.")
    continuous_standardized, factor_slices, continuous_mean, continuous_std = concatenate_factors(
        factors, split_indices["train"]
    )
    continuous_raw = np.concatenate(list(factors.values()), axis=1).astype(np.float32)
    binary_names = list(binary)
    binary_values = (
        np.stack(list(binary.values()), axis=1).astype(np.float32) if binary_names else None
    )
    phase_values = phase if len(phase) else None

    with h5py.File(args.residual_h5, "r") as residuals:
        num_layers = int(residuals.attrs["num_layers"])
        layer_convention = decode_attr(
            residuals.attrs.get("layer_convention", "decoder_block_output_before_final_norm")
        )
        expected_samples = (
            len(prompt_indices) // 2 if args.prompt_mode == "paired" else len(prompt_indices)
        )
        if int(residuals.attrs["num_images"]) != expected_samples:
            raise ValueError("Residual and label sample counts differ.")
        layers = parse_layers(args.layers, num_layers)
        feature_cache = load_feature_cache(
            residuals, args.readout, prompt_indices, args.prompt_mode
        )

    results = []
    for layer in tqdm(layers, desc=f"atlas {args.readout} {args.prompt_mode}", unit="layer"):
        states, history, training_metadata = train_layer(
            layer,
            feature_cache,
            continuous_standardized,
            binary_values,
            phase_values,
            split_indices,
            args,
            device,
        )
        result = evaluate_layer(
            feature_cache,
            layer,
            states,
            training_metadata,
            continuous_raw,
            continuous_mean,
            continuous_std,
            factor_slices,
            binary_names,
            binary_values,
            phase_values,
            split_indices,
            device,
        )
        result["layer"] = layer
        result["history"] = history
        results.append(result)
        torch.save(
            {
                "layer": layer,
                "readout": args.readout,
                "prompt_mode": args.prompt_mode,
                "states": states,
                "feature_mean": training_metadata["feature_mean"],
                "feature_std": training_metadata["feature_std"],
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "factor_slices": factor_slices,
                "binary_names": binary_names,
                "phase_names": PHASE_NAMES if phase_values is not None else (),
            },
            args.output_dir / f"layer_{layer:02d}.pt",
        )

    selected = select_best_layers(results, list(factors), binary_names)
    summary = {
        "residual_h5": str(args.residual_h5),
        "labels_h5": str(args.labels_h5),
        "source_h5": str(args.source_h5),
        "readout": args.readout,
        "prompt_mode": args.prompt_mode,
        "layer_convention": layer_convention,
        "selection_protocol": (
            "leave-task-scenes-out; excluded tasks absent from train/valid; test episodes locked"
            if exclude_task_ids
            else "layer and binary threshold selected on validation; test locked"
            if args.holdout_pair is None
            else "leave-one-pair-out; held-out pair absent from train/valid; test episodes locked"
        ),
        "holdout_pair": args.holdout_pair,
        "exclude_task_ids": exclude_task_ids,
        "split_unit": "episode",
        "split_counts": {name: len(indices) for name, indices in split_indices.items()},
        "pair_counts": {
            str(pair_id): int((metadata["pair_ids"] == pair_id).sum())
            for pair_id in np.unique(metadata["pair_ids"])
        },
        "factor_slices": factor_slices,
        "binary_names": binary_names,
        "phase_names": PHASE_NAMES if phase_values is not None else (),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
        "selected": selected,
        "layers": results,
        "elapsed_seconds": time.perf_counter() - started_at,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "layers"}, indent=2))
    print(f"Wrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
