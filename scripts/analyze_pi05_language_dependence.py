#!/usr/bin/env python3
"""Analyze locked-probe and vector-field responses to A, B, and null prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


READOUTS = {
    "prefix_prompt_end": ("prompt_end_residuals", "prompt_end_residuals"),
    "expert_action_mean": (
        "expert_action_mean_residuals",
        "expert_action_mean_residuals",
    ),
}
PRIMARY_FACTOR = "target_xyz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--labels-h5", type=Path, required=True)
    parser.add_argument("--paired-prefix-h5", type=Path, required=True)
    parser.add_argument("--paired-expert-h5", type=Path, required=True)
    parser.add_argument("--null-h5", type=Path, required=True)
    parser.add_argument("--prefix-probe-dir", type=Path, required=True)
    parser.add_argument("--expert-probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def load_summary(probe_dir: Path) -> dict[str, object]:
    return json.loads((probe_dir / "summary.json").read_text(encoding="utf-8"))


def load_checkpoint(probe_dir: Path, layer: int) -> dict[str, object]:
    return torch.load(
        probe_dir / f"layer_{layer:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )


def normalized_features(features: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return (np.asarray(features, dtype=np.float32) - mean) / std


def linear_head(features: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    weight = state["weight"].detach().cpu().numpy().astype(np.float32)
    bias = state["bias"].detach().cpu().numpy().astype(np.float32)
    return features @ weight.T + bias


def predict_continuous(features: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    standardized = linear_head(
        normalized_features(features, checkpoint), checkpoint["states"]["continuous"]
    )
    target_mean = np.asarray(checkpoint["continuous_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["continuous_std"], dtype=np.float32)
    return standardized * target_std + target_mean


def predict_phase(features: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    state = checkpoint["states"].get("phase")
    if state is None:
        raise ValueError("Selected checkpoint has no phase head.")
    logits = linear_head(normalized_features(features, checkpoint), state)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def interpolation(first: np.ndarray, second: np.ndarray, query: np.ndarray) -> np.ndarray:
    direction = second - first
    denominator = np.sum(direction * direction, axis=1)
    result = np.full(len(first), np.nan, dtype=np.float32)
    valid = denominator > 1e-12
    result[valid] = (
        np.sum((query[valid] - first[valid]) * direction[valid], axis=1)
        / denominator[valid]
    )
    return result


def regression_accuracy(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    denominator = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    active = denominator > 1e-8
    squared_error = np.sum((prediction - truth) ** 2, axis=0)
    component_r2 = np.full(truth.shape[1], np.nan, dtype=np.float64)
    component_r2[active] = 1.0 - squared_error[active] / denominator[active]
    return {
        "mean_r2_active_components": (
            float(np.mean(component_r2[active])) if active.any() else None
        ),
        "r2_components": [float(value) if np.isfinite(value) else None for value in component_r2],
        "num_r2_active_components": int(active.sum()),
        "rmse_components": np.sqrt(np.mean((prediction - truth) ** 2, axis=0)).tolist(),
    }


def clustered_mean_ci(
    values: np.ndarray,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    finite = np.isfinite(values)
    values = values[finite]
    keys = np.stack((task_ids[finite], episode_ids[finite]), axis=1)
    unique_keys = np.unique(keys, axis=0)
    clusters = [
        values[np.all(keys == key[None, :], axis=1)] for key in unique_keys
    ]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        bootstrap[index] = np.concatenate([clusters[item] for item in selected]).mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "num_states": int(len(values)),
        "num_episodes": int(len(unique_keys)),
    }


def condition_metrics(
    first: np.ndarray,
    second: np.ndarray,
    null: np.ndarray,
    object_a: np.ndarray,
    object_b: np.ndarray,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    indices: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    first = first[indices]
    second = second[indices]
    null = null[indices]
    object_a = object_a[indices]
    object_b = object_b[indices]
    selected_tasks = task_ids[indices]
    selected_episodes = episode_ids[indices]
    coefficient = interpolation(first, second, null)
    distance_to_first = np.linalg.norm(null - first, axis=1)
    distance_to_second = np.linalg.norm(null - second, axis=1)
    physical_a_distance = np.linalg.norm(null - object_a, axis=1)
    physical_b_distance = np.linalg.norm(null - object_b, axis=1)
    chooses_a = distance_to_first < distance_to_second
    task_consistent = chooses_a == (selected_tasks == 0)
    return {
        "num_states": int(len(indices)),
        "endpoint_first_accuracy": regression_accuracy(object_a, first),
        "endpoint_second_accuracy": regression_accuracy(object_b, second),
        "endpoint_mean_distance": float(np.linalg.norm(second - first, axis=1).mean()),
        "null_interpolation_first_0_second_1": clustered_mean_ci(
            coefficient,
            selected_tasks,
            selected_episodes,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "null_interpolation_median": float(np.nanmedian(coefficient)),
        "null_closer_to_first_fraction": float(chooses_a.mean()),
        "null_task_consistent_endpoint_fraction": float(task_consistent.mean()),
        "null_distance_to_first_mean": float(distance_to_first.mean()),
        "null_distance_to_second_mean": float(distance_to_second.mean()),
        "null_physical_closer_to_object_a_fraction": float(
            (physical_a_distance < physical_b_distance).mean()
        ),
        "null_physical_object_a_distance_mean": float(physical_a_distance.mean()),
        "null_physical_object_b_distance_mean": float(physical_b_distance.mean()),
        "interpolation_values": coefficient,
    }


def strip_arrays(value: object) -> object:
    if isinstance(value, dict):
        return {key: strip_arrays(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return None
    return value


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as inputs:
        task_ids = np.asarray(inputs["task_ids"], dtype=np.int64)
        episode_ids = np.asarray(inputs["episode_ids"], dtype=np.int64)
        action_steps = np.asarray(inputs["action_steps"], dtype=np.int64)
        splits = decode_strings(inputs["layout_splits"][:])
    with h5py.File(args.labels_h5, "r") as labels:
        object_a = np.asarray(labels["raw/object_a_xyz"], dtype=np.float32)
        object_b = np.asarray(labels["raw/object_b_xyz"], dtype=np.float32)
        object_a_to_eef = np.asarray(
            labels["raw/object_a_to_eef_xyz"], dtype=np.float32
        )
        object_b_to_eef = np.asarray(
            labels["raw/object_b_to_eef_xyz"], dtype=np.float32
        )

    count = len(task_ids)
    if not all(len(values) == count for values in (episode_ids, action_steps, splits, object_a)):
        raise ValueError("Input and label files are not aligned.")
    subsets = {
        "locked_test": np.flatnonzero(splits == "test"),
        "locked_test_initial": np.flatnonzero((splits == "test") & (action_steps == 0)),
        "all_initial_descriptive": np.flatnonzero(action_steps == 0),
    }
    if any(len(indices) == 0 for indices in subsets.values()):
        raise ValueError(f"An analysis subset is empty: {subsets}.")

    prefix_summary = load_summary(args.prefix_probe_dir)
    expert_summary = load_summary(args.expert_probe_dir)
    probe_sources = {
        "prefix_prompt_end": (args.paired_prefix_h5, args.prefix_probe_dir, prefix_summary),
        "expert_action_mean": (args.paired_expert_h5, args.expert_probe_dir, expert_summary),
    }

    results: dict[str, object] = {
        "protocol": {
            "conditions": ["object_a_prompt", "object_b_prompt", "target_underspecified_null"],
            "null_prompt": None,
            "selection": "all probe layers and weights locked by the pre-existing paired atlas",
            "primary_factor": PRIMARY_FACTOR,
            "primary_split": "locked_test",
            "initial_frame_reason": (
                "action_step zero removes trajectory-state cues about the demonstrated target"
            ),
            "vector_field_readout": "single denoising evaluation at t=1 with matched noise",
        },
        "subsets": {name: int(len(indices)) for name, indices in subsets.items()},
        "readouts": {},
    }
    plot_values: dict[str, dict[str, np.ndarray]] = {}

    with h5py.File(args.null_h5, "r") as null_cache:
        if not bool(null_cache.attrs["complete"]):
            raise ValueError("Null-language residual cache is incomplete.")
        if int(null_cache.attrs["num_images"]) != count:
            raise ValueError("Null cache and input sample counts differ.")
        results["protocol"]["null_prompt"] = str(null_cache.attrs["null_prompt"])

        for readout_name, (paired_path, probe_dir, probe_summary) in probe_sources.items():
            paired_dataset, null_dataset = READOUTS[readout_name]
            factor_results = {}
            plot_values[readout_name] = {}
            with h5py.File(paired_path, "r") as paired:
                for factor in ("target_xyz", "target_to_eef_xyz"):
                    layer = int(probe_summary["selected"]["regression"][factor]["layer"])
                    checkpoint = load_checkpoint(probe_dir, layer)
                    start, stop = checkpoint["factor_slices"][factor]
                    first_features = np.asarray(paired[paired_dataset][:, 0, layer], dtype=np.float32)
                    second_features = np.asarray(paired[paired_dataset][:, 1, layer], dtype=np.float32)
                    null_features = np.asarray(null_cache[null_dataset][:, layer], dtype=np.float32)
                    first_prediction = predict_continuous(first_features, checkpoint)[:, start:stop]
                    second_prediction = predict_continuous(second_features, checkpoint)[:, start:stop]
                    null_prediction = predict_continuous(null_features, checkpoint)[:, start:stop]
                    if factor == "target_xyz":
                        factor_a, factor_b = object_a, object_b
                    else:
                        factor_a, factor_b = object_a_to_eef, object_b_to_eef
                    subset_results = {}
                    for subset_offset, (subset_name, indices) in enumerate(subsets.items()):
                        metrics = condition_metrics(
                            first_prediction,
                            second_prediction,
                            null_prediction,
                            factor_a,
                            factor_b,
                            task_ids,
                            episode_ids,
                            indices,
                            bootstrap_samples=args.bootstrap_samples,
                            seed=args.seed + layer * 100 + subset_offset,
                        )
                        plot_values[readout_name][f"{factor}:{subset_name}"] = metrics.pop(
                            "interpolation_values"
                        )
                        subset_results[subset_name] = metrics
                    factor_results[factor] = {"locked_layer": layer, "subsets": subset_results}

                phase_layer = int(probe_summary["selected"]["phase_coarse"]["layer"])
                phase_checkpoint = load_checkpoint(probe_dir, phase_layer)
                first_phase = predict_phase(
                    np.asarray(paired[paired_dataset][:, 0, phase_layer], dtype=np.float32),
                    phase_checkpoint,
                )
                second_phase = predict_phase(
                    np.asarray(paired[paired_dataset][:, 1, phase_layer], dtype=np.float32),
                    phase_checkpoint,
                )
                null_phase = predict_phase(
                    np.asarray(null_cache[null_dataset][:, phase_layer], dtype=np.float32),
                    phase_checkpoint,
                )
                phase_results = {}
                for subset_name, indices in subsets.items():
                    entropy = -np.sum(
                        null_phase[indices] * np.log(null_phase[indices].clip(min=1e-12)), axis=1
                    )
                    phase_results[subset_name] = {
                        "object_a_prompt_mean_probabilities": first_phase[indices].mean(axis=0).tolist(),
                        "object_b_prompt_mean_probabilities": second_phase[indices].mean(axis=0).tolist(),
                        "null_mean_probabilities": null_phase[indices].mean(axis=0).tolist(),
                        "null_mean_entropy_nats": float(entropy.mean()),
                        "null_matches_object_a_argmax_fraction": float(
                            (null_phase[indices].argmax(axis=1) == first_phase[indices].argmax(axis=1)).mean()
                        ),
                        "null_matches_object_b_argmax_fraction": float(
                            (null_phase[indices].argmax(axis=1) == second_phase[indices].argmax(axis=1)).mean()
                        ),
                    }
                results["readouts"][readout_name] = {
                    "factors": factor_results,
                    "phase": {"locked_layer": phase_layer, "subsets": phase_results},
                }

        with h5py.File(args.paired_expert_h5, "r") as paired_expert:
            first_velocity = np.asarray(paired_expert["expert_velocity"][:, 0], dtype=np.float32)
            second_velocity = np.asarray(paired_expert["expert_velocity"][:, 1], dtype=np.float32)
            null_velocity = np.asarray(null_cache["expert_velocity"], dtype=np.float32)
        velocity_results = {}
        plot_values["expert_velocity_t1"] = {}
        flat_first = first_velocity.reshape(count, -1)
        flat_second = second_velocity.reshape(count, -1)
        flat_null = null_velocity.reshape(count, -1)
        for subset_offset, (subset_name, indices) in enumerate(subsets.items()):
            coefficient = interpolation(flat_first[indices], flat_second[indices], flat_null[indices])
            distance_a = np.linalg.norm(flat_null[indices] - flat_first[indices], axis=1)
            distance_b = np.linalg.norm(flat_null[indices] - flat_second[indices], axis=1)
            chooses_a = distance_a < distance_b
            velocity_results[subset_name] = {
                "num_states": int(len(indices)),
                "object_a_object_b_mean_distance": float(
                    np.linalg.norm(flat_second[indices] - flat_first[indices], axis=1).mean()
                ),
                "null_interpolation_object_a_0_object_b_1": clustered_mean_ci(
                    coefficient,
                    task_ids[indices],
                    episode_ids[indices],
                    samples=args.bootstrap_samples,
                    seed=args.seed + 9000 + subset_offset,
                ),
                "null_interpolation_median": float(np.nanmedian(coefficient)),
                "null_closer_to_object_a_action_fraction": float(chooses_a.mean()),
                "null_task_consistent_action_fraction": float(
                    (chooses_a == (task_ids[indices] == 0)).mean()
                ),
                "null_distance_to_object_a_action_mean": float(distance_a.mean()),
                "null_distance_to_object_b_action_mean": float(distance_b.mean()),
            }
            plot_values["expert_velocity_t1"][subset_name] = coefficient
        results["vector_field"] = velocity_results

    output_json = args.output_dir / "summary.json"
    output_json.write_text(json.dumps(strip_arrays(results), indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    names = ["Prefix probe", "Expert probe", "Expert vector field"]
    sources = [
        plot_values["prefix_prompt_end"]["target_xyz:locked_test"],
        plot_values["expert_action_mean"]["target_xyz:locked_test"],
        plot_values["expert_velocity_t1"]["locked_test"],
    ]
    grouped = []
    positions = []
    colors = []
    labels = []
    for source_index, values in enumerate(sources):
        test_indices = subsets["locked_test"]
        for task_id, color, label_suffix in ((0, "#277DA1", "task A"), (1, "#D1495B", "task B")):
            grouped.append(values[task_ids[test_indices] == task_id])
            positions.append(source_index * 3 + task_id + 1)
            colors.append(color)
            labels.append(f"{names[source_index]}\n{label_suffix}")
    boxplot = axes[0].boxplot(grouped, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    for patch, color in zip(boxplot["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0].axhline(0.0, color="#277DA1", linewidth=1, linestyle="--")
    axes[0].axhline(1.0, color="#D1495B", linewidth=1, linestyle="--")
    axes[0].axhline(0.5, color="#555555", linewidth=1, linestyle=":")
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Null interpolation (A = 0, B = 1)")
    axes[0].set_title("Where the target-unspecified prompt lands")

    subset_names = ["locked_test", "locked_test_initial", "all_initial_descriptive"]
    x = np.arange(len(subset_names))
    width = 0.24
    preference_values = []
    for readout_name in ("prefix_prompt_end", "expert_action_mean"):
        preference_values.append(
            [
                results["readouts"][readout_name]["factors"][PRIMARY_FACTOR]["subsets"][subset][
                    "null_closer_to_first_fraction"
                ]
                for subset in subset_names
            ]
        )
    preference_values.append(
        [results["vector_field"][subset]["null_closer_to_object_a_action_fraction"] for subset in subset_names]
    )
    palette = ["#277DA1", "#43AA8B", "#F4A261"]
    for index, (label, values, color) in enumerate(zip(names, preference_values, palette, strict=True)):
        axes[1].bar(x + (index - 1) * width, values, width, label=label, color=color)
    axes[1].axhline(0.5, color="#555555", linewidth=1, linestyle=":")
    axes[1].set_xticks(x, ["Held-out\nall states", "Held-out\ninitial only", "All initial\n(descriptive)"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Fraction closer to object-A condition")
    axes[1].set_title("Default target preference under null language")
    axes[1].legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"offline_language_dependence.{extension}", dpi=220)
    plt.close(figure)

    print(json.dumps(strip_arrays(results), indent=2), flush=True)
    print(f"Wrote: {output_json}", flush=True)


if __name__ == "__main__":
    main()
