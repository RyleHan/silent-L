#!/usr/bin/env python3
"""Episode-bootstrap paired-prompt OpenVLA atlas contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from train_openvla_world_state_atlas import (
    EXPERT_READOUT_DATASET,
    PHASE_NAMES,
    READOUT_POSITION,
    load_targets,
    phase_metrics,
)


READOUTS = ("prompt_end", "action_boundary", "visual_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-h5", type=Path, required=True)
    parser.add_argument("--labels-h5", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--readouts", default=",".join(READOUTS))
    parser.add_argument("--visual-readout", default="visual_mean")
    parser.add_argument(
        "--factors",
        default="target_xyz,target_to_eef_xyz,target_to_basket_xyz,target_displacement_xyz",
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_summary(root: Path, readout: str) -> dict[str, object]:
    return json.loads((root / readout / "summary.json").read_text(encoding="utf-8"))


def load_layer_features(
    residuals: h5py.File, readout: str, layer: int
) -> torch.Tensor:
    if readout == "visual_input":
        values = np.asarray(residuals["visual_input_mean"], dtype=np.float32)
        values = np.repeat(values, 2, axis=0)
    elif readout in EXPERT_READOUT_DATASET:
        dataset = residuals[EXPERT_READOUT_DATASET[readout]]
        values = np.asarray(dataset[:, :, layer, :], dtype=np.float32).reshape(
            -1, dataset.shape[-1]
        )
    elif "prompt_end_residuals" in residuals:
        if readout != "prompt_end":
            raise ValueError(f"pi0.5 prefix cache does not provide readout {readout}.")
        values = np.asarray(
            residuals["prompt_end_residuals"][:, :, layer, :], dtype=np.float32
        ).reshape(-1, residuals["prompt_end_residuals"].shape[-1])
    elif readout == "visual_mean":
        values = np.asarray(residuals["visual_mean_residuals"][:, layer], dtype=np.float32)
        values = np.repeat(values, 2, axis=0)
    else:
        position = READOUT_POSITION[readout]
        values = np.asarray(
            residuals["prompt_residuals"][:, :, layer, position, :], dtype=np.float32
        ).reshape(-1, residuals["prompt_residuals"].shape[-1])
    return torch.from_numpy(values)


def predict_continuous(
    checkpoint: dict[str, object], features: torch.Tensor
) -> np.ndarray:
    normalized_features = (
        features - checkpoint["feature_mean"]
    ) / checkpoint["feature_std"]
    state = checkpoint["states"]["continuous"]
    normalized = F.linear(
        normalized_features,
        state["weight"],
        state["bias"],
    )
    return (
        normalized * torch.from_numpy(checkpoint["continuous_std"])
        + torch.from_numpy(checkpoint["continuous_mean"])
    ).numpy()


def predict_phase(checkpoint: dict[str, object], features: torch.Tensor) -> np.ndarray:
    normalized_features = (
        features - checkpoint["feature_mean"]
    ) / checkpoint["feature_std"]
    state = checkpoint["states"]["phase"]
    return F.linear(normalized_features, state["weight"], state["bias"]).argmax(dim=1).numpy()


def mean_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    if truth.ndim == 1:
        truth = truth[:, None]
        prediction = prediction[:, None]
    denominator = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    values = 1 - np.sum((prediction - truth) ** 2, axis=0) / np.maximum(denominator, 1e-12)
    return float(values.mean())


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
        "probability_positive": float((values > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    readouts = tuple(value.strip() for value in args.readouts.split(",") if value.strip())
    if args.visual_readout not in readouts:
        raise ValueError("visual-readout must be included in readouts.")
    prompt_readouts = tuple(readout for readout in readouts if readout != args.visual_readout)
    factors_requested = [value.strip() for value in args.factors.split(",") if value.strip()]
    summaries = {readout: load_summary(args.paired_root, readout) for readout in readouts}
    with h5py.File(args.labels_h5, "r") as labels, h5py.File(args.source_h5, "r") as source:
        _, factors, _, phase, metadata = load_targets(labels, source, prompt_mode="paired")

    test_indices = np.flatnonzero(metadata["splits"] == "test")
    episode_keys = np.stack(
        (metadata["pair_ids"], metadata["task_ids"], metadata["episode_ids"]), axis=1
    )
    pair_values = np.unique(metadata["pair_ids"][test_indices]).astype(int)
    episode_groups_by_pair: dict[int, list[np.ndarray]] = {}
    unique_episodes = []
    for pair_id in pair_values:
        pair_test = test_indices[metadata["pair_ids"][test_indices] == pair_id]
        pair_episodes = np.unique(episode_keys[pair_test], axis=0)
        unique_episodes.extend(pair_episodes.tolist())
        episode_groups_by_pair[pair_id] = [
            pair_test[np.all(episode_keys[pair_test] == episode[None, :], axis=1)]
            for episode in pair_episodes
        ]

    predictions: dict[str, dict[str, np.ndarray]] = {readout: {} for readout in readouts}
    phase_predictions = {}
    with h5py.File(args.residual_h5, "r") as residuals:
        for readout in readouts:
            layer_cache: dict[int, torch.Tensor] = {}
            for factor in factors_requested:
                selected = summaries[readout]["selected"]["regression"][factor]
                layer = int(selected["layer"])
                if layer not in layer_cache:
                    layer_cache[layer] = load_layer_features(residuals, readout, layer)
                checkpoint = torch.load(
                    args.paired_root / readout / f"layer_{layer:02d}.pt",
                    map_location="cpu",
                )
                all_predictions = predict_continuous(checkpoint, layer_cache[layer])
                start, stop = checkpoint["factor_slices"][factor]
                predictions[readout][factor] = all_predictions[:, start:stop]

            phase_layer = int(summaries[readout]["selected"]["phase_coarse"]["layer"])
            if phase_layer not in layer_cache:
                layer_cache[phase_layer] = load_layer_features(residuals, readout, phase_layer)
            phase_checkpoint = torch.load(
                args.paired_root / readout / f"layer_{phase_layer:02d}.pt",
                map_location="cpu",
            )
            phase_predictions[readout] = predict_phase(
                phase_checkpoint, layer_cache[phase_layer]
            )

    rng = np.random.default_rng(args.seed)
    bootstrap_indices: list[dict[int, np.ndarray]] = []
    for _ in range(args.resamples):
        sampled: dict[int, np.ndarray] = {}
        for pair_id, groups in episode_groups_by_pair.items():
            draws = rng.integers(0, len(groups), size=len(groups))
            sampled[pair_id] = np.concatenate([groups[index] for index in draws])
        bootstrap_indices.append(sampled)

    result: dict[str, object] = {
        "protocol": (
            "paired prompts; validation-selected layers; pair-macro test aggregation; "
            "episodes resampled within pair"
        ),
        "num_test_pairs": len(pair_values),
        "num_test_episodes": sum(len(groups) for groups in episode_groups_by_pair.values()),
        "test_episodes": unique_episodes,
        "resamples": args.resamples,
        "seed": args.seed,
        "regression": {},
        "phase_coarse": {},
    }
    visual = args.visual_readout
    for readout in prompt_readouts:
        result["regression"][readout] = {}
        for factor in factors_requested:
            truth = factors[factor]
            per_pair = {}
            for pair_id in pair_values:
                indices = test_indices[metadata["pair_ids"][test_indices] == pair_id]
                prompt_score = mean_r2(
                    truth[indices], predictions[readout][factor][indices]
                )
                visual_score = mean_r2(
                    truth[indices], predictions[visual][factor][indices]
                )
                per_pair[str(pair_id)] = {
                    "prompt_aware_r2": prompt_score,
                    "visual_r2": visual_score,
                    "difference": prompt_score - visual_score,
                    "num_test_episodes": len(episode_groups_by_pair[int(pair_id)]),
                }
            point_prompt = float(
                np.mean([item["prompt_aware_r2"] for item in per_pair.values()])
            )
            point_visual = float(np.mean([item["visual_r2"] for item in per_pair.values()]))
            differences = np.empty(args.resamples, dtype=np.float64)
            for bootstrap_index, sampled in enumerate(bootstrap_indices):
                pair_differences = []
                for pair_id, indices in sampled.items():
                    pair_differences.append(
                        mean_r2(truth[indices], predictions[readout][factor][indices])
                        - mean_r2(truth[indices], predictions[visual][factor][indices])
                    )
                differences[bootstrap_index] = np.mean(pair_differences)
            result["regression"][readout][factor] = {
                "prompt_aware_layer": summaries[readout]["selected"]["regression"][factor][
                    "layer"
                ],
                "visual_layer": summaries[visual]["selected"]["regression"][factor]["layer"],
                "point_prompt_aware_r2": point_prompt,
                "point_visual_r2": point_visual,
                "point_difference": point_prompt - point_visual,
                "aggregation": "equal-weight macro average over object pairs",
                "per_pair": per_pair,
                "bootstrap_difference": interval(differences),
            }

        phase_per_pair = {}
        for pair_id in pair_values:
            indices = test_indices[metadata["pair_ids"][test_indices] == pair_id]
            prompt_score = phase_metrics(
                phase[indices], phase_predictions[readout][indices]
            )["macro_f1"]
            visual_score = phase_metrics(
                phase[indices], phase_predictions[visual][indices]
            )["macro_f1"]
            phase_per_pair[str(pair_id)] = {
                "prompt_aware_macro_f1": prompt_score,
                "visual_macro_f1": visual_score,
                "difference": prompt_score - visual_score,
                "num_test_episodes": len(episode_groups_by_pair[int(pair_id)]),
            }
        point_prompt_phase = float(
            np.mean([item["prompt_aware_macro_f1"] for item in phase_per_pair.values()])
        )
        point_visual_phase = float(
            np.mean([item["visual_macro_f1"] for item in phase_per_pair.values()])
        )
        phase_differences = np.empty(args.resamples, dtype=np.float64)
        for bootstrap_index, sampled in enumerate(bootstrap_indices):
            pair_differences = []
            for pair_id, indices in sampled.items():
                pair_differences.append(
                    phase_metrics(phase[indices], phase_predictions[readout][indices])[
                        "macro_f1"
                    ]
                    - phase_metrics(phase[indices], phase_predictions[visual][indices])[
                        "macro_f1"
                    ]
                )
            phase_differences[bootstrap_index] = np.mean(pair_differences)
        result["phase_coarse"][readout] = {
            "prompt_aware_layer": summaries[readout]["selected"]["phase_coarse"]["layer"],
            "visual_layer": summaries[visual]["selected"]["phase_coarse"]["layer"],
            "point_prompt_aware_macro_f1": point_prompt_phase,
            "point_visual_macro_f1": point_visual_phase,
            "point_difference": point_prompt_phase - point_visual_phase,
            "aggregation": "equal-weight macro average over object pairs",
            "per_pair": phase_per_pair,
            "bootstrap_difference": interval(phase_differences),
            "phase_names": PHASE_NAMES,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
