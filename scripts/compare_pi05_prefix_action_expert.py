#!/usr/bin/env python3
"""Paired episode-bootstrap comparison of pi0.5 prefix and action expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from bootstrap_openvla_world_state_atlas import (
    interval,
    load_layer_features,
    mean_r2,
    predict_continuous,
    predict_phase,
)
from train_openvla_world_state_atlas import load_targets, phase_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-residual-h5", type=Path, required=True)
    parser.add_argument("--prefix-paired-root", type=Path, required=True)
    parser.add_argument("--expert-residual-h5", type=Path, required=True)
    parser.add_argument("--expert-paired-root", type=Path, required=True)
    parser.add_argument("--labels-h5", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--factors",
        default="target_xyz,target_to_eef_xyz,target_to_basket_xyz,target_displacement_xyz",
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_summary(root: Path, readout: str) -> dict[str, object]:
    return json.loads((root / readout / "summary.json").read_text(encoding="utf-8"))


def load_predictions(
    *,
    residual_h5: Path,
    paired_root: Path,
    readout: str,
    factors: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    summary = load_summary(paired_root, readout)
    predictions = {}
    layers = {}
    with h5py.File(residual_h5, "r") as residuals:
        feature_cache: dict[int, torch.Tensor] = {}
        for factor in factors:
            layer = int(summary["selected"]["regression"][factor]["layer"])
            layers[factor] = layer
            if layer not in feature_cache:
                feature_cache[layer] = load_layer_features(residuals, readout, layer)
            checkpoint = torch.load(
                paired_root / readout / f"layer_{layer:02d}.pt", map_location="cpu"
            )
            all_predictions = predict_continuous(checkpoint, feature_cache[layer])
            start, stop = checkpoint["factor_slices"][factor]
            predictions[factor] = all_predictions[:, start:stop]

        phase_layer = int(summary["selected"]["phase_coarse"]["layer"])
        layers["phase_coarse"] = phase_layer
        if phase_layer not in feature_cache:
            feature_cache[phase_layer] = load_layer_features(
                residuals, readout, phase_layer
            )
        phase_checkpoint = torch.load(
            paired_root / readout / f"layer_{phase_layer:02d}.pt", map_location="cpu"
        )
        phase_prediction = predict_phase(phase_checkpoint, feature_cache[phase_layer])
    return predictions, phase_prediction, layers


def main() -> None:
    args = parse_args()
    factors_requested = [value.strip() for value in args.factors.split(",") if value.strip()]
    with h5py.File(args.labels_h5, "r") as labels, h5py.File(args.source_h5, "r") as source:
        _, factors, _, phase, metadata = load_targets(labels, source, prompt_mode="paired")

    specifications = {
        "prompt_end": (args.prefix_residual_h5, args.prefix_paired_root),
        "expert_action_mean": (args.expert_residual_h5, args.expert_paired_root),
        "expert_action_first": (args.expert_residual_h5, args.expert_paired_root),
    }
    regression_predictions = {}
    phase_predictions = {}
    selected_layers = {}
    for readout, (residual_h5, paired_root) in specifications.items():
        regression_predictions[readout], phase_predictions[readout], selected_layers[readout] = (
            load_predictions(
                residual_h5=residual_h5,
                paired_root=paired_root,
                readout=readout,
                factors=factors_requested,
            )
        )

    test_indices = np.flatnonzero(metadata["splits"] == "test")
    episode_keys = np.stack(
        (metadata["pair_ids"], metadata["task_ids"], metadata["episode_ids"]), axis=1
    )
    pair_values = np.unique(metadata["pair_ids"][test_indices]).astype(int)
    episode_groups_by_pair = {}
    unique_episodes = []
    for pair_id in pair_values:
        pair_test = test_indices[metadata["pair_ids"][test_indices] == pair_id]
        pair_episodes = np.unique(episode_keys[pair_test], axis=0)
        unique_episodes.extend(pair_episodes.tolist())
        episode_groups_by_pair[pair_id] = [
            pair_test[np.all(episode_keys[pair_test] == episode[None, :], axis=1)]
            for episode in pair_episodes
        ]
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = []
    for _ in range(args.resamples):
        sampled = {}
        for pair_id, groups in episode_groups_by_pair.items():
            draws = rng.integers(0, len(groups), len(groups))
            sampled[pair_id] = np.concatenate([groups[index] for index in draws])
        bootstrap_indices.append(sampled)

    result: dict[str, object] = {
        "protocol": (
            "same test episodes; validation-selected layers; expert minus prefix; "
            "equal pair weighting; episodes resampled within pair"
        ),
        "num_test_pairs": len(pair_values),
        "num_test_episodes": sum(len(groups) for groups in episode_groups_by_pair.values()),
        "test_episodes": unique_episodes,
        "resamples": args.resamples,
        "seed": args.seed,
        "selected_layers": selected_layers,
        "comparisons": {},
    }
    for expert_readout in ("expert_action_mean", "expert_action_first"):
        comparison = {"regression": {}, "phase_coarse": {}}
        for factor in factors_requested:
            truth = factors[factor]
            expert_prediction = regression_predictions[expert_readout][factor]
            prefix_prediction = regression_predictions["prompt_end"][factor]
            per_pair = {}
            for pair_id in pair_values:
                indices = test_indices[metadata["pair_ids"][test_indices] == pair_id]
                expert_score = mean_r2(truth[indices], expert_prediction[indices])
                prefix_score = mean_r2(truth[indices], prefix_prediction[indices])
                per_pair[str(pair_id)] = {
                    "expert_r2": expert_score,
                    "prefix_r2": prefix_score,
                    "difference": expert_score - prefix_score,
                }
            expert_point = float(np.mean([item["expert_r2"] for item in per_pair.values()]))
            prefix_point = float(np.mean([item["prefix_r2"] for item in per_pair.values()]))
            differences = np.asarray(
                [
                    np.mean(
                        [
                            mean_r2(truth[indices], expert_prediction[indices])
                            - mean_r2(truth[indices], prefix_prediction[indices])
                            for indices in sampled.values()
                        ]
                    )
                    for sampled in bootstrap_indices
                ],
                dtype=np.float64,
            )
            comparison["regression"][factor] = {
                "point_expert_r2": expert_point,
                "point_prefix_r2": prefix_point,
                "point_difference": expert_point - prefix_point,
                "per_pair": per_pair,
                "bootstrap_difference": interval(differences),
            }

        phase_per_pair = {}
        for pair_id in pair_values:
            indices = test_indices[metadata["pair_ids"][test_indices] == pair_id]
            expert_score = phase_metrics(
                phase[indices], phase_predictions[expert_readout][indices]
            )["macro_f1"]
            prefix_score = phase_metrics(
                phase[indices], phase_predictions["prompt_end"][indices]
            )["macro_f1"]
            phase_per_pair[str(pair_id)] = {
                "expert_macro_f1": expert_score,
                "prefix_macro_f1": prefix_score,
                "difference": expert_score - prefix_score,
            }
        expert_phase = float(
            np.mean([item["expert_macro_f1"] for item in phase_per_pair.values()])
        )
        prefix_phase = float(
            np.mean([item["prefix_macro_f1"] for item in phase_per_pair.values()])
        )
        phase_differences = np.asarray(
            [
                np.mean(
                    [
                        phase_metrics(
                            phase[indices], phase_predictions[expert_readout][indices]
                        )["macro_f1"]
                        - phase_metrics(
                            phase[indices], phase_predictions["prompt_end"][indices]
                        )["macro_f1"]
                        for indices in sampled.values()
                    ]
                )
                for sampled in bootstrap_indices
            ],
            dtype=np.float64,
        )
        comparison["phase_coarse"] = {
            "point_expert_macro_f1": expert_phase,
            "point_prefix_macro_f1": prefix_phase,
            "point_difference": expert_phase - prefix_phase,
            "per_pair": phase_per_pair,
            "bootstrap_difference": interval(phase_differences),
        }
        result["comparisons"][f"{expert_readout}_minus_prompt_end"] = comparison

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
