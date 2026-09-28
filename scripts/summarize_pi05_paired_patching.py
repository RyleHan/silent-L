#!/usr/bin/env python3
"""Summarize pi0.5 paired-prompt patching with episode-cluster uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


METRIC_NAMES = (
    "action_recovery",
    "cosine_to_natural_prompt_delta",
    "effect_norm_ratio",
    "orthogonal_distortion_ratio",
    "normalized_source_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-h5", type=Path, required=True)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--selection-exclude-layers",
        default="17",
        help="Comma-separated endpoint/control layers excluded from validation selection.",
    )
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def decode_attr(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def compute_metrics(
    natural_velocity: np.ndarray,
    patched_velocity: np.ndarray,
    *,
    base_prompt_index: int,
    source_prompt_index: int,
) -> dict[str, np.ndarray]:
    base = natural_velocity[:, base_prompt_index].reshape(len(natural_velocity), -1)
    source = natural_velocity[:, source_prompt_index].reshape(len(natural_velocity), -1)
    patched = patched_velocity.reshape(*patched_velocity.shape[:3], -1)
    natural_delta = source - base
    patch_delta = patched - base[:, None, None]
    denominator = np.sum(natural_delta**2, axis=-1)
    if np.any(denominator <= 1e-12):
        count = int(np.sum(denominator <= 1e-12))
        raise ValueError(f"Natural paired-prompt velocity delta is degenerate for {count} samples.")

    dot = np.sum(patch_delta * natural_delta[:, None, None], axis=-1)
    recovery = dot / denominator[:, None, None]
    natural_norm = np.sqrt(denominator)
    patch_norm = np.linalg.norm(patch_delta, axis=-1)
    cosine = dot / np.maximum(patch_norm * natural_norm[:, None, None], 1e-12)
    effect_norm_ratio = patch_norm / natural_norm[:, None, None]
    orthogonal = patch_delta - recovery[..., None] * natural_delta[:, None, None]
    orthogonal_ratio = np.linalg.norm(orthogonal, axis=-1) / natural_norm[:, None, None]
    source_error = np.linalg.norm(patched - source[:, None, None], axis=-1)
    normalized_source_error = source_error / natural_norm[:, None, None]
    return {
        "action_recovery": recovery,
        "cosine_to_natural_prompt_delta": cosine,
        "effect_norm_ratio": effect_norm_ratio,
        "orthogonal_distortion_ratio": orthogonal_ratio,
        "normalized_source_error": normalized_source_error,
    }


def summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_sums = np.bincount(inverse, weights=values, minlength=len(unique))
    cluster_counts = np.bincount(inverse, minlength=len(unique))
    draws = rng.integers(0, len(unique), size=(samples, len(unique)))
    bootstrap_means = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "num_clusters": int(len(unique)),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.patch_h5, "r") as patch:
        completed = np.asarray(patch["completed"][:], dtype=bool)
        if not completed.all():
            raise ValueError(f"Patch cache is incomplete: {completed.sum()}/{len(completed)}.")
        natural_velocity = np.asarray(patch["natural_velocity"][:], dtype=np.float32)
        patched_velocity = np.asarray(patch["patched_velocity"][:], dtype=np.float32)
        layers = json.loads(decode_attr(patch.attrs["patch_layers_json"]))
        alphas = json.loads(decode_attr(patch.attrs["alphas_json"]))
        base_prompt_index = int(patch.attrs["base_prompt_index"])
        source_prompt_index = int(patch.attrs["source_prompt_index"])
        audit = {
            "alpha_zero_base_velocity_max_abs_difference": float(
                patch.attrs.get("alpha_zero_base_velocity_max_abs_difference", np.nan)
            ),
            "untouched_velocity_max_abs_difference": float(
                np.asarray(patch["untouched_velocity_max_abs_difference"][:]).max()
            ),
            "natural_prompt_velocity_mean_abs_difference": float(
                np.abs(natural_velocity[:, 1] - natural_velocity[:, 0]).mean()
            ),
        }

    with h5py.File(args.input_h5, "r") as inputs:
        num_samples = len(natural_velocity)
        if len(inputs["layout_splits"]) < num_samples:
            raise ValueError("Input has fewer samples than the patch cache.")
        splits = decode_strings(inputs["layout_splits"][:num_samples])
        task_ids = np.asarray(inputs["task_ids"][:num_samples], dtype=np.int64)
        episode_ids = np.asarray(inputs["episode_ids"][:num_samples], dtype=np.int64)
    cluster_ids = np.asarray(
        [f"{task_id}:{episode_id}" for task_id, episode_id in zip(task_ids, episode_ids, strict=True)]
    )

    metrics = compute_metrics(
        natural_velocity,
        patched_velocity,
        base_prompt_index=base_prompt_index,
        source_prompt_index=source_prompt_index,
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    split_summaries: dict[str, object] = {}
    for split_name in ("train", "valid", "test"):
        selected_samples = splits == split_name
        if not selected_samples.any():
            raise ValueError(f"Split {split_name!r} is empty.")
        split_summary: dict[str, object] = {}
        for layer_offset, layer in enumerate(layers):
            layer_summary: dict[str, object] = {}
            for alpha_offset, alpha in enumerate(alphas):
                config_summary = {}
                for metric_name in METRIC_NAMES:
                    values = metrics[metric_name][selected_samples, layer_offset, alpha_offset]
                    metric_summary = summarize_values(values)
                    if split_name == "test":
                        metric_summary["episode_bootstrap"] = cluster_bootstrap_mean_ci(
                            values,
                            cluster_ids[selected_samples],
                            samples=args.bootstrap_samples,
                            rng=rng,
                        )
                    config_summary[metric_name] = metric_summary
                    rows.append(
                        {
                            "split": split_name,
                            "layer": layer,
                            "alpha": alpha,
                            "metric": metric_name,
                            **{key: value for key, value in metric_summary.items() if key != "episode_bootstrap"},
                        }
                    )
                layer_summary[str(alpha)] = config_summary
            split_summary[str(layer)] = layer_summary
        split_summaries[split_name] = split_summary

    excluded_layers = set(parse_int_list(args.selection_exclude_layers))
    candidate_layer_offsets = [
        offset for offset, layer in enumerate(layers) if layer not in excluded_layers
    ]
    if not candidate_layer_offsets:
        raise ValueError("Every patch layer was excluded from validation selection.")
    valid_error = metrics["normalized_source_error"][splits == "valid"].mean(axis=0)
    candidate_error = valid_error[candidate_layer_offsets]
    candidate_selected_offset = np.unravel_index(
        np.argmin(candidate_error), candidate_error.shape
    )
    selected_offset = (
        candidate_layer_offsets[candidate_selected_offset[0]],
        candidate_selected_offset[1],
    )
    selected_layer = layers[selected_offset[0]]
    selected_alpha = alphas[selected_offset[1]]
    selected_test = {
        metric_name: split_summaries["test"][str(selected_layer)][str(selected_alpha)][metric_name]
        for metric_name in METRIC_NAMES
    }
    summary = {
        "patch_h5": str(args.patch_h5),
        "input_h5": str(args.input_h5),
        "protocol": (
            "layer and alpha selected by minimum validation normalized source error; "
            "test reported once"
        ),
        "base_prompt_index": base_prompt_index,
        "source_prompt_index": source_prompt_index,
        "layers": layers,
        "alphas": alphas,
        "selection_excluded_layers": sorted(excluded_layers),
        "split_counts": {
            split_name: int(np.sum(splits == split_name))
            for split_name in ("train", "valid", "test")
        },
        "audit": audit,
        "selected": {
            "layer": selected_layer,
            "alpha": selected_alpha,
            "validation_normalized_source_error_mean": float(valid_error[selected_offset]),
            "test": selected_test,
        },
        "splits": split_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "splits"}, indent=2))
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
