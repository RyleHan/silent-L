#!/usr/bin/env python3
"""Summarize selected pi0.5 patch controls with paired episode bootstraps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from summarize_pi05_paired_patching import (
    METRIC_NAMES,
    cluster_bootstrap_mean_ci,
    compute_metrics,
    decode_attr,
    decode_strings,
    summarize_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls-h5", type=Path, required=True)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.controls_h5, "r") as controls:
        completed = np.asarray(controls["completed"][:], dtype=bool)
        if not completed.all():
            raise ValueError(f"Control cache is incomplete: {completed.sum()}/{len(completed)}.")
        natural_velocity = np.asarray(controls["natural_velocity"][:], dtype=np.float32)
        patched_velocity = np.asarray(controls["patched_velocity"][:], dtype=np.float32)
        norm_ratios = np.asarray(controls["patch_delta_norm_ratio"][:], dtype=np.float32)
        mode_names = json.loads(decode_attr(controls.attrs["mode_names_json"]))
        base_prompt_index = int(controls.attrs["base_prompt_index"])
        source_prompt_index = int(controls.attrs["source_prompt_index"])
        metadata = {
            "layer": int(controls.attrs["layer"]),
            "timestep": float(controls.attrs["timestep"]),
            "projection_basis": json.loads(
                decode_attr(controls.attrs["projection_basis_json"])
            ),
            "untouched_velocity_max_abs_difference": float(
                np.asarray(controls["untouched_velocity_max_abs_difference"][:]).max()
            ),
        }

    with h5py.File(args.input_h5, "r") as inputs:
        num_samples = len(natural_velocity)
        splits = decode_strings(inputs["layout_splits"][:num_samples])
        task_ids = np.asarray(inputs["task_ids"][:num_samples], dtype=np.int64)
        episode_ids = np.asarray(inputs["episode_ids"][:num_samples], dtype=np.int64)
    cluster_ids = np.asarray(
        [f"{task_id}:{episode_id}" for task_id, episode_id in zip(task_ids, episode_ids, strict=True)]
    )
    metrics = compute_metrics(
        natural_velocity,
        patched_velocity[:, :, None],
        base_prompt_index=base_prompt_index,
        source_prompt_index=source_prompt_index,
    )
    metrics = {name: values[:, :, 0] for name, values in metrics.items()}
    rng = np.random.default_rng(args.seed)
    rows = []
    split_summaries = {}
    for split_name in ("train", "valid", "test"):
        selected = splits == split_name
        split_summary = {}
        for mode_index, mode in enumerate(mode_names):
            mode_summary = {
                "patch_delta_norm_ratio": summarize_values(norm_ratios[selected, mode_index])
            }
            for metric_name in METRIC_NAMES:
                values = metrics[metric_name][selected, mode_index]
                metric_summary = summarize_values(values)
                if split_name == "test":
                    metric_summary["episode_bootstrap"] = cluster_bootstrap_mean_ci(
                        values,
                        cluster_ids[selected],
                        samples=args.bootstrap_samples,
                        rng=rng,
                    )
                mode_summary[metric_name] = metric_summary
                rows.append(
                    {
                        "split": split_name,
                        "mode": mode,
                        "metric": metric_name,
                        **{
                            key: value
                            for key, value in metric_summary.items()
                            if key != "episode_bootstrap"
                        },
                    }
                )
            split_summary[mode] = mode_summary
        split_summaries[split_name] = split_summary

    paired_index = mode_names.index("paired")
    test = splits == "test"
    test_differences = {}
    for mode_index, mode in enumerate(mode_names):
        if mode == "paired":
            continue
        mode_differences = {}
        for metric_name in ("action_recovery", "normalized_source_error"):
            difference = (
                metrics[metric_name][test, paired_index]
                - metrics[metric_name][test, mode_index]
            )
            mode_differences[metric_name] = cluster_bootstrap_mean_ci(
                difference,
                cluster_ids[test],
                samples=args.bootstrap_samples,
                rng=rng,
            )
        test_differences[f"paired_minus_{mode}"] = mode_differences

    summary = {
        "controls_h5": str(args.controls_h5),
        "input_h5": str(args.input_h5),
        "protocol": "selected block 13 and alpha=1 fixed before semantic control evaluation",
        "base_prompt_index": base_prompt_index,
        "source_prompt_index": source_prompt_index,
        "mode_names": mode_names,
        "metadata": metadata,
        "split_counts": {
            split_name: int(np.sum(splits == split_name))
            for split_name in ("train", "valid", "test")
        },
        "test_paired_differences": test_differences,
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
