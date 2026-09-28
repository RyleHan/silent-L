#!/usr/bin/env python3
"""Combine locked Stage 5 baselines with Stage 7 null-prompt rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "success": ("Success rate", 1.0),
    "target_ever_grasped": ("Target grasp rate", 1.0),
    "target_ever_in_basket": ("Target in-basket rate", 1.0),
    "alternative_ever_grasped": ("Alternative grasp rate", -1.0),
    "target_min_eef_distance": ("Target minimum EEF distance (m)", -1.0),
    "target_max_displacement": ("Target maximum displacement (m)", 1.0),
}
CONDITION_SOURCE = {"correct": "correct", "swapped": "mismatch", "null": "null"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5-root", type=Path, required=True)
    parser.add_argument("--null-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bootstrap_mean(values: np.ndarray, *, samples: int, rng: np.random.Generator) -> dict[str, float]:
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "protocol": {
            "conditions": ["correct", "swapped", "null"],
            "correct_and_swapped_source": "locked Stage 5 unmodified-policy rollouts",
            "null_source": "Stage 7 target-underspecified unmodified-policy rollouts",
            "matching": "same held-out initial-state IDs and deterministic noise protocol",
            "uncertainty": "95% paired initial-state bootstrap confidence intervals",
        },
        "metric_definitions": {
            name: {"label": label, "favorable_sign": sign}
            for name, (label, sign) in METRICS.items()
        },
        "tasks": {},
    }
    csv_rows = []

    for task_id in (0, 1):
        stage5_records = load_records(args.stage5_root / f"task-{task_id}" / "episodes.jsonl")
        null_records = load_records(args.null_root / f"task-{task_id}" / "episodes.jsonl")
        source_records = {"stage5": stage5_records, "null": null_records}
        indexed: dict[str, dict[int, dict[str, object]]] = {}
        for output_condition, source_condition in CONDITION_SOURCE.items():
            source = source_records["null" if output_condition == "null" else "stage5"]
            selected = [record for record in source if record["condition"] == source_condition]
            indexed[output_condition] = {int(record["init_state_id"]): record for record in selected}
        init_ids = sorted(indexed["correct"])
        for condition in CONDITION_SOURCE:
            if sorted(indexed[condition]) != init_ids:
                raise ValueError(f"Task {task_id} initial states differ for {condition}.")

        task_result: dict[str, object] = {
            "num_paired_initial_states": len(init_ids),
            "init_state_ids": init_ids,
            "conditions": {},
            "contrasts": {},
        }
        condition_arrays: dict[str, dict[str, np.ndarray]] = {}
        for condition in CONDITION_SOURCE:
            condition_arrays[condition] = {}
            condition_result = {}
            for metric, (label, _sign) in METRICS.items():
                values = np.asarray(
                    [indexed[condition][init_id][metric] for init_id in init_ids],
                    dtype=np.float64,
                )
                condition_arrays[condition][metric] = values
                estimate = bootstrap_mean(values, samples=args.bootstrap_samples, rng=rng)
                condition_result[metric] = estimate
                csv_rows.append(
                    {
                        "task_id": task_id,
                        "result_type": "condition",
                        "condition_or_contrast": condition,
                        "metric": metric,
                        "label": label,
                        **estimate,
                    }
                )
            task_result["conditions"][condition] = condition_result

        for contrast_name, first, second in (
            ("null_minus_correct", "null", "correct"),
            ("null_minus_swapped", "null", "swapped"),
        ):
            contrast_result = {}
            for metric, (label, favorable_sign) in METRICS.items():
                favorable_difference = favorable_sign * (
                    condition_arrays[first][metric] - condition_arrays[second][metric]
                )
                estimate = bootstrap_mean(
                    favorable_difference, samples=args.bootstrap_samples, rng=rng
                )
                contrast_result[metric] = estimate
                csv_rows.append(
                    {
                        "task_id": task_id,
                        "result_type": "favorable_contrast",
                        "condition_or_contrast": contrast_name,
                        "metric": metric,
                        "label": label,
                        **estimate,
                    }
                )
            task_result["contrasts"][contrast_name] = contrast_result
        summary["tasks"][str(task_id)] = task_result

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    colors = {"correct": "#277DA1", "swapped": "#D1495B", "null": "#F4A261"}
    conditions = list(CONDITION_SOURCE)
    x = np.arange(2)
    width = 0.23
    for offset, condition in enumerate(conditions):
        success = [
            summary["tasks"][str(task)]["conditions"][condition]["success"] for task in (0, 1)
        ]
        means = [value["mean"] for value in success]
        errors = np.asarray(
            [[value["mean"] - value["ci95_low"] for value in success],
             [value["ci95_high"] - value["mean"] for value in success]]
        )
        axes[0].bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=condition,
            color=colors[condition],
        )
    axes[0].set_xticks(x, ["Task A target", "Task B target"])
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Held-out closed-loop target selection")
    axes[0].legend(frameon=False)

    metrics = ["target_ever_grasped", "alternative_ever_grasped"]
    labels = ["Target grasp", "Alternative grasp"]
    task_offsets = np.arange(len(metrics))
    width = 0.12
    bar_index = 0
    for task_id in (0, 1):
        for condition in conditions:
            values = [
                summary["tasks"][str(task_id)]["conditions"][condition][metric]["mean"]
                for metric in metrics
            ]
            position = task_offsets + (bar_index - 2.5) * width
            axes[1].bar(
                position,
                values,
                width,
                color=colors[condition],
                alpha=1.0 if task_id == 0 else 0.55,
                label=f"task {task_id} / {condition}",
            )
            bar_index += 1
    axes[1].set_xticks(task_offsets, labels)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("Episode rate")
    axes[1].set_title("Which object is physically selected")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"closed_loop_language_dependence.{extension}", dpi=220)
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
