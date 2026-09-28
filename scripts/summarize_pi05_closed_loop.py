#!/usr/bin/env python3
"""Summarize the fixed pi0.5 closed-loop patching experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("correct", "mismatch", "paired", "random")
METRICS = {
    "success": {"label": "Success rate", "favorable_sign": 1.0},
    "target_ever_grasped": {"label": "Target grasp rate", "favorable_sign": 1.0},
    "target_ever_in_basket": {"label": "Target in-basket rate", "favorable_sign": 1.0},
    "alternative_ever_grasped": {
        "label": "Alternative grasp rate",
        "favorable_sign": -1.0,
    },
    "target_min_eef_distance": {
        "label": "Target minimum EEF distance",
        "favorable_sign": -1.0,
    },
    "target_max_displacement": {
        "label": "Target maximum displacement",
        "favorable_sign": 1.0,
    },
}
COMPARISONS = {
    "paired_minus_mismatch": ("paired", "mismatch"),
    "paired_minus_random": ("paired", "random"),
    "paired_minus_correct": ("paired", "correct"),
    "correct_minus_mismatch": ("correct", "mismatch"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-0-dir", type=Path, required=True)
    parser.add_argument("--task-1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"No episode records found in {path}.")
    return records


def validate_and_index(
    records: list[dict[str, object]], expected_task_id: int
) -> dict[str, dict[int, dict[str, object]]]:
    indexed = {condition: {} for condition in CONDITIONS}
    for record in records:
        if int(record["task_id"]) != expected_task_id:
            raise ValueError(f"Unexpected task ID in task {expected_task_id} records: {record}")
        condition = str(record["condition"])
        init_state_id = int(record["init_state_id"])
        if condition not in indexed:
            raise ValueError(f"Unexpected condition: {condition}")
        if init_state_id in indexed[condition]:
            raise ValueError(f"Duplicate task/condition/init state: {expected_task_id}/{condition}/{init_state_id}")
        indexed[condition][init_state_id] = record

    expected_ids = set(indexed[CONDITIONS[0]])
    for condition in CONDITIONS:
        if set(indexed[condition]) != expected_ids:
            raise ValueError(f"Condition {condition} does not have the same paired initial states.")
    return indexed


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def bootstrap_mean(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, dict[str, float]]:
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), percentile_interval(means)


def summarize_task(
    indexed: dict[str, dict[int, dict[str, object]]],
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    init_state_ids = np.asarray(sorted(indexed[CONDITIONS[0]]), dtype=np.int64)
    values = {
        condition: {
            metric: np.asarray(
                [indexed[condition][int(state)][metric] for state in init_state_ids],
                dtype=np.float64,
            )
            for metric in METRICS
        }
        for condition in CONDITIONS
    }

    condition_summaries = {}
    for condition in CONDITIONS:
        condition_summaries[condition] = {}
        for metric in METRICS:
            mean, interval = bootstrap_mean(values[condition][metric], samples, rng)
            condition_summaries[condition][metric] = {"mean": mean, **interval}

    comparisons = {}
    for name, (left, right) in COMPARISONS.items():
        comparisons[name] = {}
        for metric, spec in METRICS.items():
            favorable_difference = spec["favorable_sign"] * (
                values[left][metric] - values[right][metric]
            )
            mean, interval = bootstrap_mean(favorable_difference, samples, rng)
            comparisons[name][metric] = {
                "mean_favorable_difference": mean,
                **interval,
            }

    return {
        "num_paired_initial_states": int(len(init_state_ids)),
        "init_state_ids": init_state_ids.tolist(),
        "conditions": condition_summaries,
        "comparisons": comparisons,
    }


def summarize_stratified(
    task_summaries: list[dict[str, object]],
    task_indexes: list[dict[str, dict[int, dict[str, object]]]],
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    del task_summaries
    output = {}
    for name, (left, right) in COMPARISONS.items():
        output[name] = {}
        for metric, spec in METRICS.items():
            task_differences = []
            for indexed in task_indexes:
                ids = sorted(indexed[CONDITIONS[0]])
                task_differences.append(
                    spec["favorable_sign"]
                    * np.asarray(
                        [
                            float(indexed[left][state][metric])
                            - float(indexed[right][state][metric])
                            for state in ids
                        ]
                    )
                )
            point = float(np.mean([difference.mean() for difference in task_differences]))
            bootstrap = np.empty(samples, dtype=np.float64)
            for sample_index in range(samples):
                task_means = []
                for difference in task_differences:
                    draw = rng.integers(0, len(difference), size=len(difference))
                    task_means.append(float(difference[draw].mean()))
                bootstrap[sample_index] = np.mean(task_means)
            output[name][metric] = {
                "mean_favorable_difference": point,
                **percentile_interval(bootstrap),
            }
    return output


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    task_dirs = (args.task_0_dir, args.task_1_dir)
    task_indexes = [
        validate_and_index(load_jsonl(path / "episodes.jsonl"), task_id)
        for task_id, path in enumerate(task_dirs)
    ]
    task_summaries = [
        summarize_task(indexed, args.bootstrap_samples, rng) for indexed in task_indexes
    ]
    source_summaries = [
        json.loads((path / "summary.json").read_text(encoding="utf-8")) for path in task_dirs
    ]
    for task_id, source in enumerate(source_summaries):
        audits = source.get("sampler_audits", {})
        if not audits or any(float(error) != 0.0 for error in audits.values()):
            raise ValueError(f"Task {task_id} does not have exact sampler audits: {audits}")

    summary = {
        "protocol": {
            "selection": "block 13 and alpha=1 fixed by offline validation before closed-loop evaluation",
            "primary_comparisons": ["paired_minus_mismatch", "paired_minus_random"],
            "uncertainty": "95% paired initial-state bootstrap confidence intervals",
            "positive_favorable_difference": True,
        },
        "metric_definitions": METRICS,
        "tasks": {str(task_id): task for task_id, task in enumerate(task_summaries)},
        "stratified_equal_task_average": summarize_stratified(
            task_summaries, task_indexes, args.bootstrap_samples, rng
        ),
        "sampler_audits": {
            str(task_id): source["sampler_audits"]
            for task_id, source in enumerate(source_summaries)
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    rows = []
    for task_id, task in summary["tasks"].items():
        for condition, metrics in task["conditions"].items():
            for metric, values in metrics.items():
                rows.append(
                    {
                        "scope": f"task_{task_id}",
                        "kind": "condition",
                        "name": condition,
                        "metric": metric,
                        **values,
                    }
                )
        for comparison, metrics in task["comparisons"].items():
            for metric, values in metrics.items():
                rows.append(
                    {
                        "scope": f"task_{task_id}",
                        "kind": "comparison",
                        "name": comparison,
                        "metric": metric,
                        "mean": values["mean_favorable_difference"],
                        "ci95_low": values["ci95_low"],
                        "ci95_high": values["ci95_high"],
                    }
                )
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["stratified_equal_task_average"], indent=2))
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
