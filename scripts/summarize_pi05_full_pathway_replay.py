#!/usr/bin/env python3
"""Summarize the fixed full-expert replay positive control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITION_SOURCE = {
    "correct": ("stage5", "correct"),
    "mismatch": ("stage5", "mismatch"),
    "single_block": ("stage5", "paired"),
    "full_pathway": ("replay", "pathway"),
}
METRICS = {
    "success": ("Success", 1.0),
    "target_ever_grasped": ("Target grasp", 1.0),
    "target_ever_in_basket": ("Target in basket", 1.0),
    "alternative_ever_grasped": ("Alternative grasp", -1.0),
    "target_min_eef_distance": ("Target minimum EEF distance", -1.0),
    "target_max_displacement": ("Target maximum displacement", 1.0),
}
TRAJECTORY_AUDIT_METRICS = (
    "success",
    "target_ever_grasped",
    "target_ever_in_basket",
    "alternative_ever_grasped",
    "target_min_eef_distance",
    "alternative_min_eef_distance",
    "target_max_displacement",
    "alternative_max_displacement",
    "executed_steps",
    "num_policy_queries",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage5-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--exact-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bootstrap_mean(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> dict[str, float]:
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.exact_tolerance < 0:
        raise ValueError("Bootstrap samples must be positive and tolerance nonnegative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "protocol": {
            "purpose": "operator positive control for Stage 5 single-block replacement",
            "base": "mismatched prompt",
            "source": "correct prompt on the same observation, robot state, and noise",
            "intervention": (
                "overwrite all 18 action-expert block outputs at every denoising step"
            ),
            "selection": "no layer or strength sweep",
            "exact_tolerance": args.exact_tolerance,
            "uncertainty": "95% paired initial-state bootstrap confidence intervals",
        },
        "metric_definitions": {
            name: {"label": label, "favorable_sign": sign}
            for name, (label, sign) in METRICS.items()
        },
        "tasks": {},
    }
    csv_rows = []
    all_action_errors = []
    all_velocity_errors = []
    all_hook_mins = []
    all_hook_maxes = []
    all_source_official_errors = []
    all_replay_official_errors = []

    for task_id in (0, 1):
        stage5_records = load_records(args.stage5_root / f"task-{task_id}" / "episodes.jsonl")
        replay_records = load_records(args.replay_root / f"task-{task_id}" / "episodes.jsonl")
        source_records = {"stage5": stage5_records, "replay": replay_records}
        indexed = {}
        for output_condition, (source_name, source_condition) in CONDITION_SOURCE.items():
            selected = [
                record
                for record in source_records[source_name]
                if record["condition"] == source_condition
            ]
            indexed[output_condition] = {
                int(record["init_state_id"]): record for record in selected
            }
        init_ids = sorted(indexed["correct"])
        for condition in CONDITION_SOURCE:
            if sorted(indexed[condition]) != init_ids:
                raise ValueError(f"Task {task_id} initial states differ for {condition}.")

        condition_arrays = {}
        condition_results = {}
        for condition in CONDITION_SOURCE:
            condition_arrays[condition] = {}
            condition_results[condition] = {}
            for metric, (label, _sign) in METRICS.items():
                values = np.asarray(
                    [indexed[condition][state][metric] for state in init_ids],
                    dtype=np.float64,
                )
                condition_arrays[condition][metric] = values
                estimate = bootstrap_mean(values, samples=args.bootstrap_samples, rng=rng)
                condition_results[condition][metric] = estimate
                csv_rows.append(
                    {
                        "task_id": task_id,
                        "kind": "condition",
                        "name": condition,
                        "metric": metric,
                        "label": label,
                        **estimate,
                    }
                )

        contrasts = {}
        for contrast_name, first, second in (
            ("full_pathway_minus_single_block", "full_pathway", "single_block"),
            ("full_pathway_minus_mismatch", "full_pathway", "mismatch"),
            ("full_pathway_minus_correct", "full_pathway", "correct"),
        ):
            contrasts[contrast_name] = {}
            for metric, (label, favorable_sign) in METRICS.items():
                differences = favorable_sign * (
                    condition_arrays[first][metric] - condition_arrays[second][metric]
                )
                estimate = bootstrap_mean(
                    differences, samples=args.bootstrap_samples, rng=rng
                )
                contrasts[contrast_name][metric] = estimate
                csv_rows.append(
                    {
                        "task_id": task_id,
                        "kind": "favorable_contrast",
                        "name": contrast_name,
                        "metric": metric,
                        "label": label,
                        **estimate,
                    }
                )

        trajectory_agreement = {}
        for metric in TRAJECTORY_AUDIT_METRICS:
            correct = np.asarray(
                [indexed["correct"][state][metric] for state in init_ids], dtype=np.float64
            )
            replay = np.asarray(
                [indexed["full_pathway"][state][metric] for state in init_ids],
                dtype=np.float64,
            )
            trajectory_agreement[metric] = {
                "max_abs_difference": float(np.abs(replay - correct).max()),
                "exact_match_fraction": float((replay == correct).mean()),
            }

        task_replay_records = [indexed["full_pathway"][state] for state in init_ids]
        action_errors = np.asarray(
            [record["pathway_action_max_abs_difference"] for record in task_replay_records],
            dtype=np.float64,
        )
        velocity_errors = np.asarray(
            [record["pathway_velocity_max_abs_difference"] for record in task_replay_records],
            dtype=np.float64,
        )
        hook_mins = np.asarray(
            [record["pathway_layer_hook_count_min"] for record in task_replay_records],
            dtype=np.int64,
        )
        hook_maxes = np.asarray(
            [record["pathway_layer_hook_count_max"] for record in task_replay_records],
            dtype=np.int64,
        )
        replay_source_summary = json.loads(
            (args.replay_root / f"task-{task_id}" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        sampler_audit = replay_source_summary["sampler_audits"]["pathway"]
        source_official_error = float(
            sampler_audit["source_vs_official_max_abs_difference"]
        )
        replay_official_error = float(
            sampler_audit["replay_vs_official_source_max_abs_difference"]
        )
        all_action_errors.extend(action_errors.tolist())
        all_velocity_errors.extend(velocity_errors.tolist())
        all_hook_mins.extend(hook_mins.tolist())
        all_hook_maxes.extend(hook_maxes.tolist())
        all_source_official_errors.append(source_official_error)
        all_replay_official_errors.append(replay_official_error)

        summary["tasks"][str(task_id)] = {
            "num_paired_initial_states": len(init_ids),
            "init_state_ids": init_ids,
            "conditions": condition_results,
            "contrasts": contrasts,
            "trajectory_agreement_with_correct": trajectory_agreement,
            "operator_fidelity": {
                "action_max_abs_difference": float(action_errors.max()),
                "velocity_max_abs_difference": float(velocity_errors.max()),
                "layer_hook_count_min": int(hook_mins.min()),
                "layer_hook_count_max": int(hook_maxes.max()),
                "source_vs_official_max_abs_difference": source_official_error,
                "replay_vs_official_source_max_abs_difference": replay_official_error,
            },
        }

    global_fidelity = {
        "action_max_abs_difference": float(max(all_action_errors)),
        "velocity_max_abs_difference": float(max(all_velocity_errors)),
        "layer_hook_count_min": int(min(all_hook_mins)),
        "layer_hook_count_max": int(max(all_hook_maxes)),
        "source_vs_official_max_abs_difference": float(max(all_source_official_errors)),
        "replay_vs_official_source_max_abs_difference": float(
            max(all_replay_official_errors)
        ),
    }
    expected_hook_count = 10
    exact_replay = (
        global_fidelity["action_max_abs_difference"] <= args.exact_tolerance
        and global_fidelity["velocity_max_abs_difference"] <= args.exact_tolerance
        and global_fidelity["source_vs_official_max_abs_difference"] == 0.0
        and global_fidelity["replay_vs_official_source_max_abs_difference"]
        <= args.exact_tolerance
    )
    hooks_complete = (
        global_fidelity["layer_hook_count_min"] == expected_hook_count
        and global_fidelity["layer_hook_count_max"] == expected_hook_count
    )
    behavior_matches_correct = all(
        summary["tasks"][str(task_id)]["conditions"]["full_pathway"]["success"][
            "mean"
        ]
        == summary["tasks"][str(task_id)]["conditions"]["correct"]["success"]["mean"]
        for task_id in (0, 1)
    )
    summary["global_operator_fidelity"] = global_fidelity
    summary["positive_control_gate"] = {
        "passed": bool(exact_replay and hooks_complete and behavior_matches_correct),
        "exact_action_and_velocity_replay": bool(exact_replay),
        "all_18_hooks_called_once_per_denoising_step": bool(hooks_complete),
        "full_pathway_success_matches_correct_prompt": bool(behavior_matches_correct),
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    conditions = list(CONDITION_SOURCE)
    colors = ["#277DA1", "#8D99AE", "#D1495B", "#43AA8B"]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True, constrained_layout=True)
    for task_id, axis in enumerate(axes):
        x = np.arange(len(conditions))
        means = [
            summary["tasks"][str(task_id)]["conditions"][condition]["success"]["mean"]
            for condition in conditions
        ]
        low = [
            means[index]
            - summary["tasks"][str(task_id)]["conditions"][condition]["success"][
                "ci95_low"
            ]
            for index, condition in enumerate(conditions)
        ]
        high = [
            summary["tasks"][str(task_id)]["conditions"][condition]["success"][
                "ci95_high"
            ]
            - means[index]
            for index, condition in enumerate(conditions)
        ]
        axis.bar(x, means, yerr=np.asarray([low, high]), capsize=3, color=colors)
        axis.set_xticks(x, ["Correct", "Mismatch", "Single block", "Full pathway"], rotation=20)
        axis.set_ylim(0.0, 1.08)
        axis.set_title("Target A" if task_id == 0 else "Target B")
        axis.grid(axis="y", color="#D8DADD", linewidth=0.7)
    axes[0].set_ylabel("Closed-loop success rate")
    figure.suptitle("Full expert-pathway replay positive control")
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"full_pathway_positive_control.{extension}", dpi=220)
    plt.close(figure)
    print(json.dumps(summary["positive_control_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
