#!/usr/bin/env python3
"""Summarize the matched ten-task OpenVLA FFN steering pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("baseline", "up_10", "random_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def load_rows(input_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for path in sorted(input_root.glob("task-*-state-*-job-*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition_map = {item["condition"]: item for item in payload["conditions"]}
        missing = set(CONDITIONS) - set(condition_map)
        if missing:
            raise ValueError(f"{path} is missing conditions: {sorted(missing)}")
        for condition in CONDITIONS:
            item = condition_map[condition]
            rows.append(
                {
                    "task_id": int(payload["task_id"]),
                    "init_state_id": int(payload["init_state_id"]),
                    "task": item["task"],
                    "condition": condition,
                    "success": int(item["success"]),
                    "action_steps": int(item["action_steps"]),
                    "mean_step_delta_x": float(item["mean_step_delta_xyz"][0]),
                    "mean_step_delta_y": float(item["mean_step_delta_xyz"][1]),
                    "mean_step_delta_z": float(item["mean_step_delta_xyz"][2]),
                    "net_delta_y": float(item["net_delta_xyz"][1]),
                    "path_length": float(item["path_length"]),
                }
            )
        audits.append(
            {
                "task_id": int(payload["task_id"]),
                "init_state_id": int(payload["init_state_id"]),
                **payload["audits"],
            }
        )
    return rows, audits


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, audits = load_rows(args.input_root)
    task_ids = sorted({int(row["task_id"]) for row in rows})
    if len(task_ids) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} tasks, found {task_ids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "audits.csv", audits)

    by_task: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_task.setdefault(int(row["task_id"]), {})[str(row["condition"])] = row

    effect_rows = []
    for task_id in task_ids:
        task_rows = by_task[task_id]
        baseline_y = float(task_rows["baseline"]["mean_step_delta_y"])
        up_y = float(task_rows["up_10"]["mean_step_delta_y"])
        random_y = float(task_rows["random_10"]["mean_step_delta_y"])
        effect_rows.append(
            {
                "task_id": task_id,
                "task": task_rows["baseline"]["task"],
                "up_minus_baseline_mean_step_delta_y": up_y - baseline_y,
                "up_minus_random_mean_step_delta_y": up_y - random_y,
                "up_minus_baseline_path_length": float(task_rows["up_10"]["path_length"])
                - float(task_rows["baseline"]["path_length"]),
                "up_minus_random_path_length": float(task_rows["up_10"]["path_length"])
                - float(task_rows["random_10"]["path_length"]),
            }
        )
    write_csv(args.output_dir / "paired_effects.csv", effect_rows)

    primary = {}
    for name in (
        "up_minus_baseline_mean_step_delta_y",
        "up_minus_random_mean_step_delta_y",
    ):
        values = np.asarray([float(row[name]) for row in effect_rows])
        ci_low, ci_high = bootstrap_mean_ci(values, args.bootstrap_samples, args.seed)
        primary[name] = {
            "mean": float(values.mean()),
            "ci95": [ci_low, ci_high],
            "per_task": values.tolist(),
        }

    implementation_valid = all(
        float(audit["up_10_initial_eef_max_abs_diff"]) == 0.0
        and float(audit["random_10_initial_eef_max_abs_diff"]) == 0.0
        and bool(audit["up_10_all_selected_layers_fired"])
        and bool(audit["random_10_all_selected_layers_fired"])
        and int(audit["up_10_num_differing_raw_actions"]) > 0
        for audit in audits
    )
    up_vs_baseline = primary["up_minus_baseline_mean_step_delta_y"]
    up_vs_random = primary["up_minus_random_mean_step_delta_y"]
    pilot_gate_passed = bool(
        implementation_valid
        and up_vs_baseline["mean"] > 0.0
        and up_vs_random["ci95"][0] > 0.0
    )
    summary = {
        "num_tasks": len(task_ids),
        "num_conditions": len(CONDITIONS),
        "implementation_valid": implementation_valid,
        "pilot_gate": {
            "passed": pilot_gate_passed,
            "rule": "mean(up-baseline)>0 and 95% task-bootstrap CI(up-random)>0",
        },
        "primary": primary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    y_by_condition = {
        condition: np.asarray(
            [float(by_task[task_id][condition]["mean_step_delta_y"]) for task_id in task_ids]
        )
        for condition in CONDITIONS
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(CONDITIONS))
    for task_index, task_id in enumerate(task_ids):
        axes[0].plot(
            x,
            [y_by_condition[condition][task_index] for condition in CONDITIONS],
            color="#9ca3af",
            alpha=0.55,
            linewidth=1,
        )
    axes[0].scatter(
        x,
        [y_by_condition[condition].mean() for condition in CONDITIONS],
        color=["#374151", "#0f766e", "#b45309"],
        s=55,
        zorder=3,
    )
    axes[0].set_xticks(x, ["Baseline", "Official up_10", "Matched random"])
    axes[0].set_ylabel("Mean signed EEF Y displacement per step (m)")
    axes[0].set_title("Matched trajectories across LIBERO-Long tasks")
    axes[0].grid(axis="y", alpha=0.25)

    effect_names = (
        "up_minus_baseline_mean_step_delta_y",
        "up_minus_random_mean_step_delta_y",
    )
    colors = ("#2563eb", "#0f766e")
    for index, (name, color) in enumerate(zip(effect_names, colors)):
        values = np.asarray([float(row[name]) for row in effect_rows])
        jitter = np.linspace(-0.12, 0.12, len(values))
        axes[1].scatter(np.full(len(values), index) + jitter, values, color=color, alpha=0.75)
        mean = primary[name]["mean"]
        low, high = primary[name]["ci95"]
        axes[1].errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black", capsize=5)
    axes[1].axhline(0.0, color="#4b5563", linewidth=1)
    axes[1].set_xticks([0, 1], ["up_10 - baseline", "up_10 - random"])
    axes[1].set_ylabel("Paired mean-step Y effect (m)")
    axes[1].set_title("Preregistered directional effects")
    axes[1].grid(axis="y", alpha=0.25)

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"pilot_directional_effects.{suffix}", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
