#!/usr/bin/env python3
"""Plot the fixed pi0.5 closed-loop patching results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("correct", "mismatch", "paired", "random")
COLORS = {
    "correct": "#3A6EA5",
    "mismatch": "#8B8C89",
    "paired": "#D1495B",
    "random": "#E9C46A",
}
OUTCOMES = (
    ("success", "Task success"),
    ("target_ever_grasped", "Target grasped"),
    ("target_ever_in_basket", "Target in basket"),
)
METRIC_COLORS = ("#3A6EA5", "#E07A1F", "#2A9D31")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(summary: dict[str, object], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharey=True)
    task_names = ("A target: alphabet soup", "B target: cream cheese")
    positions = np.arange(len(CONDITIONS))
    for task_id, task_name in enumerate(task_names):
        task = summary["tasks"][str(task_id)]
        for outcome_index, (metric, title) in enumerate(OUTCOMES):
            axis = axes[task_id, outcome_index]
            means = []
            low = []
            high = []
            for condition in CONDITIONS:
                values = task["conditions"][condition][metric]
                means.append(values["mean"])
                low.append(values["mean"] - values["ci95_low"])
                high.append(values["ci95_high"] - values["mean"])
            axis.bar(
                positions,
                means,
                yerr=np.asarray([low, high]),
                capsize=3,
                color=[COLORS[condition] for condition in CONDITIONS],
                edgecolor="#202124",
                linewidth=0.6,
            )
            axis.scatter(
                positions,
                means,
                s=18,
                color="#202124",
                zorder=3,
            )
            axis.set_ylim(-0.03, 1.05)
            axis.set_xticks(
                positions,
                [condition.capitalize() for condition in CONDITIONS],
                rotation=24,
                ha="right",
            )
            axis.grid(axis="y", color="#D8DADD", linewidth=0.7)
            if task_id == 0:
                axis.set_title(title)
            if outcome_index == 0:
                axis.set_ylabel(f"{task_name}\nEpisode rate")
    fig.suptitle("Closed-loop target switching with natural block-13 patching", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "closed_loop_outcomes")


def plot_primary_effects(summary: dict[str, object], output_dir: Path) -> None:
    comparisons = ("paired_minus_mismatch", "paired_minus_random")
    labels = ("Paired vs mismatch", "Paired vs random")
    metrics = ("success", "target_ever_grasped", "target_ever_in_basket")
    metric_labels = ("Success", "Target grasp", "Target in basket")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    width = 0.24
    base_positions = np.arange(len(comparisons))
    for task_id, axis in enumerate(axes):
        task = summary["tasks"][str(task_id)]
        for metric_index, (metric, metric_label) in enumerate(zip(metrics, metric_labels, strict=True)):
            means = []
            low = []
            high = []
            for comparison in comparisons:
                values = task["comparisons"][comparison][metric]
                mean = values["mean_favorable_difference"]
                means.append(mean)
                low.append(mean - values["ci95_low"])
                high.append(values["ci95_high"] - mean)
            positions = base_positions + (metric_index - 1) * width
            axis.bar(
                positions,
                means,
                width=width,
                yerr=np.asarray([low, high]),
                capsize=3,
                label=metric_label,
                color=METRIC_COLORS[metric_index],
                edgecolor="#202124",
                linewidth=0.5,
            )
            axis.scatter(
                positions,
                means,
                s=24,
                color=METRIC_COLORS[metric_index],
                edgecolor="#202124",
                linewidth=0.5,
                zorder=3,
            )
        axis.axhline(0.0, color="#202124", linewidth=0.8)
        axis.set_ylim(-0.85, 0.05)
        axis.set_xticks(base_positions, labels)
        axis.set_title("A to B" if task_id == 1 else "B to A")
        axis.grid(axis="y", color="#D8DADD", linewidth=0.7)
    axes[0].set_ylabel("Favorable paired difference (episode-rate points)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Pre-specified closed-loop causal comparisons (95% paired bootstrap CI)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "closed_loop_primary_effects")


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    plot_outcomes(summary, args.output_dir)
    plot_primary_effects(summary, args.output_dir)
    manifest = {
        "closed_loop_outcomes": "four fixed conditions on held-out initial states",
        "closed_loop_primary_effects": "paired-vs-mismatch and paired-vs-random comparisons",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
