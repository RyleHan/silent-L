#!/usr/bin/env python3
"""Plot pi0.5 natural paired-patching dose response and semantic controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    0: "#6B7280",
    5: "#2A9D8F",
    9: "#E9C46A",
    13: "#D1495B",
    17: "#3A6EA5",
}
MODE_COLORS = {
    "paired": "#D1495B",
    "goal": "#3A6EA5",
    "null": "#2A9D8F",
    "shuffled": "#E9C46A",
    "random": "#8B8C89",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-0-summary", type=Path, required=True)
    parser.add_argument("--base-1-summary", type=Path, required=True)
    parser.add_argument("--control-base-0-summary", type=Path, required=True)
    parser.add_argument("--control-base-1-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dose_response(summaries: list[dict[str, object]], output_dir: Path) -> None:
    layers = summaries[0]["layers"]
    alphas = np.asarray(summaries[0]["alphas"], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    direction_names = ("A to B", "B to A")
    for axis, summary, direction in zip(axes, summaries, direction_names, strict=True):
        validation = summary["splits"]["valid"]
        for layer in layers:
            values = [
                validation[str(layer)][str(float(alpha))]["action_recovery"]["mean"]
                for alpha in alphas
            ]
            label = f"Block {layer}"
            if layer == 17:
                label += " (endpoint control)"
            axis.plot(
                alphas,
                values,
                marker="o",
                linewidth=2,
                markersize=4.5,
                color=COLORS[layer],
                label=label,
            )
        axis.axhline(0.0, color="#202124", linewidth=0.8)
        axis.set_title(direction)
        axis.set_xlabel("Patch strength alpha")
        axis.set_xticks(alphas)
        axis.grid(axis="y", color="#D8DADD", linewidth=0.7)
    axes[0].set_ylabel("Validation action recovery")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Natural counterfactual patching: validation dose response", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "validation_dose_response")


def plot_controls(summaries: list[dict[str, object]], output_dir: Path) -> None:
    modes = summaries[0]["mode_names"]
    direction_names = ("A to B", "B to A")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for axis, summary, direction in zip(axes, summaries, direction_names, strict=True):
        test = summary["splits"]["test"]
        means = []
        low_errors = []
        high_errors = []
        for mode in modes:
            metric = test[mode]["action_recovery"]
            bootstrap = metric["episode_bootstrap"]
            means.append(metric["mean"])
            low_errors.append(metric["mean"] - bootstrap["ci95_low"])
            high_errors.append(bootstrap["ci95_high"] - metric["mean"])
        positions = np.arange(len(modes))
        axis.bar(
            positions,
            means,
            yerr=np.asarray([low_errors, high_errors]),
            capsize=3,
            color=[MODE_COLORS[mode] for mode in modes],
            edgecolor="#202124",
            linewidth=0.6,
        )
        axis.axhline(0.0, color="#202124", linewidth=0.8)
        axis.set_xticks(positions, [mode.capitalize() for mode in modes], rotation=24, ha="right")
        axis.set_title(direction)
        axis.grid(axis="y", color="#D8DADD", linewidth=0.7)
    axes[0].set_ylabel("Locked-test action recovery")
    fig.suptitle("Block 13 semantic and matched controls (95% episode-bootstrap CI)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "locked_test_controls")


def main() -> None:
    args = parse_args()
    paired = [load(args.base_0_summary), load(args.base_1_summary)]
    controls = [load(args.control_base_0_summary), load(args.control_base_1_summary)]
    plot_dose_response(paired, args.output_dir)
    plot_controls(controls, args.output_dir)
    manifest = {
        "validation_dose_response": "validation layer/alpha sweep; endpoint block 17 is a control",
        "locked_test_controls": "selected block 13, alpha=1; test episodes with cluster-bootstrap CI",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
