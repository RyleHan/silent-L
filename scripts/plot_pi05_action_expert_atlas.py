#!/usr/bin/env python3
"""Plot pi0.5 prefix-to-action-expert world-state transfer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREFIX_COLOR = "#2563A5"
EXPERT_MEAN_COLOR = "#C84C32"
EXPERT_FIRST_COLOR = "#D69E2E"
VISUAL_COLOR = "#158F83"


FACTORS = (
    ("regression", "target_xyz", "Target XYZ"),
    ("regression", "target_to_eef_xyz", "Target to EEF"),
    ("regression", "target_to_basket_xyz", "Target to basket"),
    ("regression", "target_displacement_xyz", "Target displacement"),
    ("phase", "phase_coarse", "Task phase"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-root", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_metric(result: dict[str, object], family: str, factor: str) -> float:
    if family == "regression":
        return float(result["selected"][family][factor]["test"]["mean_r2"])
    return float(result["selected"]["phase_coarse"]["test"]["macro_f1"])


def layer_value(layer: dict[str, object], family: str, factor: str) -> float:
    if family == "regression":
        return float(layer["splits"]["valid"]["regression"][factor]["mean_r2"])
    return float(layer["splits"]["valid"]["phase_coarse"]["macro_f1"])


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_headline(
    prefix: dict[str, object],
    expert_mean: dict[str, object],
    expert_first: dict[str, object],
    visual: dict[str, object],
    output_dir: Path,
) -> list[str]:
    series = (
        ("PaliGemma prompt end", prefix, PREFIX_COLOR),
        ("Expert action-token mean", expert_mean, EXPERT_MEAN_COLOR),
        ("Expert first action token", expert_first, EXPERT_FIRST_COLOR),
        ("SigLIP visual control", visual, VISUAL_COLOR),
    )
    x = np.arange(len(FACTORS))
    width = 0.2
    figure, axis = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    for series_index, (label, result, color) in enumerate(series):
        values = [selected_metric(result, family, factor) for family, factor, _ in FACTORS]
        offset = (series_index - 1.5) * width
        bars = axis.bar(x + offset, values, width, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axis.set_title("Goal-state transfer from pi0.5 prefix to action expert at t=1.0")
    axis.set_ylabel("Held-out test score")
    axis.set_xticks(x, [label for _, _, label in FACTORS])
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    return save_figure(figure, output_dir, "prefix_expert_headline_scores")


def plot_layer_curves(
    prefix: dict[str, object],
    expert_mean: dict[str, object],
    expert_first: dict[str, object],
    visual: dict[str, object],
    output_dir: Path,
) -> list[str]:
    curve_factors = (
        ("regression", "target_xyz", "Prompt-relative target position", "Validation R2"),
        ("regression", "target_to_eef_xyz", "Prompt-relative target to EEF", "Validation R2"),
        ("regression", "target_displacement_xyz", "Prompt-relative displacement", "Validation R2"),
        ("phase", "phase_coarse", "Prompt-relative task phase", "Validation macro-F1"),
    )
    layers = np.arange(18)
    visual_layer = visual["layers"][0]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), sharex=True, constrained_layout=True)
    for axis, (family, factor, title, ylabel) in zip(axes.flat, curve_factors, strict=True):
        prefix_values = [layer_value(layer, family, factor) for layer in prefix["layers"]]
        mean_values = [layer_value(layer, family, factor) for layer in expert_mean["layers"]]
        first_values = [layer_value(layer, family, factor) for layer in expert_first["layers"]]
        control = layer_value(visual_layer, family, factor)
        axis.plot(layers, prefix_values, color=PREFIX_COLOR, linewidth=2.0, label="PaliGemma")
        axis.plot(
            layers, mean_values, color=EXPERT_MEAN_COLOR, linewidth=2.0, label="Expert mean"
        )
        axis.plot(
            layers,
            first_values,
            color=EXPERT_FIRST_COLOR,
            linewidth=1.7,
            label="Expert first",
        )
        axis.axhline(
            control, color=VISUAL_COLOR, linewidth=1.6, linestyle="--", label="SigLIP control"
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Block index within module")
        axis.set_xlim(0, 17)
        axis.set_xticks(np.arange(0, 18, 2))
        axis.set_ylim(0, 1.02)
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "PaliGemma and action-expert layers are separate 18-block modules", fontsize=14.5
    )
    return save_figure(figure, output_dir, "prefix_expert_layer_curves")


def plot_expert_minus_prefix(
    comparison: dict[str, object], output_dir: Path
) -> list[str]:
    readouts = (
        ("expert_action_mean_minus_prompt_end", "Expert mean", EXPERT_MEAN_COLOR),
        ("expert_action_first_minus_prompt_end", "Expert first", EXPERT_FIRST_COLOR),
    )
    y = np.arange(len(FACTORS))[::-1]
    offsets = (0.08, -0.08)
    figure, axis = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    for offset, (key, label, color) in zip(offsets, readouts, strict=True):
        points, lower, upper = [], [], []
        item = comparison["comparisons"][key]
        for family, factor, _ in FACTORS:
            metric = item["phase_coarse"] if family == "phase" else item["regression"][factor]
            points.append(float(metric["point_difference"]))
            lower.append(float(metric["bootstrap_difference"]["lower_95"]))
            upper.append(float(metric["bootstrap_difference"]["upper_95"]))
        points_array = np.asarray(points)
        axis.errorbar(
            points_array,
            y + offset,
            xerr=np.vstack(
                (points_array - np.asarray(lower), np.asarray(upper) - points_array)
            ),
            fmt="o",
            color=color,
            capsize=4,
            linewidth=1.7,
            markersize=5.5,
            label=label,
        )
    axis.axvline(0, color="#111827", linewidth=1.0)
    axis.set_yticks(y, [label for _, _, label in FACTORS])
    axis.set_xlabel("Action-expert minus PaliGemma prompt-end test score")
    axis.set_title("Episode-bootstrap change across the policy interface")
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    return save_figure(figure, output_dir, "expert_minus_prefix_bootstrap")


def plot_expert_heatmap(expert_mean: dict[str, object], output_dir: Path) -> list[str]:
    heatmap_factors = (
        ("regression", "object_a_xyz", "Object A XYZ"),
        ("regression", "eef_xyz", "EEF XYZ"),
        ("regression", "target_xyz", "Target XYZ"),
        ("regression", "target_to_eef_xyz", "Target to EEF"),
        ("regression", "target_to_basket_xyz", "Target to basket"),
        ("regression", "target_displacement_xyz", "Target displacement"),
        ("regression", "normalized_action_progress", "Action progress"),
        ("phase", "phase_coarse", "Task phase"),
    )
    matrix = np.asarray(
        [
            [layer_value(layer, family, factor) for layer in expert_mean["layers"]]
            for family, factor, _ in heatmap_factors
        ]
    )
    figure, axis = plt.subplots(figsize=(11.4, 5.9), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    axis.set_title("pi0.5 action-token mean world-state atlas at t=1.0")
    axis.set_xlabel("Gemma action-expert block")
    axis.set_xticks(np.arange(0, 18, 2))
    axis.set_yticks(np.arange(len(heatmap_factors)), [label for _, _, label in heatmap_factors])
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Validation score (R2 or macro-F1)")
    return save_figure(figure, output_dir, "expert_action_mean_heatmap")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    prefix = load_json(args.prefix_root / "paired" / "prompt_end" / "summary.json")
    expert_mean = load_json(args.expert_root / "paired" / "expert_action_mean" / "summary.json")
    expert_first = load_json(
        args.expert_root / "paired" / "expert_action_first" / "summary.json"
    )
    visual = load_json(args.expert_root / "paired" / "visual_input" / "summary.json")
    comparison = load_json(args.expert_root / "paired" / "expert_minus_prefix_bootstrap.json")

    figures = []
    figures += plot_headline(prefix, expert_mean, expert_first, visual, args.output_dir)
    figures += plot_layer_curves(prefix, expert_mean, expert_first, visual, args.output_dir)
    figures += plot_expert_minus_prefix(comparison, args.output_dir)
    figures += plot_expert_heatmap(expert_mean, args.output_dir)
    manifest = {
        "prefix_root": str(args.prefix_root),
        "expert_root": str(args.expert_root),
        "protocol": "paired prompt; t=1.0; validation-selected layers; held-out episode test",
        "figures": figures,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
