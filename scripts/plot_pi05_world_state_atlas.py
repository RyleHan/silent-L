#!/usr/bin/env python3
"""Create publication-ready figures for the paired-prompt pi0.5 atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROMPT_COLOR = "#2563A5"
VISUAL_COLOR = "#158F83"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_metric(result: dict[str, object], family: str, factor: str) -> tuple[float, int]:
    if family == "regression":
        item = result["selected"][family][factor]
        return float(item["test"]["mean_r2"]), int(item["layer"])
    item = result["selected"]["phase_coarse"]
    return float(item["test"]["macro_f1"]), int(item["layer"])


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
    prompt: dict[str, object], visual: dict[str, object], output_dir: Path
) -> list[str]:
    factors = (
        ("regression", "target_xyz", "Target XYZ"),
        ("regression", "target_to_eef_xyz", "Target to EEF"),
        ("regression", "target_to_basket_xyz", "Target to basket"),
        ("regression", "target_displacement_xyz", "Target displacement"),
        ("phase", "phase_coarse", "Task phase"),
    )
    x = np.arange(len(factors))
    width = 0.36
    prompt_values, prompt_layers = zip(
        *(selected_metric(prompt, family, factor) for family, factor, _ in factors)
    )
    visual_values = [selected_metric(visual, family, factor)[0] for family, factor, _ in factors]
    figure, axis = plt.subplots(figsize=(11.0, 5.6), constrained_layout=True)
    prompt_bars = axis.bar(
        x - width / 2,
        prompt_values,
        width,
        color=PROMPT_COLOR,
        label="Prompt-end residual",
    )
    visual_bars = axis.bar(
        x + width / 2,
        visual_values,
        width,
        color=VISUAL_COLOR,
        label="SigLIP visual-input control",
    )
    for bar, value, layer in zip(prompt_bars, prompt_values, prompt_layers):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.2f}\nL{layer}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, value in zip(visual_bars, visual_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_title("pi0.5 paired-prompt decoding on held-out LIBERO episodes")
    axis.set_ylabel("Held-out test score")
    axis.set_xticks(x, [label for _, _, label in factors])
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    return save_figure(figure, output_dir, "paired_headline_scores")


def plot_layer_curves(
    prompt: dict[str, object], visual: dict[str, object], output_dir: Path
) -> list[str]:
    factors = (
        ("regression", "object_a_xyz", "Absolute object A position", "Validation R2"),
        ("regression", "target_xyz", "Prompt-relative target position", "Validation R2"),
        ("regression", "target_to_eef_xyz", "Prompt-relative target to EEF", "Validation R2"),
        ("phase", "phase_coarse", "Prompt-relative task phase", "Validation macro-F1"),
    )
    layers = [int(item["layer"]) for item in prompt["layers"]]
    visual_layer = visual["layers"][0]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True, constrained_layout=True)
    for axis, (family, factor, title, ylabel) in zip(axes.flat, factors):
        values = [layer_value(item, family, factor) for item in prompt["layers"]]
        control = layer_value(visual_layer, family, factor)
        axis.plot(layers, values, color=PROMPT_COLOR, linewidth=2.1, label="Prompt end")
        axis.axhline(
            control,
            color=VISUAL_COLOR,
            linewidth=1.8,
            linestyle="--",
            label="SigLIP visual input",
        )
        best = int(np.argmax(values))
        axis.scatter(layers[best], values[best], color=PROMPT_COLOR, s=30, zorder=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("PaliGemma decoder block")
        axis.set_xlim(0, 17)
        axis.set_xticks(np.arange(0, 18, 2))
        axis.set_ylim(0, 1.02)
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.suptitle("Goal-conditioned state across pi0.5 PaliGemma layers", fontsize=15)
    return save_figure(figure, output_dir, "paired_layer_curves")


def plot_heatmap(prompt: dict[str, object], output_dir: Path) -> list[str]:
    factors = (
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
            [layer_value(layer, family, factor) for layer in prompt["layers"]]
            for family, factor, _ in factors
        ]
    )
    figure, axis = plt.subplots(figsize=(11.4, 5.9), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    axis.set_title("pi0.5 prompt-end paired world-state atlas")
    axis.set_xlabel("PaliGemma decoder block")
    axis.set_xticks(np.arange(0, 18, 2))
    axis.set_yticks(np.arange(len(factors)), [label for _, _, label in factors])
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.02)
    colorbar.set_label("Validation score (R2 or macro-F1)")
    return save_figure(figure, output_dir, "paired_layer_heatmap")


def plot_bootstrap(bootstrap: dict[str, object], output_dir: Path) -> list[str]:
    factors = (
        ("target_xyz", "Target XYZ R2"),
        ("target_to_eef_xyz", "Target to EEF R2"),
        ("target_to_basket_xyz", "Target to basket R2"),
        ("target_displacement_xyz", "Target displacement R2"),
        ("phase_coarse", "Task phase macro-F1"),
    )
    points, lower, upper = [], [], []
    for factor, _ in factors:
        item = (
            bootstrap["phase_coarse"]["prompt_end"]
            if factor == "phase_coarse"
            else bootstrap["regression"]["prompt_end"][factor]
        )
        points.append(float(item["point_difference"]))
        lower.append(float(item["bootstrap_difference"]["lower_95"]))
        upper.append(float(item["bootstrap_difference"]["upper_95"]))
    points = np.asarray(points)
    y = np.arange(len(factors))[::-1]
    figure, axis = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    axis.errorbar(
        points,
        y,
        xerr=np.vstack((points - np.asarray(lower), np.asarray(upper) - points)),
        fmt="o",
        color=PROMPT_COLOR,
        capsize=4,
        linewidth=1.8,
        markersize=6,
    )
    axis.axvline(0, color="#111827", linewidth=1.0)
    axis.set_yticks(y, [label for _, label in factors])
    axis.set_xlabel("Prompt-end minus SigLIP-control test score")
    axis.set_title("Episode-bootstrap advantage of instruction-conditioned residuals")
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    return save_figure(figure, output_dir, "paired_episode_bootstrap")


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
    prompt = load_json(args.atlas_root / "paired" / "prompt_end" / "summary.json")
    visual = load_json(args.atlas_root / "paired" / "visual_input" / "summary.json")
    bootstrap = load_json(args.atlas_root / "paired" / "episode_bootstrap.json")
    figures = []
    figures += plot_headline(prompt, visual, args.output_dir)
    figures += plot_layer_curves(prompt, visual, args.output_dir)
    figures += plot_heatmap(prompt, args.output_dir)
    figures += plot_bootstrap(bootstrap, args.output_dir)
    manifest = {
        "atlas_root": str(args.atlas_root),
        "protocol": "paired prompt; validation-selected layers; held-out episode test",
        "figures": figures,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

