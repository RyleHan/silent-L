#!/usr/bin/env python3
"""Create publication-ready figures for the OpenVLA world-state atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


READOUTS = ("prompt_end", "action_boundary", "visual_mean")
READOUT_LABELS = {
    "prompt_end": "Prompt end",
    "action_boundary": "Action boundary",
    "visual_mean": "Visual mean control",
}
COLORS = {
    "prompt_end": "#2563A5",
    "action_boundary": "#D55E00",
    "visual_mean": "#158F83",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(root: Path, protocol: str) -> dict[str, dict[str, object]]:
    if protocol == "paired":
        return {
            readout: load_json(root / "paired" / readout / "summary.json")
            for readout in READOUTS
        }
    return {
        readout: load_json(root / f"{readout}_actual" / "summary.json")
        for readout in READOUTS
    }


def selected_metric(result: dict[str, object], family: str, factor: str) -> tuple[float, int]:
    if family == "regression":
        item = result["selected"][family][factor]
        return float(item["test"]["mean_r2"]), int(item["layer"])
    if family == "phase_coarse":
        item = result["selected"]["phase_coarse"]
        return float(item["test"]["macro_f1"]), int(item["layer"])
    raise ValueError(f"Unsupported family: {family}")


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_headline_bars(results: dict[str, dict[str, object]], output_dir: Path) -> list[str]:
    factor_specs = (
        ("regression", "target_xyz", "Target XYZ", "R2"),
        ("regression", "target_to_eef_xyz", "Target to EEF", "R2"),
        ("regression", "target_to_basket_xyz", "Target to basket", "R2"),
        ("regression", "target_displacement_xyz", "Target displacement", "R2"),
        ("phase_coarse", "phase_coarse", "Task phase", "Macro-F1"),
    )
    x = np.arange(len(factor_specs))
    width = 0.24
    figure, axis = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    for readout_index, readout in enumerate(READOUTS):
        values = []
        layers = []
        for family, factor, _, _ in factor_specs:
            value, layer = selected_metric(results[readout], family, factor)
            values.append(value)
            layers.append(layer)
        positions = x + (readout_index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width,
            color=COLORS[readout],
            label=READOUT_LABELS[readout],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value, layer in zip(bars, values, layers):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}\nL{layer}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_title("Paired-prompt decoding: goal-conditioned state requires prompt-aware residuals")
    axis.set_ylabel("Held-out test score")
    axis.set_xlabel("World-state factor")
    axis.set_xticks(x, [spec[2] for spec in factor_specs])
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    return save_figure(figure, output_dir, "paired_headline_scores")


def layer_value(
    layer_result: dict[str, object], family: str, factor: str, split: str = "valid"
) -> float:
    if family == "regression":
        return float(layer_result["splits"][split]["regression"][factor]["mean_r2"])
    return float(layer_result["splits"][split]["phase_coarse"]["macro_f1"])


def plot_layer_curves(results: dict[str, dict[str, object]], output_dir: Path) -> list[str]:
    factor_specs = (
        ("regression", "object_a_xyz", "Absolute object A position", "Validation R2"),
        ("regression", "target_xyz", "Prompt-relative target position", "Validation R2"),
        ("regression", "target_to_eef_xyz", "Prompt-relative target to EEF", "Validation R2"),
        ("phase", "phase_coarse", "Prompt-relative task phase", "Validation macro-F1"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True, constrained_layout=True)
    for axis, (family, factor, title, ylabel) in zip(axes.flat, factor_specs):
        for readout in READOUTS:
            layers = [int(item["layer"]) for item in results[readout]["layers"]]
            values = [layer_value(item, family, factor) for item in results[readout]["layers"]]
            axis.plot(
                layers,
                values,
                color=COLORS[readout],
                linewidth=2.0,
                label=READOUT_LABELS[readout],
            )
            best = int(np.argmax(values))
            axis.scatter(layers[best], values[best], color=COLORS[readout], s=28, zorder=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Decoder block")
        axis.set_xlim(0, 31)
        axis.set_ylim(0, 1.02)
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, loc="lower left")
    figure.suptitle("Where world-state information appears across OpenVLA layers", fontsize=15)
    return save_figure(figure, output_dir, "paired_layer_curves")


def plot_layer_heatmap(results: dict[str, dict[str, object]], output_dir: Path) -> list[str]:
    factor_specs = (
        ("regression", "object_a_xyz", "Object A XYZ"),
        ("regression", "eef_xyz", "EEF XYZ"),
        ("regression", "target_xyz", "Target XYZ"),
        ("regression", "target_to_eef_xyz", "Target to EEF"),
        ("regression", "target_to_basket_xyz", "Target to basket"),
        ("regression", "target_displacement_xyz", "Target displacement"),
        ("regression", "normalized_action_progress", "Action progress"),
        ("phase", "phase_coarse", "Task phase"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 5.8), constrained_layout=True)
    image = None
    for axis, readout in zip(axes, READOUTS):
        matrix = np.asarray(
            [
                [layer_value(layer, family, factor) for layer in results[readout]["layers"]]
                for family, factor, _ in factor_specs
            ]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=0, vmax=1)
        axis.set_title(READOUT_LABELS[readout])
        axis.set_xlabel("Decoder block")
        axis.set_xticks(np.arange(0, 32, 4))
        axis.set_xticklabels(np.arange(0, 32, 4))
        axis.set_yticks(np.arange(len(factor_specs)))
        axis.set_yticklabels([spec[2] for spec in factor_specs] if readout == "prompt_end" else [])
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Validation score (R2 or macro-F1)")
    figure.suptitle("Paired-prompt world-state atlas across 32 decoder blocks", fontsize=15)
    return save_figure(figure, output_dir, "paired_layer_heatmap")


def plot_bootstrap_forest(bootstrap: dict[str, object], output_dir: Path) -> list[str]:
    factors = (
        ("target_xyz", "Target XYZ R2"),
        ("target_to_eef_xyz", "Target to EEF R2"),
        ("target_to_basket_xyz", "Target to basket R2"),
        ("target_displacement_xyz", "Target displacement R2"),
        ("phase_coarse", "Task phase macro-F1"),
    )
    figure, axis = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)
    base_y = np.arange(len(factors))[::-1]
    offsets = {"prompt_end": 0.12, "action_boundary": -0.12}
    for readout in ("prompt_end", "action_boundary"):
        points = []
        lower = []
        upper = []
        for factor, _ in factors:
            item = (
                bootstrap["phase_coarse"][readout]
                if factor == "phase_coarse"
                else bootstrap["regression"][readout][factor]
            )
            points.append(float(item["point_difference"]))
            interval = item["bootstrap_difference"]
            lower.append(float(interval["lower_95"]))
            upper.append(float(interval["upper_95"]))
        points_array = np.asarray(points)
        axis.errorbar(
            points_array,
            base_y + offsets[readout],
            xerr=np.vstack((points_array - np.asarray(lower), np.asarray(upper) - points_array)),
            fmt="o",
            color=COLORS[readout],
            capsize=4,
            linewidth=1.8,
            markersize=6,
            label=READOUT_LABELS[readout],
        )
    axis.axvline(0, color="#111827", linewidth=1.0)
    axis.set_yticks(base_y, [label for _, label in factors])
    axis.set_xlabel("Prompt-aware minus visual-control test score")
    axis.set_ylabel("World-state factor")
    axis.set_title("Goal-conditioned decoding advantage is stable across held-out episodes")
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
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
    paired = load_results(args.atlas_root, "paired")
    bootstrap = load_json(args.atlas_root / "paired_episode_bootstrap.json")
    figures = []
    figures += plot_headline_bars(paired, args.output_dir)
    figures += plot_layer_curves(paired, args.output_dir)
    figures += plot_layer_heatmap(paired, args.output_dir)
    figures += plot_bootstrap_forest(bootstrap, args.output_dir)
    manifest = {
        "atlas_root": str(args.atlas_root),
        "protocol": "paired prompt; validation-selected layers; held-out episode test",
        "figures": figures,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
