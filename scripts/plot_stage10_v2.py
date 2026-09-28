#!/usr/bin/env python3
"""Plot the final multi-pair confirmation and causal-closure figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


INK = "#172331"
MUTED = "#657180"
GRID = "#D9DEE5"
BLUE = "#326B9B"
TEAL = "#16847D"
GOLD = "#D3A13D"
CORAL = "#C65757"
GRAY = "#A4ADB8"
PALE = "#F4F6F8"
SEEDS = (42, 43, 44)
PAIR_LABELS = ("Soup / cheese", "Dressing / ketchup", "BBQ / pudding", "Tomato / butter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8.5)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=13, weight="bold", color=INK)
    ax.text(0.055, 1.08, title, transform=ax.transAxes, fontsize=10.5, weight="bold", color=INK)


def selected_target_r2(path: Path) -> float:
    item = load(path)["selected"]["regression"]["target_xyz"]
    return float(item["test"]["mean_r2"])


def pooled_values(run_root: Path) -> dict[str, dict[str, object]]:
    specs = {
        "OpenVLA": (
            run_root / "openvla_world_state_atlas_v2/pooled",
            "prompt_end",
            "visual_mean",
        ),
        r"$\pi$0.5 prefix": (
            run_root / "pi05_world_state_atlas_v2/pooled",
            "prompt_end",
            "visual_input",
        ),
        r"$\pi$0.5 expert": (
            run_root / "pi05_action_expert_atlas_v2/pooled",
            "expert_action_mean",
            "visual_input",
        ),
    }
    result = {}
    for label, (root, prompt_readout, visual_readout) in specs.items():
        prompt = [
            selected_target_r2(root / f"seed-{seed}/{prompt_readout}/summary.json")
            for seed in SEEDS
        ]
        visual = [
            selected_target_r2(root / f"seed-{seed}/{visual_readout}/summary.json")
            for seed in SEEDS
        ]
        pair_differences = np.empty((len(SEEDS), 4), dtype=np.float64)
        for seed_index, seed in enumerate(SEEDS):
            bootstrap = load(root / f"seed-{seed}/pair_macro_episode_bootstrap.json")
            per_pair = bootstrap["regression"][prompt_readout]["target_xyz"]["per_pair"]
            pair_differences[seed_index] = [
                float(per_pair[str(pair_id)]["difference"]) for pair_id in range(4)
            ]
        result[label] = {
            "prompt": np.asarray(prompt),
            "visual": np.asarray(visual),
            "pair_differences": pair_differences,
        }
    return result


def plot_coordinate(ax: plt.Axes, summary: dict[str, object]) -> None:
    labels = ("Prompt end", "Action boundary", "Visual mean")
    readouts = ("prompt_end", "action_boundary", "visual_mean")
    y = np.arange(3)[::-1]
    for row, readout in enumerate(readouts):
        item = summary["aggregates"][
            f"coordinate_relative_minus_absolute/openvla/{readout}/foreground_macro_f1"
        ]
        point = float(item["mean_point"])
        low, high = map(float, item["envelope"])
        ax.errorbar(
            point,
            y[row],
            xerr=[[point - low], [high - point]],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            capsize=3,
            markersize=6,
            linewidth=1.6,
        )
        for seed_point in item["points_by_seed"].values():
            ax.scatter(float(seed_point), y[row], s=12, color="white", edgecolor=BLUE, zorder=4)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Relative minus absolute foreground macro-F1", color=INK, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    panel_title(ax, "a", "Othello-style coordinate advantage")
    style(ax)


def plot_pooled(ax: plt.Axes, pooled: dict[str, dict[str, object]]) -> None:
    labels = list(pooled)
    y = np.arange(len(labels))[::-1]
    for row, label in enumerate(labels):
        visual = float(np.mean(pooled[label]["visual"]))
        prompt = float(np.mean(pooled[label]["prompt"]))
        ax.plot([visual, prompt], [y[row], y[row]], color=GRID, linewidth=4, solid_capstyle="round")
        ax.scatter(visual, y[row], s=55, color=GRAY, edgecolor="white", zorder=3)
        ax.scatter(prompt, y[row], s=65, color=TEAL, edgecolor="white", zorder=3)
        ax.text(visual, y[row] + 0.18, f"{visual:.2f}", ha="center", color=MUTED, fontsize=8)
        ax.text(prompt, y[row] + 0.18, f"{prompt:.2f}", ha="center", color=TEAL, fontsize=8, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"Held-out target XYZ $R^2$", color=INK, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.scatter([], [], color=GRAY, label="Visual control")
    ax.scatter([], [], color=TEAL, label="Prompt-aware")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    panel_title(ax, "b", "Goal state survives the 7.9x scale-up")
    style(ax)


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, threshold: float) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) >= threshold else INK,
                fontsize=8.5,
                weight="bold",
            )


def plot_pair_heatmap(ax: plt.Axes, pooled: dict[str, dict[str, object]]) -> None:
    labels = list(pooled)
    matrix = np.stack(
        [np.mean(pooled[label]["pair_differences"], axis=0) for label in labels]
    )
    cmap = LinearSegmentedColormap.from_list("positive", ["#EEF2F4", "#8FC4BB", "#08736B"])
    image = ax.imshow(matrix, vmin=0, vmax=0.65, cmap=cmap, aspect="auto")
    annotate_heatmap(ax, matrix, 0.38)
    ax.set_xticks(range(4), PAIR_LABELS, rotation=24, ha="right")
    ax.set_yticks(range(3), labels)
    panel_title(ax, "c", "Pooled gain is positive for every pair")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"Prompt-aware - visual $R^2$")
    style(ax)


def plot_lopo(ax: plt.Axes, summary: dict[str, object]) -> None:
    labels = ("OpenVLA", r"$\pi$0.5 prefix", r"$\pi$0.5 expert")
    models = ("openvla", "pi05_prefix", "pi05_action_expert")
    matrix = np.asarray(
        [
            [
                float(summary["lopo"][model]["target_xyz"]["difference_by_heldout_pair"][str(pair)])
                for pair in range(4)
            ]
            for model in models
        ]
    )
    bound = max(0.35, float(np.abs(matrix).max()))
    cmap = LinearSegmentedColormap.from_list("signed", [CORAL, "#F7F7F5", TEAL])
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
        aspect="auto",
    )
    annotate_heatmap(ax, matrix, 0.18)
    ax.set_xticks(range(4), PAIR_LABELS, rotation=24, ha="right")
    ax.set_yticks(range(3), labels)
    panel_title(ax, "d", "Leave-one-pair-out transfer")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"Prompt-aware - visual $R^2$")
    style(ax)


def plot_expert_prefix(ax: plt.Axes, summary: dict[str, object]) -> None:
    readouts = ("expert_action_mean", "expert_action_first")
    labels = ("Expert mean", "Expert first token")
    y = np.arange(2)[::-1]
    for row, (readout, label) in enumerate(zip(readouts, labels)):
        item = summary["aggregates"][
            f"action_expert_minus_prefix/pi05/{readout}/target_xyz"
        ]
        for seed_index, seed in enumerate(SEEDS):
            point = float(item["points_by_seed"][str(seed)])
            low, high = map(float, item["bootstrap_ci_by_seed"][str(seed)])
            yy = y[row] + (seed_index - 1) * 0.11
            ax.errorbar(
                point,
                yy,
                xerr=[[point - low], [high - point]],
                fmt="o",
                color=(BLUE, TEAL, GOLD)[seed_index],
                capsize=2,
                markersize=4.5,
                linewidth=1.1,
                label=f"seed {seed}" if row == 0 else None,
            )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"Action expert - prefix target XYZ $R^2$", color=INK, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    panel_title(ax, "e", "Propagation is real; extra decodability is modest")
    style(ax)


def plot_causal(ax: plt.Axes, run_root: Path) -> None:
    replay = load(run_root / "pi05_full_pathway_replay/summary/summary.json")
    replay_conditions = ("correct", "single_block", "full_pathway")
    replay_values = [
        np.mean(
            [
                replay["tasks"][task]["conditions"][condition]["success"]["mean"]
                for task in ("0", "1")
            ]
        )
        for condition in replay_conditions
    ]
    coast = load(run_root / "pi05_coast_pilot/summary/summary.json")
    coast_conditions = ("baseline", "random", "coast")
    coast_values = [float(coast["conditions"][condition]["mean"]) for condition in coast_conditions]
    coast_low = [float(coast["conditions"][condition]["ci95_low"]) for condition in coast_conditions]
    coast_high = [float(coast["conditions"][condition]["ci95_high"]) for condition in coast_conditions]

    x = np.asarray([0, 1, 2, 3.5, 4.5, 5.5])
    values = np.asarray(replay_values + coast_values)
    colors = [BLUE, CORAL, TEAL, BLUE, GRAY, TEAL]
    bars = ax.bar(x, values, width=0.68, color=colors, edgecolor="white")
    ax.errorbar(
        x[3:],
        coast_values,
        yerr=np.vstack((np.asarray(coast_values) - coast_low, np.asarray(coast_high) - coast_values)),
        fmt="none",
        ecolor=INK,
        capsize=2.5,
        linewidth=1,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.0%}", ha="center", fontsize=8, weight="bold", color=INK)
    ax.axvline(2.75, color=GRID, linewidth=1)
    ax.set_xticks(x, ["Correct", "Block 13", "Full path", "Baseline", "Random", "COAST"], rotation=20, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Closed-loop success", color=INK, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.text(1, 1.12, "paired-target replay", ha="center", color=MUTED, fontsize=8)
    ax.text(4.5, 1.12, "KS3 soft gate", ha="center", color=MUTED, fontsize=8)
    panel_title(ax, "f", "Causal use depends on pathway-compatible operators")
    style(ax)


def main() -> None:
    args = parse_args()
    run_root = args.work_root / "runs"
    summary = load(run_root / "stage10_v2_summary/summary.json")
    pooled = pooled_values(run_root)
    dataset = summary["dataset"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.8))
    figure.subplots_adjust(
        left=0.07,
        right=0.95,
        bottom=0.09,
        top=0.86,
        wspace=0.72,
        hspace=0.5,
    )
    plot_coordinate(axes[0, 0], summary)
    plot_pooled(axes[0, 1], pooled)
    plot_pair_heatmap(axes[0, 2], pooled)
    plot_lopo(axes[1, 0], summary)
    plot_expert_prefix(axes[1, 1], summary)
    plot_causal(axes[1, 2], run_root)
    figure.suptitle(
        "Goal-Centric World State in VLA Models: Multi-Pair Confirmation and Causal Boundaries",
        fontsize=17,
        weight="bold",
        color=INK,
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        (
            f"4 object pairs | 8 tasks | {dataset['num_episodes']} rollouts | "
            f"{dataset['num_samples']:,} states | {dataset['split_episode_counts']['test']} locked test episodes | 3 probe seeds"
        ),
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    figure.text(
        0.995,
        0.005,
        "Panel f combines two preregistered causal tests; COAST uses a different task and checkpoint.",
        ha="right",
        color=MUTED,
        fontsize=7.5,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = args.output_dir / f"stage10_multi_pair_confirmation.{suffix}"
        figure.savefig(path, dpi=320, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(figure)
    manifest = {
        "summary": str(run_root / "stage10_v2_summary/summary.json"),
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
