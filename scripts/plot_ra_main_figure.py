#!/usr/bin/env python3
"""Build the paper-style overview figure for the VLA coordinate project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "paper_main_figure"

INK = "#132238"
MUTED = "#667085"
GRID = "#D9E0E8"
BLUE = "#4169A1"
TEAL = "#1B9E8A"
GOLD = "#E2B13C"
CORAL = "#D85A5A"
PALE = "#F5F7FA"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def style_axis(ax: mpl.axes.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.9)
    ax.tick_params(colors=INK, labelsize=8.5)


def panel_title(ax: mpl.axes.Axes, label: str, title: str) -> None:
    ax.text(
        0,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    ax.text(
        0.055,
        1.04,
        title,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="bottom",
    )


def world_state_panel(ax: mpl.axes.Axes) -> None:
    openvla = load_json(
        ARTIFACTS / "openvla_world_state_atlas" / "summary" / "atlas_summary.json"
    )["headline"]
    pi_visual = load_json(
        ARTIFACTS / "pi05_world_state_atlas" / "visual_input_summary.json"
    )["selected"]
    pi_prefix = load_json(
        ARTIFACTS / "pi05_world_state_atlas" / "prompt_end_summary.json"
    )["selected"]
    pi_expert = load_json(
        ARTIFACTS
        / "pi05_action_expert_atlas"
        / "expert_action_mean_summary.json"
    )["selected"]

    values = np.array(
        [
            [
                openvla["visual_target_xyz_test_r2"],
                openvla["prompt_end_target_xyz_test_r2"],
                pi_visual["regression"]["target_xyz"]["test"]["mean_r2"],
                pi_prefix["regression"]["target_xyz"]["test"]["mean_r2"],
                pi_expert["regression"]["target_xyz"]["test"]["mean_r2"],
            ],
            [
                openvla["visual_target_to_eef_test_r2"],
                openvla["prompt_end_target_to_eef_test_r2"],
                pi_visual["regression"]["target_to_eef_xyz"]["test"]["mean_r2"],
                pi_prefix["regression"]["target_to_eef_xyz"]["test"]["mean_r2"],
                pi_expert["regression"]["target_to_eef_xyz"]["test"]["mean_r2"],
            ],
            [
                openvla["visual_phase_test_macro_f1"],
                openvla["prompt_end_phase_test_macro_f1"],
                pi_visual["phase_coarse"]["test"]["macro_f1"],
                pi_prefix["phase_coarse"]["test"]["macro_f1"],
                pi_expert["phase_coarse"]["test"]["macro_f1"],
            ],
        ]
    )
    cmap = LinearSegmentedColormap.from_list(
        "world_state", ["#183A63", "#4F7F8F", "#D6B94F", "#F4DF61"]
    )
    ax.imshow(values, vmin=0.2, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(
        ["Visual\nonly", "Prompt\nend", "Visual\nonly", "Prefix", "Action\nexpert"]
    )
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Target XYZ\n$R^2$", "Target-to-EEF\n$R^2$", "Task phase\nmacro-F1"])
    ax.axvline(1.5, color="white", linewidth=5)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] < 0.52 else INK
            ax.text(
                col,
                row,
                f"{values[row, col]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                fontweight="bold",
            )
    ax.text(0.5, -0.76, "OpenVLA", ha="center", va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.text(3.0, -0.76, r"$\pi$0.5", ha="center", va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.plot([-0.42, 1.42], [-0.60, -0.60], color=BLUE, linewidth=2.3, clip_on=False)
    ax.plot([1.58, 4.42], [-0.60, -0.60], color=TEAL, linewidth=2.3, clip_on=False)
    ax.text(
        0.5,
        -0.26,
        "+0.60",
        ha="center",
        va="center",
        color=BLUE,
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        2.5,
        -0.26,
        "+0.39",
        ha="center",
        va="center",
        color=TEAL,
        fontsize=8.5,
        fontweight="bold",
    )
    panel_title(ax, "a", "Goal-centric state emerges after language conditioning")
    style_axis(ax)


def patching_panel(ax: mpl.axes.Axes) -> None:
    paths = [
        ARTIFACTS / "pi05_paired_patching" / "control-base-0" / "summary.json",
        ARTIFACTS / "pi05_paired_patching" / "control-base-1" / "summary.json",
    ]
    summaries = [load_json(path)["splits"]["test"] for path in paths]
    modes = ["paired", "goal", "null", "shuffled", "random"]
    labels = ["Natural\npaired", "Probe goal\nsubspace", "Probe\nnullspace", "Episode\nshuffled", "Norm-matched\nrandom"]
    colors = [CORAL, BLUE, TEAL, GOLD, "#8E99A8"]
    offsets = [-0.09, 0.09]
    markers = ["o", "s"]
    directions = ["A -> B", "B -> A"]

    for direction, (summary, offset, marker) in enumerate(zip(summaries, offsets, markers)):
        means, lows, highs = [], [], []
        for mode in modes:
            metric = summary[mode]["action_recovery"]["episode_bootstrap"]
            means.append(metric["mean"])
            lows.append(metric["mean"] - metric["ci95_low"])
            highs.append(metric["ci95_high"] - metric["mean"])
        x = np.arange(len(modes)) + offset
        for idx in range(len(modes)):
            ax.errorbar(
                x[idx],
                means[idx],
                yerr=np.array([[lows[idx]], [highs[idx]]]),
                fmt=marker,
                markersize=6.2,
                color=colors[idx],
                markeredgecolor="white",
                markeredgewidth=0.8,
                ecolor=colors[idx],
                elinewidth=1.4,
                capsize=2.5,
                zorder=3,
                label=directions[direction] if idx == 0 else None,
            )
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_ylim(-0.018, 0.315)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Locked-test action recovery", color=INK, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, handletextpad=0.3)
    ax.text(
        0.02,
        0.92,
        "Block 13",
        transform=ax.transAxes,
        color=CORAL,
        fontsize=8.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": CORAL, "boxstyle": "round,pad=0.22", "linewidth": 0.8},
    )
    ax.text(
        0.50,
        0.75,
        "~22% local action recovery\nsurvives removal of probe space",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=8.3,
        fontweight="bold",
    )
    panel_title(ax, "b", "A single natural residual is locally causal, but not probe-localized")
    style_axis(ax)


def pathway_panel(ax: mpl.axes.Axes) -> None:
    summary = load_json(
        ARTIFACTS / "pi05_full_pathway_replay" / "summary" / "summary.json"
    )
    conditions = ["correct", "mismatch", "single_block", "full_pathway"]
    values = np.array(
        [
            [summary["tasks"][task]["conditions"][condition]["success"]["mean"] for condition in conditions]
            for task in ["0", "1"]
        ]
    )
    cmap = LinearSegmentedColormap.from_list(
        "closed_loop", ["#B94A55", "#E5C66A", "#2A9D78"]
    )
    ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto", extent=(-0.5, 3.5, 1.5, -0.5))
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Correct\nprompt", "Mismatch", "Block 13\nreplace", "18-block\nreplay"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Target A\nalphabet soup", "Target B\ncream cheese"])
    for row in range(2):
        for col in range(4):
            ax.text(
                col,
                row,
                f"{values[row, col]:.0%}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if values[row, col] < 0.25 or values[row, col] > 0.75 else INK,
            )
    ax.text(
        2,
        1.78,
        "hybrid computation",
        ha="center",
        va="center",
        fontsize=8,
        color=CORAL,
        fontweight="bold",
    )
    ax.text(
        3,
        1.78,
        r"exact source replay: max $|\Delta a|=0$",
        ha="center",
        va="center",
        fontsize=8,
        color=TEAL,
        fontweight="bold",
    )
    panel_title(ax, "c", "Closed loop exposes path incompatibility")
    style_axis(ax)

def coast_panel(ax: mpl.axes.Axes) -> None:
    summary = load_json(
        ARTIFACTS / "pi05_coast_pilot" / "summary" / "summary.json"
    )
    order = ["baseline", "random", "coast"]
    labels = ["Baseline", "Random\nspectrum", "COAST"]
    colors = [BLUE, "#8E99A8", TEAL]
    means = [summary["conditions"][key]["mean"] for key in order]
    lows = [summary["conditions"][key]["ci95_low"] for key in order]
    highs = [summary["conditions"][key]["ci95_high"] for key in order]
    x = np.array([0.0, 0.85, 1.70])
    for idx, key in enumerate(order):
        ax.errorbar(
            x[idx],
            means[idx],
            yerr=np.array([[means[idx] - lows[idx]], [highs[idx] - means[idx]]]),
            fmt="o",
            markersize=8,
            color=colors[idx],
            markeredgecolor="white",
            markeredgewidth=1,
            elinewidth=2,
            capsize=3,
            zorder=4,
        )
        ax.text(x[idx], means[idx] + 0.055, f"{means[idx]:.0%}", ha="center", color=INK, fontsize=8.5, fontweight="bold")

    paper_x = np.array([2.85, 3.65])
    paper_y = np.array([0.53, 0.93])
    for idx in range(2):
        ax.scatter(
            paper_x[idx],
            paper_y[idx],
            s=62,
            facecolors="white",
            edgecolors=[BLUE, TEAL][idx],
            linewidths=2,
            zorder=4,
        )
        ax.text(paper_x[idx], paper_y[idx] + 0.055, f"{paper_y[idx]:.0%}", ha="center", color=INK, fontsize=8.5, fontweight="bold")

    ax.axvline(2.28, color=GRID, linewidth=1)
    ax.set_xlim(-0.45, 4.1)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([*x, *paper_x])
    ax.set_xticklabels([*labels, "Paper\nbaseline", "Paper\nCOAST"])
    ax.set_ylabel("Closed-loop success rate", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.text(0.28, 0.995, "Our held-out pilot (n=30)", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(0.82, 0.995, "Reported KS3 (n=30)", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=MUTED, fontweight="bold")
    ax.text(
        0.02,
        0.08,
        "+13.3 pp  [95% CI -3.3, +30.0]\n6 rescued / 2 harmed",
        transform=ax.transAxes,
        fontsize=8.2,
        color=INK,
        fontweight="bold",
        va="bottom",
    )
    panel_title(ax, "d", "Soft subspace steering is promising, not yet resolved")
    style_axis(ax)

    image_paths = [
        OUTPUT / "assets" / "coast-baseline-end.png",
        OUTPUT / "assets" / "coast-success-end.png",
        OUTPUT / "assets" / "coast-random-end.png",
    ]
    image_labels = ["Baseline: fail", "COAST: success", "Random: fail"]
    border_colors = [CORAL, TEAL, CORAL]
    for idx, (path, label, border) in enumerate(zip(image_paths, image_labels, border_colors)):
        image_ax = ax.inset_axes([0.02 + idx * 0.20, -0.54, 0.17, 0.36], transform=ax.transAxes)
        image_ax.imshow(plt.imread(path))
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_color(border)
            spine.set_linewidth(2)
        image_ax.set_title(label, fontsize=7.5, color=border, fontweight="bold", pad=3)
    ax.text(
        0.62,
        -0.36,
        "Same smoke initial state\nOne qualitative rescue",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(18, 11.1), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.975,
        bottom=0.22,
        top=0.86,
        wspace=0.23,
        hspace=0.44,
    )
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    world_state_panel(axes[0])
    patching_panel(axes[1])
    pathway_panel(axes[2])
    coast_panel(axes[3])

    figure.text(
        0.06,
        0.958,
        "From Decodable World State to Closed-Loop Control",
        fontsize=23,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.06,
        0.918,
        r"A paired-prompt causal atlas of OpenVLA and $\pi$0.5 on LIBERO",
        fontsize=12.5,
        color=MUTED,
        va="top",
    )
    figure.add_artist(
        mpl.lines.Line2D([0.06, 0.975], [0.89, 0.89], transform=figure.transFigure, color=INK, linewidth=1.1)
    )
    figure.text(
        0.5,
        0.046,
        "Readable state is not automatically a usable control handle: effective interventions must preserve the model's computation path.",
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.975,
        0.014,
        "Paired episodes • locked test splits • matched controls",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )

    for suffix, kwargs in [
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ]:
        figure.savefig(
            OUTPUT / f"vla_world_state_causal_story.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
