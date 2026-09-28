#!/usr/bin/env python3
"""Render the paper/homepage Figure 1 from held-out VLA probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


INK = "#172126"
MUTED = "#687479"
PAPER = "#F7F8F5"
WHITE = "#FFFFFF"
GRID = "#D9DEDC"
OBJECT_A = "#08A9B5"
OBJECT_A_DARK = "#087783"
OBJECT_B = "#EF6A4D"
OBJECT_B_DARK = "#B74332"
ACCENT = "#E5B93F"
PHASE_COLORS = ("#DDE8E5", "#F1D77A", "#B5D8D0")
PHASE_NAMES = ("APPROACH", "PREGRASP", "MANIPULATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def clean_object_name(value: str) -> str:
    return value.replace("_1", "").replace("_", " ").title()


def add_panel_label(ax: plt.Axes, label: str, title: str, *, y: float = 1.12) -> None:
    ax.text(
        -0.012,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=WHITE,
        ha="right",
        va="center",
        bbox={"boxstyle": "square,pad=0.28", "facecolor": INK, "edgecolor": "none"},
    )
    ax.text(
        0.005,
        y,
        title,
        transform=ax.transAxes,
        fontsize=13.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )


def overlay_mask(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32) / 255.0
    for channel, color in ((0, OBJECT_A), (1, OBJECT_B)):
        mask = masks[..., channel] > 0
        rgb = np.asarray(mpl.colors.to_rgb(color), dtype=np.float32)
        result[mask] = 0.66 * result[mask] + 0.34 * rgb
        edge = mask & (
            ~np.roll(mask, 1, axis=0)
            | ~np.roll(mask, -1, axis=0)
            | ~np.roll(mask, 1, axis=1)
            | ~np.roll(mask, -1, axis=1)
        )
        result[edge] = rgb
    return np.clip(result, 0, 1)


def phase_boundaries(phases: np.ndarray) -> list[int]:
    return (np.flatnonzero(np.diff(phases) != 0) + 1).tolist()


def draw_prompt_legend(ax: plt.Axes, object_a: str, object_b: str) -> None:
    ax.axis("off")
    ax.text(0.0, 0.55, "IDENTICAL RGB + ROBOT STATE", color=MUTED, fontsize=8.5, fontweight="bold")
    ax.text(0.265, 0.55, "PROMPT A", color=OBJECT_A_DARK, fontsize=8.5, fontweight="bold")
    ax.text(0.335, 0.55, f"Pick up {object_a}", color=INK, fontsize=10.2, fontweight="bold")
    ax.text(0.57, 0.55, "PROMPT B", color=OBJECT_B_DARK, fontsize=8.5, fontweight="bold")
    ax.text(0.64, 0.55, f"Pick up {object_b}", color=INK, fontsize=10.2, fontweight="bold")
    ax.add_patch(Circle((0.248, 0.58), 0.010, transform=ax.transAxes, color=OBJECT_A))
    ax.add_patch(Circle((0.553, 0.58), 0.010, transform=ax.transAxes, color=OBJECT_B))
    ax.add_patch(
        FancyArrowPatch(
            (0.205, 0.58),
            (0.235, 0.58),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            color=MUTED,
        )
    )


def combined_preference(preference: np.ndarray) -> np.ndarray:
    num_layers = preference.shape[1]
    separator = np.full((1, preference.shape[2]), np.nan, dtype=np.float32)
    return np.concatenate((preference[0], separator, preference[1]), axis=0), num_layers


def draw_ribbon(
    ax: plt.Axes,
    preference: np.ndarray,
    phases: np.ndarray,
    cmap: LinearSegmentedColormap,
) -> mpl.image.AxesImage:
    matrix, num_layers = combined_preference(preference)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-1, vmax=1)
    ax.set_facecolor(INK)
    ax.axhline(num_layers - 0.5, color=PAPER, linewidth=2.4)
    ax.axhline(num_layers + 0.5, color=PAPER, linewidth=2.4)
    for boundary in phase_boundaries(phases):
        ax.axvline(boundary - 0.5, color=WHITE, alpha=0.72, linewidth=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(1.012, 0.73, "PROMPT A", transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=OBJECT_A_DARK, va="center")
    ax.text(1.012, 0.25, "PROMPT B", transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=OBJECT_B_DARK, va="center")
    return image


def draw_tomography(
    ax: plt.Axes,
    predictions: np.ndarray,
    object_a: np.ndarray,
    object_b: np.ndarray,
    eef: np.ndarray,
    basket: np.ndarray,
    title: str,
    num_blocks: int,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    ax.set_facecolor(WHITE)
    ax.scatter(basket[0], basket[1], s=170, marker="s", facecolor="#E7E9E6", edgecolor=MUTED, linewidth=1.0, zorder=1)
    ax.text(basket[0], basket[1], "BASKET", fontsize=6.0, color=MUTED, ha="center", va="center", zorder=2)
    ax.scatter(eef[0], eef[1], s=58, marker="D", facecolor=INK, edgecolor=WHITE, linewidth=0.8, zorder=5)
    ax.text(eef[0], eef[1] - 0.027, "EEF", fontsize=6.2, color=INK, ha="center", va="top")

    for prompt, color, dark in ((0, OBJECT_A, OBJECT_A_DARK), (1, OBJECT_B, OBJECT_B_DARK)):
        points = predictions[prompt, :, :2]
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.42, linewidth=1.35, zorder=3)
        depth = np.linspace(0.2, 1.0, len(points))
        ax.scatter(points[:, 0], points[:, 1], s=10 + 22 * depth, c=color, alpha=depth, edgecolors="none", zorder=4)
        ax.scatter(points[-1, 0], points[-1, 1], s=86, marker="*", facecolor=color, edgecolor=dark, linewidth=0.8, zorder=6)

    for point, color, letter in ((object_a, OBJECT_A, "A"), (object_b, OBJECT_B, "B")):
        ax.scatter(point[0], point[1], s=145, facecolor=WHITE, edgecolor=color, linewidth=2.2, zorder=7)
        ax.text(point[0], point[1], letter, fontsize=8.5, fontweight="bold", color=color, ha="center", va="center", zorder=8)

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, linewidth=0.55, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    ax.text(
        0.04,
        0.96,
        title,
        transform=ax.transAxes,
        fontsize=9.8,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
        bbox={"boxstyle": "square,pad=0.18", "facecolor": WHITE, "edgecolor": "none", "alpha": 0.86},
        zorder=10,
    )
    ax.text(0.98, 0.04, f"{num_blocks} blocks", transform=ax.transAxes, fontsize=6.8, color=MUTED, ha="right")


def main() -> None:
    args = parse_args()
    archive = np.load(args.data, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    object_a_name = clean_object_name(metadata["object_a"])
    object_b_name = clean_object_name(metadata["object_b"])
    phases = archive["phases"]
    frame_positions = archive["frame_positions"]
    images = archive["images"]
    masks = archive["masks"]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "latent_goal", [OBJECT_B_DARK, OBJECT_B, "#F3F2EE", OBJECT_A, OBJECT_A_DARK], N=256
    )
    cmap.set_bad(INK)

    fig = plt.figure(figsize=(18, 10.3), facecolor=PAPER)
    grid = fig.add_gridspec(
        9,
        14,
        height_ratios=[0.72, 1.52, 0.27, 0.38, 0.92, 0.92, 0.92, 0.22, 2.55],
        width_ratios=[0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.65, 0.55],
        left=0.045,
        right=0.97,
        top=0.955,
        bottom=0.055,
        hspace=0.52,
        wspace=0.16,
    )

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.68, "ONE SCENE, TWO LATENT WORLDS", fontsize=27, fontweight="bold", color=INK, va="center")
    title_ax.text(
        0.0,
        0.15,
        "Counterfactual prompts remap the same visual state into goal-centric representations that remain readable inside action computation.",
        fontsize=11.5,
        color=MUTED,
        va="center",
    )
    title_ax.text(
        1.0,
        0.68,
        "HELD-OUT LIBERO · PAIRED PROMPTS",
        fontsize=8.4,
        fontweight="bold",
        color=MUTED,
        ha="right",
        va="center",
    )

    frame_grid = grid[1, 1:13].subgridspec(1, len(frame_positions), wspace=0.035)
    for index, (image, mask, position) in enumerate(zip(images, masks, frame_positions)):
        ax = fig.add_subplot(frame_grid[0, index])
        ax.imshow(overlay_mask(image, mask))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(WHITE)
            spine.set_linewidth(1.5)
        phase = int(phases[position])
        ax.add_patch(Rectangle((0, 0), 1, 0.10, transform=ax.transAxes, color=PHASE_COLORS[phase], alpha=0.96, zorder=4))
        ax.text(0.04, 0.05, f"t={int(archive['action_steps'][position])}", transform=ax.transAxes, fontsize=6.7, fontweight="bold", color=INK, ha="left", va="center", zorder=5)
        if index == 0 or int(phases[frame_positions[index - 1]]) != phase:
            ax.text(0.96, 0.05, PHASE_NAMES[phase], transform=ax.transAxes, fontsize=6.0, fontweight="bold", color=INK, ha="right", va="center", zorder=5)
    first_frame_ax = fig.axes[-len(frame_positions)]
    add_panel_label(first_frame_ax, "A", "One physical trajectory; two counterfactual goals")

    timeline_ax = fig.add_subplot(grid[2, 1:13])
    timeline_ax.set_xlim(0, len(phases))
    timeline_ax.set_ylim(0, 1)
    timeline_ax.axis("off")
    start = 0
    for boundary in phase_boundaries(phases) + [len(phases)]:
        phase = int(phases[start])
        timeline_ax.add_patch(Rectangle((start, 0.22), boundary - start, 0.42, color=PHASE_COLORS[phase], ec="none"))
        timeline_ax.text((start + boundary) / 2, 0.43, PHASE_NAMES[phase], fontsize=6.8, fontweight="bold", color=INK, ha="center", va="center")
        start = boundary
    timeline_ax.add_patch(FancyArrowPatch((0, 0.03), (len(phases), 0.03), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=MUTED))
    timeline_ax.text(-0.6, 0.03, "time", fontsize=7.0, fontweight="bold", color=MUTED, ha="right", va="center")

    prompt_ax = fig.add_subplot(grid[3, 1:13])
    draw_prompt_legend(prompt_ax, object_a_name, object_b_name)

    ribbon_specs = (
        ("openvla", "OpenVLA", "visual-language policy · 32 blocks"),
        ("pi05_prefix", "pi0.5 VLM", "PaliGemma prefix · 18 blocks"),
        ("pi05_expert", "pi0.5 Expert", "action-token mean · 18 blocks"),
    )
    ribbon_axes = []
    last_image = None
    for row, (key, title, subtitle) in enumerate(ribbon_specs, start=4):
        label_ax = fig.add_subplot(grid[row, 0:3])
        label_ax.axis("off")
        label_ax.text(0.95, 0.64, title, ha="right", va="center", fontsize=11.2, fontweight="bold", color=INK)
        label_ax.text(0.95, 0.35, subtitle, ha="right", va="center", fontsize=7.4, color=MUTED)
        ax = fig.add_subplot(grid[row, 3:12])
        ribbon_axes.append(ax)
        last_image = draw_ribbon(ax, archive[f"{key}_preference"], phases, cmap)
    add_panel_label(ribbon_axes[0], "B", "Layer-by-time tomography of decoded target identity")

    colorbar_ax = fig.add_subplot(grid[7, 4:11])
    colorbar = fig.colorbar(last_image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels([f"B · {object_b_name}", "ambiguous", f"A · {object_a_name}"])
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(labelsize=7.0, length=0, pad=3, colors=MUTED)
    colorbar.outline.set_visible(False)

    representative = int(np.flatnonzero(phases == 1)[0]) if np.any(phases == 1) else len(phases) // 2
    decoded_xy = np.concatenate(
        [archive[f"{key}_target_xyz"][:, :, representative, :2].reshape(-1, 2) for key, _, _ in ribbon_specs],
        axis=0,
    )
    all_xy = np.concatenate(
        (
            archive["object_a_xyz"][:, :2],
            archive["object_b_xyz"][:, :2],
            archive["eef_xyz"][:, :2],
            archive["basket_xyz"][:, :2],
            decoded_xy,
        ),
        axis=0,
    )
    center = (all_xy.min(axis=0) + all_xy.max(axis=0)) / 2
    span = max(float(np.ptp(all_xy[:, 0])), float(np.ptp(all_xy[:, 1])), 0.28) * 1.18
    limits = ((center[0] - span / 2, center[0] + span / 2), (center[1] - span / 2, center[1] + span / 2))
    tomography_grid = grid[8, 1:13].subgridspec(1, 3, wspace=0.11)
    tomography_axes = []
    for column, (key, title, _) in enumerate(ribbon_specs):
        ax = fig.add_subplot(tomography_grid[0, column])
        tomography_axes.append(ax)
        prediction = archive[f"{key}_target_xyz"][:, :, representative, :]
        draw_tomography(
            ax,
            prediction,
            archive["object_a_xyz"][representative],
            archive["object_b_xyz"][representative],
            archive["eef_xyz"][representative],
            archive["basket_xyz"][representative],
            title,
            prediction.shape[1],
            limits,
        )
    add_panel_label(
        tomography_axes[0],
        "C",
        "At one pregrasp state, decoded goal coordinates bifurcate across depth",
        y=1.055,
    )
    tomography_axes[1].text(
        0.5,
        -0.16,
        "circles = physical objects   ·   trails = per-block decoded target   ·   stars = final block",
        transform=tomography_axes[1].transAxes,
        fontsize=7.1,
        color=MUTED,
        ha="center",
        va="top",
    )

    fig.text(
        0.965,
        0.017,
        "Decode, not attention · test episode selected without probe outputs · causal use remains unresolved",
        fontsize=7.0,
        color=MUTED,
        ha="right",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "one_scene_two_latent_worlds"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix(".svg"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
