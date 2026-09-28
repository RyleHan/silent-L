#!/usr/bin/env python3
"""Plot first-query decoded goal position for the Stage 12 anchor pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = (
    ("laso_expert", "Expert b13, LASO\n(primary)"),
    ("lopo_expert", "Expert b13, LOPO"),
    ("laso_prefix", "Prefix b12, LASO"),
    ("laso_expert_layer_descriptive", "Expert b16, LASO"),
)
GROUPS = (
    ("0", "0", "Prompt A, grasp A", "#3A6EA5"),
    ("1", "0", "Prompt B, grasp A (wrong object)", "#D1495B"),
    ("1", "1", "Prompt B, grasp B", "#2A9D31"),
    ("1", "-1", "Prompt B, no grasp", "#8B8C89"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.summary_dir / "summary.json").read_text(encoding="utf-8"))
    with (args.summary_dir / "reads.csv").open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["pair_id"] == args.pair_id]
    families = summary["pairs"][args.pair_id]["families"]

    figure, axes = plt.subplots(1, len(FAMILIES), figsize=(13.5, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for axis, (family, title) in zip(axes, FAMILIES):
        axis.axvspan(-1.0, 0.5, color="#3A6EA5", alpha=0.05)
        axis.axvspan(0.5, 2.0, color="#2A9D31", alpha=0.05)
        axis.axvline(0.0, color="#3A6EA5", lw=0.8, ls=":")
        axis.axvline(1.0, color="#2A9D31", lw=0.8, ls=":")
        for index, (prompt, grasp, label, color) in enumerate(GROUPS):
            values = np.asarray(
                [
                    float(row[f"{family}_interpolation"])
                    for row in rows
                    if row["prompt_side"] == prompt and row["first_grasp_side"] == grasp
                ]
            )
            if not len(values):
                continue
            jitter = rng.uniform(-0.18, 0.18, len(values))
            axis.scatter(values, index + jitter, s=14, color=color, alpha=0.75, lw=0)
            axis.plot([np.median(values)] * 2, [index - 0.3, index + 0.3], color="black", lw=1.6)
        dissociation = families[family]["dissociation"]
        axis.set_title(
            f"{title}\nwrong-object reads at instructed: "
            f"{dissociation['probe_points_at_instructed']}/{dissociation['determinate']}",
            fontsize=9,
        )
        axis.set_xlim(-0.8, 1.6)
        axis.set_xlabel("decoded target  (0 = object A, 1 = object B)", fontsize=8)
        axis.tick_params(labelsize=8)
    axes[0].set_yticks(range(len(GROUPS)), [group[2] for group in GROUPS], fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle(
        "Stage 12, pair 0 (alphabet soup / cream cheese), first policy query: "
        f"preregistered decision = {summary['decision']}",
        fontsize=10,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(args.summary_dir / f"stage12_first_query_goal_read.{suffix}", dpi=200)


if __name__ == "__main__":
    main()
