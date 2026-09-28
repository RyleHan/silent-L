#!/usr/bin/env python3
"""Create the locked, cross-seed summary for the final multi-pair study."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FACTORS = (
    "target_xyz",
    "target_to_eef_xyz",
    "target_to_basket_xyz",
    "target_displacement_xyz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_seed_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    points = [float(row["point"]) for row in rows]
    lowers = [float(row["ci_lower"]) for row in rows]
    uppers = [float(row["ci_upper"]) for row in rows]
    return {
        "mean_point": statistics.fmean(points),
        "seed_std": statistics.stdev(points) if len(points) > 1 else 0.0,
        "points_by_seed": {str(row["seed"]): row["point"] for row in rows},
        "bootstrap_ci_by_seed": {
            str(row["seed"]): [row["ci_lower"], row["ci_upper"]] for row in rows
        },
        "all_seed_points_positive": all(value > 0 for value in points),
        "all_seed_bootstrap_lower_bounds_positive": all(value > 0 for value in lowers),
        "envelope": [min(lowers), max(uppers)],
    }


def bootstrap_row(
    *,
    family: str,
    model: str,
    readout: str,
    factor: str,
    seed: int,
    item: dict[str, object],
) -> dict[str, object]:
    interval = item["bootstrap_difference"]
    return {
        "family": family,
        "model": model,
        "readout": readout,
        "factor": factor,
        "seed": seed,
        "point": float(item["point_difference"]),
        "ci_lower": float(interval["lower_95"]),
        "ci_upper": float(interval["upper_95"]),
        "probability_positive": float(interval["probability_positive"]),
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    run_root = args.work_root / "runs"
    rows: list[dict[str, object]] = []

    coordinate_root = run_root / "openvla_trajectory_coordinate_probes_v2/pooled"
    for seed in seeds:
        summary = load(coordinate_root / f"seed-{seed}/pair_macro_episode_bootstrap.json")
        for readout, item in summary["readouts"].items():
            lower, upper = item["bootstrap_delta_95_percent_ci"]
            rows.append(
                {
                    "family": "coordinate_relative_minus_absolute",
                    "model": "openvla",
                    "readout": readout,
                    "factor": "foreground_macro_f1",
                    "seed": seed,
                    "point": float(item["test_delta"]),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "probability_positive": float(
                        item["bootstrap_probability_delta_positive"]
                    ),
                }
            )

    atlas_specs = {
        "openvla": (
            run_root / "openvla_world_state_atlas_v2/pooled",
            ("prompt_end", "action_boundary"),
        ),
        "pi05_prefix": (
            run_root / "pi05_world_state_atlas_v2/pooled",
            ("prompt_end",),
        ),
        "pi05_action_expert": (
            run_root / "pi05_action_expert_atlas_v2/pooled",
            ("expert_action_mean", "expert_action_first"),
        ),
    }
    for model, (root, readouts) in atlas_specs.items():
        for seed in seeds:
            summary = load(root / f"seed-{seed}/pair_macro_episode_bootstrap.json")
            for readout in readouts:
                for factor in FACTORS:
                    rows.append(
                        bootstrap_row(
                            family="prompt_aware_minus_visual",
                            model=model,
                            readout=readout,
                            factor=factor,
                            seed=seed,
                            item=summary["regression"][readout][factor],
                        )
                    )
                rows.append(
                    bootstrap_row(
                        family="prompt_aware_minus_visual",
                        model=model,
                        readout=readout,
                        factor="phase_coarse",
                        seed=seed,
                        item=summary["phase_coarse"][readout],
                    )
                )

    compare_root = run_root / "pi05_action_expert_atlas_v2/pooled"
    for seed in seeds:
        summary = load(compare_root / f"seed-{seed}/expert_minus_prefix_pair_macro.json")
        for comparison, contents in summary["comparisons"].items():
            readout = comparison.removesuffix("_minus_prompt_end")
            for factor in FACTORS:
                rows.append(
                    bootstrap_row(
                        family="action_expert_minus_prefix",
                        model="pi05",
                        readout=readout,
                        factor=factor,
                        seed=seed,
                        item=contents["regression"][factor],
                    )
                )
            rows.append(
                bootstrap_row(
                    family="action_expert_minus_prefix",
                    model="pi05",
                    readout=readout,
                    factor="phase_coarse",
                    seed=seed,
                    item=contents["phase_coarse"],
                )
            )

    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in ("family", "model", "readout", "factor"))
        groups.setdefault(key, []).append(row)
    aggregates = {
        "/".join(key): aggregate_seed_rows(group) for key, group in sorted(groups.items())
    }

    lopo = {
        model: load(path)["aggregate"]
        for model, path in {
            "openvla": run_root / "openvla_world_state_atlas_v2/lopo/summary/summary.json",
            "pi05_prefix": run_root / "pi05_world_state_atlas_v2/lopo/summary/summary.json",
            "pi05_action_expert": run_root
            / "pi05_action_expert_atlas_v2/lopo/summary/summary.json",
        }.items()
    }
    dataset = load(args.work_root / "datasets/libero_object_trajectories_v2/metadata.json")[
        "audit"
    ]
    result = {
        "protocol": (
            "four object pairs; 50 episodes per task; validation-selected layers; "
            "locked test; pair-macro episode bootstrap; three probe seeds"
        ),
        "seeds": seeds,
        "dataset": dataset,
        "aggregates": aggregates,
        "lopo": lopo,
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "seed_metrics.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"dataset": dataset, "aggregates": aggregates}, indent=2))
    print(f"Wrote: {json_path}", flush=True)


if __name__ == "__main__":
    main()
