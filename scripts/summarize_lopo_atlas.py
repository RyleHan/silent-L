#!/usr/bin/env python3
"""Aggregate leave-one-object-pair-out atlas results across folds and seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_FACTORS = (
    "target_xyz",
    "target_to_eef_xyz",
    "target_to_basket_xyz",
    "target_displacement_xyz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-readout", required=True)
    parser.add_argument("--visual-readout", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--pairs", default="0,1,2,3")
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS))
    return parser.parse_args()


def load_summary(root: Path, seed: int, pair_id: int, readout: str) -> dict[str, object]:
    path = root / f"seed-{seed}/pair-{pair_id}/{readout}/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if int(summary["holdout_pair"]) != pair_id:
        raise ValueError(f"Holdout mismatch in {path}.")
    return summary


def regression_score(summary: dict[str, object], factor: str) -> tuple[int, float]:
    selected = summary["selected"]["regression"][factor]
    return int(selected["layer"]), float(selected["test"]["mean_r2"])


def phase_score(summary: dict[str, object]) -> tuple[int, float]:
    selected = summary["selected"]["phase_coarse"]
    return int(selected["layer"]), float(selected["test"]["macro_f1"])


def aggregate(rows: list[dict[str, object]], seeds: list[int], pairs: list[int]) -> dict[str, object]:
    factors = sorted(set(str(row["factor"]) for row in rows))
    result = {}
    for factor in factors:
        selected = [row for row in rows if row["factor"] == factor]
        seed_macro = {
            str(seed): float(
                np.mean([row["difference"] for row in selected if row["seed"] == seed])
            )
            for seed in seeds
        }
        pair_means = {
            str(pair_id): float(
                np.mean([row["difference"] for row in selected if row["pair_id"] == pair_id])
            )
            for pair_id in pairs
        }
        prompt_seed_macro = {
            str(seed): float(
                np.mean([row["prompt_score"] for row in selected if row["seed"] == seed])
            )
            for seed in seeds
        }
        visual_seed_macro = {
            str(seed): float(
                np.mean([row["visual_score"] for row in selected if row["seed"] == seed])
            )
            for seed in seeds
        }
        result[factor] = {
            "prompt_pair_macro_mean": float(np.mean(list(prompt_seed_macro.values()))),
            "visual_pair_macro_mean": float(np.mean(list(visual_seed_macro.values()))),
            "difference_pair_macro_mean": float(np.mean(list(seed_macro.values()))),
            "difference_seed_std": float(np.std(list(seed_macro.values()), ddof=1)),
            "difference_by_seed": seed_macro,
            "difference_by_heldout_pair": pair_means,
            "all_heldout_pairs_positive": bool(all(value > 0 for value in pair_means.values())),
        }
    return result


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    pairs = [int(value) for value in args.pairs.split(",") if value.strip()]
    factors = [value.strip() for value in args.factors.split(",") if value.strip()]
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for pair_id in pairs:
            prompt = load_summary(args.root, seed, pair_id, args.prompt_readout)
            visual = load_summary(args.root, seed, pair_id, args.visual_readout)
            for factor in factors:
                prompt_layer, prompt_value = regression_score(prompt, factor)
                visual_layer, visual_value = regression_score(visual, factor)
                rows.append(
                    {
                        "model": args.model,
                        "seed": seed,
                        "pair_id": pair_id,
                        "family": "regression",
                        "factor": factor,
                        "prompt_layer": prompt_layer,
                        "visual_layer": visual_layer,
                        "prompt_score": prompt_value,
                        "visual_score": visual_value,
                        "difference": prompt_value - visual_value,
                    }
                )
            prompt_layer, prompt_value = phase_score(prompt)
            visual_layer, visual_value = phase_score(visual)
            rows.append(
                {
                    "model": args.model,
                    "seed": seed,
                    "pair_id": pair_id,
                    "family": "phase",
                    "factor": "phase_coarse",
                    "prompt_layer": prompt_layer,
                    "visual_layer": visual_layer,
                    "prompt_score": prompt_value,
                    "visual_score": visual_value,
                    "difference": prompt_value - visual_value,
                }
            )

    summary = {
        "model": args.model,
        "protocol": (
            "leave-one-object-pair-out; held-out pair absent from train and validation; "
            "validation-selected layers; locked test episodes; equal pair weighting"
        ),
        "seeds": seeds,
        "pairs": pairs,
        "prompt_readout": args.prompt_readout,
        "visual_readout": args.visual_readout,
        "aggregate": aggregate(rows, seeds, pairs),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "fold_metrics.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"Wrote: {json_path}", flush=True)


if __name__ == "__main__":
    main()
