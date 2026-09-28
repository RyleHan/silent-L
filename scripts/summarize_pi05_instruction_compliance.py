#!/usr/bin/env python3
"""Summarize the fixed multi-pair pi0.5 instruction-compliance survey."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


OUTCOMES = ("a", "b", "both", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--pair-ids", type=int, nargs="+")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial.")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "mean": float(proportion),
        "ci95_low": float(center - radius),
        "ci95_high": float(center + radius),
        "successes": int(successes),
        "trials": int(trials),
    }


def paired_bootstrap(
    differences: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float]:
    if differences.ndim != 1 or not len(differences):
        raise ValueError("Paired bootstrap expects a nonempty vector.")
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[draws].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def outcome_key(record: dict[str, object]) -> str:
    side = int(record["first_grasp_side"])
    return {0: "a", 1: "b", 2: "both", -1: "none"}[side]


def load_pair(run_root: Path, pair_id: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    pair_dir = run_root / f"pair-{pair_id}"
    summary = json.loads((pair_dir / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (pair_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if summary.get("patching") is not False or summary.get("hooks_registered") is not False:
        raise ValueError(f"Pair {pair_id} is not an unmodified-policy run.")
    return summary, records


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    pair_ids = args.pair_ids if args.pair_ids is not None else list(range(args.num_pairs))
    if not pair_ids or len(set(pair_ids)) != len(pair_ids) or min(pair_ids) < 0:
        raise ValueError("pair-ids must be a nonempty list of unique nonnegative integers.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pair_results = []
    metric_rows = []
    pair_differences = []

    for pair_id in pair_ids:
        run_summary, records = load_pair(args.run_root, pair_id)
        pair = run_summary["pair"]
        object_names = pair["object_names"]
        indexed = {
            prompt_side: {
                int(record["init_state_id"]): record
                for record in records
                if int(record["prompt_side"]) == prompt_side
            }
            for prompt_side in (0, 1)
        }
        init_state_ids = sorted(indexed[0])
        if not init_state_ids or sorted(indexed[1]) != init_state_ids:
            raise ValueError(f"Pair {pair_id} prompt conditions are not paired by initial state.")

        compliance = {}
        ever_grasp = {}
        in_basket = {}
        outcome_distributions = {}
        compliance_arrays = {}
        for prompt_side in (0, 1):
            selected = [indexed[prompt_side][state] for state in init_state_ids]
            compliance_array = np.asarray(
                [record["commanded_first_grasp"] for record in selected], dtype=np.float64
            )
            compliance_arrays[prompt_side] = compliance_array
            compliance[str(prompt_side)] = wilson_interval(
                int(compliance_array.sum()), len(compliance_array)
            )
            ever_values = np.asarray(
                [record["commanded_ever_grasped"] for record in selected], dtype=np.int64
            )
            basket_values = np.asarray(
                [record["commanded_in_basket"] for record in selected], dtype=np.int64
            )
            ever_grasp[str(prompt_side)] = wilson_interval(
                int(ever_values.sum()), len(ever_values)
            )
            in_basket[str(prompt_side)] = wilson_interval(
                int(basket_values.sum()), len(basket_values)
            )
            outcomes = [outcome_key(record) for record in selected]
            outcome_distributions[str(prompt_side)] = {
                outcome: float(np.mean(np.asarray(outcomes) == outcome))
                for outcome in OUTCOMES
            }
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_name": pair["name"],
                    "scene_side": run_summary["scene_side"],
                    "prompt_side": prompt_side,
                    "prompt_object": object_names[prompt_side],
                    "condition": (
                        "aligned" if prompt_side == run_summary["scene_side"] else "swapped"
                    ),
                    "first_grasp_compliance": compliance[str(prompt_side)]["mean"],
                    "ci95_low": compliance[str(prompt_side)]["ci95_low"],
                    "ci95_high": compliance[str(prompt_side)]["ci95_high"],
                    "commanded_ever_grasp": ever_grasp[str(prompt_side)]["mean"],
                    "commanded_in_basket": in_basket[str(prompt_side)]["mean"],
                    **{
                        f"first_grasp_{outcome}": outcome_distributions[str(prompt_side)][outcome]
                        for outcome in OUTCOMES
                    },
                }
            )

        aligned_side = int(run_summary["scene_side"])
        swapped_side = 1 - aligned_side
        difference = compliance_arrays[aligned_side] - compliance_arrays[swapped_side]
        contrast = paired_bootstrap(difference, rng, args.bootstrap_samples)
        pair_differences.append(difference)
        pair_results.append(
            {
                "pair_id": pair_id,
                "pair_name": pair["name"],
                "object_names": object_names,
                "task_id": run_summary["task_id"],
                "scene_side": aligned_side,
                "num_paired_initial_states": len(init_state_ids),
                "commanded_first_grasp": compliance,
                "commanded_ever_grasp": ever_grasp,
                "commanded_in_basket": in_basket,
                "first_grasp_distribution": outcome_distributions,
                "aligned_minus_swapped_first_grasp_compliance": contrast,
            }
        )

    macro_draws = np.empty(args.bootstrap_samples, dtype=np.float64)
    for bootstrap_index in range(args.bootstrap_samples):
        pair_means = []
        for differences in pair_differences:
            indices = rng.integers(0, len(differences), size=len(differences))
            pair_means.append(float(differences[indices].mean()))
        macro_draws[bootstrap_index] = float(np.mean(pair_means))
    macro_point = float(np.mean([values.mean() for values in pair_differences]))
    macro_contrast = {
        "mean": macro_point,
        "ci95_low": float(np.quantile(macro_draws, 0.025)),
        "ci95_high": float(np.quantile(macro_draws, 0.975)),
    }

    scene_sides = sorted({int(result["scene_side"]) for result in pair_results})
    summary = {
        "protocol": {
            "model": "unmodified pi0.5 LIBERO policy",
            "design": "pair-local anchor scenes; same initial states under prompt A and prompt B",
            "primary_endpoint": "first-grasp command compliance",
            "uncertainty": "Wilson intervals for rates; paired state bootstrap for contrasts",
            "scope": f"pair ids {pair_ids}; scene sides {scene_sides}",
        },
        "pair_ids": pair_ids,
        "num_pairs": len(pair_ids),
        "num_paired_initial_states": int(sum(len(values) for values in pair_differences)),
        "num_rollouts": int(2 * sum(len(values) for values in pair_differences)),
        "pairs": pair_results,
        "pair_macro_aligned_minus_swapped_compliance": macro_contrast,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(pair_ids), figsize=(4.0 * len(pair_ids), 7.0))
    if len(pair_ids) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    colors = ["#2A708B", "#D0A33C"]
    for column_index, pair_result in enumerate(pair_results):
        object_names = pair_result["object_names"]
        rates = [
            pair_result["commanded_first_grasp"][str(side)]["mean"] for side in (0, 1)
        ]
        lower = [
            rates[side] - pair_result["commanded_first_grasp"][str(side)]["ci95_low"]
            for side in (0, 1)
        ]
        upper = [
            pair_result["commanded_first_grasp"][str(side)]["ci95_high"] - rates[side]
            for side in (0, 1)
        ]
        axes[0, column_index].bar(
            ["Prompt A", "Prompt B"],
            rates,
            color=colors,
            yerr=np.asarray([lower, upper]),
            capsize=4,
        )
        axes[0, column_index].set_ylim(0, 1.08)
        axes[0, column_index].set_title(f"{object_names[0]} /\n{object_names[1]}")
        axes[0, column_index].grid(axis="y", alpha=0.25)
        if column_index == 0:
            axes[0, column_index].set_ylabel("First-grasp command compliance")

        matrix = np.asarray(
            [
                [
                    pair_result["first_grasp_distribution"][str(side)][outcome]
                    for outcome in OUTCOMES
                ]
                for side in (0, 1)
            ]
        )
        image = axes[1, column_index].imshow(
            matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto"
        )
        axes[1, column_index].set_xticks(range(4), ["A", "B", "Both", "None"])
        axes[1, column_index].set_yticks(range(2), ["Prompt A", "Prompt B"])
        axes[1, column_index].set_xlabel("First grasp")
        for row in range(2):
            for column in range(4):
                axes[1, column_index].text(
                    column,
                    row,
                    f"{matrix[row, column]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > 0.55 else "#17212B",
                )
    figure.colorbar(image, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)
    figure.suptitle("pi0.5 instruction compliance under fixed pair-local scenes")
    figure.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.95, hspace=0.42, wspace=0.3)
    figure.savefig(args.output_dir / "instruction_compliance.png", dpi=220)
    figure.savefig(args.output_dir / "instruction_compliance.pdf")
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
