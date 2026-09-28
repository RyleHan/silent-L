#!/usr/bin/env python3
"""Summarize the fixed held-out COAST replication pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ("baseline", "coast", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, samples: int) -> dict[str, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def paired_transitions(
    reference: np.ndarray, coast: np.ndarray
) -> dict[str, float | int]:
    both_failure = int(np.sum((reference == 0) & (coast == 0)))
    rescued_by_coast = int(np.sum((reference == 0) & (coast == 1)))
    harmed_by_coast = int(np.sum((reference == 1) & (coast == 0)))
    both_success = int(np.sum((reference == 1) & (coast == 1)))
    discordant = rescued_by_coast + harmed_by_coast
    if discordant:
        smaller = min(rescued_by_coast, harmed_by_coast)
        exact_p = min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(smaller + 1))
            / (2**discordant),
        )
    else:
        exact_p = 1.0
    return {
        "both_failure": both_failure,
        "rescued_by_coast": rescued_by_coast,
        "harmed_by_coast": harmed_by_coast,
        "both_success": both_success,
        "discordant_states": discordant,
        "net_success_gain": rescued_by_coast - harmed_by_coast,
        "exact_two_sided_sign_p": exact_p,
    }


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in (args.run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    run_summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    fit_summary = json.loads((args.fit_dir / "summary.json").read_text(encoding="utf-8"))
    indexed = {
        condition: {
            int(record["init_state_id"]): record
            for record in records
            if record["condition"] == condition
        }
        for condition in CONDITIONS
    }
    init_state_ids = sorted(indexed["baseline"])
    if any(sorted(indexed[condition]) != init_state_ids for condition in CONDITIONS):
        raise ValueError("Conditions do not share identical held-out initial states.")
    rng = np.random.default_rng(args.seed)
    arrays = {
        condition: np.asarray(
            [indexed[condition][state]["success"] for state in init_state_ids],
            dtype=np.float64,
        )
        for condition in CONDITIONS
    }
    estimates = {
        condition: bootstrap_mean(values, rng, args.bootstrap_samples)
        for condition, values in arrays.items()
    }
    contrasts = {}
    for name, first, second in (
        ("coast_minus_baseline", "coast", "baseline"),
        ("coast_minus_random", "coast", "random"),
        ("random_minus_baseline", "random", "baseline"),
    ):
        contrasts[name] = bootstrap_mean(
            arrays[first] - arrays[second], rng, args.bootstrap_samples
        )
    transitions = {
        "coast_vs_baseline": paired_transitions(arrays["baseline"], arrays["coast"]),
        "coast_vs_random": paired_transitions(arrays["random"], arrays["coast"]),
    }

    audit = run_summary["sampler_audits"]
    steered_records = [
        record for record in records if record["condition"] in {"coast", "random"}
    ]
    operator_valid = (
        audit.get("beta_zero_vs_official_max_abs_difference") == 0.0
        and audit.get("beta_zero_hook_count") == 0
        and min(record["hook_count_min"] for record in steered_records) == 10
        and max(record["hook_count_max"] for record in steered_records) == 10
        and fit_summary["diagnostics"]["steering"]["min_eigenvalue"] >= -1e-5
        and fit_summary["diagnostics"]["steering"]["max_eigenvalue"] <= 1.0 + 1e-5
    )
    improves_baseline = contrasts["coast_minus_baseline"]["mean"] > 0
    primary_strong = (
        operator_valid and contrasts["coast_minus_baseline"]["ci95_low"] > 0
    )
    direction_specific = (
        operator_valid and contrasts["coast_minus_random"]["ci95_low"] > 0
    )
    if primary_strong and direction_specific:
        decision = "strong_specific_replication"
    elif primary_strong:
        decision = "primary_replication_without_random_specificity"
    elif operator_valid and improves_baseline:
        decision = "positive_but_underpowered"
    else:
        decision = "not_replicated"

    summary = {
        "protocol": {
            "task": "LIBERO-10 KS3",
            "fit": "15 public on-policy rollouts: 8 success / 7 failure",
            "test": "30 disjoint initial states",
            "configuration": "global, expert layer 5, aperture 0.5, beta 0.1",
            "selection": "published configuration; no local test sweep",
            "control": "matched-eigenvalue-spectrum random rotation",
            "uncertainty": "paired initial-state bootstrap, 95% CI",
        },
        "num_heldout_states": len(init_state_ids),
        "init_state_ids": init_state_ids,
        "conditions": estimates,
        "contrasts": contrasts,
        "paired_transitions": transitions,
        "operator_valid": operator_valid,
        "decision": decision,
        "decision_rules": {
            "strong_specific_replication": "COAST beats baseline and random with paired CI lower bounds > 0",
            "primary_replication_without_random_specificity": "COAST beats baseline with CI lower bound > 0, but not the stronger random control",
            "positive_but_underpowered": "COAST-baseline point estimate > 0 but its CI includes 0",
            "not_replicated": "invalid operator or COAST-baseline point estimate is non-positive",
        },
        "run_sampler_audits": audit,
        "fit_diagnostics": fit_summary["diagnostics"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "name", "mean", "ci95_low", "ci95_high"])
        writer.writeheader()
        for name, estimate in estimates.items():
            writer.writerow({"kind": "condition", "name": name, **estimate})
        for name, estimate in contrasts.items():
            writer.writerow({"kind": "contrast", "name": name, **estimate})

    figure, (rate_axis, transition_axis) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    labels = ["Baseline", "COAST", "Random spectrum"]
    means = [estimates[name]["mean"] for name in CONDITIONS]
    errors = np.asarray(
        [
            [means[i] - estimates[name]["ci95_low"] for i, name in enumerate(CONDITIONS)],
            [estimates[name]["ci95_high"] - means[i] for i, name in enumerate(CONDITIONS)],
        ]
    )
    rate_axis.bar(
        labels,
        means,
        color=["#3182A4", "#45A98C", "#929DB2"],
        yerr=errors,
        capsize=5,
    )
    rate_axis.set_ylim(0, 1.08)
    rate_axis.set_ylabel("Closed-loop success rate")
    rate_axis.set_title("Held-out success (30 states)")
    rate_axis.grid(axis="y", alpha=0.3)

    comparison_labels = ["vs baseline", "vs random"]
    rescued = [
        transitions["coast_vs_baseline"]["rescued_by_coast"],
        transitions["coast_vs_random"]["rescued_by_coast"],
    ]
    harmed = [
        transitions["coast_vs_baseline"]["harmed_by_coast"],
        transitions["coast_vs_random"]["harmed_by_coast"],
    ]
    positions = np.arange(len(comparison_labels))
    width = 0.34
    transition_axis.bar(
        positions - width / 2,
        rescued,
        width,
        color="#45A98C",
        label="Rescued by COAST",
    )
    transition_axis.bar(
        positions + width / 2,
        harmed,
        width,
        color="#C95B53",
        label="Harmed by COAST",
    )
    transition_axis.set_xticks(positions, comparison_labels)
    transition_axis.set_ylabel("Number of initial states")
    transition_axis.set_title("Paired outcome changes")
    transition_axis.legend(frameon=False)
    transition_axis.grid(axis="y", alpha=0.3)
    figure.suptitle("COAST paper-guided reimplementation: pi0.5 on LIBERO-10 KS3")
    figure.tight_layout()
    figure.savefig(args.output_dir / "coast_pilot.png", dpi=220)
    figure.savefig(args.output_dir / "coast_pilot.pdf")
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
