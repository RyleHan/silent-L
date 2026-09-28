#!/usr/bin/env python3
"""Stage 13: relate the instruction-induced goal shift to swapped-instruction compliance."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from vla_coordinates.goal_probes import ProbeFamily, assign


CONDITIONS = ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (3, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--laso-root", type=Path, required=True)
    parser.add_argument("--lopo-root", type=Path, required=True)
    parser.add_argument("--stage12-reads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-layer", type=int, default=13)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--indeterminate-margin", type=float, default=0.02)
    parser.add_argument("--validity-floor", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2 + 1
        start = stop + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def exact_permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    observed = spearman(x, y)
    rhos = [spearman(x, np.asarray(perm)) for perm in itertools.permutations(y)]
    return float(np.mean([rho >= observed - 1e-12 for rho in rhos]))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def family(root: Path, name: str, layer: int, seeds: list[int], subdir: str) -> ProbeFamily:
    return ProbeFamily(
        name,
        "expert_action_mean_t1",
        layer,
        [root / f"seed-{seed}" / subdir / "expert_action_mean" / f"layer_{layer:02d}.pt" for seed in seeds],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    rng = np.random.default_rng(20260928)

    conditions = []
    rows = []
    for pair_id, scene_side in CONDITIONS:
        condition_dir = args.run_root / f"pair-{pair_id}-side-{scene_side}"
        summary = json.loads((condition_dir / "summary.json").read_text(encoding="utf-8"))
        with (condition_dir / "episodes.jsonl").open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        task_id = int(summary["task_id"])
        families = {
            "laso": family(args.laso_root / f"anchor-task-{task_id}", "laso", args.expert_layer, seeds, ""),
            "lopo": family(args.lopo_root, "lopo", args.expert_layer, seeds, f"pair-{pair_id}"),
        }
        swapped_side = 1 - scene_side
        per_state: dict[int, dict] = {}
        for record in records:
            residuals = np.load(condition_dir / record["residual_file"])
            object_a = np.asarray(record["initial_object_a_xyz"], dtype=np.float32)
            object_b = np.asarray(record["initial_object_b_xyz"], dtype=np.float32)
            entry = per_state.setdefault(record["init_state_id"], {})
            reads = {}
            for name, probe in families.items():
                features = residuals[probe.readout_key][0, probe.layer].astype(np.float32)
                decoded, _ = probe.decode(features[None])
                read = assign(decoded[0], object_a, object_b, args.indeterminate_margin)
                # Orient the axis from the scene's native object (0) to the swapped object (1).
                towards_swapped = (
                    read["interpolation_a0_b1"] if scene_side == 0 else 1 - read["interpolation_a0_b1"]
                )
                reads[name] = {"side": read["side"], "towards_swapped": towards_swapped,
                               "interpolation_a0_b1": read["interpolation_a0_b1"]}
            entry[record["prompt_side"]] = {"record": record, "reads": reads}

        condition = {
            "pair_id": pair_id,
            "pair_name": summary["pair"]["name"],
            "object_names": summary["pair"]["object_names"],
            "scene_side": scene_side,
            "task_id": task_id,
            "audits": summary["audits"],
            "probes": {name: probe.provenance() for name, probe in families.items()},
        }
        swapped_records = [entry[swapped_side]["record"] for entry in per_state.values()]
        condition["swapped_compliance"] = float(
            np.mean([record["commanded_first_grasp"] for record in swapped_records])
        )
        condition["native_compliance"] = float(
            np.mean([entry[scene_side]["record"]["commanded_first_grasp"] for entry in per_state.values()])
        )
        condition["swapped_first_grasp"] = {
            key: int(sum(record["first_grasp_side"] == code for record in swapped_records))
            for key, code in (("native_object", scene_side), ("instructed_object", swapped_side),
                              ("both", 2), ("none", -1))
        }
        for name in families:
            compliant = [
                entry[scene_side]["reads"][name]["side"]
                for entry in per_state.values()
                if entry[scene_side]["record"]["first_grasp_side"] == scene_side
            ]
            determinate = [side for side in compliant if side is not None]
            correct = sum(side == scene_side for side in determinate)
            rate = correct / len(determinate) if determinate else float("nan")
            shifts = np.asarray([
                entry[swapped_side]["reads"][name]["towards_swapped"]
                - entry[scene_side]["reads"][name]["towards_swapped"]
                for entry in per_state.values()
            ])
            condition[name] = {
                "validity_floor": {
                    "compliant_aligned": len(compliant),
                    "determinate": len(determinate),
                    "correct": correct,
                    "rate": rate,
                    "passed": bool(determinate) and rate >= args.validity_floor
                    and len(compliant) - len(determinate) <= len(compliant) / 2,
                },
                "mean_shift": float(shifts.mean()),
                "shift_bootstrap_95": [
                    float(np.percentile(boot, q))
                    for boot in [np.asarray([rng.choice(shifts, len(shifts)).mean()
                                             for _ in range(args.bootstrap_samples)])]
                    for q in (2.5, 97.5)
                ],
                "positive_shift_states": int((shifts > 0).sum()),
                "mean_native_prompt_position": float(np.mean(
                    [entry[scene_side]["reads"][name]["towards_swapped"] for entry in per_state.values()])),
                "mean_swapped_prompt_position": float(np.mean(
                    [entry[swapped_side]["reads"][name]["towards_swapped"] for entry in per_state.values()])),
            }
        conditions.append(condition)
        for state_id, entry in sorted(per_state.items()):
            for name in families:
                rows.append({
                    "pair_id": pair_id,
                    "scene_side": scene_side,
                    "init_state_id": state_id,
                    "probe": name,
                    "native_position": entry[scene_side]["reads"][name]["towards_swapped"],
                    "swapped_position": entry[swapped_side]["reads"][name]["towards_swapped"],
                    "shift": entry[swapped_side]["reads"][name]["towards_swapped"]
                    - entry[scene_side]["reads"][name]["towards_swapped"],
                    "swapped_first_grasp_side": entry[swapped_side]["record"]["first_grasp_side"],
                    "swapped_commanded": int(entry[swapped_side]["record"]["commanded_first_grasp"]),
                })

    # Stage 12 consistency: conditions 0A and 3A must reproduce the Stage 12 first-query reads.
    stage12 = {}
    with args.stage12_reads.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stage12[(int(row["pair_id"]), int(row["init_state_id"]), int(row["prompt_side"]))] = float(
                row["laso_expert_interpolation"])
    stage12_max_difference = 0.0
    stage12_matched = 0
    for row in rows:
        if row["probe"] != "laso" or row["scene_side"] != 0 or row["pair_id"] not in (0, 3):
            continue
        for prompt_side, position in ((0, row["native_position"]), (1, row["swapped_position"])):
            reference = stage12[(row["pair_id"], row["init_state_id"], prompt_side)]
            stage12_max_difference = max(stage12_max_difference, abs(reference - position))
            stage12_matched += 1

    results = {}
    for name in ("laso", "lopo"):
        included = [condition for condition in conditions if condition[name]["validity_floor"]["passed"]]
        shifts = np.asarray([condition[name]["mean_shift"] for condition in included])
        compliance = np.asarray([condition["swapped_compliance"] for condition in included])
        result = {
            "included_conditions": [f"{c['pair_id']}{'AB'[c['scene_side']]}" for c in included],
            "excluded_conditions": [f"{c['pair_id']}{'AB'[c['scene_side']]}" for c in conditions
                                    if c not in included],
        }
        if len(included) >= 3:
            result["spearman_rho"] = spearman(shifts, compliance)
            result["exact_permutation_p_one_sided"] = exact_permutation_p(shifts, compliance)
        rollouts = [row for row in rows if row["probe"] == name
                    and any(row["pair_id"] == c["pair_id"] and row["scene_side"] == c["scene_side"]
                            for c in included)]
        scores = np.asarray([row["shift"] for row in rollouts])
        labels = np.asarray([row["swapped_commanded"] for row in rollouts])
        result["pooled_rollout_auc"] = auc(scores, labels)
        keys = sorted({(row["pair_id"], row["scene_side"]) for row in rollouts})
        boots = []
        for _ in range(args.bootstrap_samples):
            sample = [keys[index] for index in rng.integers(0, len(keys), len(keys))]
            chosen = [row for key in sample for row in rollouts if (row["pair_id"], row["scene_side"]) == key]
            value = auc(np.asarray([row["shift"] for row in chosen]),
                        np.asarray([row["swapped_commanded"] for row in chosen]))
            if not np.isnan(value):
                boots.append(value)
        result["pooled_rollout_auc_condition_bootstrap_95"] = (
            [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
        )
        within = {}
        for key in keys:
            subset = [row for row in rollouts if (row["pair_id"], row["scene_side"]) == key]
            subset_labels = np.asarray([row["swapped_commanded"] for row in subset])
            if subset_labels.sum() >= 5 and (1 - subset_labels).sum() >= 5:
                within[f"{key[0]}{'AB'[key[1]]}"] = auc(
                    np.asarray([row["shift"] for row in subset]), subset_labels)
        result["within_condition_auc"] = within
        results[name] = result

    output = {
        "protocol": "docs/protocols/stage13_language_shift.md (exploratory, analysis fixed in advance)",
        "all_outcomes_reproduce_stage11": all(
            c["audits"]["all_stage11_outcomes_reproduced"] for c in conditions),
        "max_hooked_vs_unhooked_action_error": max(
            c["audits"]["max_first_query_hooked_vs_unhooked_action_error"] for c in conditions),
        "stage12_consistency": {"reads_compared": stage12_matched,
                                "max_abs_difference": stage12_max_difference},
        "primary": results["laso"],
        "robustness_lopo": results["lopo"],
        "conditions": conditions,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: output[key] for key in ("all_outcomes_reproduce_stage11",
                      "max_hooked_vs_unhooked_action_error", "stage12_consistency",
                      "primary", "robustness_lopo")}, indent=2))
    for condition in conditions:
        print(condition["pair_id"], "AB"[condition["scene_side"]],
              f"compliance={condition['swapped_compliance']:.2f}",
              f"laso shift={condition['laso']['mean_shift']:.2f} floor={condition['laso']['validity_floor']['passed']}",
              f"lopo shift={condition['lopo']['mean_shift']:.2f} floor={condition['lopo']['validity_floor']['passed']}")


if __name__ == "__main__":
    main()
