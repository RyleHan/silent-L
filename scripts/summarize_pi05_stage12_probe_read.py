#!/usr/bin/env python3
"""Apply stored goal-state probes to Stage 12 first-query residuals and apply the decision rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from vla_coordinates.goal_probes import ProbeFamily, assign, wilson_interval


FIRST_GRASP_NONE = -1
FIRST_GRASP_BOTH = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--laso-root", type=Path, required=True)
    parser.add_argument("--lopo-root", type=Path, required=True)
    parser.add_argument("--pooled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-pair", type=int, default=0)
    parser.add_argument("--secondary-pairs", default="3")
    parser.add_argument("--expert-layer", type=int, default=13)
    parser.add_argument("--descriptive-expert-layer", type=int, default=16)
    parser.add_argument("--prefix-layer", type=int, default=12)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--indeterminate-margin", type=float, default=0.02)
    parser.add_argument("--validity-floor", type=float, default=0.90)
    parser.add_argument("--dissociation-threshold", type=float, default=0.70)
    parser.add_argument("--tracking-threshold", type=float, default=0.30)
    return parser.parse_args()


def build_families(args: argparse.Namespace, pair_anchor_tasks: dict[int, int]) -> dict:
    seeds = [int(value) for value in args.seeds.split(",")]
    families: dict[int, dict[str, ProbeFamily]] = {}
    for pair_id, anchor_task in pair_anchor_tasks.items():
        laso = args.laso_root / f"anchor-task-{anchor_task}"
        families[pair_id] = {
            "laso_expert": ProbeFamily(
                "laso_expert",
                "expert_action_mean_t1",
                args.expert_layer,
                [
                    laso / f"seed-{seed}" / "expert_action_mean" / f"layer_{args.expert_layer:02d}.pt"
                    for seed in seeds
                ],
            ),
            "lopo_expert": ProbeFamily(
                "lopo_expert",
                "expert_action_mean_t1",
                args.expert_layer,
                [
                    args.lopo_root
                    / f"seed-{seed}"
                    / f"pair-{pair_id}"
                    / "expert_action_mean"
                    / f"layer_{args.expert_layer:02d}.pt"
                    for seed in seeds
                ],
            ),
            "pooled_expert_contaminated": ProbeFamily(
                "pooled_expert_contaminated",
                "expert_action_mean_t1",
                args.expert_layer,
                [
                    args.pooled_root
                    / f"seed-{seed}"
                    / "expert_action_mean"
                    / f"layer_{args.expert_layer:02d}.pt"
                    for seed in seeds
                ],
            ),
            "laso_expert_layer_descriptive": ProbeFamily(
                "laso_expert_layer_descriptive",
                "expert_action_mean_t1",
                args.descriptive_expert_layer,
                [
                    laso
                    / f"seed-{seed}"
                    / "expert_action_mean"
                    / f"layer_{args.descriptive_expert_layer:02d}.pt"
                    for seed in seeds
                ],
            ),
            "laso_prefix": ProbeFamily(
                "laso_prefix",
                "prefix_prompt_end",
                args.prefix_layer,
                [
                    laso / f"seed-{seed}" / "prompt_end" / f"layer_{args.prefix_layer:02d}.pt"
                    for seed in seeds
                ],
            ),
        }
    return families


def read_rollouts(pair_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((pair_dir / "summary.json").read_text(encoding="utf-8"))
    with (pair_dir / "episodes.jsonl").open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    return summary, records


def probe_reads(
    records: list[dict],
    pair_dir: Path,
    families: dict[str, ProbeFamily],
    margin: float,
) -> list[dict]:
    reads = []
    for record in records:
        residuals = np.load(pair_dir / record["residual_file"])
        object_a = np.asarray(record["initial_object_a_xyz"], dtype=np.float32)
        object_b = np.asarray(record["initial_object_b_xyz"], dtype=np.float32)
        read = {
            "init_state_id": record["init_state_id"],
            "prompt_side": record["prompt_side"],
            "condition": record["condition"],
            "first_grasp_side": record["first_grasp_side"],
            "object_ab_distance": float(np.linalg.norm(object_b - object_a)),
            "families": {},
            "trajectory": {},
        }
        for name, family in families.items():
            features = residuals[family.readout_key][:, family.layer].astype(np.float32)
            decoded_ensemble, decoded_seeds = family.decode(features)
            first = assign(decoded_ensemble[0], object_a, object_b, margin)
            first["decoded_xyz"] = decoded_ensemble[0].tolist()
            first["per_seed_sides"] = [
                assign(decoded_seeds[index, 0], object_a, object_b, margin)["side"]
                for index in range(decoded_seeds.shape[0])
            ]
            read["families"][name] = first
            # Trajectory reads compare against the objects' current positions at each query.
            query_objects = residuals["query_object_xyz"]
            read["trajectory"][name] = [
                assign(decoded_ensemble[query], query_objects[query, 0], query_objects[query, 1], margin)[
                    "side"
                ]
                for query in range(len(decoded_ensemble))
            ]
        reads.append(read)
    return reads


def validity_floor(reads: list[dict], family: str, scene_side: int, floor: float) -> dict:
    compliant = [
        read
        for read in reads
        if read["prompt_side"] == scene_side and read["first_grasp_side"] == scene_side
    ]
    sides = [read["families"][family]["side"] for read in compliant]
    determinate = [side for side in sides if side is not None]
    correct = sum(side == scene_side for side in determinate)
    rate = correct / len(determinate) if determinate else float("nan")
    indeterminate = len(sides) - len(determinate)
    passed = bool(determinate) and rate >= floor and indeterminate <= len(sides) / 2
    return {
        "compliant_aligned_rollouts": len(sides),
        "determinate": len(determinate),
        "indeterminate": indeterminate,
        "probe_points_at_instructed": correct,
        "rate": rate,
        "wilson_95": wilson_interval(correct, len(determinate)),
        "floor": floor,
        "passed": passed,
    }


def cross_tab(reads: list[dict], family: str, prompt_side: int) -> dict:
    table: dict[str, int] = {}
    for read in reads:
        if read["prompt_side"] != prompt_side:
            continue
        grasp = read["first_grasp_side"]
        behaviour = (
            "grasp_instructed"
            if grasp == prompt_side
            else "grasp_other"
            if grasp == 1 - prompt_side
            else "grasp_both"
            if grasp == FIRST_GRASP_BOTH
            else "grasp_none"
        )
        side = read["families"][family]["side"]
        probe = (
            "indeterminate"
            if side is None
            else "probe_instructed"
            if side == prompt_side
            else "probe_other"
        )
        key = f"{behaviour}|{probe}"
        table[key] = table.get(key, 0) + 1
    return dict(sorted(table.items()))


def dissociation(reads: list[dict], family: str, prompt_side: int, args: argparse.Namespace) -> dict:
    wrong = [
        read
        for read in reads
        if read["prompt_side"] == prompt_side and read["first_grasp_side"] == 1 - prompt_side
    ]
    sides = [read["families"][family]["side"] for read in wrong]
    determinate = [side for side in sides if side is not None]
    dissociated = sum(side == prompt_side for side in determinate)
    rate = dissociated / len(determinate) if determinate else float("nan")
    interval = wilson_interval(dissociated, len(determinate))
    if not determinate:
        decision = "inconclusive"
    elif interval[0] > args.dissociation_threshold:
        decision = "dissociation"
    elif interval[1] < args.tracking_threshold:
        decision = "probe_tracks_behavior"
    else:
        decision = "inconclusive"
    per_seed = []
    for seed_index in range(len(wrong[0]["families"][family]["per_seed_sides"]) if wrong else 0):
        seed_sides = [
            read["families"][family]["per_seed_sides"][seed_index]
            for read in wrong
            if read["families"][family]["per_seed_sides"][seed_index] is not None
        ]
        per_seed.append(
            {
                "determinate": len(seed_sides),
                "rate": (
                    sum(side == prompt_side for side in seed_sides) / len(seed_sides)
                    if seed_sides
                    else float("nan")
                ),
            }
        )
    return {
        "wrong_object_rollouts": len(sides),
        "determinate": len(determinate),
        "indeterminate": len(sides) - len(determinate),
        "probe_points_at_instructed": dissociated,
        "dissociation_rate": rate,
        "wilson_95": interval,
        "decision": decision,
        "per_seed": per_seed,
    }


def trajectory_profile(reads: list[dict], family: str, prompt_side: int, grasp_side: int) -> dict:
    selected = [
        read
        for read in reads
        if read["prompt_side"] == prompt_side and read["first_grasp_side"] == grasp_side
    ]
    max_queries = max((len(read["trajectory"][family]) for read in selected), default=0)
    profile = []
    for query in range(max_queries):
        sides = [
            read["trajectory"][family][query]
            for read in selected
            if query < len(read["trajectory"][family])
        ]
        determinate = [side for side in sides if side is not None]
        profile.append(
            {
                "query": query,
                "rollouts": len(sides),
                "determinate": len(determinate),
                "fraction_instructed": (
                    sum(side == prompt_side for side in determinate) / len(determinate)
                    if determinate
                    else float("nan")
                ),
            }
        )
    return {"prompt_side": prompt_side, "first_grasp_side": grasp_side, "profile": profile}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = [args.primary_pair] + [int(value) for value in args.secondary_pairs.split(",") if value]

    pair_data = {}
    for pair_id in pair_ids:
        pair_dir = args.run_root / f"pair-{pair_id}"
        summary, records = read_rollouts(pair_dir)
        pair_data[pair_id] = (pair_dir, summary, records)
    anchor_tasks = {pair_id: int(data[1]["task_id"]) for pair_id, data in pair_data.items()}
    families = build_families(args, anchor_tasks)

    audits = {}
    for pair_id, (_, summary, records) in pair_data.items():
        audit = summary["audits"]
        audits[str(pair_id)] = {
            "rollouts": len(records),
            "all_stage11_outcomes_reproduced": audit["all_stage11_outcomes_reproduced"],
            "stage11_mismatches": [
                {
                    "init_state_id": record["init_state_id"],
                    "prompt_side": record["prompt_side"],
                    "mismatches": record["stage11_mismatches"],
                }
                for record in records
                if not record["stage11_match"]
            ],
            "max_first_query_hooked_vs_unhooked_action_error": audit[
                "max_first_query_hooked_vs_unhooked_action_error"
            ],
            "patching": audit["patching"],
        }
    valid_run = all(
        value["all_stage11_outcomes_reproduced"]
        and value["max_first_query_hooked_vs_unhooked_action_error"] == 0.0
        and value["patching"] is False
        for value in audits.values()
    )

    results = {}
    rows = []
    for pair_id, (pair_dir, summary, records) in pair_data.items():
        scene_side = int(summary["scene_side"])
        swapped_side = 1 - scene_side
        reads = probe_reads(records, pair_dir, families[pair_id], args.indeterminate_margin)
        pair_result = {"task_id": anchor_tasks[pair_id], "scene_side": scene_side, "families": {}}
        for name in families[pair_id]:
            pair_result["families"][name] = {
                "provenance": families[pair_id][name].provenance(),
                "validity_floor": validity_floor(reads, name, scene_side, args.validity_floor),
                "dissociation": dissociation(reads, name, swapped_side, args),
                "cross_tab_swapped": cross_tab(reads, name, swapped_side),
                "cross_tab_aligned": cross_tab(reads, name, scene_side),
                "trajectory_wrong_object": trajectory_profile(
                    reads, name, swapped_side, scene_side
                ),
                "trajectory_aligned_compliant": trajectory_profile(
                    reads, name, scene_side, scene_side
                ),
            }
        results[str(pair_id)] = pair_result
        for read in reads:
            row = {
                "pair_id": pair_id,
                "init_state_id": read["init_state_id"],
                "prompt_side": read["prompt_side"],
                "condition": read["condition"],
                "first_grasp_side": read["first_grasp_side"],
                "object_ab_distance": read["object_ab_distance"],
            }
            for name, value in read["families"].items():
                row[f"{name}_side"] = "" if value["side"] is None else value["side"]
                row[f"{name}_interpolation"] = value["interpolation_a0_b1"]
                row[f"{name}_distance_a"] = value["distance_a"]
                row[f"{name}_distance_b"] = value["distance_b"]
            rows.append(row)

    primary = results[str(args.primary_pair)]["families"]
    primary_floor = primary["laso_expert"]["validity_floor"]
    primary_dissociation = primary["laso_expert"]["dissociation"]
    lopo_floor = primary["lopo_expert"]["validity_floor"]
    lopo_dissociation = primary["lopo_expert"]["dissociation"]
    if not valid_run:
        decision = "invalid_reproduction_or_hook_audit"
        reason = "Stage 11 outcomes or hook audits did not reproduce exactly."
    elif not primary_floor["passed"]:
        decision = "probe_transfer_failure"
        reason = "LASO probe failed the validity floor on aligned compliant rollouts."
    else:
        decision = primary_dissociation["decision"]
        reason = "LASO primary rule applied."
        if lopo_floor["passed"] and not math.isnan(lopo_dissociation["dissociation_rate"]):
            primary_direction = primary_dissociation["dissociation_rate"] > 0.5
            lopo_direction = lopo_dissociation["dissociation_rate"] > 0.5
            if primary_direction != lopo_direction:
                decision = "inconclusive"
                reason = "LASO and LOPO disagree in direction; LOPO passed its validity floor."
            else:
                reason += " LOPO passed its floor and agrees in direction."
        else:
            reason += " LOPO failed its validity floor and has no direction veto."

    output = {
        "protocol": "docs/protocols/stage12_probe_read.md with Amendment 1 (2026-09-28)",
        "decision": decision,
        "decision_reason": reason,
        "primary": {
            "pair_id": args.primary_pair,
            "probe": "laso_expert",
            "expert_layer": args.expert_layer,
            "validity_floor": primary_floor,
            "dissociation": primary_dissociation,
            "lopo_validity_floor": lopo_floor,
            "lopo_dissociation": lopo_dissociation,
        },
        "parameters": {
            "indeterminate_margin_m": args.indeterminate_margin,
            "validity_floor": args.validity_floor,
            "dissociation_threshold": args.dissociation_threshold,
            "tracking_threshold": args.tracking_threshold,
            "seeds": args.seeds,
            "decoded_xyz": "mean over seed probes",
        },
        "audits": audits,
        "valid_run": valid_run,
        "pairs": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "reads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: output[key] for key in ("decision", "decision_reason", "primary", "valid_run")}, indent=2))


if __name__ == "__main__":
    main()
