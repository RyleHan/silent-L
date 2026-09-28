#!/usr/bin/env python3
"""Audit configured object pairs against LIBERO's runtime task ordering and instances."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv

from vla_coordinates.libero_object_pairs import load_pair_config, task_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pair_config(args.pair_config)["pairs"]
    configured_tasks = task_lookup(pairs)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    records = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = SegmentationRenderEnv(
            bddl_file_name=bddl_path,
            camera_names=["agentview"],
            camera_heights=64,
            camera_widths=64,
            camera_segmentations="instance",
        )
        try:
            env.reset()
            instances = sorted(env.instance_to_id)
        finally:
            env.close()
        record: dict[str, object] = {
            "task_id": task_id,
            "language": task.language,
            "bddl_file": task.bddl_file,
            "instances": instances,
        }
        if task_id in configured_tasks:
            pair, side = configured_tasks[task_id]
            expected = pair["object_instances"]
            missing = sorted(set(expected) - set(instances))
            target_name = pair["object_names"][side]
            record.update(
                {
                    "configured_pair_id": pair["pair_id"],
                    "configured_target_side": side,
                    "configured_target_name": target_name,
                    "missing_pair_instances": missing,
                    "language_matches_target": target_name in task.language.lower(),
                    "valid": not missing and target_name in task.language.lower(),
                }
            )
        records.append(record)

    result = {
        "suite": "libero_object",
        "num_tasks": suite.n_tasks,
        "all_configured_tasks_valid": all(
            bool(record.get("valid", True)) for record in records
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["all_configured_tasks_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
