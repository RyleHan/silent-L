#!/usr/bin/env python3
"""Inspect LIBERO semantic observations after replaying cached simulator states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=3)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)),
            "max": float(np.nanmax(value)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def representative_indices(indices: np.ndarray, count: int) -> list[int]:
    if len(indices) <= count:
        return indices.astype(int).tolist()
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return indices[positions].astype(int).tolist()


def model_names(model: Any, kind: str) -> list[str]:
    names = getattr(model, f"{kind}_names", None)
    if names is not None:
        return [str(name) for name in names]
    count = int(getattr(model, f"n{kind}"))
    id_to_name = getattr(model, f"{kind}_id2name")
    return [str(id_to_name(index)) for index in range(count)]


def state_interfaces(env: SegmentationRenderEnv) -> dict[str, Any]:
    result = {}
    for name, state in env.env.object_states_dict.items():
        result[name] = {
            "type": type(state).__name__,
            "public_methods": [
                method
                for method in dir(state)
                if not method.startswith("_") and callable(getattr(state, method))
            ],
        }
    return result


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    report: dict[str, Any] = {
        "source_h5": str(args.source_h5),
        "samples_per_task": args.samples_per_task,
        "tasks": [],
    }

    with h5py.File(args.source_h5, "r") as source:
        task_ids = np.asarray(source["task_ids"])
        states = source["simulator_states"]
        action_steps = np.asarray(source["action_steps"])
        episode_ids = np.asarray(source["episode_ids"])
        for task_id in sorted(np.unique(task_ids).astype(int).tolist()):
            task = suite.get_task(task_id)
            bddl_path = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            env = SegmentationRenderEnv(
                bddl_file_name=bddl_path,
                camera_names=["agentview"],
                camera_heights=256,
                camera_widths=256,
                camera_segmentations="instance",
            )
            try:
                task_indices = np.flatnonzero(task_ids == task_id)
                selected = representative_indices(task_indices, args.samples_per_task)
                samples = []
                for index in selected:
                    env.reset()
                    observation = env.set_init_state(np.asarray(states[index]))
                    semantic_observations = {
                        key: jsonable(np.asarray(value))
                        for key, value in observation.items()
                        if "image" not in key and "segmentation" not in key
                    }
                    samples.append(
                        {
                            "index": index,
                            "episode_id": int(episode_ids[index]),
                            "action_step": int(action_steps[index]),
                            "success": bool(env.check_success()),
                            "semantic_observations": semantic_observations,
                        }
                    )

                base_env = env.env
                model = env.sim.model
                report["tasks"].append(
                    {
                        "task_id": task_id,
                        "language": task.language,
                        "bddl_path": bddl_path,
                        "obj_of_interest": jsonable(base_env.obj_of_interest),
                        "parsed_problem": jsonable(base_env.parsed_problem),
                        "objects": sorted(base_env.objects_dict),
                        "fixtures": sorted(base_env.fixtures_dict),
                        "object_sites": sorted(base_env.object_sites_dict),
                        "object_state_interfaces": state_interfaces(env),
                        "instance_to_id": jsonable(env.instance_to_id),
                        "body_names": model_names(model, "body"),
                        "site_names": model_names(model, "site"),
                        "joint_names": model_names(model, "joint"),
                        "samples": samples,
                    }
                )
            finally:
                env.close()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Wrote: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
