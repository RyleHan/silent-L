#!/usr/bin/env python3
"""Run matched LIBERO episodes with the official OpenVLA FFN intervention."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from libero_experiments.hooks import apply_gate_proj_hooks
from libero_experiments.interventions import load_intervention_dict
from vla_coordinates.openvla_runtime import (
    build_openvla_prompt,
    load_openvla,
    predict_openvla_action,
    prepare_openvla_inputs,
)


DEFAULT_CHECKPOINT = "openvla/openvla-7b-finetuned-libero-10"
DEFAULT_REVISION = "80970322773f81baa2e22fe495d0487b93a05cfa"
INTERMEDIATE_SIZE = 11008


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--conditions", default="baseline,up_10,random_10")
    parser.add_argument("--semantic-dict", default="up_10")
    parser.add_argument("--coefficient", type=float, default=4.0)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--max-action-steps", type=int, default=80)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--interventions",
        type=Path,
        default=Path(os.environ.get("VLA_MECH_REPO", "."))
        / "openvla/libero_experiments/configs/interventions/dictionaries.yaml",
    )
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layer_neurons(flat_indices: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for flat_index in flat_indices:
        layer = flat_index // INTERMEDIATE_SIZE
        neuron = flat_index % INTERMEDIATE_SIZE
        grouped.setdefault(layer, []).append(neuron)
    return grouped


def make_layer_matched_random(
    semantic: dict[int, list[str]], seed: int
) -> dict[int, list[str]]:
    rng = np.random.default_rng(seed)
    semantic_by_layer = layer_neurons(list(semantic))
    random_dict: dict[int, list[str]] = {}
    for layer, semantic_neurons in sorted(semantic_by_layer.items()):
        excluded = set(semantic_neurons)
        candidates = np.asarray(
            [index for index in range(INTERMEDIATE_SIZE) if index not in excluded],
            dtype=np.int64,
        )
        chosen = rng.choice(candidates, size=len(semantic_neurons), replace=False)
        for neuron in chosen.tolist():
            random_dict[layer * INTERMEDIATE_SIZE + int(neuron)] = ["layer-matched random control"]
    return random_dict


def resolve_unnorm_key(model: Any, suite: str) -> str:
    norm_stats = getattr(model, "norm_stats", {})
    for candidate in (suite, f"{suite}_no_noops"):
        if candidate in norm_stats:
            return candidate
    raise KeyError(f"Checkpoint has no normalization statistics for {suite}; keys={sorted(norm_stats)}")


def register_hook_counters(model: Any, intervention: dict[int, list[str]]) -> tuple[list[Any], dict[int, int]]:
    counts = {layer: 0 for layer in layer_neurons(list(intervention))}
    handles = []
    for layer in counts:
        module = model.language_model.model.layers[layer].mlp.down_proj

        def count_hook(_module: Any, _inputs: Any, _output: Any, *, layer_index: int = layer) -> None:
            counts[layer_index] += 1

        handles.append(module.register_forward_hook(count_hook))
    return handles, counts


def trajectory_metrics(eef_positions: list[list[float]]) -> dict[str, Any]:
    positions = np.asarray(eef_positions, dtype=np.float64)
    if len(positions) < 2:
        return {
            "num_eef_transitions": 0,
            "mean_step_delta_xyz": [0.0, 0.0, 0.0],
            "net_delta_xyz": [0.0, 0.0, 0.0],
            "path_length": 0.0,
        }
    deltas = np.diff(positions, axis=0)
    return {
        "num_eef_transitions": int(len(deltas)),
        "mean_step_delta_xyz": deltas.mean(axis=0).tolist(),
        "net_delta_xyz": (positions[-1] - positions[0]).tolist(),
        "path_length": float(np.linalg.norm(deltas, axis=1).sum()),
        "mean_step_distance": float(np.linalg.norm(deltas, axis=1).mean()),
        "mean_eef_y": float(positions[:, 1].mean()),
        "max_eef_y": float(positions[:, 1].max()),
        "min_eef_y": float(positions[:, 1].min()),
    }


def run_condition(
    *,
    condition: str,
    model: Any,
    processor: Any,
    task: Any,
    initial_state: np.ndarray,
    prompt: str,
    unnorm_key: str,
    intervention: dict[int, list[str]] | None,
    coefficient: float,
    wait_steps: int,
    max_action_steps: int,
    save_video: bool,
    output_dir: Path,
) -> dict[str, Any]:
    official_handles: list[Any] = []
    counter_handles: list[Any] = []
    hook_counts: dict[int, int] = {}
    if intervention:
        official_handles = apply_gate_proj_hooks(model, intervention, coef=coefficient)
        counter_handles, hook_counts = register_hook_counters(model, intervention)

    env, task_description = get_libero_env(task, "openvla", resolution=256)
    frames: list[np.ndarray] = []
    raw_actions: list[list[float]] = []
    executed_actions: list[list[float]] = []
    eef_positions: list[list[float]] = []
    inference_seconds: list[float] = []
    success = False
    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(wait_steps):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        eef_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())

        for action_step in range(max_action_steps):
            image = get_libero_image(obs, 224)
            if save_video:
                frames.append(image)
            inputs = prepare_openvla_inputs(processor, prompt, image, torch.device("cuda:0"), dtype=torch.float16)
            started = time.perf_counter()
            raw_action = predict_openvla_action(model, inputs, unnorm_key=unnorm_key)
            inference_seconds.append(time.perf_counter() - started)
            if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
                raise RuntimeError(f"Invalid action at step {action_step}: {raw_action}")

            action = normalize_gripper_action(raw_action.copy(), binarize=True)
            action = invert_gripper_action(action)
            obs, _, success, _ = env.step(action.tolist())
            raw_actions.append(raw_action.tolist())
            executed_actions.append(np.asarray(action).tolist())
            eef_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())
            if success:
                break
    finally:
        env.close()
        for handle in counter_handles:
            handle.remove()
        for handle in official_handles:
            handle.remove()

    result = {
        "condition": condition,
        "task": task_description,
        "success": bool(success),
        "action_steps": len(raw_actions),
        "coefficient": coefficient if intervention else None,
        "num_intervention_neurons": len(intervention or {}),
        "intervention_flat_indices": sorted(intervention or {}),
        "hook_counts_by_layer": hook_counts,
        "mean_inference_seconds": float(np.mean(inference_seconds)) if inference_seconds else None,
        "raw_actions": raw_actions,
        "executed_actions": executed_actions,
        "eef_positions": eef_positions,
        **trajectory_metrics(eef_positions),
    }
    result_path = output_dir / f"{condition}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if save_video and frames:
        video_path = output_dir / f"{condition}.mp4"
        with imageio.get_writer(video_path, fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)
        result["video"] = str(video_path)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script inside a GPU allocation.")
    set_determinism(args.random_seed)

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    allowed = {"baseline", args.semantic_dict, "random_10"}
    unknown = set(conditions) - allowed
    if unknown:
        raise ValueError(f"Unknown conditions {sorted(unknown)}; allowed={sorted(allowed)}")

    semantic = load_intervention_dict(args.semantic_dict, args.interventions)
    random_control = make_layer_matched_random(semantic, args.random_seed)
    if len(semantic) != 10 or len(random_control) != len(semantic):
        raise ValueError(
            f"Expected a 10-neuron semantic cluster and matched random control; "
            f"got semantic={len(semantic)}, random={len(random_control)}"
        )

    output_dir = args.output_dir or (
        Path(os.environ["VLA_WORK_ROOT"])
        / "runs/mechanistic_steering_openvla"
        / f"job-{os.environ.get('SLURM_JOB_ID', 'local')}-task-{args.task_id}-state-{args.init_state_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_openvla(
        args.checkpoint,
        args.revision,
        torch.device("cuda:0"),
        dtype=torch.float16,
        attention_implementation="eager",
    )
    unnorm_key = resolve_unnorm_key(model, args.suite)
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state_id < len(initial_states):
        raise IndexError(f"init-state-id {args.init_state_id} outside [0, {len(initial_states)})")
    prompt = build_openvla_prompt(task.language)

    interventions = {
        "baseline": None,
        args.semantic_dict: semantic,
        "random_10": random_control,
    }
    results = []
    for condition in conditions:
        set_determinism(args.random_seed)
        result = run_condition(
            condition=condition,
            model=model,
            processor=processor,
            task=task,
            initial_state=initial_states[args.init_state_id],
            prompt=prompt,
            unnorm_key=unnorm_key,
            intervention=interventions[condition],
            coefficient=args.coefficient,
            wait_steps=args.wait_steps,
            max_action_steps=args.max_action_steps,
            save_video=args.save_video,
            output_dir=output_dir,
        )
        print(json.dumps({key: value for key, value in result.items() if key not in {"raw_actions", "executed_actions", "eef_positions"}}, indent=2), flush=True)
        results.append(result)

    by_condition = {result["condition"]: result for result in results}
    baseline = by_condition.get("baseline")
    audits: dict[str, Any] = {}
    if baseline is not None:
        baseline_start = np.asarray(baseline["eef_positions"][0])
        for condition, result in by_condition.items():
            start = np.asarray(result["eef_positions"][0])
            audits[f"{condition}_initial_eef_max_abs_diff"] = float(
                np.max(np.abs(start - baseline_start))
            )
            if condition != "baseline" and baseline["raw_actions"] and result["raw_actions"]:
                baseline_actions = np.asarray(baseline["raw_actions"])
                condition_actions = np.asarray(result["raw_actions"])
                num_matched_steps = min(len(baseline_actions), len(condition_actions))
                action_l2 = np.linalg.norm(
                    condition_actions[:num_matched_steps] - baseline_actions[:num_matched_steps],
                    axis=1,
                )
                differing = np.flatnonzero(action_l2 > 1e-12)
                audits[f"{condition}_first_raw_action_l2_from_baseline"] = float(action_l2[0])
                audits[f"{condition}_mean_raw_action_l2_from_baseline"] = float(action_l2.mean())
                audits[f"{condition}_max_raw_action_l2_from_baseline"] = float(action_l2.max())
                audits[f"{condition}_num_differing_raw_actions"] = int(len(differing))
                audits[f"{condition}_num_matched_action_steps"] = int(num_matched_steps)
                audits[f"{condition}_first_differing_action_step"] = (
                    int(differing[0]) if len(differing) else None
                )
    for condition, result in by_condition.items():
        if condition != "baseline":
            counts = result["hook_counts_by_layer"]
            audits[f"{condition}_all_selected_layers_fired"] = bool(
                counts and all(int(count) > 0 for count in counts.values())
            )

    summary = {
        "status": "ok",
        "official_repository_revision": "559c0f25a3cc5a20fc8b804774a88415404e0d22",
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.revision,
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "prompt": prompt,
        "unnorm_key": unnorm_key,
        "dtype": "float16",
        "attention_implementation": "eager",
        "semantic_dict": args.semantic_dict,
        "coefficient": args.coefficient,
        "random_seed": args.random_seed,
        "audits": audits,
        "conditions": results,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
