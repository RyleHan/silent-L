#!/usr/bin/env python3
"""Fit a COAST contrastive conceptor from public pi0.5 rollout activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_coordinates.conceptors import (
    conceptor_diagnostics,
    conceptor_overlap,
    contrastive_conceptor,
    fit_conceptor,
    random_spectrum_control,
)


PUBLIC_LAYER_INDICES = (0, 5, 11, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--aperture", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--expected-episodes", type=int, default=15)
    parser.add_argument("--expected-successes", type=int, default=8)
    parser.add_argument("--expected-failures", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer not in PUBLIC_LAYER_INDICES:
        raise ValueError(f"Layer must be one of {PUBLIC_LAYER_INDICES} for the public cache.")
    if not 0 <= args.beta <= 1:
        raise ValueError("Beta must be in [0, 1].")
    task_dir = args.activation_root / "openpi-libero-2000" / args.task_name
    episode_dirs = sorted(task_dir.glob("episode_*_env_*"))
    if len(episode_dirs) != args.expected_episodes:
        raise ValueError(
            f"Expected {args.expected_episodes} episodes, found {len(episode_dirs)}."
        )

    layer_offset = PUBLIC_LAYER_INDICES.index(args.layer)
    class_vectors: dict[bool, list[np.ndarray]] = {True: [], False: []}
    episode_summary = []
    for episode_dir in episode_dirs:
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        success = bool(metadata["episode_success"])
        residual_paths = sorted(episode_dir.glob("step_*/suffix_residual.npz"))
        if len(residual_paths) != int(metadata["total_inference_steps"]):
            raise ValueError(
                f"{episode_dir.name}: expected {metadata['total_inference_steps']} residuals, "
                f"found {len(residual_paths)}."
            )
        episode_vectors = []
        for residual_path in residual_paths:
            with np.load(residual_path) as payload:
                residual = np.asarray(payload["all_suffix_residual"], dtype=np.float32)
            if residual.ndim != 4 or residual.shape[1] != len(PUBLIC_LAYER_INDICES):
                raise ValueError(f"Unexpected residual shape {residual.shape} in {residual_path}.")
            # [denoise, layer, action_token, hidden] -> [denoise, hidden]
            episode_vectors.append(residual[:, layer_offset].mean(axis=1))
        vectors = np.concatenate(episode_vectors, axis=0)
        class_vectors[success].append(vectors)
        episode_summary.append(
            {
                "episode": episode_dir.name,
                "success": success,
                "num_policy_queries": len(residual_paths),
                "num_activation_vectors": len(vectors),
            }
        )

    successes = np.concatenate(class_vectors[True], axis=0)
    failures = np.concatenate(class_vectors[False], axis=0)
    num_success_episodes = sum(item["success"] for item in episode_summary)
    num_failure_episodes = len(episode_summary) - num_success_episodes
    if (num_success_episodes, num_failure_episodes) != (
        args.expected_successes,
        args.expected_failures,
    ):
        raise ValueError(
            "Public class-count audit failed: "
            f"{num_success_episodes} success, {num_failure_episodes} failure."
        )

    success_conceptor = fit_conceptor(successes, args.aperture)
    failure_conceptor = fit_conceptor(failures, args.aperture)
    steering_conceptor = contrastive_conceptor(success_conceptor, failure_conceptor)
    random_conceptor = random_spectrum_control(steering_conceptor, args.random_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "coast_conceptor.npz",
        success_conceptor=success_conceptor,
        failure_conceptor=failure_conceptor,
        steering_conceptor=steering_conceptor,
        random_conceptor=random_conceptor,
    )
    summary = {
        "method": "COAST paper-guided independent reimplementation",
        "task_name": args.task_name,
        "checkpoint": "brandonyang/openpi-libero-2000",
        "strategy": "global contrastive conceptor",
        "layer": args.layer,
        "public_layer_indices": list(PUBLIC_LAYER_INDICES),
        "aperture": args.aperture,
        "beta": args.beta,
        "random_seed": args.random_seed,
        "num_episodes": len(episode_summary),
        "num_success_episodes": num_success_episodes,
        "num_failure_episodes": num_failure_episodes,
        "num_success_vectors": len(successes),
        "num_failure_vectors": len(failures),
        "success_failure_overlap": conceptor_overlap(
            success_conceptor, failure_conceptor
        ),
        "diagnostics": {
            "success": conceptor_diagnostics(success_conceptor),
            "failure": conceptor_diagnostics(failure_conceptor),
            "steering": conceptor_diagnostics(steering_conceptor),
            "random": conceptor_diagnostics(random_conceptor),
        },
        "episodes": episode_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
