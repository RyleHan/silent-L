#!/usr/bin/env python3
"""Cache exact per-token paired-prompt deltas at one pi0.5 expert block."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from extract_pi05_paired_patching import decode_attr, deterministic_noise
from vla_coordinates.pi05_runtime import (
    DEFAULT_OPENPI_REVISION,
    Pi05ActionExpertResidualExtractor,
    load_pi05_policy,
    prepare_pi05_observation_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def create_output(
    path: Path,
    *,
    num_images: int,
    action_horizon: int,
    hidden_size: int,
    source_path: Path,
    checkpoint: Path,
    layer: int,
    timestep: float,
    noise_seed: int,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, "w")
    output.create_dataset(
        "paired_token_delta_b_minus_a",
        shape=(num_images, action_horizon, hidden_size),
        dtype=np.float16,
        chunks=(1, action_horizon, hidden_size),
    )
    output.create_dataset("completed", shape=(num_images,), dtype=np.bool_, fillvalue=False)
    output.attrs["source_h5"] = str(source_path)
    output.attrs["checkpoint"] = str(checkpoint)
    output.attrs["openpi_revision"] = DEFAULT_OPENPI_REVISION
    output.attrs["layer"] = layer
    output.attrs["timestep"] = timestep
    output.attrs["noise_seed"] = noise_seed
    output.attrs["delta_convention"] = "prompt_1_minus_prompt_0_per_action_token"
    output.attrs["num_images"] = num_images
    output.attrs["action_horizon"] = action_horizon
    output.attrs["hidden_size"] = hidden_size
    output.attrs["complete"] = False
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.output_h5.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_h5}. Use --resume.")
    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    expert = model.paligemma_with_expert.gemma_expert.model
    if args.layer < 0 or args.layer >= len(expert.layers):
        raise ValueError(f"--layer must be in [0, {len(expert.layers) - 1}].")
    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim
    hidden_size = expert.config.hidden_size
    started_at = time.perf_counter()

    with h5py.File(args.input_h5, "r") as source:
        available = source["base_images"].shape[0]
        num_images = min(available, args.max_images or available)
        prompts = [decode_attr(source.attrs["prompt_a"]), decode_attr(source.attrs["prompt_b"])]
        if args.output_h5.exists():
            output = h5py.File(args.output_h5, "r+")
            expected = (num_images, action_horizon, hidden_size)
            if output["paired_token_delta_b_minus_a"].shape != expected:
                output.close()
                raise ValueError("Existing token-delta cache has an incompatible shape.")
            for key, value in (
                ("layer", args.layer),
                ("timestep", args.timestep),
                ("noise_seed", args.noise_seed),
            ):
                if float(output.attrs[key]) != float(value):
                    output.close()
                    raise ValueError(f"Existing output uses a different {key}.")
        else:
            output = create_output(
                args.output_h5,
                num_images=num_images,
                action_horizon=action_horizon,
                hidden_size=hidden_size,
                source_path=args.input_h5,
                checkpoint=args.checkpoint,
                layer=args.layer,
                timestep=args.timestep,
                noise_seed=args.noise_seed,
            )

        try:
            pending = np.flatnonzero(~output["completed"][:])
            extractor = Pi05ActionExpertResidualExtractor(model)
            for progress_index, image_index in enumerate(
                tqdm(pending, desc=f"pi05 expert block {args.layer} token deltas", unit="image"),
                start=1,
            ):
                base_image = np.asarray(source["base_images"][image_index])
                wrist_image = np.asarray(source["wrist_images"][image_index])
                state = np.asarray(source["robot_states"][image_index])
                observation = prepare_pi05_observation_batch(
                    policy,
                    base_images=[base_image, base_image],
                    wrist_images=[wrist_image, wrist_image],
                    states=[state, state],
                    prompts=prompts,
                    device=device,
                )
                noise = deterministic_noise(
                    image_index=int(image_index),
                    seed=args.noise_seed,
                    horizon=action_horizon,
                    action_dim=action_dim,
                    device=device,
                )
                readout = extractor.extract_layer_tokens(
                    observation,
                    noise=noise,
                    timestep=args.timestep,
                    layer_index=args.layer,
                )
                action_tokens = readout["action_tokens"]
                if args.verify_determinism and progress_index == 1:
                    repeated = extractor.extract_layer_tokens(
                        observation,
                        noise=noise,
                        timestep=args.timestep,
                        layer_index=args.layer,
                    )
                    token_error = float(
                        (action_tokens - repeated["action_tokens"]).abs().max().item()
                    )
                    velocity_error = float(
                        (readout["velocity"] - repeated["velocity"]).abs().max().item()
                    )
                    output.attrs["determinism_max_abs_difference_json"] = json.dumps(
                        {"action_tokens": token_error, "velocity": velocity_error}
                    )
                    if token_error != 0.0 or velocity_error != 0.0:
                        raise RuntimeError("Selected-layer token extraction is not deterministic.")

                delta = action_tokens[1] - action_tokens[0]
                output["paired_token_delta_b_minus_a"][image_index] = delta.numpy().astype(
                    np.float16
                )
                output["completed"][image_index] = True
                if progress_index == 1:
                    output.attrs["first_delta_mean_abs"] = float(delta.abs().mean().item())
                    output.attrs["first_delta_max_abs"] = float(delta.abs().max().item())
                if progress_index % args.flush_every == 0:
                    output.flush()

            output.attrs["complete"] = bool(output["completed"][:].all())
            output.attrs["elapsed_seconds_latest_run"] = time.perf_counter() - started_at
            output.attrs["max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (
                1024**3
            )
            output.attrs["job_id"] = os.environ.get("SLURM_JOB_ID", "")
            output.flush()
            print(
                json.dumps(
                    {
                        "output_h5": str(args.output_h5),
                        "num_completed": int(output["completed"][:].sum()),
                        "layer": args.layer,
                        "elapsed_seconds": float(output.attrs["elapsed_seconds_latest_run"]),
                        "max_cuda_memory_gib": float(output.attrs["max_cuda_memory_gib"]),
                    },
                    indent=2,
                )
            )
        finally:
            output.close()


if __name__ == "__main__":
    main()
