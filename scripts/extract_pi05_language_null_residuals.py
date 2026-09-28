#!/usr/bin/env python3
"""Cache pi0.5 residuals for one target-underspecified LIBERO instruction."""

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

from vla_coordinates.pi05_runtime import (
    DEFAULT_OPENPI_REVISION,
    Pi05ActionExpertResidualExtractor,
    load_pi05_policy,
    prepare_pi05_observation_batch,
)


DEFAULT_NULL_PROMPT = "pick up an object and place it in the basket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--null-prompt", default=DEFAULT_NULL_PROMPT)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def deterministic_noise(
    *, image_index: int, seed: int, horizon: int, action_dim: int, device: torch.device
) -> torch.Tensor:
    sequence = np.random.SeedSequence([seed, image_index])
    rng = np.random.default_rng(sequence)
    values = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    return torch.from_numpy(values).to(device)


def create_output(
    path: Path,
    *,
    num_images: int,
    prefix_layers: int,
    prefix_hidden_size: int,
    expert_layers: int,
    expert_hidden_size: int,
    action_horizon: int,
    action_dim: int,
    args: argparse.Namespace,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, "w")
    output.create_dataset(
        "prompt_end_residuals",
        shape=(num_images, prefix_layers, prefix_hidden_size),
        dtype=np.float16,
        chunks=(1, prefix_layers, prefix_hidden_size),
    )
    for name in ("expert_action_mean_residuals", "expert_action_first_residuals"):
        output.create_dataset(
            name,
            shape=(num_images, expert_layers, expert_hidden_size),
            dtype=np.float16,
            chunks=(1, expert_layers, expert_hidden_size),
        )
    output.create_dataset(
        "expert_velocity",
        shape=(num_images, action_horizon, action_dim),
        dtype=np.float16,
        chunks=(1, action_horizon, action_dim),
    )
    output.create_dataset("completed", shape=(num_images,), dtype=np.bool_, fillvalue=False)
    output.attrs["source_h5"] = str(args.input_h5)
    output.attrs["checkpoint"] = str(args.checkpoint)
    output.attrs["openpi_revision"] = DEFAULT_OPENPI_REVISION
    output.attrs["language_condition"] = "target_underspecified_null"
    output.attrs["null_prompt"] = args.null_prompt
    output.attrs["noise_protocol"] = (
        "per_state_seeded_Gaussian_identical_to_paired_action_expert_cache"
    )
    output.attrs["noise_seed"] = args.noise_seed
    output.attrs["timestep"] = args.timestep
    output.attrs["prefix_layer_convention"] = (
        "PaliGemma_decoder_block_output_before_final_norm"
    )
    output.attrs["expert_layer_convention"] = (
        "Gemma_action_expert_block_output_before_final_norm"
    )
    output.attrs["num_images"] = num_images
    output.attrs["num_prefix_layers"] = prefix_layers
    output.attrs["prefix_hidden_size"] = prefix_hidden_size
    output.attrs["num_expert_layers"] = expert_layers
    output.attrs["expert_hidden_size"] = expert_hidden_size
    output.attrs["action_horizon"] = action_horizon
    output.attrs["action_dim"] = action_dim
    output.attrs["complete"] = False
    return output


def max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max().item())


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.timestep <= 1.0:
        raise ValueError("--timestep must be in [0, 1].")
    if not args.null_prompt.strip():
        raise ValueError("The null condition must remain a nonempty grammatical instruction.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.output_h5.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_h5}. Use --resume to continue.")

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    prefix_model = model.paligemma_with_expert.paligemma.language_model
    expert_model = model.paligemma_with_expert.gemma_expert.model
    prefix_layers = len(prefix_model.layers)
    prefix_hidden_size = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    expert_layers = len(expert_model.layers)
    expert_hidden_size = expert_model.config.hidden_size
    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim
    started_at = time.perf_counter()

    with h5py.File(args.input_h5, "r") as source:
        available = source["base_images"].shape[0]
        num_images = min(available, args.max_images or available)
        if args.output_h5.exists():
            output = h5py.File(args.output_h5, "r+")
            expected = (num_images, prefix_layers, prefix_hidden_size)
            if output["prompt_end_residuals"].shape != expected:
                existing = output["prompt_end_residuals"].shape
                output.close()
                raise ValueError(f"Existing prefix shape {existing} != {expected}.")
            if str(output.attrs["null_prompt"]) != args.null_prompt:
                output.close()
                raise ValueError("Existing output uses a different null prompt.")
            if int(output.attrs["noise_seed"]) != args.noise_seed:
                output.close()
                raise ValueError("Existing output uses a different noise seed.")
        else:
            output = create_output(
                args.output_h5,
                num_images=num_images,
                prefix_layers=prefix_layers,
                prefix_hidden_size=prefix_hidden_size,
                expert_layers=expert_layers,
                expert_hidden_size=expert_hidden_size,
                action_horizon=action_horizon,
                action_dim=action_dim,
                args=args,
            )

        extractor = Pi05ActionExpertResidualExtractor(model)
        try:
            pending = np.flatnonzero(~output["completed"][:])
            for progress_index, image_index in enumerate(
                tqdm(pending, desc="pi05 null-language residuals", unit="image"), start=1
            ):
                base_image = np.asarray(source["base_images"][image_index])
                wrist_image = np.asarray(source["wrist_images"][image_index])
                state = np.asarray(source["robot_states"][image_index])
                observation = prepare_pi05_observation_batch(
                    policy,
                    base_images=[base_image],
                    wrist_images=[wrist_image],
                    states=[state],
                    prompts=[args.null_prompt],
                    device=device,
                )
                noise = deterministic_noise(
                    image_index=int(image_index),
                    seed=args.noise_seed,
                    horizon=action_horizon,
                    action_dim=action_dim,
                    device=device,
                )
                readout = extractor.extract(
                    observation,
                    noise=noise,
                    timestep=args.timestep,
                    capture_prefix=True,
                )

                if args.verify_determinism and progress_index == 1:
                    repeated = extractor.extract(
                        observation,
                        noise=noise,
                        timestep=args.timestep,
                        capture_prefix=True,
                    )
                    determinism = {
                        key: max_abs_difference(readout[key], repeated[key])
                        for key in ("prompt_end", "action_mean", "action_first", "velocity")
                    }
                    output.attrs["determinism_max_abs_difference_json"] = json.dumps(
                        determinism
                    )
                    if any(value != 0.0 for value in determinism.values()):
                        raise RuntimeError(f"Null-language extraction is not exact: {determinism}.")

                output["prompt_end_residuals"][image_index] = (
                    readout["prompt_end"][0].float().numpy().astype(np.float16)
                )
                output["expert_action_mean_residuals"][image_index] = (
                    readout["action_mean"][0].float().numpy().astype(np.float16)
                )
                output["expert_action_first_residuals"][image_index] = (
                    readout["action_first"][0].float().numpy().astype(np.float16)
                )
                output["expert_velocity"][image_index] = (
                    readout["velocity"][0].float().numpy().astype(np.float16)
                )
                output["completed"][image_index] = True

                if progress_index == 1 and "null_prompt_input_ids" not in output.attrs:
                    tokens = readout["language_tokens"][0].numpy()
                    mask = readout["language_masks"][0].numpy().astype(bool)
                    output.attrs["null_prompt_input_ids"] = json.dumps(tokens[mask].tolist())
                    output.attrs["num_image_tokens"] = int(readout["num_image_tokens"])
                    output.attrs["num_action_tokens"] = int(readout["num_action_tokens"])
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
                        "null_prompt": args.null_prompt,
                        "num_completed": int(output["completed"][:].sum()),
                        "prompt_end_shape": list(output["prompt_end_residuals"].shape),
                        "expert_mean_shape": list(
                            output["expert_action_mean_residuals"].shape
                        ),
                        "velocity_shape": list(output["expert_velocity"].shape),
                        "elapsed_seconds": float(output.attrs["elapsed_seconds_latest_run"]),
                        "max_cuda_memory_gib": float(output.attrs["max_cuda_memory_gib"]),
                    },
                    indent=2,
                ),
                flush=True,
            )
        finally:
            output.close()


if __name__ == "__main__":
    main()
