#!/usr/bin/env python3
"""Run same-state paired-prompt activation patching in the pi0.5 action expert."""

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


DEFAULT_LAYERS = "0,5,9,13,17"
DEFAULT_ALPHAS = "0,0.25,0.5,0.75,1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--base-prompt-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--verify-audits", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("--layers must contain unique comma-separated integers.")
    return parsed


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or not all(np.isfinite(item) for item in parsed):
        raise ValueError("--alphas must contain finite comma-separated values.")
    return parsed


def decode_attr(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def deterministic_noise(
    *, image_index: int, seed: int, horizon: int, action_dim: int, device: torch.device
) -> torch.Tensor:
    seed_sequence = np.random.SeedSequence([seed, image_index])
    rng = np.random.default_rng(seed_sequence)
    single = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    return torch.from_numpy(np.repeat(single, 2, axis=0)).to(device)


def create_output(
    path: Path,
    *,
    num_images: int,
    layer_indices: list[int],
    alphas: list[float],
    hidden_size: int,
    action_horizon: int,
    action_dim: int,
    source_path: Path,
    checkpoint: Path,
    timestep: float,
    noise_seed: int,
    base_prompt_index: int,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, "w")
    num_layers = len(layer_indices)
    num_alphas = len(alphas)
    output.create_dataset(
        "natural_velocity",
        shape=(num_images, 2, action_horizon, action_dim),
        dtype=np.float32,
        chunks=(1, 2, action_horizon, action_dim),
    )
    output.create_dataset(
        "natural_final_mean_residuals",
        shape=(num_images, 2, hidden_size),
        dtype=np.float16,
        chunks=(1, 2, hidden_size),
    )
    output.create_dataset(
        "patched_velocity",
        shape=(num_images, num_layers, num_alphas, action_horizon, action_dim),
        dtype=np.float32,
        chunks=(1, 1, num_alphas, action_horizon, action_dim),
    )
    for name in ("patched_target_mean_residuals", "patched_final_mean_residuals"):
        output.create_dataset(
            name,
            shape=(num_images, num_layers, num_alphas, hidden_size),
            dtype=np.float16,
            chunks=(1, 1, num_alphas, hidden_size),
        )
    output.create_dataset(
        "untouched_velocity_max_abs_difference",
        shape=(num_images, num_layers, num_alphas),
        dtype=np.float32,
        chunks=(1, num_layers, num_alphas),
    )
    output.create_dataset("completed", shape=(num_images,), dtype=np.bool_, fillvalue=False)
    output.attrs["source_h5"] = str(source_path)
    output.attrs["checkpoint"] = str(checkpoint)
    output.attrs["openpi_revision"] = DEFAULT_OPENPI_REVISION
    output.attrs["layer_convention"] = "Gemma_action_expert_block_output_before_final_norm"
    output.attrs["patch_protocol"] = (
        "per_state_all_action_tokens_base_plus_alpha_times_source_minus_base"
    )
    output.attrs["patch_layers_json"] = json.dumps(layer_indices)
    output.attrs["alphas_json"] = json.dumps(alphas)
    output.attrs["base_prompt_index"] = base_prompt_index
    output.attrs["source_prompt_index"] = 1 - base_prompt_index
    output.attrs["noise_protocol"] = "per_state_seeded_Gaussian_shared_within_prompt_pair"
    output.attrs["noise_seed"] = noise_seed
    output.attrs["timestep"] = timestep
    output.attrs["num_images"] = num_images
    output.attrs["hidden_size"] = hidden_size
    output.attrs["action_horizon"] = action_horizon
    output.attrs["action_dim"] = action_dim
    output.attrs["complete"] = False
    return output


def validate_resume(
    output: h5py.File,
    *,
    num_images: int,
    layer_indices: list[int],
    alphas: list[float],
    timestep: float,
    noise_seed: int,
    base_prompt_index: int,
) -> None:
    expected = {
        "num_images": num_images,
        "patch_layers_json": json.dumps(layer_indices),
        "alphas_json": json.dumps(alphas),
        "timestep": timestep,
        "noise_seed": noise_seed,
        "base_prompt_index": base_prompt_index,
    }
    for key, value in expected.items():
        existing = output.attrs[key]
        if isinstance(value, float):
            matches = float(existing) == value
        elif isinstance(value, int):
            matches = int(existing) == value
        else:
            matches = decode_attr(existing) == value
        if not matches:
            raise ValueError(f"Existing output attribute {key}={existing!r} != {value!r}.")


def max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def main() -> None:
    args = parse_args()
    layer_indices = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    if not 0.0 <= args.timestep <= 1.0:
        raise ValueError("--timestep must be in [0, 1].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.output_h5.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_h5}. Use --resume.")

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    expert = model.paligemma_with_expert.gemma_expert.model
    hidden_size = expert.config.hidden_size
    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim
    if any(index < 0 or index >= len(expert.layers) for index in layer_indices):
        raise ValueError(f"Patch layers must be in [0, {len(expert.layers) - 1}].")
    started_at = time.perf_counter()

    with h5py.File(args.input_h5, "r") as source:
        available = source["base_images"].shape[0]
        num_images = min(available, args.max_images or available)
        prompts = [decode_attr(source.attrs["prompt_a"]), decode_attr(source.attrs["prompt_b"])]
        if args.output_h5.exists():
            output = h5py.File(args.output_h5, "r+")
            validate_resume(
                output,
                num_images=num_images,
                layer_indices=layer_indices,
                alphas=alphas,
                timestep=args.timestep,
                noise_seed=args.noise_seed,
                base_prompt_index=args.base_prompt_index,
            )
        else:
            output = create_output(
                args.output_h5,
                num_images=num_images,
                layer_indices=layer_indices,
                alphas=alphas,
                hidden_size=hidden_size,
                action_horizon=action_horizon,
                action_dim=action_dim,
                source_path=args.input_h5,
                checkpoint=args.checkpoint,
                timestep=args.timestep,
                noise_seed=args.noise_seed,
                base_prompt_index=args.base_prompt_index,
            )

        try:
            pending = np.flatnonzero(~output["completed"][:])
            extractor = Pi05ActionExpertResidualExtractor(model)
            alpha_zero_index = alphas.index(0.0) if 0.0 in alphas else None
            for progress_index, image_index in enumerate(
                tqdm(pending, desc="pi05 natural paired patching", unit="image"), start=1
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
                readout = extractor.extract_paired_patch_grid(
                    observation,
                    noise=noise,
                    timestep=args.timestep,
                    layer_indices=layer_indices,
                    alphas=alphas,
                    base_prompt_index=args.base_prompt_index,
                    source_prompt_index=1 - args.base_prompt_index,
                )

                natural_velocity = readout["natural_velocity"].numpy()
                patched_velocity = readout["patched_velocity"].numpy()
                untouched_difference = readout[
                    "untouched_velocity_max_abs_difference"
                ].numpy()
                if args.verify_audits:
                    if untouched_difference.max() != 0.0:
                        raise RuntimeError(
                            "Patching changed the untouched source branch: "
                            f"max abs difference={untouched_difference.max()}."
                        )
                    if alpha_zero_index is not None:
                        alpha_zero = patched_velocity[:, alpha_zero_index]
                        natural_base = natural_velocity[args.base_prompt_index][None]
                        difference = max_abs_difference(alpha_zero, natural_base)
                        if difference != 0.0:
                            raise RuntimeError(
                                f"alpha=0 changed the base velocity: max abs difference={difference}."
                            )

                output["natural_velocity"][image_index] = natural_velocity
                output["natural_final_mean_residuals"][image_index] = readout[
                    "natural_final_mean"
                ].numpy().astype(np.float16)
                output["patched_velocity"][image_index] = patched_velocity
                output["patched_target_mean_residuals"][image_index] = readout[
                    "patched_target_mean"
                ].numpy().astype(np.float16)
                output["patched_final_mean_residuals"][image_index] = readout[
                    "patched_final_mean"
                ].numpy().astype(np.float16)
                output["untouched_velocity_max_abs_difference"][image_index] = (
                    untouched_difference
                )
                output["completed"][image_index] = True

                if progress_index == 1:
                    output.attrs["alpha_zero_base_velocity_max_abs_difference"] = (
                        0.0
                        if alpha_zero_index is None
                        else max_abs_difference(
                            patched_velocity[:, alpha_zero_index],
                            natural_velocity[args.base_prompt_index][None],
                        )
                    )
                    output.attrs["untouched_velocity_max_abs_difference_first"] = float(
                        untouched_difference.max()
                    )
                    output.attrs["natural_prompt_velocity_mean_abs_difference"] = float(
                        np.abs(natural_velocity[1] - natural_velocity[0]).mean()
                    )

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
                        "patch_layers": layer_indices,
                        "alphas": alphas,
                        "base_prompt_index": args.base_prompt_index,
                        "elapsed_seconds": float(output.attrs["elapsed_seconds_latest_run"]),
                        "max_cuda_memory_gib": float(output.attrs["max_cuda_memory_gib"]),
                        "alpha_zero_base_velocity_max_abs_difference": float(
                            output.attrs["alpha_zero_base_velocity_max_abs_difference"]
                        ),
                        "untouched_velocity_max_abs_difference_first": float(
                            output.attrs["untouched_velocity_max_abs_difference_first"]
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )
        finally:
            output.close()


if __name__ == "__main__":
    main()
