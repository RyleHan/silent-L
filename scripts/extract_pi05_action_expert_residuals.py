#!/usr/bin/env python3
"""Cache paired-prompt pi0.5 action-expert residuals at fixed time."""

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

from vla_coordinates.libero_object_pairs import prompts_for_sample, read_pair_ids, read_pairs
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
    seed_sequence = np.random.SeedSequence([seed, image_index])
    rng = np.random.default_rng(seed_sequence)
    single = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    paired = np.repeat(single, 2, axis=0)
    return torch.from_numpy(paired).to(device)


def create_output(
    path: Path,
    *,
    num_images: int,
    num_layers: int,
    hidden_size: int,
    prefix_hidden_size: int,
    action_horizon: int,
    action_dim: int,
    source_path: Path,
    checkpoint: Path,
    timestep: float,
    noise_seed: int,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, "w")
    for name in ("expert_action_mean_residuals", "expert_action_first_residuals"):
        output.create_dataset(
            name,
            shape=(num_images, 2, num_layers, hidden_size),
            dtype=np.float16,
            chunks=(1, 1, num_layers, hidden_size),
        )
    output.create_dataset(
        "expert_velocity",
        shape=(num_images, 2, action_horizon, action_dim),
        dtype=np.float16,
        chunks=(1, 1, action_horizon, action_dim),
    )
    output.create_dataset(
        "visual_input_mean",
        shape=(num_images, prefix_hidden_size),
        dtype=np.float16,
        chunks=(1, prefix_hidden_size),
    )
    output.create_dataset("completed", shape=(num_images,), dtype=np.bool_, fillvalue=False)
    output.attrs["source_h5"] = str(source_path)
    output.attrs["checkpoint"] = str(checkpoint)
    output.attrs["openpi_revision"] = DEFAULT_OPENPI_REVISION
    output.attrs["layer_convention"] = "Gemma_action_expert_block_output_before_final_norm"
    output.attrs["readouts"] = json.dumps(
        {
            "expert_action_mean": "mean_over_10_action_tokens",
            "expert_action_first": "first_action_token",
        }
    )
    output.attrs["noise_protocol"] = "per_state_seeded_Gaussian_shared_within_prompt_pair"
    output.attrs["noise_seed"] = noise_seed
    output.attrs["timestep"] = timestep
    output.attrs["denoising_protocol"] = "single_vector_field_evaluation_at_fixed_x_t"
    output.attrs["pi05_state_token_present"] = False
    output.attrs["num_images"] = num_images
    output.attrs["num_layers"] = num_layers
    output.attrs["hidden_size"] = hidden_size
    output.attrs["prefix_hidden_size"] = prefix_hidden_size
    output.attrs["action_horizon"] = action_horizon
    output.attrs["action_dim"] = action_dim
    output.attrs["complete"] = False
    return output


def max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first.float() - second.float())).item())


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.timestep <= 1.0:
        raise ValueError("--timestep must be in [0, 1].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.output_h5.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_h5}. Use --resume to continue it.")

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    expert = model.paligemma_with_expert.gemma_expert.model
    num_layers = len(expert.layers)
    hidden_size = expert.config.hidden_size
    prefix_hidden_size = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim
    started_at = time.perf_counter()

    with h5py.File(args.input_h5, "r") as source:
        available = source["base_images"].shape[0]
        num_images = min(available, args.max_images or available)
        pairs = read_pairs(source)
        pairs_by_id = {int(pair["pair_id"]): pair for pair in pairs}
        pair_ids = read_pair_ids(source, num_images)
        first_index_by_pair = {
            int(pair_id): int(np.flatnonzero(pair_ids == pair_id)[0])
            for pair_id in np.unique(pair_ids)
        }

        if args.output_h5.exists():
            output = h5py.File(args.output_h5, "r+")
            expected_shape = (num_images, 2, num_layers, hidden_size)
            existing_shape = output["expert_action_mean_residuals"].shape
            if existing_shape != expected_shape:
                output.close()
                raise ValueError(f"Existing output shape {existing_shape} != {expected_shape}.")
            if float(output.attrs["timestep"]) != args.timestep:
                output.close()
                raise ValueError("Existing output uses a different timestep.")
            if int(output.attrs["noise_seed"]) != args.noise_seed:
                output.close()
                raise ValueError("Existing output uses a different noise seed.")
        else:
            output = create_output(
                args.output_h5,
                num_images=num_images,
                num_layers=num_layers,
                hidden_size=hidden_size,
                prefix_hidden_size=prefix_hidden_size,
                action_horizon=action_horizon,
                action_dim=action_dim,
                source_path=args.input_h5,
                checkpoint=args.checkpoint,
                timestep=args.timestep,
                noise_seed=args.noise_seed,
            )

        try:
            pending = np.flatnonzero(~output["completed"][:])
            extractor = Pi05ActionExpertResidualExtractor(model)
            for progress_index, image_index in enumerate(
                tqdm(pending, desc="pi05 paired action-expert residuals", unit="image"), start=1
            ):
                base_image = np.asarray(source["base_images"][image_index])
                wrist_image = np.asarray(source["wrist_images"][image_index])
                state = np.asarray(source["robot_states"][image_index])
                pair_id = int(pair_ids[image_index])
                prompts = prompts_for_sample(
                    source,
                    pairs_by_id,
                    pair_ids,
                    int(image_index),
                    kind="plain",
                )
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
                if max_abs_difference(noise[0], noise[1]) != 0.0:
                    raise RuntimeError("Noise changed within a paired-prompt example.")
                readout = extractor.extract(observation, noise=noise, timestep=args.timestep)

                if args.verify_determinism and progress_index == 1:
                    repeated = extractor.extract(observation, noise=noise, timestep=args.timestep)
                    determinism = {
                        key: max_abs_difference(readout[key], repeated[key])
                        for key in ("action_mean", "action_first", "velocity")
                    }
                    output.attrs["determinism_max_abs_difference_json"] = json.dumps(determinism)
                    if any(value != 0.0 for value in determinism.values()):
                        raise RuntimeError(f"Action-expert extraction is not exact: {determinism}.")

                action_mean = readout["action_mean"].float().numpy()
                action_first = readout["action_first"].float().numpy()
                velocity = readout["velocity"].float().numpy()
                visual_control = readout["visual_input_mean"].float().numpy()
                visual_difference = np.abs(visual_control[0] - visual_control[1])
                if visual_difference.max() != 0.0:
                    raise RuntimeError(
                        "Prompt-blind visual control changed across paired prompts: "
                        f"max abs difference={visual_difference.max()}"
                    )

                output["expert_action_mean_residuals"][image_index] = action_mean.astype(np.float16)
                output["expert_action_first_residuals"][image_index] = action_first.astype(np.float16)
                output["expert_velocity"][image_index] = velocity.astype(np.float16)
                output["visual_input_mean"][image_index] = visual_control[0].astype(np.float16)
                output["completed"][image_index] = True

                if image_index == first_index_by_pair[pair_id]:
                    language_tokens = readout["language_tokens"].numpy()
                    language_masks = readout["language_masks"].numpy().astype(bool)
                    for prompt_index in range(2):
                        valid_tokens = language_tokens[prompt_index, language_masks[prompt_index]]
                        output.attrs[f"pair_{pair_id}_prompt_{prompt_index}_input_ids"] = json.dumps(
                            valid_tokens.tolist()
                        )
                    output.attrs["num_image_tokens"] = int(readout["num_image_tokens"])
                    output.attrs["num_action_tokens"] = int(readout["num_action_tokens"])
                    output.attrs[f"pair_{pair_id}_visual_prompt_pair_max_abs_difference"] = float(
                        visual_difference.max()
                    )
                    output.attrs[f"pair_{pair_id}_expert_mean_pair_mean_abs_difference"] = float(
                        np.abs(action_mean[0] - action_mean[1]).mean()
                    )
                    output.attrs[f"pair_{pair_id}_expert_first_pair_mean_abs_difference"] = float(
                        np.abs(action_first[0] - action_first[1]).mean()
                    )
                    output.attrs[f"pair_{pair_id}_velocity_pair_mean_abs_difference"] = float(
                        np.abs(velocity[0] - velocity[1]).mean()
                    )

                if progress_index % args.flush_every == 0:
                    output.flush()

            output.attrs["complete"] = bool(output["completed"][:].all())
            output.attrs["elapsed_seconds_latest_run"] = time.perf_counter() - started_at
            output.attrs["max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            output.attrs["job_id"] = os.environ.get("SLURM_JOB_ID", "")
            output.flush()
            print(
                json.dumps(
                    {
                        "output_h5": str(args.output_h5),
                        "mean_shape": list(output["expert_action_mean_residuals"].shape),
                        "first_shape": list(output["expert_action_first_residuals"].shape),
                        "velocity_shape": list(output["expert_velocity"].shape),
                        "num_completed": int(output["completed"][:].sum()),
                        "num_pairs": len(pairs),
                        "timestep": float(output.attrs["timestep"]),
                        "expert_mean_pair_mean_abs_difference": {
                            str(pair_id): float(
                                output.attrs[
                                    f"pair_{pair_id}_expert_mean_pair_mean_abs_difference"
                                ]
                            )
                            for pair_id in np.unique(pair_ids)
                        },
                        "velocity_pair_mean_abs_difference": {
                            str(pair_id): float(
                                output.attrs[
                                    f"pair_{pair_id}_velocity_pair_mean_abs_difference"
                                ]
                            )
                            for pair_id in np.unique(pair_ids)
                        },
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
