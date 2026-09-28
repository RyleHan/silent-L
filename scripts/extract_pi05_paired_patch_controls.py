#!/usr/bin/env python3
"""Run fixed semantic and matched controls for selected pi0.5 paired patching."""

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


GOAL_FACTORS = (
    "target_xyz",
    "alternative_xyz",
    "target_to_eef_xyz",
    "alternative_to_eef_xyz",
    "target_to_basket_xyz",
    "eef_target_distance",
    "eef_alternative_distance",
    "target_basket_distance",
    "target_displacement_xyz",
    "target_speed_per_action_step",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--token-delta-h5", type=Path, required=True)
    parser.add_argument("--reference-patch-h5", type=Path, required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--base-prompt-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--shuffle-seed", type=int, default=20260811)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--verify-audits", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def build_episode_matched_shuffle(source: h5py.File, seed: int) -> np.ndarray:
    splits = decode_strings(source["layout_splits"][:])
    task_ids = np.asarray(source["task_ids"][:], dtype=np.int64)
    episode_ids = np.asarray(source["episode_ids"][:], dtype=np.int64)
    action_steps = np.asarray(source["action_steps"][:], dtype=np.int64)
    shuffled = np.full(len(splits), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for split in np.unique(splits):
        for task_id in np.unique(task_ids[splits == split]):
            group = np.flatnonzero((splits == split) & (task_ids == task_id))
            episodes = np.unique(episode_ids[group])
            if len(episodes) < 2:
                raise ValueError(f"Need two episodes for shuffle group {split}/{task_id}.")
            episode_order = rng.permutation(episodes)
            episode_mapping = {
                int(episode): int(episode_order[(offset + 1) % len(episode_order)])
                for offset, episode in enumerate(episode_order)
            }
            for episode in episodes:
                current = group[episode_ids[group] == episode]
                source_episode = episode_mapping[int(episode)]
                candidates = group[episode_ids[group] == source_episode]
                current = current[np.argsort(action_steps[current])]
                candidates = candidates[np.argsort(action_steps[candidates])]
                if len(current) == 1:
                    candidate_offsets = np.zeros(1, dtype=np.int64)
                else:
                    candidate_offsets = np.rint(
                        np.linspace(0, len(candidates) - 1, len(current))
                    ).astype(np.int64)
                shuffled[current] = candidates[candidate_offsets]

    if np.any(shuffled < 0):
        raise RuntimeError("Failed to assign every shuffled source index.")
    if np.any(episode_ids[shuffled] == episode_ids):
        raise RuntimeError("Episode-matched shuffle retained a source episode.")
    if np.any(task_ids[shuffled] != task_ids) or np.any(splits[shuffled] != splits):
        raise RuntimeError("Shuffled controls crossed task or split boundaries.")
    return shuffled


def load_goal_basis(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    feature_std = checkpoint["feature_std"].float().clamp_min(1e-8)
    factor_slices = checkpoint["factor_slices"]
    continuous_weight = checkpoint["states"]["continuous"]["weight"].float()
    rows = []
    selected_rows = {}
    for factor in GOAL_FACTORS:
        if factor not in factor_slices:
            raise KeyError(f"Goal factor {factor!r} is absent from the probe checkpoint.")
        start, end = factor_slices[factor]
        rows.append(continuous_weight[start:end])
        selected_rows[factor] = [int(start), int(end)]
    for head_name in ("binary", "phase"):
        state = checkpoint["states"].get(head_name)
        if state is not None:
            rows.append(state["weight"].float())

    effective_rows = torch.cat(rows, dim=0) / feature_std.unsqueeze(0)
    _, singular_values, vh = torch.linalg.svd(effective_rows, full_matrices=False)
    tolerance = (
        singular_values.max()
        * max(effective_rows.shape)
        * torch.finfo(singular_values.dtype).eps
    )
    rank = int((singular_values > tolerance).sum().item())
    basis = vh[:rank].T.contiguous()
    metadata = {
        "goal_factors": list(GOAL_FACTORS),
        "continuous_factor_rows": selected_rows,
        "includes_binary_head": checkpoint["states"].get("binary") is not None,
        "includes_phase_head": checkpoint["states"].get("phase") is not None,
        "effective_row_count": int(effective_rows.shape[0]),
        "basis_rank": rank,
        "hidden_size": int(effective_rows.shape[1]),
    }
    return basis, metadata


def create_output(
    path: Path,
    *,
    num_images: int,
    mode_names: list[str],
    action_horizon: int,
    action_dim: int,
    shuffled_indices: np.ndarray,
    args: argparse.Namespace,
    basis_metadata: dict[str, object],
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, "w")
    output.create_dataset(
        "natural_velocity",
        shape=(num_images, 2, action_horizon, action_dim),
        dtype=np.float32,
        chunks=(1, 2, action_horizon, action_dim),
    )
    output.create_dataset(
        "patched_velocity",
        shape=(num_images, len(mode_names), action_horizon, action_dim),
        dtype=np.float32,
        chunks=(1, len(mode_names), action_horizon, action_dim),
    )
    for name in ("patch_delta_norm_ratio", "untouched_velocity_max_abs_difference"):
        output.create_dataset(
            name,
            shape=(num_images, len(mode_names)),
            dtype=np.float32,
            chunks=(1, len(mode_names)),
        )
    output.create_dataset("shuffled_source_indices", data=shuffled_indices[:num_images])
    output.create_dataset("completed", shape=(num_images,), dtype=np.bool_, fillvalue=False)
    output.attrs["source_h5"] = str(args.input_h5)
    output.attrs["token_delta_h5"] = str(args.token_delta_h5)
    output.attrs["reference_patch_h5"] = str(args.reference_patch_h5)
    output.attrs["probe_checkpoint"] = str(args.probe_checkpoint)
    output.attrs["checkpoint"] = str(args.checkpoint)
    output.attrs["openpi_revision"] = DEFAULT_OPENPI_REVISION
    output.attrs["layer"] = args.layer
    output.attrs["timestep"] = args.timestep
    output.attrs["noise_seed"] = args.noise_seed
    output.attrs["shuffle_seed"] = args.shuffle_seed
    output.attrs["random_seed"] = args.random_seed
    output.attrs["base_prompt_index"] = args.base_prompt_index
    output.attrs["source_prompt_index"] = 1 - args.base_prompt_index
    output.attrs["mode_names_json"] = json.dumps(mode_names)
    output.attrs["projection_basis_json"] = json.dumps(basis_metadata)
    output.attrs["complete"] = False
    return output


def reference_offsets(reference: h5py.File, layer: int) -> tuple[int, int]:
    layers = json.loads(decode_attr(reference.attrs["patch_layers_json"]))
    alphas = json.loads(decode_attr(reference.attrs["alphas_json"]))
    if layer not in layers or 1.0 not in alphas:
        raise ValueError("Reference patch cache lacks selected layer/alpha=1.")
    return layers.index(layer), alphas.index(1.0)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.output_h5.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_h5}. Use --resume.")

    device = torch.device("cuda:0")
    basis, basis_metadata = load_goal_basis(args.probe_checkpoint)
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim
    started_at = time.perf_counter()

    with (
        h5py.File(args.input_h5, "r") as source,
        h5py.File(args.token_delta_h5, "r") as token_cache,
        h5py.File(args.reference_patch_h5, "r") as reference,
    ):
        available = source["base_images"].shape[0]
        num_images = min(available, args.max_images or available)
        if token_cache["paired_token_delta_b_minus_a"].shape[0] < num_images:
            raise ValueError("Token-delta cache has fewer samples than requested.")
        if reference["patched_velocity"].shape[0] < num_images:
            raise ValueError("Reference patch cache has fewer samples than requested.")
        if int(token_cache.attrs["layer"]) != args.layer:
            raise ValueError("Token-delta cache uses a different expert block.")
        if int(reference.attrs["base_prompt_index"]) != args.base_prompt_index:
            raise ValueError("Reference patch cache uses a different base prompt.")
        layer_offset, alpha_offset = reference_offsets(reference, args.layer)
        shuffled_indices = build_episode_matched_shuffle(source, args.shuffle_seed)
        prompts = [decode_attr(source.attrs["prompt_a"]), decode_attr(source.attrs["prompt_b"])]
        mode_names = ["paired", "goal", "null", "shuffled", "random"]

        if args.output_h5.exists():
            output = h5py.File(args.output_h5, "r+")
            if output["patched_velocity"].shape[:2] != (num_images, len(mode_names)):
                output.close()
                raise ValueError("Existing control output has an incompatible shape.")
        else:
            output = create_output(
                args.output_h5,
                num_images=num_images,
                mode_names=mode_names,
                action_horizon=action_horizon,
                action_dim=action_dim,
                shuffled_indices=shuffled_indices,
                args=args,
                basis_metadata=basis_metadata,
            )

        try:
            pending = np.flatnonzero(~output["completed"][:])
            extractor = Pi05ActionExpertResidualExtractor(model)
            for progress_index, image_index in enumerate(
                tqdm(pending, desc="pi05 selected paired-patch controls", unit="image"), start=1
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
                shuffled_index = int(shuffled_indices[image_index])
                shuffled_delta = torch.from_numpy(
                    np.asarray(
                        token_cache["paired_token_delta_b_minus_a"][shuffled_index],
                        dtype=np.float32,
                    )
                )
                if args.base_prompt_index == 1:
                    shuffled_delta = -shuffled_delta
                readout = extractor.extract_paired_patch_controls(
                    observation,
                    noise=noise,
                    timestep=args.timestep,
                    layer_index=args.layer,
                    projection_basis=basis,
                    shuffled_delta=shuffled_delta,
                    random_seed=args.random_seed + int(image_index) * 2 + args.base_prompt_index,
                    base_prompt_index=args.base_prompt_index,
                    source_prompt_index=1 - args.base_prompt_index,
                )
                if readout["mode_names"] != mode_names:
                    raise RuntimeError("Runtime control mode order changed unexpectedly.")

                patched_velocity = readout["patched_velocity"].numpy()
                untouched = readout["untouched_velocity_max_abs_difference"].numpy()
                reference_velocity = np.asarray(
                    reference["patched_velocity"][
                        image_index, layer_offset, alpha_offset
                    ],
                    dtype=np.float32,
                )
                paired_reference_error = float(
                    np.max(np.abs(patched_velocity[0] - reference_velocity))
                )
                if args.verify_audits:
                    if untouched.max() != 0.0:
                        raise RuntimeError(
                            f"Control patch changed source branch: {untouched.max()}."
                        )
                    if paired_reference_error != 0.0:
                        raise RuntimeError(
                            "Paired control failed to reproduce locked sweep: "
                            f"max abs difference={paired_reference_error}."
                        )

                output["natural_velocity"][image_index] = readout["natural_velocity"].numpy()
                output["patched_velocity"][image_index] = patched_velocity
                output["patch_delta_norm_ratio"][image_index] = readout[
                    "patch_delta_norm_ratio"
                ].numpy()
                output["untouched_velocity_max_abs_difference"][image_index] = untouched
                output["completed"][image_index] = True
                if progress_index == 1:
                    output.attrs["paired_reference_velocity_max_abs_difference_first"] = (
                        paired_reference_error
                    )
                    output.attrs["untouched_velocity_max_abs_difference_first"] = float(
                        untouched.max()
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
                        "layer": args.layer,
                        "base_prompt_index": args.base_prompt_index,
                        "mode_names": mode_names,
                        "projection_basis": basis_metadata,
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
