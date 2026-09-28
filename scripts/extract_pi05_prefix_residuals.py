#!/usr/bin/env python3
"""Cache paired-prompt pi0.5 PaliGemma prefix residuals."""

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
    Pi05PrefixResidualExtractor,
    load_pi05_policy,
    prepare_pi05_observation_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def create_output(
    path: Path,
    *,
    num_images: int,
    num_layers: int,
    hidden_size: int,
    source_path: Path,
    checkpoint: Path,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, "w")
    output.create_dataset(
        "prompt_end_residuals",
        shape=(num_images, 2, num_layers, hidden_size),
        dtype=np.float16,
        chunks=(1, 1, num_layers, hidden_size),
    )
    output.create_dataset(
        "visual_input_mean",
        shape=(num_images, hidden_size),
        dtype=np.float16,
        chunks=(1, hidden_size),
    )
    output.create_dataset("completed", shape=(num_images,), dtype=np.bool_, fillvalue=False)
    output.attrs["source_h5"] = str(source_path)
    output.attrs["checkpoint"] = str(checkpoint)
    output.attrs["openpi_revision"] = DEFAULT_OPENPI_REVISION
    output.attrs["layer_convention"] = "PaliGemma_decoder_block_output_before_final_norm"
    output.attrs["readout"] = "last_valid_language_token"
    output.attrs["visual_control"] = (
        "mean_of_valid_projected_SigLIP_tokens_before_PaliGemma_language_blocks"
    )
    output.attrs["num_images"] = num_images
    output.attrs["num_layers"] = num_layers
    output.attrs["hidden_size"] = hidden_size
    output.attrs["complete"] = False
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.output_h5.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_h5}. Use --resume to continue it.")

    device = torch.device("cuda:0")
    policy = load_pi05_policy(args.checkpoint, device)
    model = policy._model  # noqa: SLF001
    model.eval()
    num_layers = len(model.paligemma_with_expert.paligemma.language_model.layers)
    hidden_size = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
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
            if output["prompt_end_residuals"].shape != expected_shape:
                existing_shape = output["prompt_end_residuals"].shape
                output.close()
                raise ValueError(f"Existing output shape {existing_shape} != {expected_shape}.")
        else:
            output = create_output(
                args.output_h5,
                num_images=num_images,
                num_layers=num_layers,
                hidden_size=hidden_size,
                source_path=args.input_h5,
                checkpoint=args.checkpoint,
            )

        try:
            pending = np.flatnonzero(~output["completed"][:])
            extractor = Pi05PrefixResidualExtractor(model)
            for progress_index, image_index in enumerate(
                tqdm(pending, desc="pi05 paired prefix residuals", unit="image"), start=1
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
                readout = extractor.extract(observation)
                prompt_end = readout["prompt_end"].float().numpy()
                visual_control = readout["visual_input_mean"].float().numpy()
                visual_difference = np.abs(visual_control[0] - visual_control[1])
                if visual_difference.max() != 0.0:
                    raise RuntimeError(
                        "Prompt-blind visual control changed across paired prompts: "
                        f"max abs difference={visual_difference.max()}"
                    )

                output["prompt_end_residuals"][image_index] = prompt_end.astype(np.float16)
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
                        output.attrs[f"pair_{pair_id}_prompt_{prompt_index}_token_count"] = len(valid_tokens)
                    output.attrs["num_image_tokens"] = int(readout["num_image_tokens"])
                    output.attrs[f"pair_{pair_id}_visual_prompt_pair_max_abs_difference"] = float(
                        visual_difference.max()
                    )
                    output.attrs[f"pair_{pair_id}_prompt_readout_pair_mean_abs_difference"] = float(
                        np.abs(prompt_end[0] - prompt_end[1]).mean()
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
                        "shape": list(output["prompt_end_residuals"].shape),
                        "visual_control_shape": list(output["visual_input_mean"].shape),
                        "num_completed": int(output["completed"][:].sum()),
                        "num_pairs": len(pairs),
                        "num_image_tokens": int(output.attrs["num_image_tokens"]),
                        "prompt_pair_mean_abs_difference": {
                            str(pair_id): float(
                                output.attrs[
                                    f"pair_{pair_id}_prompt_readout_pair_mean_abs_difference"
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
