#!/usr/bin/env python3
"""Cache paired-prompt OpenVLA residuals for coordinate-system probes."""

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
from vla_coordinates.openvla_residuals import OpenVLAResidualExtractor
from vla_coordinates.openvla_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_REVISION,
    load_openvla,
    prepare_openvla_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
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
    checkpoint: str,
    revision: str,
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    handle.create_dataset(
        "prompt_residuals",
        shape=(num_images, 2, num_layers, 2, hidden_size),
        dtype=np.float16,
        chunks=(1, 1, num_layers, 2, hidden_size),
    )
    handle.create_dataset(
        "visual_mean_residuals",
        shape=(num_images, num_layers, hidden_size),
        dtype=np.float16,
        chunks=(1, num_layers, hidden_size),
    )
    handle.create_dataset("completed", shape=(num_images,), dtype=np.bool_, fillvalue=False)
    handle.attrs["source_h5"] = str(source_path)
    handle.attrs["checkpoint"] = checkpoint
    handle.attrs["revision"] = revision
    handle.attrs["layer_convention"] = "decoder_block_output_before_final_norm"
    handle.attrs["prompt_position_names"] = json.dumps(["prompt_end", "action_boundary"])
    handle.attrs["visual_readout"] = "mean_over_visual_tokens"
    handle.attrs["num_images"] = num_images
    handle.attrs["num_layers"] = num_layers
    handle.attrs["hidden_size"] = hidden_size
    handle.attrs["complete"] = False
    return handle


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    if args.output_h5.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_h5}. Use --resume to continue it.")

    device = torch.device("cuda:0")
    dtype = torch.float16
    model, processor = load_openvla(
        args.checkpoint,
        args.revision,
        device,
        dtype=dtype,
        attention_implementation="eager",
        local_files_only=True,
    )
    hidden_size = model.config.text_config.hidden_size
    num_layers = model.config.text_config.num_hidden_layers
    started_at = time.perf_counter()

    with h5py.File(args.input_h5, "r") as source:
        available_images = source["images"].shape[0]
        num_images = min(available_images, args.max_images or available_images)
        pairs = read_pairs(source)
        pairs_by_id = {int(pair["pair_id"]): pair for pair in pairs}
        pair_ids = read_pair_ids(source, num_images)
        first_index_by_pair = {
            int(pair_id): int(np.flatnonzero(pair_ids == pair_id)[0])
            for pair_id in np.unique(pair_ids)
        }

        if args.output_h5.exists():
            output = h5py.File(args.output_h5, "r+")
            expected_shape = (num_images, 2, num_layers, 2, hidden_size)
            if output["prompt_residuals"].shape != expected_shape:
                existing_shape = output["prompt_residuals"].shape
                output.close()
                raise ValueError(
                    f"Existing output shape {existing_shape} != expected {expected_shape}."
                )
        else:
            output = create_output(
                args.output_h5,
                num_images=num_images,
                num_layers=num_layers,
                hidden_size=hidden_size,
                source_path=args.input_h5,
                checkpoint=args.checkpoint,
                revision=args.revision,
            )

        try:
            completed = output["completed"][:]
            pending = np.flatnonzero(~completed)
            with OpenVLAResidualExtractor(model) as extractor:
                progress = tqdm(pending, desc="paired residuals", unit="image")
                for progress_index, image_index in enumerate(progress, start=1):
                    image = source["images"][image_index]
                    pair_id = int(pair_ids[image_index])
                    prompts = prompts_for_sample(
                        source,
                        pairs_by_id,
                        pair_ids,
                        int(image_index),
                        kind="openvla",
                    )
                    verify_visual_causality = image_index == first_index_by_pair[pair_id]
                    visual_mean = None
                    for prompt_index, prompt in enumerate(prompts):
                        inputs = prepare_openvla_inputs(
                            processor,
                            prompt,
                            image,
                            device,
                            dtype=dtype,
                            center_crop=False,
                        )
                        if verify_visual_causality:
                            token_ids = inputs["input_ids"][0].detach().cpu().tolist()
                            output.attrs[f"pair_{pair_id}_prompt_{prompt_index}_input_ids"] = json.dumps(token_ids)
                            output.attrs[f"pair_{pair_id}_prompt_{prompt_index}_tokens"] = json.dumps(
                                processor.tokenizer.convert_ids_to_tokens(token_ids)
                            )
                        readout = extractor.extract(
                            inputs,
                            include_visual=prompt_index == 0 or verify_visual_causality,
                        )
                        output["prompt_residuals"][image_index, prompt_index] = (
                            readout.prompt_positions.numpy().astype(np.float16, copy=False)
                        )
                        output.attrs[f"pair_{pair_id}_multimodal_sequence_length_prompt_{prompt_index}"] = (
                            readout.multimodal_sequence_length
                        )
                        if prompt_index == 0:
                            assert readout.visual_mean is not None
                            visual_mean = readout.visual_mean
                            output.attrs["num_visual_tokens"] = readout.num_visual_tokens
                        elif verify_visual_causality:
                            assert visual_mean is not None and readout.visual_mean is not None
                            visual_difference = (visual_mean.float() - readout.visual_mean.float()).abs()
                            output.attrs[f"pair_{pair_id}_visual_prompt_pair_max_abs_difference"] = visual_difference.max().item()
                            output.attrs[f"pair_{pair_id}_visual_prompt_pair_mean_abs_difference"] = visual_difference.mean().item()

                    assert visual_mean is not None
                    output["visual_mean_residuals"][image_index] = visual_mean.numpy().astype(
                        np.float16, copy=False
                    )
                    if verify_visual_causality:
                        prompt_pair = output["prompt_residuals"][image_index]
                        output.attrs[f"pair_{pair_id}_prompt_readout_pair_mean_abs_difference"] = float(
                            np.abs(prompt_pair[0].astype(np.float32) - prompt_pair[1].astype(np.float32)).mean()
                        )
                    output["completed"][image_index] = True
                    if progress_index % args.flush_every == 0:
                        output.flush()

            output.attrs["complete"] = bool(output["completed"][:].all())
            output.attrs["elapsed_seconds_latest_run"] = time.perf_counter() - started_at
            output.attrs["max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            output.attrs["job_id"] = os.environ.get("SLURM_JOB_ID", "")
            output.flush()
            summary = {
                "output_h5": str(args.output_h5),
                "shape": list(output["prompt_residuals"].shape),
                "visual_shape": list(output["visual_mean_residuals"].shape),
                "num_completed": int(output["completed"][:].sum()),
                "num_pairs": len(pairs),
                "num_visual_tokens": int(output.attrs["num_visual_tokens"]),
                "elapsed_seconds": output.attrs["elapsed_seconds_latest_run"],
                "max_cuda_memory_gib": output.attrs["max_cuda_memory_gib"],
            }
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            output.close()


if __name__ == "__main__":
    main()
