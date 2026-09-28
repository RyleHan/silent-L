#!/usr/bin/env python3
"""Generate the legal Othello sequences expected by the official probe code."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data.othello import get_ood_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--valid-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_model_tokens(board_sequences: torch.Tensor) -> torch.Tensor:
    model_tokens = board_sequences.clone()
    model_tokens[board_sequences < 29] += 1
    model_tokens[(board_sequences >= 29) & (board_sequences <= 34)] -= 1
    model_tokens[board_sequences > 34] -= 3
    return model_tokens


def main() -> None:
    args = parse_args()
    requested_size = args.train_size + args.valid_size
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accepted: list[list[int]] = []
    attempted = 0
    started_at = time.perf_counter()
    progress = tqdm(total=requested_size, desc="full 60-move games")
    while len(accepted) < requested_size:
        sequence = get_ood_game(attempted)
        attempted += 1
        if len(sequence) == 60:
            accepted.append(sequence)
            progress.update(1)
    progress.close()

    board_sequences = torch.tensor(accepted, dtype=torch.long)
    permutation = torch.randperm(requested_size, generator=torch.Generator().manual_seed(args.seed))
    board_sequences = board_sequences[permutation]
    model_tokens = to_model_tokens(board_sequences)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "board_seqs_string_train.pth": board_sequences[: args.train_size],
        "board_seqs_int_train.pth": model_tokens[: args.train_size],
        "board_seqs_string_valid.pth": board_sequences[args.train_size :],
        "board_seqs_int_valid.pth": model_tokens[args.train_size :],
    }
    for filename, tensor in outputs.items():
        torch.save(tensor, args.output_dir / filename)

    result = {
        "seed": args.seed,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
        "sequence_length": board_sequences.shape[1],
        "attempted_games": attempted,
        "acceptance_rate": requested_size / attempted,
        "elapsed_seconds": time.perf_counter() - started_at,
        "files": {name: list(tensor.shape) for name, tensor in outputs.items()},
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

