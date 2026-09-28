#!/usr/bin/env python3
"""Reproduce absolute versus MINE/YOURS linear probes on OthelloGPT."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--valid-size", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def absolute_labels(state_stack: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(state_stack, dtype=torch.long)
    labels[state_stack == 1] = 1
    labels[state_stack == -1] = 2
    return labels


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    encoded = labels.reshape(-1) * num_classes + predictions.reshape(-1)
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    confusion = confusion.to(torch.float64)
    true_positives = confusion.diag()
    precision = true_positives / confusion.sum(dim=0).clamp_min(1)
    recall = true_positives / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).eps)
    return {
        "accuracy": (true_positives.sum() / confusion.sum()).item(),
        "macro_f1": f1.mean().item(),
        "per_class_f1": f1.tolist(),
        "confusion": confusion.to(torch.long).tolist(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # The official repository hard-codes OTHELLO_HOME at import time.
    import constants

    constants.OTHELLO_HOME = str(args.reference_root)
    from mech_int.tl_othello_utils import (
        build_state_stack,
        load_hooked_model,
        state_stack_to_one_hot_threeway,
    )

    data_dir = args.reference_root / "data"
    train_int = torch.load(data_dir / "board_seqs_int_train.pth")[: args.train_size]
    train_string = torch.load(data_dir / "board_seqs_string_train.pth")[: args.train_size]
    valid_int = torch.load(data_dir / "board_seqs_int_valid.pth")[: args.valid_size]
    valid_string = torch.load(data_dir / "board_seqs_string_valid.pth")[: args.valid_size]
    if len(train_int) < args.train_size or len(valid_int) < args.valid_size:
        raise ValueError("Requested more games than the prepared dataset contains.")

    device = torch.device("cuda:0")
    model = load_hooked_model("synthetic").to(device)
    model.eval()
    layers = list(range(model.cfg.n_layers))
    modes = ["absolute", "mine_yours"]

    probes = torch.nn.Parameter(
        torch.randn(
            len(modes),
            len(layers),
            model.cfg.d_model,
            8,
            8,
            3,
            device=device,
        )
        / np.sqrt(model.cfg.d_model)
    )
    optimizer = torch.optim.AdamW([probes], lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    started_at = time.perf_counter()
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(train_int), generator=torch.Generator().manual_seed(args.seed + epoch))
        epoch_losses = torch.zeros(len(modes), len(layers), dtype=torch.float64)
        batches = 0
        progress = tqdm(range(0, len(train_int), args.batch_size), desc=f"probe epoch {epoch + 1}")
        for start in progress:
            indices = permutation[start : start + args.batch_size]
            games_int = train_int[indices].to(device)
            state_stack = build_state_stack(train_string[indices])[:, : model.cfg.n_ctx]
            labels = torch.stack(
                [
                    absolute_labels(state_stack),
                    state_stack_to_one_hot_threeway(state_stack)[0].argmax(dim=-1),
                ],
                dim=0,
            ).to(device)

            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    games_int[:, :-1],
                    return_type=None,
                    names_filter=lambda name: name.endswith("hook_resid_post"),
                )

            optimizer.zero_grad(set_to_none=True)
            losses = []
            for layer_index, layer in enumerate(layers):
                residual = cache["resid_post", layer].clone()
                for mode_index in range(len(modes)):
                    logits = torch.einsum(
                        "bpd,drcq->bprcq",
                        residual,
                        probes[mode_index, layer_index],
                    )
                    loss = F.cross_entropy(logits.reshape(-1, 3), labels[mode_index].reshape(-1))
                    epoch_losses[mode_index, layer_index] += loss.detach().cpu()
                    losses.append(loss)

            torch.stack(losses).sum().backward()
            optimizer.step()
            batches += 1
            progress.set_postfix(mean_loss=f"{torch.stack(losses).mean().item():.4f}")

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "mean_losses": {
                        mode: (epoch_losses[mode_index] / batches).tolist()
                        for mode_index, mode in enumerate(modes)
                    },
                }
            ),
            flush=True,
        )

    confusions = torch.zeros(len(modes), len(layers), 3, 3, dtype=torch.long)
    with torch.inference_mode():
        for start in tqdm(range(0, len(valid_int), args.batch_size), desc="probe validation"):
            games_int = valid_int[start : start + args.batch_size].to(device)
            state_stack = build_state_stack(valid_string[start : start + args.batch_size])[:, : model.cfg.n_ctx]
            labels = torch.stack(
                [
                    absolute_labels(state_stack),
                    state_stack_to_one_hot_threeway(state_stack)[0].argmax(dim=-1),
                ],
                dim=0,
            ).to(device)
            _, cache = model.run_with_cache(
                games_int[:, :-1],
                return_type=None,
                names_filter=lambda name: name.endswith("hook_resid_post"),
            )
            for layer_index, layer in enumerate(layers):
                residual = cache["resid_post", layer]
                for mode_index in range(len(modes)):
                    logits = torch.einsum(
                        "bpd,drcq->bprcq",
                        residual,
                        probes[mode_index, layer_index],
                    )
                    predictions = logits.argmax(dim=-1)
                    confusions[mode_index, layer_index] += confusion_matrix(
                        predictions,
                        labels[mode_index],
                    ).cpu()

    layer_metrics = {
        mode: {
            str(layer): metrics_from_confusion(confusions[mode_index, layer_index])
            for layer_index, layer in enumerate(layers)
        }
        for mode_index, mode in enumerate(modes)
    }
    best_layers = {
        mode: max(metrics, key=lambda layer: metrics[layer]["macro_f1"])
        for mode, metrics in layer_metrics.items()
    }
    result = {
        "config": vars(args) | {"reference_root": str(args.reference_root), "output_dir": str(args.output_dir)},
        "model": {
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_ctx": model.cfg.n_ctx,
        },
        "labels": {
            "absolute": ["empty", "black", "white"],
            "mine_yours": ["empty", "mine", "yours"],
        },
        "metrics": layer_metrics,
        "best_layers_by_macro_f1": best_layers,
        "elapsed_seconds": time.perf_counter() - started_at,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probes.detach().cpu(), args.output_dir / "linear_probes.pt")
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Wrote: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
