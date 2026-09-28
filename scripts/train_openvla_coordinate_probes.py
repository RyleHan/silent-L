#!/usr/bin/env python3
"""Train matched OthelloGPT-style linear probes on cached OpenVLA residuals."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


MODE_NAMES = ("absolute", "target_alternative")
READOUT_POSITION = {"prompt_end": 0, "action_boundary": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-h5", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--readout",
        choices=["prompt_end", "action_boundary", "visual_mean"],
        required=True,
    )
    parser.add_argument("--layers", default="all", help="Comma-separated zero-based layers or 'all'.")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--min-mask-pixels", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_layers(specification: str, num_layers: int) -> list[int]:
    if specification == "all":
        return list(range(num_layers))
    layers = sorted({int(value) for value in specification.split(",")})
    if not layers or layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError(f"Layers must be in [0, {num_layers - 1}], got {layers}.")
    return layers


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])


def masks_to_patch_labels(
    masks: np.ndarray,
    *,
    grid_size: int,
    min_mask_pixels: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if masks.ndim != 4 or masks.shape[-1] != 2 or masks.shape[1] != masks.shape[2]:
        raise ValueError(f"Expected masks [image, height, width, 2], got {masks.shape}.")
    image_size = masks.shape[1]
    if image_size % grid_size:
        raise ValueError(f"Image size {image_size} is not divisible by grid size {grid_size}.")
    patch_size = image_size // grid_size
    pooled = masks.reshape(
        len(masks), grid_size, patch_size, grid_size, patch_size, 2
    ).sum(axis=(2, 4))
    winner = pooled.argmax(axis=-1)
    winner_pixels = pooled.max(axis=-1)
    labels = np.zeros(winner.shape, dtype=np.int64)
    foreground = winner_pixels >= min_mask_pixels
    labels[foreground] = winner[foreground] + 1

    per_image_a = (labels == 1).sum(axis=(1, 2))
    per_image_b = (labels == 2).sum(axis=(1, 2))
    eligible = (per_image_a > 0) & (per_image_b > 0)
    stats = {
        "patch_size": patch_size,
        "min_mask_pixels": min_mask_pixels,
        "num_images_total": len(labels),
        "num_images_eligible": int(eligible.sum()),
        "num_images_excluded": int((~eligible).sum()),
        "excluded_source_image_indices": np.flatnonzero(~eligible).tolist(),
        "empty_object_a_images": int((per_image_a == 0).sum()),
        "empty_object_b_images": int((per_image_b == 0).sum()),
        "patch_conflicts": int(np.logical_and(pooled[..., 0] >= min_mask_pixels, pooled[..., 1] >= min_mask_pixels).sum()),
        "object_a_patches_min_median_max": [
            int(per_image_a.min()),
            float(np.median(per_image_a)),
            int(per_image_a.max()),
        ],
        "object_b_patches_min_median_max": [
            int(per_image_b.min()),
            float(np.median(per_image_b)),
            int(per_image_b.max()),
        ],
    }
    return labels, eligible, stats


def build_paired_labels(image_labels: np.ndarray) -> np.ndarray:
    absolute = np.repeat(image_labels[:, None], 2, axis=1)
    swapped = image_labels.copy()
    swapped[image_labels == 1] = 2
    swapped[image_labels == 2] = 1
    relative = np.stack((image_labels, swapped), axis=1)
    return np.stack((absolute, relative), axis=0).reshape(
        2, 2 * len(image_labels), image_labels.shape[1], image_labels.shape[2]
    )


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    encoded = labels.reshape(-1) * num_classes + predictions.reshape(-1)
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    confusion = confusion.to(torch.float64)
    true_positives = confusion.diag()
    precision = true_positives / confusion.sum(dim=0).clamp_min(1)
    recall = true_positives / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).eps)
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positives
    iou = true_positives / union.clamp_min(1)
    return {
        "accuracy": (true_positives.sum() / confusion.sum()).item(),
        "macro_f1": f1.mean().item(),
        "foreground_macro_f1": f1[1:].mean().item(),
        "mean_iou": iou.mean().item(),
        "foreground_mean_iou": iou[1:].mean().item(),
        "per_class_f1": f1.tolist(),
        "per_class_iou": iou.tolist(),
        "confusion": confusion.to(torch.long).tolist(),
    }


def evaluate(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    probes: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    confusions = torch.zeros(2, 3, 3, dtype=torch.long)
    probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_features = features[batch_indices].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = torch.einsum("bd,mdhwc->mbhwc", batch_features, probes)
            predictions = logits.argmax(dim=-1).cpu()
            batch_labels = labels[:, batch_indices]
            for mode_index in range(2):
                confusions[mode_index] += confusion_matrix(predictions[mode_index], batch_labels[mode_index])
            probabilities.append(logits.float().softmax(dim=-1).cpu())
    return [metrics_from_confusion(confusions[index]) for index in range(2)], torch.cat(probabilities, dim=1)


def paired_consistency(probabilities: torch.Tensor, split_indices: torch.Tensor) -> list[dict[str, float]]:
    if len(split_indices) % 2:
        raise ValueError("Paired split has an odd number of prompt samples.")
    if not torch.all(split_indices[0::2] + 1 == split_indices[1::2]):
        raise ValueError("Split indices no longer preserve adjacent same-image prompt pairs.")

    results = []
    for mode_index in range(2):
        prompt_a = probabilities[mode_index, 0::2]
        prompt_b = probabilities[mode_index, 1::2]
        if mode_index == 1:
            prompt_b = prompt_b[..., [0, 2, 1]]
        results.append(
            {
                "mean_absolute_probability_error": (prompt_a - prompt_b).abs().mean().item(),
                "prediction_agreement": (prompt_a.argmax(dim=-1) == prompt_b.argmax(dim=-1)).float().mean().item(),
            }
        )
    return results


def load_layer_features(
    residuals: h5py.File,
    readout: str,
    layer: int,
    source_image_indices: np.ndarray,
) -> torch.Tensor:
    if readout == "visual_mean":
        image_features = residuals["visual_mean_residuals"][source_image_indices, layer, :]
        paired = np.repeat(image_features[:, None, :], 2, axis=1)
    else:
        paired = residuals["prompt_residuals"][
            source_image_indices, :, layer, READOUT_POSITION[readout], :
        ]
    return torch.from_numpy(np.asarray(paired, dtype=np.float32).reshape(-1, paired.shape[-1])).pin_memory()


def train_layer(
    *,
    layer: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    split_indices: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object], dict[str, np.ndarray]]:
    hidden_size = features.shape[-1]
    generator = torch.Generator().manual_seed(args.seed + layer)
    initial = torch.randn(
        hidden_size,
        args.grid_size,
        args.grid_size,
        3,
        generator=generator,
    ) / np.sqrt(hidden_size)
    probes = torch.nn.Parameter(initial.to(device).unsqueeze(0).repeat(2, 1, 1, 1, 1))
    optimizer = torch.optim.AdamW(
        [probes],
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler()
    best_score = -float("inf")
    best_probes = probes.detach().cpu().clone()
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    train_indices = split_indices["train"]

    for epoch in range(args.epochs):
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(args.seed + layer * 1000 + epoch))
        ]
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[start : start + args.batch_size]
            batch_features = features[batch_indices].to(device, non_blocking=True)
            batch_labels = labels[:, batch_indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = torch.einsum("bd,mdhwc->mbhwc", batch_features, probes)
                losses = [
                    F.cross_entropy(
                        logits[mode_index].reshape(-1, 3),
                        batch_labels[mode_index].reshape(-1),
                        weight=class_weights,
                    )
                    for mode_index in range(2)
                ]
                loss = torch.stack(losses).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.detach().item()
            batches += 1

        validation, _ = evaluate(
            features,
            labels,
            split_indices["valid"],
            probes,
            batch_size=args.batch_size,
            device=device,
        )
        score = float(np.mean([metrics["foreground_macro_f1"] for metrics in validation]))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / batches,
                "validation_foreground_macro_f1_mean": score,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_probes = probes.detach().cpu().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    best_probes_gpu = best_probes.to(device)
    split_metrics: dict[str, object] = {}
    split_predictions: dict[str, np.ndarray] = {}
    for split_name in ("valid", "test"):
        metrics, probabilities = evaluate(
            features,
            labels,
            split_indices[split_name],
            best_probes_gpu,
            batch_size=args.batch_size,
            device=device,
        )
        consistency = paired_consistency(probabilities, split_indices[split_name])
        split_predictions[split_name] = probabilities.argmax(dim=-1).numpy().astype(np.uint8)
        split_metrics[split_name] = {
            MODE_NAMES[mode_index]: metrics[mode_index] | {"paired_consistency": consistency[mode_index]}
            for mode_index in range(2)
        }

    result = {
        "layer": layer,
        "best_epoch": best_epoch,
        "best_validation_foreground_macro_f1_mean": best_score,
        "history": history,
        "metrics": split_metrics,
    }
    return best_probes, result, split_predictions


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script in a GPU allocation.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = args.output_dir / "probes"
    probe_dir.mkdir(exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"

    with h5py.File(args.source_h5, "r") as source:
        image_labels, eligible_images, label_stats = masks_to_patch_labels(
            source["masks"][:],
            grid_size=args.grid_size,
            min_mask_pixels=args.min_mask_pixels,
        )
        source_image_indices = np.flatnonzero(eligible_images).astype(np.int64)
        labels = torch.from_numpy(
            build_paired_labels(image_labels[source_image_indices])
        ).long().pin_memory()
        image_splits = decode_strings(source["layout_splits"][source_image_indices])
        source_pair_ids = (
            np.asarray(source["pair_ids"][source_image_indices], dtype=np.int64)
            if "pair_ids" in source
            else np.zeros(len(source_image_indices), dtype=np.int64)
        )
        source_splits = decode_strings(source["layout_splits"][:])
        excluded = ~eligible_images
        label_stats["excluded_by_split"] = {
            split: int((excluded & (source_splits == split)).sum())
            for split in ("train", "valid", "test")
        }
        label_stats["eligible_by_pair"] = {
            str(pair_id): int((source_pair_ids == pair_id).sum())
            for pair_id in np.unique(source_pair_ids)
        }
        paired_splits = np.repeat(image_splits, 2)
        split_indices = {
            split_name: torch.from_numpy(np.flatnonzero(paired_splits == split_name)).long()
            for split_name in ("train", "valid", "test")
        }

    train_absolute = labels[0, split_indices["train"]]
    foreground_count = int((train_absolute != 0).sum())
    background_count = int((train_absolute == 0).sum())
    class_weights = torch.tensor(
        [foreground_count / background_count, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    with h5py.File(args.residual_h5, "r") as residuals:
        if not bool(residuals.attrs.get("complete", False)):
            raise ValueError(f"Residual cache is not marked complete: {args.residual_h5}")
        num_layers = residuals["prompt_residuals"].shape[2]
        layers = parse_layers(args.layers, num_layers)
        existing: dict[str, object] = {}
        if metrics_path.exists() and not args.overwrite:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        layer_results = dict(existing.get("layers", {}))
        started_at = time.perf_counter()

        for layer in tqdm(layers, desc=f"{args.readout} probes", unit="layer"):
            layer_key = str(layer)
            probe_path = probe_dir / f"layer_{layer:02d}.pt"
            prediction_path = prediction_dir / f"layer_{layer:02d}.npz"
            if (
                layer_key in layer_results
                and probe_path.exists()
                and prediction_path.exists()
                and not args.overwrite
            ):
                continue
            features = load_layer_features(
                residuals,
                args.readout,
                layer,
                source_image_indices,
            )
            best_probes, result, split_predictions = train_layer(
                layer=layer,
                features=features,
                labels=labels,
                split_indices=split_indices,
                class_weights=class_weights,
                args=args,
                device=device,
            )
            torch.save(best_probes.half(), probe_path)
            np.savez_compressed(
                prediction_path,
                valid_indices=split_indices["valid"].numpy(),
                valid_predictions=split_predictions["valid"],
                test_indices=split_indices["test"].numpy(),
                test_predictions=split_predictions["test"],
                source_image_indices=source_image_indices,
            )
            layer_results[layer_key] = result
            partial = {
                "config": {
                    **vars(args),
                    "residual_h5": str(args.residual_h5),
                    "source_h5": str(args.source_h5),
                    "output_dir": str(args.output_dir),
                },
                "label_stats": label_stats,
                "class_weights": class_weights.detach().cpu().tolist(),
                "split_prompt_samples": {key: len(value) for key, value in split_indices.items()},
                "layers": layer_results,
            }
            metrics_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    selected_layers = {
        mode_name: max(
            layer_results,
            key=lambda layer: layer_results[layer]["metrics"]["valid"][mode_name]["foreground_macro_f1"],
        )
        for mode_name in MODE_NAMES
    }
    common_layer = max(
        layer_results,
        key=lambda layer: np.mean(
            [
                layer_results[layer]["metrics"]["valid"][mode_name]["foreground_macro_f1"]
                for mode_name in MODE_NAMES
            ]
        ),
    )
    selection_summary = {
        mode_name: {
            "layer": selected_layers[mode_name],
            "validation": layer_results[selected_layers[mode_name]]["metrics"]["valid"][mode_name],
            "test": layer_results[selected_layers[mode_name]]["metrics"]["test"][mode_name],
        }
        for mode_name in MODE_NAMES
    }
    final_result = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_result["validation_selected_layers"] = selection_summary
    final_result["common_validation_selected_layer"] = {
        "layer": common_layer,
        "validation": layer_results[common_layer]["metrics"]["valid"],
        "test": layer_results[common_layer]["metrics"]["test"],
    }
    final_result["elapsed_seconds_latest_run"] = time.perf_counter() - started_at
    final_result["max_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
    metrics_path.write_text(json.dumps(final_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "readout": args.readout,
        "validation_selected_layers": selected_layers,
        "common_validation_selected_layer": common_layer,
        "metrics_path": str(metrics_path),
        "elapsed_seconds": final_result["elapsed_seconds_latest_run"],
        "max_cuda_memory_gib": final_result["max_cuda_memory_gib"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
