#!/usr/bin/env python3
"""Merge deterministic per-task trajectory shards into the v2 paired dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default="0,1,2,4,3,8,5,6")
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def create_like(output: h5py.File, name: str, source: h5py.Dataset, total: int) -> None:
    shape = (total, *source.shape[1:])
    kwargs: dict[str, object] = {"shape": shape, "dtype": source.dtype}
    if source.chunks is not None:
        kwargs["chunks"] = (min(source.chunks[0], total), *source.chunks[1:])
    if source.compression is not None:
        kwargs["compression"] = source.compression
        if source.compression_opts is not None:
            kwargs["compression_opts"] = source.compression_opts
    output.create_dataset(name, **kwargs)


def main() -> None:
    args = parse_args()
    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    shard_paths = [args.shard_root / f"task-{task_id}/paired_trajectories.h5" for task_id in task_ids]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectory shards: {missing}")
    output_h5 = args.output_dir / "paired_trajectories.h5"
    output_metadata = args.output_dir / "metadata.json"
    if output_h5.exists() or output_metadata.exists():
        raise FileExistsError(f"Refusing to replace existing v2 output under {args.output_dir}.")

    lengths = []
    metadata_records: list[dict[str, object]] = []
    reference_attrs: dict[str, object] | None = None
    dataset_names: list[str] | None = None
    for task_id, path in zip(task_ids, shard_paths):
        with h5py.File(path, "r") as shard:
            shard_tasks = np.unique(shard["task_ids"][:]).astype(int).tolist()
            if shard_tasks != [task_id]:
                raise ValueError(f"Shard {path} contains task IDs {shard_tasks}, expected {[task_id]}.")
            if not bool(shard.attrs.get("complete", False)):
                raise ValueError(f"Shard is incomplete: {path}")
            lengths.append(int(shard["images"].shape[0]))
            current_names = sorted(shard.keys())
            if dataset_names is None:
                dataset_names = current_names
                reference_attrs = dict(shard.attrs)
            elif current_names != dataset_names:
                raise ValueError(f"Dataset schema mismatch in {path}.")
        metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_records.extend(metadata["records"])

    assert dataset_names is not None and reference_attrs is not None
    total = int(sum(lengths))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(shard_paths[0], "r") as first, h5py.File(output_h5, "w") as output:
        for name in dataset_names:
            create_like(output, name, first[name], total)
        for key, value in reference_attrs.items():
            output.attrs[key] = value
        output.attrs["complete"] = False
        output.attrs["num_samples"] = total
        output.attrs["merged_task_ids_json"] = json.dumps(task_ids)

        offset = 0
        for path, length in zip(shard_paths, lengths):
            with h5py.File(path, "r") as shard:
                for name in dataset_names:
                    output[name][offset : offset + length] = shard[name][:]
            offset += length
        output.attrs["complete"] = True

    with h5py.File(output_h5, "r") as output:
        merged_task_ids = np.asarray(output["task_ids"], dtype=np.int16)
        pair_ids = np.asarray(output["pair_ids"], dtype=np.int16)
        target_sides = np.asarray(output["target_sides"], dtype=np.int8)
        episode_ids = np.asarray(output["episode_ids"], dtype=np.int16)
        splits = decode_strings(output["layout_splits"][:])
        episode_keys = set(zip(merged_task_ids.tolist(), episode_ids.tolist()))
        expected_episodes = sum(
            1 for record in metadata_records if int(record["num_samples"]) > 0
        )
        if len(episode_keys) != expected_episodes:
            raise ValueError(
                f"Merged episode count {len(episode_keys)} != metadata count {expected_episodes}."
            )
        audit = {
            "num_samples": total,
            "num_episodes": len(episode_keys),
            "num_successes": int(sum(bool(record["success"]) for record in metadata_records)),
            "task_sample_counts": {
                str(task_id): int((merged_task_ids == task_id).sum()) for task_id in task_ids
            },
            "pair_sample_counts": {
                str(pair_id): int((pair_ids == pair_id).sum()) for pair_id in np.unique(pair_ids)
            },
            "target_side_sample_counts": {
                str(side): int((target_sides == side).sum()) for side in (0, 1)
            },
            "split_sample_counts": dict(Counter(splits.tolist())),
            "split_episode_counts": {
                split: len(
                    set(
                        zip(
                            merged_task_ids[splits == split].tolist(),
                            episode_ids[splits == split].tolist(),
                        )
                    )
                )
                for split in ("train", "valid", "test")
            },
        }
    output_metadata.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "shards": [str(path) for path in shard_paths],
                "audit": audit,
                "records": metadata_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Wrote: {output_h5}", flush=True)


if __name__ == "__main__":
    main()
