#!/usr/bin/env python3
"""Download the public COAST checkpoint parameters and one activation task."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import threading

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-repo", default="brandonyang/openpi-libero-2000")
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument(
        "--activation-repo",
        default="brandonyang/pi05-libero-activations-v1-2000-15env",
    )
    parser.add_argument("--activation-output", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def list_files(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
) -> list[str]:
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        recursive=True,
        expand=False,
    )
    return sorted(
        entry.path for entry in entries if entry.__class__.__name__ == "RepoFile"
    )


def download_files(
    *,
    repo_id: str,
    repo_type: str,
    filenames: list[str],
    output: Path,
    max_workers: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    lock = threading.Lock()

    def download(filename: str) -> None:
        nonlocal completed
        if not (output / filename).is_file():
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=output,
            )
        with lock:
            completed += 1
            if completed == len(filenames) or completed % 50 == 0:
                print(f"{repo_id}: downloaded {completed}/{len(filenames)} files", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(download, filenames))


def main() -> None:
    args = parse_args()
    api = HfApi()
    checkpoint_files = ["_CHECKPOINT_METADATA"]
    checkpoint_files.extend(
        list_files(
            api,
            repo_id=args.checkpoint_repo,
            repo_type="model",
            path_in_repo="params",
        )
    )
    checkpoint_files.extend(
        list_files(
            api,
            repo_id=args.checkpoint_repo,
            repo_type="model",
            path_in_repo="assets",
        )
    )
    checkpoint_files = sorted(set(checkpoint_files))
    download_files(
        repo_id=args.checkpoint_repo,
        repo_type="model",
        filenames=checkpoint_files,
        output=args.checkpoint_output,
        max_workers=args.max_workers,
    )

    task_root = f"openpi-libero-2000/{args.task_name}"
    task_files = list_files(
        api,
        repo_id=args.activation_repo,
        repo_type="dataset",
        path_in_repo=task_root,
    )
    activation_files = []
    for filename in task_files:
        relative = filename.removeprefix(f"{task_root}/")
        is_episode_metadata = (
            relative.count("/") == 1 and relative.endswith("/metadata.json")
        )
        if is_episode_metadata or filename.endswith("/suffix_residual.npz"):
            activation_files.append(filename)
    download_files(
        repo_id=args.activation_repo,
        repo_type="dataset",
        filenames=activation_files,
        output=args.activation_output,
        max_workers=args.max_workers,
    )
    episode_metadata = sorted(
        (args.activation_output / task_root).glob("episode_*/metadata.json")
    )
    manifest = {
        "checkpoint_repo": args.checkpoint_repo,
        "checkpoint_path": str(args.checkpoint_output),
        "checkpoint_has_params": (args.checkpoint_output / "params").is_dir(),
        "checkpoint_file_count": len(checkpoint_files),
        "activation_repo": args.activation_repo,
        "activation_path": str(args.activation_output),
        "task_name": args.task_name,
        "activation_file_count": len(activation_files),
        "num_episodes": len(episode_metadata),
        "num_suffix_residual_files": len(
            list(
                (args.activation_output / task_root).glob(
                    "episode_*/step_*/suffix_residual.npz"
                )
            )
        ),
    }
    args.activation_output.mkdir(parents=True, exist_ok=True)
    (args.activation_output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
