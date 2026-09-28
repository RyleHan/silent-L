#!/usr/bin/env python3
"""Download and inventory one pinned Hugging Face model snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE")
    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=cache_dir,
            max_workers=args.max_workers,
        )
    )
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(snapshot)), "size_bytes": path.stat().st_size})
    result = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot": str(snapshot),
        "num_files": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
