#!/usr/bin/env python3
"""Download a public GCS prefix with resumable files and checksum verification."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def list_objects(bucket: str, prefix: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    page_token = None
    while True:
        query = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken"}
        if page_token:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o?"
            + urllib.parse.urlencode(query)
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    if not items:
        raise FileNotFoundError(f"No public GCS objects found at gs://{bucket}/{prefix}")
    return items


def md5_base64(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - GCS supplies MD5 for integrity, not security.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def download_object(
    *,
    bucket: str,
    item: dict[str, object],
    prefix: str,
    staging: Path,
    retries: int,
) -> tuple[str, int]:
    name = str(item["name"])
    expected_size = int(item["size"])
    expected_md5 = str(item.get("md5Hash", ""))
    relative = name.removeprefix(prefix)
    if not relative or relative == name:
        raise ValueError(f"Object {name!r} is not below prefix {prefix!r}.")
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.exists():
        if destination.stat().st_size == expected_size and (
            not expected_md5 or md5_base64(destination) == expected_md5
        ):
            return relative, expected_size
        raise RuntimeError(f"Existing staged object failed verification: {destination}")

    object_url = (
        f"https://storage.googleapis.com/{urllib.parse.quote(bucket, safe='')}/"
        f"{urllib.parse.quote(name, safe='/')}"
    )
    for attempt in range(1, retries + 1):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            if existing > expected_size:
                with partial.open("wb"):
                    pass
                existing = 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            request = urllib.request.Request(object_url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = existing > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise IOError(
                    f"Size mismatch for {relative}: {partial.stat().st_size} != {expected_size}"
                )
            if expected_md5 and md5_base64(partial) != expected_md5:
                with partial.open("wb"):
                    pass
                raise IOError(f"MD5 mismatch for {relative}")
            os.replace(partial, destination)
            return relative, expected_size
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    prefix = args.prefix.strip("/") + "/"
    objects = list_objects(args.bucket, prefix)
    expected = {str(item["name"]).removeprefix(prefix): int(item["size"]) for item in objects}
    expected_bytes = sum(expected.values())

    if args.output_dir.exists():
        actual = {
            str(path.relative_to(args.output_dir)): path.stat().st_size
            for path in args.output_dir.rglob("*")
            if path.is_file()
        }
        if actual == expected:
            print(f"Already complete: {args.output_dir} ({expected_bytes} bytes)", flush=True)
            return
        raise RuntimeError(f"Output exists but does not match public GCS listing: {args.output_dir}")

    staging = args.output_dir.with_name(args.output_dir.name + ".download")
    staging.mkdir(parents=True, exist_ok=True)
    completed_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_object,
                bucket=args.bucket,
                item=item,
                prefix=prefix,
                staging=staging,
                retries=args.retries,
            )
            for item in objects
        ]
        for future in concurrent.futures.as_completed(futures):
            relative, size = future.result()
            completed_bytes += size
            print(
                f"verified {relative}: {size} bytes "
                f"({completed_bytes / expected_bytes:.1%} of checkpoint)",
                flush=True,
            )

    actual = {
        str(path.relative_to(staging)): path.stat().st_size
        for path in staging.rglob("*")
        if path.is_file() and not path.name.endswith(".part")
    }
    if actual != expected:
        raise RuntimeError("Staged checkpoint does not exactly match the public GCS listing.")
    staging.rename(args.output_dir)
    print(f"Downloaded and verified: {args.output_dir} ({expected_bytes} bytes)", flush=True)


if __name__ == "__main__":
    main()

