"""Stored target-XYZ atlas probes and the two-object readout used from Stage 12 on."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StoredProbe:
    """Target-XYZ readout of one stored atlas layer; weights are loaded, never refit."""

    def __init__(self, path: Path):
        # Atlas layer files are local artifacts written by train_openvla_world_state_atlas.py.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        start, stop = payload["factor_slices"]["target_xyz"]
        continuous = payload["states"]["continuous"]
        self.path = path
        self.sha256 = sha256(path)
        self.layer = int(payload["layer"])
        self.readout = str(payload["readout"])
        self.weight = continuous["weight"].float().numpy()[start:stop]
        self.bias = continuous["bias"].float().numpy()[start:stop]
        self.feature_mean = payload["feature_mean"].float().numpy()
        self.feature_std = payload["feature_std"].float().numpy()
        self.target_mean = np.asarray(payload["continuous_mean"], dtype=np.float32)[start:stop]
        self.target_std = np.asarray(payload["continuous_std"], dtype=np.float32)[start:stop]

    def decode(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_std
        return (standardized @ self.weight.T + self.bias) * self.target_std + self.target_mean


class ProbeFamily:
    """Seed ensemble whose decoded XYZ is the mean over seed probes."""

    def __init__(self, name: str, readout_key: str, layer: int, paths: list[Path]):
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing probe files for {name}: {missing}")
        self.name = name
        self.readout_key = readout_key
        self.layer = layer
        self.probes = [StoredProbe(path) for path in paths]
        if any(probe.layer != layer for probe in self.probes):
            raise ValueError(f"{name}: stored layers {[p.layer for p in self.probes]} != {layer}.")

    def decode(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        per_seed = np.stack([probe.decode(features) for probe in self.probes])
        return per_seed.mean(axis=0), per_seed

    def provenance(self) -> list[dict[str, object]]:
        return [
            {"path": str(probe.path), "sha256": probe.sha256, "layer": probe.layer}
            for probe in self.probes
        ]


def assign(decoded: np.ndarray, object_a: np.ndarray, object_b: np.ndarray, margin: float) -> dict:
    """Nearest-object assignment with an indeterminate margin, plus the A=0 / B=1 projection."""
    distance_a = float(np.linalg.norm(decoded - object_a))
    distance_b = float(np.linalg.norm(decoded - object_b))
    axis = object_b - object_a
    interpolation = float(np.dot(decoded - object_a, axis) / max(float(np.dot(axis, axis)), 1e-12))
    if abs(distance_a - distance_b) < margin:
        side = None
    else:
        side = 0 if distance_a < distance_b else 1
    return {
        "side": side,
        "distance_a": distance_a,
        "distance_b": distance_b,
        "interpolation_a0_b1": interpolation,
    }
