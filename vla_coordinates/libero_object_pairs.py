"""Pair-aware metadata helpers for the multi-pair LIBERO-Object protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_OBJECTS = ("alphabet_soup_1", "cream_cheese_1")
LEGACY_NAMES = ("alphabet soup", "cream cheese")


def plain_prompt(object_name: str) -> str:
    return f"pick up the {object_name} and place it in the basket"


def openvla_prompt(object_name: str) -> str:
    return f"In: What action should the robot take to {plain_prompt(object_name)}?\nOut:"


def _normalize_pair(pair: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(pair)
    normalized["pair_id"] = int(normalized["pair_id"])
    normalized["task_ids"] = [int(value) for value in normalized["task_ids"]]
    normalized["object_instances"] = [str(value) for value in normalized["object_instances"]]
    normalized["object_names"] = [str(value) for value in normalized["object_names"]]
    normalized["prompts_plain"] = [plain_prompt(value) for value in normalized["object_names"]]
    normalized["prompts_openvla"] = [openvla_prompt(value) for value in normalized["object_names"]]
    return normalized


def validate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [_normalize_pair(pair) for pair in pairs]
    pair_ids = [pair["pair_id"] for pair in normalized]
    if pair_ids != list(range(len(normalized))):
        raise ValueError(f"pair_id values must be contiguous from zero, got {pair_ids}.")
    task_ids: list[int] = []
    for pair in normalized:
        for key in ("task_ids", "object_instances", "object_names"):
            if len(pair[key]) != 2:
                raise ValueError(f"Pair {pair['pair_id']} field {key} must contain two values.")
        if pair["task_ids"][0] == pair["task_ids"][1]:
            raise ValueError(f"Pair {pair['pair_id']} repeats one task ID.")
        if pair["object_instances"][0] == pair["object_instances"][1]:
            raise ValueError(f"Pair {pair['pair_id']} repeats one object instance.")
        task_ids.extend(pair["task_ids"])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("A task may belong to only one pair.")
    return normalized


def load_pair_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["pairs"] = validate_pairs(config["pairs"])
    return config


def legacy_pairs(source: Any | None = None) -> list[dict[str, Any]]:
    object_instances = list(LEGACY_OBJECTS)
    prompts_openvla = [openvla_prompt(value) for value in LEGACY_NAMES]
    if source is not None:
        object_instances = [
            _decode(source.attrs.get("object_a_instance", object_instances[0])),
            _decode(source.attrs.get("object_b_instance", object_instances[1])),
        ]
        prompts_openvla = [
            _decode(source.attrs.get("prompt_a", prompts_openvla[0])),
            _decode(source.attrs.get("prompt_b", prompts_openvla[1])),
        ]
    pair = _normalize_pair(
        {
            "pair_id": 0,
            "name": "alphabet_soup_vs_cream_cheese",
            "task_ids": [0, 1],
            "object_instances": object_instances,
            "object_names": list(LEGACY_NAMES),
        }
    )
    pair["prompts_openvla"] = prompts_openvla
    return [pair]


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_pairs(source: Any) -> list[dict[str, Any]]:
    if "pair_metadata_json" not in source.attrs:
        return legacy_pairs(source)
    return validate_pairs(json.loads(_decode(source.attrs["pair_metadata_json"])))


def read_pair_ids(source: Any, count: int | None = None) -> np.ndarray:
    size = int(count if count is not None else source["task_ids"].shape[0])
    if "pair_ids" in source:
        return np.asarray(source["pair_ids"][:size], dtype=np.int16)
    return np.zeros(size, dtype=np.int16)


def read_target_sides(source: Any, count: int | None = None) -> np.ndarray:
    size = int(count if count is not None else source["task_ids"].shape[0])
    if "target_sides" in source:
        return np.asarray(source["target_sides"][:size], dtype=np.int8)
    task_ids = np.asarray(source["task_ids"][:size], dtype=np.int16)
    if not np.all(np.isin(task_ids, (0, 1))):
        raise ValueError("Multi-pair data must include target_sides.")
    return task_ids.astype(np.int8)


def task_lookup(pairs: list[dict[str, Any]]) -> dict[int, tuple[dict[str, Any], int]]:
    return {
        int(task_id): (pair, side)
        for pair in pairs
        for side, task_id in enumerate(pair["task_ids"])
    }


def prompts_for_pair(pair: dict[str, Any], kind: str) -> list[str]:
    key = {"plain": "prompts_plain", "openvla": "prompts_openvla"}[kind]
    return [str(value) for value in pair[key]]


def prompts_for_sample(
    source: Any,
    pairs_by_id: dict[int, dict[str, Any]],
    pair_ids: np.ndarray,
    index: int,
    *,
    kind: str,
) -> list[str]:
    pair_id = int(pair_ids[index])
    if pair_id not in pairs_by_id:
        raise KeyError(f"Sample {index} refers to unknown pair_id={pair_id}.")
    return prompts_for_pair(pairs_by_id[pair_id], kind)


def copy_pair_metadata(source: Any, output: Any, count: int) -> None:
    pairs = read_pairs(source)
    output.create_dataset("pair_ids", data=read_pair_ids(source, count))
    output.create_dataset("target_sides", data=read_target_sides(source, count))
    output.attrs["pair_metadata_json"] = json.dumps(pairs)
    output.attrs["pair_schema_version"] = 2 if "pair_metadata_json" in source.attrs else 1
