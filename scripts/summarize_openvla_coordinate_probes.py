#!/usr/bin/env python3
"""Summarize layerwise OpenVLA coordinate probes without test-set selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


READOUTS = ("prompt_end", "action_boundary", "visual_mean")
MODES = ("absolute", "target_alternative")
METRIC = "foreground_macro_f1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "selection_rule": "Select layers using validation data only; report held-out test at that fixed layer.",
        "metric": METRIC,
        "readouts": {},
    }
    rows: list[dict[str, object]] = []

    for readout in READOUTS:
        metrics_path = args.probe_root / readout / "metrics.json"
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        layers = sorted(result["layers"], key=int)
        curves: dict[str, dict[str, list[float]]] = {
            split: {mode: [] for mode in MODES} for split in ("valid", "test")
        }
        for layer in layers:
            for split in ("valid", "test"):
                for mode in MODES:
                    value = result["layers"][layer]["metrics"][split][mode][METRIC]
                    curves[split][mode].append(value)
            rows.append(
                {
                    "readout": readout,
                    "layer": int(layer),
                    "valid_absolute": curves["valid"]["absolute"][-1],
                    "valid_target_alternative": curves["valid"]["target_alternative"][-1],
                    "valid_delta": curves["valid"]["target_alternative"][-1] - curves["valid"]["absolute"][-1],
                    "test_absolute": curves["test"]["absolute"][-1],
                    "test_target_alternative": curves["test"]["target_alternative"][-1],
                    "test_delta": curves["test"]["target_alternative"][-1] - curves["test"]["absolute"][-1],
                }
            )

        validation_delta = np.asarray(curves["valid"]["target_alternative"]) - np.asarray(
            curves["valid"]["absolute"]
        )
        selected_index = int(validation_delta.argmax())
        selected_layer = layers[selected_index]
        summary["readouts"][readout] = {
            "validation_delta_selected_layer": int(selected_layer),
            "validation_at_selected_layer": {
                mode: curves["valid"][mode][selected_index] for mode in MODES
            }
            | {"delta": float(validation_delta[selected_index])},
            "test_at_selected_layer": {
                mode: curves["test"][mode][selected_index] for mode in MODES
            }
            | {
                "delta": curves["test"]["target_alternative"][selected_index]
                - curves["test"]["absolute"][selected_index]
            },
            "coordinate_specific_validation_selected_layers": result["validation_selected_layers"],
            "common_validation_selected_layer": result["common_validation_selected_layer"],
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "layer_curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote: {summary_path}", flush=True)
    print(f"Wrote: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
