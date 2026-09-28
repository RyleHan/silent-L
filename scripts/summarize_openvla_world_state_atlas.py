#!/usr/bin/env python3
"""Summarize actual-task and paired-prompt OpenVLA atlas results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


READOUTS = ("prompt_end", "action_boundary", "visual_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_results(root: Path, suffix: str) -> dict[str, dict[str, object]]:
    return {
        readout: json.loads((root / f"{readout}{suffix}").read_text(encoding="utf-8"))
        for readout in READOUTS
    }


def selected_rows(protocol: str, results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for readout, result in results.items():
        selected = result["selected"]
        for factor, item in selected["regression"].items():
            rows.append(
                {
                    "protocol": protocol,
                    "readout": readout,
                    "family": "regression",
                    "factor": factor,
                    "layer": item["layer"],
                    "metric": "mean_r2",
                    "validation": item["validation"]["mean_r2"],
                    "test": item["test"]["mean_r2"],
                }
            )
        for factor, item in selected["binary"].items():
            rows.append(
                {
                    "protocol": protocol,
                    "readout": readout,
                    "family": "binary",
                    "factor": factor,
                    "layer": item["layer"],
                    "metric": "auc",
                    "validation": item["validation"]["auc"],
                    "test": item["test"]["auc"],
                }
            )
        phase = selected["phase_coarse"]
        rows.append(
            {
                "protocol": protocol,
                "readout": readout,
                "family": "phase",
                "factor": "phase_coarse",
                "layer": phase["layer"],
                "metric": "macro_f1",
                "validation": phase["validation"]["macro_f1"],
                "test": phase["test"]["macro_f1"],
            }
        )
    return rows


def layer_curve_rows(
    protocol: str, results: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for readout, result in results.items():
        for layer_result in result["layers"]:
            layer = layer_result["layer"]
            for split in ("valid", "test"):
                split_result = layer_result["splits"][split]
                for factor, metrics in split_result["regression"].items():
                    rows.append(
                        {
                            "protocol": protocol,
                            "readout": readout,
                            "layer": layer,
                            "split": split,
                            "family": "regression",
                            "factor": factor,
                            "metric": "mean_r2",
                            "value": metrics["mean_r2"],
                        }
                    )
                for factor, metrics in split_result["binary"].items():
                    rows.append(
                        {
                            "protocol": protocol,
                            "readout": readout,
                            "layer": layer,
                            "split": split,
                            "family": "binary",
                            "factor": factor,
                            "metric": "auc",
                            "value": metrics["auc"],
                        }
                    )
                rows.append(
                    {
                        "protocol": protocol,
                        "readout": readout,
                        "layer": layer,
                        "split": split,
                        "family": "phase",
                        "factor": "phase_coarse",
                        "metric": "macro_f1",
                        "value": split_result["phase_coarse"]["macro_f1"],
                    }
                )
    return rows


def onset_layers(results: dict[str, dict[str, object]]) -> dict[str, object]:
    profiles = {}
    for readout, result in results.items():
        profiles[readout] = {"regression": {}, "binary": {}}
        for family, metric in (("regression", "mean_r2"), ("binary", "auc")):
            for factor in result["selected"][family]:
                values = [
                    (
                        layer_result["layer"],
                        layer_result["splits"]["valid"][family][factor][metric],
                    )
                    for layer_result in result["layers"]
                ]
                peak = max(value for _, value in values)
                threshold = 0.9 * peak
                first = next(layer for layer, value in values if value >= threshold)
                profiles[readout][family][factor] = {
                    "first_layer_at_90_percent_validation_peak": first,
                    "validation_peak": peak,
                    "threshold": threshold,
                }
        phase_values = [
            (
                layer_result["layer"],
                layer_result["splits"]["valid"]["phase_coarse"]["macro_f1"],
            )
            for layer_result in result["layers"]
        ]
        phase_peak = max(value for _, value in phase_values)
        profiles[readout]["phase_coarse"] = {
            "first_layer_at_90_percent_validation_peak": next(
                layer for layer, value in phase_values if value >= 0.9 * phase_peak
            ),
            "validation_peak": phase_peak,
            "threshold": 0.9 * phase_peak,
        }
    return profiles


def metric_value(result: dict[str, object], family: str, factor: str) -> float:
    item = result["selected"][family][factor]
    if family == "regression":
        return float(item["test"]["mean_r2"])
    if family == "binary":
        return float(item["test"]["auc"])
    raise ValueError(f"Unsupported family: {family}")


def main() -> None:
    args = parse_args()
    actual = load_results(args.actual_root, "_actual_summary.json")
    paired = load_results(args.paired_root, "_summary.json")
    rows = selected_rows("actual", actual) + selected_rows("paired", paired)
    curve_rows = layer_curve_rows("actual", actual) + layer_curve_rows("paired", paired)

    contrast_factors = (
        "target_xyz",
        "alternative_xyz",
        "target_to_eef_xyz",
        "alternative_to_eef_xyz",
        "target_to_basket_xyz",
        "eef_target_distance",
        "target_basket_distance",
        "target_displacement_xyz",
    )
    contrasts = {}
    for readout in ("prompt_end", "action_boundary"):
        contrasts[readout] = {
            factor: {
                "prompt_aware_test_r2": metric_value(paired[readout], "regression", factor),
                "visual_control_test_r2": metric_value(
                    paired["visual_mean"], "regression", factor
                ),
                "difference": metric_value(paired[readout], "regression", factor)
                - metric_value(paired["visual_mean"], "regression", factor),
            }
            for factor in contrast_factors
        }

    phase_contrasts = {}
    visual_phase = paired["visual_mean"]["selected"]["phase_coarse"]["test"]["macro_f1"]
    for readout in ("prompt_end", "action_boundary"):
        prompt_phase = paired[readout]["selected"]["phase_coarse"]["test"]["macro_f1"]
        phase_contrasts[readout] = {
            "prompt_aware_test_macro_f1": prompt_phase,
            "visual_control_test_macro_f1": visual_phase,
            "difference": prompt_phase - visual_phase,
        }

    summary = {
        "protocols": {
            "actual": "one residual per image under the rollout instruction",
            "paired": (
                "same image under prompt A and B; absolute labels repeat and "
                "target-relative labels swap within each pair"
            ),
        },
        "paired_prompt_minus_visual": contrasts,
        "paired_phase_minus_visual": phase_contrasts,
        "paired_onset_layers": onset_layers(paired),
        "headline": {
            "prompt_end_target_xyz_test_r2": metric_value(
                paired["prompt_end"], "regression", "target_xyz"
            ),
            "visual_target_xyz_test_r2": metric_value(
                paired["visual_mean"], "regression", "target_xyz"
            ),
            "prompt_end_target_to_eef_test_r2": metric_value(
                paired["prompt_end"], "regression", "target_to_eef_xyz"
            ),
            "visual_target_to_eef_test_r2": metric_value(
                paired["visual_mean"], "regression", "target_to_eef_xyz"
            ),
            "prompt_end_phase_test_macro_f1": paired["prompt_end"]["selected"][
                "phase_coarse"
            ]["test"]["macro_f1"],
            "visual_phase_test_macro_f1": visual_phase,
        },
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "atlas_summary.json"
    csv_path = args.output_dir / "atlas_selected_metrics.csv"
    curves_path = args.output_dir / "atlas_layer_curves.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(json.dumps(summary["headline"], indent=2))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {curves_path}")


if __name__ == "__main__":
    main()
