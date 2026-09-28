import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_pi05_instruction_compliance.py"
SPEC = importlib.util.spec_from_file_location("instruction_compliance_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_wilson_interval_contains_observed_rate() -> None:
    estimate = MODULE.wilson_interval(40, 50)
    assert estimate["mean"] == 0.8
    assert estimate["ci95_low"] < 0.8 < estimate["ci95_high"]
    assert 0.0 <= estimate["ci95_low"] <= estimate["ci95_high"] <= 1.0


def test_paired_bootstrap_reports_paired_mean() -> None:
    differences = np.asarray([1.0, 0.0, -1.0, 1.0])
    estimate = MODULE.paired_bootstrap(
        differences, np.random.default_rng(7), samples=2000
    )
    assert estimate["mean"] == 0.25
    assert estimate["ci95_low"] <= 0.25 <= estimate["ci95_high"]


def test_outcome_key_covers_all_recorded_codes() -> None:
    assert MODULE.outcome_key({"first_grasp_side": 0}) == "a"
    assert MODULE.outcome_key({"first_grasp_side": 1}) == "b"
    assert MODULE.outcome_key({"first_grasp_side": 2}) == "both"
    assert MODULE.outcome_key({"first_grasp_side": -1}) == "none"


def test_pair_ids_argument_accepts_noncontiguous_subset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize",
            "--run-root",
            "/tmp/run",
            "--output-dir",
            "/tmp/output",
            "--pair-ids",
            "0",
            "3",
        ],
    )
    args = MODULE.parse_args()
    assert args.pair_ids == [0, 3]
