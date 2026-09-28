import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_pi05_stage12_probe_read.py"
SPEC = importlib.util.spec_from_file_location("stage12_probe_read_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def decision_args() -> argparse.Namespace:
    return argparse.Namespace(dissociation_threshold=0.70, tracking_threshold=0.30)


def read(prompt_side: int, grasp_side: int, probe_side: int | None) -> dict:
    return {
        "prompt_side": prompt_side,
        "first_grasp_side": grasp_side,
        "families": {"probe": {"side": probe_side, "per_seed_sides": [probe_side]}},
    }


def test_wilson_interval_matches_reference_values() -> None:
    low, high = MODULE.wilson_interval(40, 50)
    assert low == pytest.approx(0.6696, abs=1e-4)
    assert high == pytest.approx(0.8876, abs=1e-4)
    assert MODULE.wilson_interval(0, 10)[0] == 0.0
    assert MODULE.wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_assign_uses_indeterminate_margin() -> None:
    object_a = np.zeros(3, dtype=np.float32)
    object_b = np.asarray([0.2, 0.0, 0.0], dtype=np.float32)
    near_a = MODULE.assign(np.asarray([0.02, 0.0, 0.0]), object_a, object_b, 0.02)
    assert near_a["side"] == 0
    assert near_a["interpolation_a0_b1"] == pytest.approx(0.1)
    midpoint = MODULE.assign(np.asarray([0.105, 0.0, 0.0]), object_a, object_b, 0.02)
    assert midpoint["side"] is None


def test_dissociation_counts_only_wrong_object_rollouts() -> None:
    reads = [read(1, 0, 1)] * 38 + [read(1, 0, 0)] * 2 + [read(1, 1, 0)] * 5 + [read(1, -1, 0)] * 5
    result = MODULE.dissociation(reads, "probe", 1, decision_args())
    assert result["wrong_object_rollouts"] == 40
    assert result["dissociation_rate"] == pytest.approx(0.95)
    assert result["decision"] == "dissociation"


def test_dissociation_reports_tracking_and_inconclusive() -> None:
    tracking = [read(1, 0, 0)] * 38 + [read(1, 0, 1)] * 2
    assert MODULE.dissociation(tracking, "probe", 1, decision_args())["decision"] == "probe_tracks_behavior"
    mixed = [read(1, 0, 0)] * 20 + [read(1, 0, 1)] * 20
    assert MODULE.dissociation(mixed, "probe", 1, decision_args())["decision"] == "inconclusive"


def test_validity_floor_requires_mostly_determinate_reads() -> None:
    reads = [read(0, 0, 0)] * 4 + [read(0, 0, None)] * 6
    result = MODULE.validity_floor(reads, "probe", 0, 0.90)
    assert result["rate"] == 1.0
    assert result["passed"] is False
