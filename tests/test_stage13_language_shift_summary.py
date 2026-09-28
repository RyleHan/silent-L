import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_pi05_stage13_language_shift.py"
SPEC = importlib.util.spec_from_file_location("stage13_language_shift_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_spearman_handles_ties_and_sign() -> None:
    x = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert MODULE.spearman(x, x) == pytest.approx(1.0)
    assert MODULE.spearman(x, -x) == pytest.approx(-1.0)
    assert MODULE.spearman(np.asarray([1.0, 2.0, 2.0, 3.0]), x) == pytest.approx(0.9486833, abs=1e-6)


def test_exact_permutation_p_is_one_sided() -> None:
    x = np.arange(6, dtype=float)
    assert MODULE.exact_permutation_p(x, x) == pytest.approx(1 / 720)
    assert MODULE.exact_permutation_p(x, -x) == pytest.approx(1.0)


def test_auc_counts_ties_as_half() -> None:
    scores = np.asarray([0.1, 0.5, 0.5, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert MODULE.auc(scores, labels) == pytest.approx(0.875)
    assert np.isnan(MODULE.auc(scores, np.zeros(4, dtype=int)))
