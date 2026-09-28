from __future__ import annotations

import unittest

import numpy as np

from vla_coordinates.conceptors import (
    conceptor_diagnostics,
    contrastive_conceptor,
    fit_conceptor,
    random_spectrum_control,
)


class ConceptorTests(unittest.TestCase):
    def test_fit_is_symmetric_psd_contraction(self) -> None:
        rng = np.random.default_rng(3)
        activations = rng.normal(size=(80, 12)) @ np.diag(np.linspace(0.1, 2.0, 12))
        conceptor = fit_conceptor(activations, aperture=0.5)
        diagnostics = conceptor_diagnostics(conceptor)
        self.assertEqual(conceptor.shape, (12, 12))
        self.assertLess(diagnostics["symmetry_max_abs_error"], 1e-6)
        self.assertGreaterEqual(diagnostics["min_eigenvalue"], -1e-6)
        self.assertLessEqual(diagnostics["max_eigenvalue"], 1.0 + 1e-6)

    def test_contrastive_and_random_control_preserve_valid_spectrum(self) -> None:
        rng = np.random.default_rng(7)
        success = fit_conceptor(rng.normal(size=(100, 10)), aperture=1.0)
        failure = fit_conceptor(rng.normal(size=(100, 10)) * 0.5, aperture=1.0)
        contrastive = contrastive_conceptor(success, failure)
        random = random_spectrum_control(contrastive, seed=9)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(contrastive),
            np.linalg.eigvalsh(random),
            atol=1e-5,
        )
        for matrix in (contrastive, random):
            eigenvalues = np.linalg.eigvalsh(matrix)
            self.assertGreaterEqual(float(eigenvalues.min()), -1e-6)
            self.assertLessEqual(float(eigenvalues.max()), 1.0 + 1e-6)

    def test_invalid_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            fit_conceptor(np.ones((1, 3)), aperture=1.0)
        with self.assertRaises(ValueError):
            fit_conceptor(np.ones((2, 3)), aperture=0.0)


if __name__ == "__main__":
    unittest.main()
