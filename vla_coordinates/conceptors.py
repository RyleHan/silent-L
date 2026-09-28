"""Closed-form conceptor operators for activation steering."""

from __future__ import annotations

import numpy as np


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def fit_conceptor(activations: np.ndarray, aperture: float) -> np.ndarray:
    """Fit C = R(R + aperture^-2 I)^-1 to mean-centered activations."""
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Activations must have shape [samples, hidden] with >=2 samples.")
    if not np.isfinite(values).all():
        raise ValueError("Activations contain non-finite values.")
    if not np.isfinite(aperture) or aperture <= 0:
        raise ValueError("Aperture must be finite and positive.")

    centered = values - values.mean(axis=0, keepdims=True)
    covariance = _symmetric(centered.T @ centered / len(centered))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    weights = eigenvalues / (eigenvalues + aperture**-2)
    return _symmetric((eigenvectors * weights) @ eigenvectors.T).astype(np.float32)


def contrastive_conceptor(success: np.ndarray, failure: np.ndarray) -> np.ndarray:
    """Return the canonical Boolean conceptor ``success AND NOT failure``."""
    success = _symmetric(success)
    failure = _symmetric(failure)
    if success.shape != failure.shape or success.ndim != 2:
        raise ValueError("Success and failure conceptors must be same-shaped matrices.")
    identity = np.eye(success.shape[0], dtype=np.float64)
    not_failure = identity - failure
    combined = np.linalg.pinv(
        np.linalg.pinv(success, hermitian=True)
        + np.linalg.pinv(not_failure, hermitian=True)
        - identity,
        hermitian=True,
    )
    # COAST keeps the full canonical AND-NOT result; it does not truncate its spectrum.
    return _symmetric(combined).astype(np.float32)


def random_spectrum_control(conceptor: np.ndarray, seed: int) -> np.ndarray:
    """Randomize eigenvectors while preserving a conceptor's eigenvalue spectrum."""
    matrix = _symmetric(conceptor)
    eigenvalues = np.linalg.eigvalsh(matrix)
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal(matrix.shape)
    orthogonal, triangular = np.linalg.qr(random_matrix)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    orthogonal *= signs
    randomized = (orthogonal * eigenvalues) @ orthogonal.T
    return _symmetric(randomized).astype(np.float32)


def conceptor_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    numerator = np.trace(first @ second)
    denominator = np.sqrt(np.trace(first @ first) * np.trace(second @ second))
    return float(numerator / max(float(denominator), np.finfo(np.float64).eps))


def conceptor_diagnostics(conceptor: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(conceptor, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(_symmetric(matrix))
    return {
        "hidden_size": int(matrix.shape[0]),
        "quota": float(np.trace(matrix) / matrix.shape[0]),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "effective_rank": float(np.exp(-np.sum(
            np.where(
                eigenvalues > 0,
                (eigenvalues / max(float(eigenvalues.sum()), np.finfo(np.float64).eps))
                * np.log(
                    np.maximum(
                        eigenvalues / max(float(eigenvalues.sum()), np.finfo(np.float64).eps),
                        np.finfo(np.float64).tiny,
                    )
                ),
                0.0,
            )
        ))),
        "symmetry_max_abs_error": float(np.abs(matrix - matrix.T).max()),
    }
