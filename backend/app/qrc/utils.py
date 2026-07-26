"""QRC helpers — normalization, metrics, seeding.

Deliberately dependency-light (NumPy only) so it imports even when the
heavy quantum stack (QuTiP/JAX) isn't installed. The training module
adds a Torch-GPU path on top of the metrics defined here.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> np.random.Generator:
    """Seed NumPy's global state and return a fresh Generator.

    Torch is seeded lazily in :mod:`app.qrc.training` (import guarded) so
    this stays NumPy-only.
    """
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Normalization (plan §8.2)
# ---------------------------------------------------------------------------


def normalize(
    x: np.ndarray,
    lo: float | None = None,
    hi: float | None = None,
    clip: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Min-max scale ``x`` into ``[0, 1]``.

    Returns ``(scaled, lo, hi)`` so predictions can be denormalized
    later. If ``lo``/``hi`` are omitted they're taken from the data.
    """
    x = np.asarray(x, dtype=float)
    lo = float(np.min(x)) if lo is None else lo
    hi = float(np.max(x)) if hi is None else hi
    if hi == lo:
        return np.zeros_like(x), lo, hi
    s = (x - lo) / (hi - lo)
    if clip:
        s = np.clip(s, 0.0, 1.0)
    return s, lo, hi


def denormalize(s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Invert :func:`normalize`."""
    return np.asarray(s, dtype=float) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def squared_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Squared Pearson correlation — the memory-capacity convention.

    The standard QRC memory-capacity metric (Jaeger 2002, Nakajima 2018)
    uses ``corr(y, ŷ)²`` rather than R². For a well-fit linear readout on
    the training distribution the two coincide, but corr² is bounded to
    ``[0, 1]`` and is the number reported in the QRC literature, so MC
    curves stay comparable to published figures.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    c = np.corrcoef(y_true, y_pred)[0, 1]
    return float(c * c)


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean squared error (Paper 4's NARMA metric)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    var = np.var(y_true)
    if var == 0:
        return float(np.mean((y_true - y_pred) ** 2))
    return float(np.mean((y_true - y_pred) ** 2) / var)
