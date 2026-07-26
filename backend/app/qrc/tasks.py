"""QRC benchmark tasks — NARMA, memory capacity, weather (plan §10).

Each task returns input/target arrays (and, for MC, a per-delay curve
helper). The heavy lifting — running the reservoir and fitting the
readout — lives in :mod:`app.qrc.benchmarks`; this module only defines
the *tasks* so they can be reused across systems of any qubit count.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# NARMA (Nonlinear Auto-Regressive Moving Average) — plan §10.1
# ---------------------------------------------------------------------------


def narma_sequence(n: int, order: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a NARMA-``order`` input/target pair of length ``n``.

    Standard formulation (Atiya & Parlos 2000; used by Paper 4). Input
    ``u_k`` ~ Uniform[0, 0.5]; the target follows the NARMA recurrence.
    For order 2 the classic quadratic variant is used; for higher orders
    the general NARMA-m recurrence. Inputs are already in the reservoir's
    ``[0, 1]`` domain (max 0.5), so no extra normalization is needed.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n)
    y = np.zeros(n)
    if order == 2:
        for k in range(1, n - 1):
            y[k + 1] = (
                0.4 * y[k]
                + 0.4 * y[k] * y[k - 1]
                + 0.6 * u[k] ** 3
                + 0.1
            )
    else:
        m = order
        for k in range(m - 1, n - 1):
            y[k + 1] = (
                0.3 * y[k]
                + 0.05 * y[k] * np.sum(y[k - m + 1 : k + 1])
                + 1.5 * u[k - m + 1] * u[k]
                + 0.1
            )
    return u, y


# ---------------------------------------------------------------------------
# Memory capacity — plan §5.5 / §10.3  (THE fading-memory test)
# ---------------------------------------------------------------------------


def memory_task(n: int, seed: int = 0) -> np.ndarray:
    """Random input sequence for the short-term memory-capacity test."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n)


def delayed_targets(u: np.ndarray, max_delay: int) -> dict[int, np.ndarray]:
    """For each delay ``d`` in ``0..max_delay``, target ``y_k = u_{k-d}``.

    The memory-capacity metric is ``MC = Σ_d corr²(u_{k-d}, ŷ_d)`` where
    ``ŷ_d`` is the reservoir's best linear reconstruction of the input
    delayed by ``d`` (plan §5.5.1). A well-behaved reservoir gives corr²≈1
    at d=0 and a smooth decay toward 0 — the fading-memory signature.
    """
    out: dict[int, np.ndarray] = {}
    for d in range(max_delay + 1):
        yd = np.zeros_like(u)
        if d == 0:
            yd = u.copy()
        else:
            yd[d:] = u[:-d]
        out[d] = yd
    return out


# ---------------------------------------------------------------------------
# Weather forecasting — plan §10.2 (Delhi climate, Kaggle)
# ---------------------------------------------------------------------------


def load_weather(
    csv_path: str,
    columns: tuple[str, ...] = ("meantemp", "humidity"),
) -> np.ndarray:
    """Load the Delhi daily climate CSV into a ``(n_days, n_vars)`` array.

    Needs pandas (installed with the ``[qrc]`` extra). The dataset is the
    Kaggle "Daily Delhi Climate" file with columns ``meantemp``,
    ``humidity``, ``wind_speed``, ``meanpressure``. Normalization to
    ``[0,1]`` happens in the benchmark, not here.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"weather CSV missing columns {missing}; has {list(df.columns)}")
    return df[list(columns)].to_numpy(dtype=float)
