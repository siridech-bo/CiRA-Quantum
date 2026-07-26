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


def narma_input_sine(n: int, seed: int = 0) -> np.ndarray:
    """Superposition-of-sines NARMA driving input, normalized to ``[0, 1]``.

    Paper 4 drives NARMA with a smooth multi-tone signal rather than the
    classic i.i.d. uniform noise. We sum a few incommensurate sinusoids
    (non-harmonic periods so the composite doesn't repeat over the
    sequence) with seeded phases, then min-max scale to ``[0, 1]`` — the
    reservoir's input domain. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(n, dtype=float)
    periods = (20.0, 33.0, 51.0)          # incommensurate → quasi-periodic
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(periods))
    s = np.zeros(n)
    for p, ph in zip(periods, phases, strict=True):
        s += np.sin(2.0 * np.pi * k / p + ph)
    s -= s.min()
    peak = s.max()
    if peak > 0:
        s /= peak
    return s


def narma_sequence_sine(
    n: int, order: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-Eq.2 NARMA target driven by the sine input (plan §4).

    Uses the standard NARMA-``order`` recurrence
    ``y_{k+1} = α y_k + β y_k Σ_{i} y_{k-i} + γ s_{k-n+1} s_k + δ`` with
    ``α=.3, β=.05, γ=1.5, δ=.1``; order 2 keeps the classic quadratic
    variant. The sine input from :func:`narma_input_sine` is rescaled to
    ``[0, 0.5]`` before entering the recurrence — the usual NARMA stability
    range. The returned input is that same ``[0, 0.5]`` sequence so it
    matches the target exactly when driven through the reservoir.

    For ``order >= 10`` an outer ``tanh`` saturates the recurrence
    (Rodan & Tiňo 2011; the standard NARMA-10/20 form). Without it the
    higher-order sum term ``β y_k Σ y_{k-i}`` grows without bound and the
    target overflows — the ``tanh`` keeps it in ``(-1, 1)`` while
    preserving the nonlinear-memory character of the task.
    """
    s = narma_input_sine(n, seed=seed) * 0.5
    y = np.zeros(n)
    saturate = order >= 10
    if order == 2:
        for k in range(1, n - 1):
            y[k + 1] = (
                0.4 * y[k]
                + 0.4 * y[k] * y[k - 1]
                + 0.6 * s[k] ** 3
                + 0.1
            )
    else:
        m = order
        for k in range(m - 1, n - 1):
            val = (
                0.3 * y[k]
                + 0.05 * y[k] * np.sum(y[k - m + 1 : k + 1])
                + 1.5 * s[k - m + 1] * s[k]
                + 0.1
            )
            y[k + 1] = np.tanh(val) if saturate else val
    return s, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def nmse_paper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Paper-4 NARMA error: ``Σ(y-ŷ)² / Σy²``.

    Unlike :func:`app.qrc.utils.nmse` (which normalizes by the *variance*
    of the target), Paper 4 normalizes the squared error by the target's
    raw second moment ``Σy²``. Both are reported for NARMA so results are
    comparable to the paper's Table I as well as to the QRC literature.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = float(np.sum(y_true**2))
    if denom == 0.0:
        return float(np.sum((y_true - y_pred) ** 2))
    return float(np.sum((y_true - y_pred) ** 2) / denom)


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
