"""QRC offline feature-evaluation harness (Phase 0.B / Stage A §2).

The trace-cache generator (:mod:`scripts.qrc_gen_traces`) dumps the per-step
raw complex FID for a weather or NARMA reservoir pass into an ``.npz`` (schema
in ``docs/QRC/QRC_StageA_Contracts.md`` §1). Feature *builders* (Coder B's
:func:`app.qrc.feature_methods.build_features`) turn that FID tensor into a
design matrix ``X``; this module *scores* such an ``X`` against the trace's
task, targets, and split — implementing the Evaluation Protocol from the
next-stage experiment plan so every feature method reports comparable numbers.

Two public entry points:

* :func:`load_trace` — load a trace ``.npz`` into a plain dict (arrays kept as
  arrays, scalars/strings/JSON decoded to Python types).
* :func:`evaluate_features` — given a feature matrix ``X`` and a loaded trace
  dict, fit the same ridge readout the reproduction uses and return the
  protocol dict ``{experiment, feature_count, weather_r2, narma10_nmse,
  wall_clock_s, delta_vs_baseline}``.

The heavy reservoir pass never runs here: this operates entirely on the cached
FID-derived features, which is what makes the Phase-1/3/4 feature experiments
cheap (minutes instead of the ~15 h evolution).
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from app.qrc.benchmarks import _split, forecast_from_X
from app.qrc.config import TrainingConfig
from app.qrc.tasks import nmse_paper
from app.qrc.training import train_readout

# Baselines to diff against (next-stage experiment plan §"Baseline to beat";
# weather = temperature R² for the FID-653 magnitude ridge readout, v2).
BASELINE_WEATHER_R2: dict[int, float] = {1: 0.950, 10: 0.888, 20: 0.860,
                                         30: 0.812, 45: 0.786}
BASELINE_NARMA10_NMSE = 2.46e-5

# Canonical horizons whose R² the protocol table reports.
CANONICAL_HORIZONS = (1, 10, 20, 30, 45)


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_trace(path: str) -> dict[str, Any]:
    """Load a trace ``.npz`` (Stage A §1 schema) into a plain dict.

    Array fields (``fids``, ``weather_norm``, ``narma_input``, ``split``,
    ``horizons``, ``orders``) stay as NumPy arrays; ``fid_dwell``/``seed``
    become Python scalars, ``task`` a ``str``, and ``meta`` is JSON-decoded to
    a dict. Missing optional fields are simply absent from the returned dict.
    """
    npz = np.load(path, allow_pickle=False)
    out: dict[str, Any] = {}
    out["fids"] = np.asarray(npz["fids"])
    out["fid_dwell"] = float(npz["fid_dwell"])
    out["task"] = str(npz["task"])
    out["split"] = np.asarray(npz["split"], dtype=int)
    out["seed"] = int(npz["seed"])
    if "weather_norm" in npz.files:
        out["weather_norm"] = np.asarray(npz["weather_norm"], dtype=float)
    if "horizons" in npz.files:
        out["horizons"] = np.asarray(npz["horizons"], dtype=int)
    if "narma_input" in npz.files:
        out["narma_input"] = np.asarray(npz["narma_input"], dtype=float)
    if "orders" in npz.files:
        out["orders"] = np.asarray(npz["orders"], dtype=int)
    if "meta" in npz.files:
        try:
            out["meta"] = json.loads(str(npz["meta"]))
        except (ValueError, TypeError):
            out["meta"] = {}
    return out


# ---------------------------------------------------------------------------
# NARMA target reconstruction (from the cached driving input)
# ---------------------------------------------------------------------------


def narma_target_from_input(s: np.ndarray, order: int) -> np.ndarray:
    """Regenerate the NARMA-``order`` target from a cached driving input ``s``.

    Mirrors :func:`app.qrc.tasks.narma_sequence_sine` exactly but takes the
    stored ``narma_input`` (the ``[0, 0.5]`` sequence that actually drove the
    reservoir) directly, so the target is reproduced from the trace rather than
    re-derived from a seed. ``order >= 10`` applies the standard outer ``tanh``
    saturation (Rodan & Tiňo 2011).
    """
    s = np.asarray(s, dtype=float).ravel()
    n = s.size
    y = np.zeros(n)
    if order == 2:
        for k in range(1, n - 1):
            y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * s[k] ** 3 + 0.1
        return y
    m = order
    saturate = order >= 10
    for k in range(m - 1, n - 1):
        val = (
            0.3 * y[k]
            + 0.05 * y[k] * np.sum(y[k - m + 1: k + 1])
            + 1.5 * s[k - m + 1] * s[k]
            + 0.1
        )
        y[k + 1] = np.tanh(val) if saturate else val
    return y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _tr_cfg_from_split(split, seed: int) -> TrainingConfig:
    """Build a :class:`TrainingConfig` carrying the trace's own split.

    ``cv_folds`` is capped at ``n_train`` so tiny validation traces (where
    ``n_train`` may be < 10) don't produce empty CV folds; the default
    log-spaced λ grid and CUDA-with-CPU-fallback device are kept."""
    washout, n_train, n_test = (int(split[0]), int(split[1]), int(split[2]))
    folds = max(2, min(10, n_train))
    return TrainingConfig(
        washout=washout, n_train=n_train, n_test=n_test,
        cv_folds=folds, seed=seed,
    )


def _weather_scores(X, npz, tr_cfg) -> dict[str, float]:
    """Temperature R² per horizon on the test block (baseline metric)."""
    weather_norm = npz["weather_norm"]
    horizons = [int(h) for h in npz["horizons"]]
    forecast = forecast_from_X(X, weather_norm, horizons, tr_cfg, use_rbf=False)
    return {f"h{h}": float(forecast[h]["temp"]["r2"]) for h in horizons}


def _narma10_score(X, npz, tr_cfg) -> tuple[float, int]:
    """NARMA-10 NMSE (Σ(y-ŷ)²/Σy²) on the test block; returns (nmse, order)."""
    orders = [int(o) for o in npz["orders"]]
    order = 10 if 10 in orders else orders[0]
    y = narma_target_from_input(npz["narma_input"], order)
    Xtr, ytr, Xte, yte = _split(
        X, y, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test
    )
    model = train_readout(Xtr, ytr, tr_cfg)
    return nmse_paper(yte, model.predict(Xte)), order


def evaluate_features(X: np.ndarray, npz: dict, method_name: str) -> dict:
    """Score a feature matrix ``X`` against a loaded trace (Stage A §2).

    ``X`` is ``(n_steps, n_feat)`` built by a feature method from the trace's
    ``fids``; ``npz`` is a dict from :func:`load_trace`; ``method_name`` labels
    the experiment. The trace's ``task`` selects which target/metric is
    computed: a weather trace populates ``weather_r2`` (temperature R² per
    horizon), a NARMA trace populates ``narma10_nmse``. Both are keyed against
    the same washout/train/test split stored in the trace.

    Returns the Evaluation-Protocol dict::

        {experiment, feature_count, weather_r2:{hN:..}, narma10_nmse,
         wall_clock_s, delta_vs_baseline:{..}}
    """
    X = np.asarray(X, dtype=float)
    task = str(npz["task"])
    tr_cfg = _tr_cfg_from_split(npz["split"], int(npz["seed"]))

    result: dict[str, Any] = {
        "experiment": method_name,
        "feature_count": int(X.shape[1]),
        "weather_r2": {},
        "narma10_nmse": None,
        "wall_clock_s": None,
        "delta_vs_baseline": {},
    }

    t0 = time.time()
    if task == "weather":
        weather_r2 = _weather_scores(X, npz, tr_cfg)
        result["weather_r2"] = weather_r2
        delta: dict[str, str] = {}
        for h in CANONICAL_HORIZONS:
            key = f"h{h}"
            if key in weather_r2 and h in BASELINE_WEATHER_R2:
                delta[key] = f"{weather_r2[key] - BASELINE_WEATHER_R2[h]:+.4f}"
        result["delta_vs_baseline"] = delta
    elif task == "narma":
        nmse_val, order = _narma10_score(X, npz, tr_cfg)
        result["narma10_nmse"] = float(nmse_val)
        result["delta_vs_baseline"] = {
            "narma10_nmse": f"{nmse_val - BASELINE_NARMA10_NMSE:+.3e}",
            "order": order,
        }
    else:
        raise ValueError(f"unknown trace task {task!r} (expected weather|narma)")
    result["wall_clock_s"] = round(time.time() - t0, 4)
    return result
