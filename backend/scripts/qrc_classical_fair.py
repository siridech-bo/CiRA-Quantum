"""Fair classical baselines for the QRC-vs-classical comparison (SOP §6.5).

Fixes the two weaknesses of ``qrc_classical_baseline.py``: the ESN there swept
only reservoir size (spectral radius / leak / input scale / ridge fixed) and every
classical number came from data seed 1 only, while the QRC results are means over
data seeds 1-3.

Here, on the SAME tasks, data seeds, T, washout and leakage-free 3-way split as
the QRC campaigns:
  * ESN  — full grid: size x spectral radius x leak x input scale x ridge alpha,
           each config averaged over 3 reservoir inits; selected on VALIDATION;
           test reported at the selected config (mean +/- s.d. over inits).
           Also the best size-matched ESN (n_res <= 350 ~ QRC feature count).
  * LSTM — the tuned sweep of qrc_classical_baseline.tuned_lstm (hidden x layers x
           lr, multi-init, cosine LR, grad clip, val early stop).
Per task: mean +/- s.d. over data seeds 1-3 of each model's test NMSE.

  python backend/scripts/qrc_classical_fair.py --models esn  --workers 32 --out esn.json
  python backend/scripts/qrc_classical_fair.py --models lstm --out lstm.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per ESN worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qrc_learnable_9spin import _three_way, make_task  # noqa: E402

TASKS = [("narma2", 1), ("narma10", 1), ("mackey_glass", 10)]
SEEDS = (1, 2, 3)
ESN_SIZES = (200, 350, 1000)
ESN_SR = (0.5, 0.8, 0.95, 1.1, 1.25)
ESN_LEAK = (0.1, 0.3, 0.6, 1.0)
ESN_INSCALE = (0.1, 0.3, 1.0, 3.0)
ESN_ALPHA = (1e-8, 1e-6, 1e-4, 1e-2)
ESN_INITS = (0, 1, 2)
MATCHED_MAX = 350


def _key(task, h):
    return f"{task}_h{h}" if h != 1 else task


def _nmse(pred, y):
    v = float(np.var(y))
    return float(np.mean((pred - y) ** 2) / v) if v > 0 else float("nan")


def _esn_job(args):
    """All (sr, leak, inscale, alpha) configs for one (task, seed, size, init):
    returns {(sr, leak, inscale, alpha): (val, test)}."""
    task, h, seed, T, washout, n_res, init = args
    u, y = make_task(task, T, seed, horizon=h)
    u = np.asarray(u, float)
    yu = np.asarray(y, float)[washout:]
    tr, va, te = _three_way(len(yu))
    rng = np.random.default_rng(1000 * init + n_res)
    W = rng.uniform(-1, 1, (n_res, n_res))
    W /= np.max(np.abs(np.linalg.eigvals(W)))       # unit spectral radius
    Win = rng.uniform(-1, 1, (n_res, 2))             # [bias, u]
    out = {}
    for sr, leak, ins in itertools.product(ESN_SR, ESN_LEAK, ESN_INSCALE):
        Ws, Wi = sr * W, ins * Win
        x = np.zeros(n_res)
        X = np.empty((len(u), n_res))
        for k in range(len(u)):
            x = (1 - leak) * x + leak * np.tanh(Wi[:, 0] + Wi[:, 1] * u[k] + Ws @ x)
            X[k] = x
        Xb = np.hstack([X, np.ones((len(u), 1))])[washout:]
        G = Xb[tr].T @ Xb[tr]
        b = Xb[tr].T @ yu[tr]
        for a in ESN_ALPHA:
            w = np.linalg.solve(G + a * np.eye(G.shape[0]), b)
            out[(sr, leak, ins, a)] = (_nmse(Xb[va] @ w, yu[va]), _nmse(Xb[te] @ w, yu[te]))
    return (task, h, seed, n_res, init), out


def run_esn(T, washout, workers, log):
    jobs = [(t, h, s, T, washout, m, i) for (t, h) in TASKS for s in SEEDS
            for m in ESN_SIZES for i in ESN_INITS]
    res = {}
    t0 = time.time()
    ex = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for k, (key, out) in enumerate((ex.map if ex else map)(_esn_job, jobs)):
            res[key] = out
            log(f"ESN {k + 1}/{len(jobs)} {key} ({time.time() - t0:.0f}s)")
    finally:
        if ex:
            ex.shutdown()
    summary = {}
    for (t, h) in TASKS:
        per_seed = {}
        for s in SEEDS:
            def select(max_size):
                best = None
                for m in ESN_SIZES:
                    if m > max_size:
                        continue
                    for cfg in res[(t, h, s, m, 0)]:
                        vals = [res[(t, h, s, m, i)][cfg][0] for i in ESN_INITS]
                        tests = [res[(t, h, s, m, i)][cfg][1] for i in ESN_INITS]
                        mv = float(np.mean(vals))
                        if best is None or mv < best["val_mean"]:
                            best = {"n_res": m, "sr": cfg[0], "leak": cfg[1], "inscale": cfg[2],
                                    "alpha": cfg[3], "val_mean": mv,
                                    "test_mean": float(np.mean(tests)),
                                    "test_std": float(np.std(tests))}
                return best
            per_seed[str(s)] = {"esn_best": select(10 ** 9), "esn_matched": select(MATCHED_MAX)}
        summary[_key(t, h)] = {"per_seed": per_seed,
                               "esn_best": _agg([per_seed[str(s)]["esn_best"]["test_mean"] for s in SEEDS]),
                               "esn_matched": _agg([per_seed[str(s)]["esn_matched"]["test_mean"] for s in SEEDS])}
    return summary


def run_lstm(T, washout, inits, epochs, log):
    import torch
    from qrc_classical_baseline import tuned_lstm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}
    for (t, h) in TASKS:
        per_seed = {}
        for s in SEEDS:
            u, y = make_task(t, T, s, horizon=h)
            t0 = time.time()
            per_seed[str(s)] = tuned_lstm(u, y, device, washout, inits, epochs)
            log(f"LSTM {_key(t, h)} seed {s}: test {per_seed[str(s)]['test_mean']:.5f} "
                f"({time.time() - t0:.0f}s)")
        summary[_key(t, h)] = {"per_seed": per_seed,
                               "lstm_best": _agg([per_seed[str(s)]["test_mean"] for s in SEEDS])}
        yield summary


def _agg(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "per_seed": [float(x) for x in xs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", choices=["esn", "lstm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--washout", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--inits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=500)
    a = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    meta = {"experiment": "classical_fair", "models": a.models, "T": a.T, "washout": a.washout,
            "data_seeds": list(SEEDS), "split": "leakage-free 3-way 50/25/25, select on val",
            "tasks": [_key(t, h) for t, h in TASKS]}
    if a.models == "esn":
        meta["grid"] = {"size": ESN_SIZES, "spectral_radius": ESN_SR, "leak": ESN_LEAK,
                        "input_scale": ESN_INSCALE, "ridge_alpha": ESN_ALPHA,
                        "inits": len(ESN_INITS), "matched_max_size": MATCHED_MAX}
        meta["results"] = run_esn(a.T, a.washout, a.workers, log)
        Path(a.out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        meta["grid"] = {"hidden": [32, 64, 128, 256], "layers": [1, 2], "lr": [1e-3, 3e-3, 1e-2],
                        "inits": a.inits, "epochs": a.epochs}
        for partial in run_lstm(a.T, a.washout, a.inits, a.epochs, log):
            meta["results"] = partial                        # crash-safe after every task
            Path(a.out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
