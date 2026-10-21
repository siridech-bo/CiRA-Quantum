"""Fair ESN baseline for the Delhi weather forecasting task (SOP §6.5).

The Paper-4 weather comparison (QRC beats ESN(500-10000), reproduced in
QRC_Reproduction_Results.md §5) used an ESN with no leak rate and a fixed or
radius/connectivity-only tuned recurrent matrix. Leak sets the ESN's memory
timescale, which matters most for 30-45-day (seasonal) forecasting. This script
re-runs the comparison with a fully tuned ESN on EXACTLY the data the QRC saw.

Data: the saved weather trace (``artifacts/traces/weather_full.npz``): its
``weather_norm`` series and the 653-peak FID readout rebuilt from its ``fids``
(|FFT| at the 653 strongest bins of the first step, as features.from_fid does).

Protocol (leakage-free; the test block is never used for any choice):
  * main split = the reproduction's: washout 374, train 600, test 500;
    horizons 1,5,10,15,20,30,45; targets temp and humidity; metric R².
  * QRC: ridge readout per (horizon, var), lambda by CV on the 600 train days
    (train_readout — the original protocol). R² checked against the saved run.
  * ESN: 2-D input [bias, temp, humidity]; grid size x spectral radius x leak x
    input scale x ridge, 3 random inits per config. Selection on a validation
    slice = last 150 of the 600 train days (fit on the first 450), then refit on
    all 600 and score test. Reported variants:
      single   - one reservoir config for all targets (alpha per target) = the
                 fair analogue of the QRC's one reservoir pass
      flexible - config chosen per (horizon, var)
      matched  - single, size <= 653 (the QRC feature count)
      no_leak  - single, leak fixed at 1 (the Paper-4 / reproduction ESN form)
  * Uncertainty: moving-block bootstrap (block 30 days) of R²(QRC) - R²(ESN
    single) on the main test block; rolling origins (train 600 / test 300 at
    three start offsets) with selection redone per origin.

  python backend/scripts/qrc_weather_fair.py --trace artifacts/traces/weather_full.npz \
      --workers 32 --out weather_fair.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
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

HORIZONS = (1, 5, 10, 15, 20, 30, 45)
VARS = ("temp", "humidity")
SIZES = (300, 650, 1000, 2000)
SR = (0.5, 0.8, 0.95, 1.1, 1.3)
LEAK = (0.02, 0.05, 0.1, 0.3, 0.6, 1.0)
INSCALE = (0.1, 0.3, 1.0, 3.0)
ALPHA = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
INITS = (0, 1, 2)
MATCHED_MAX = 653
N_VAL = 150
# (washout/start, n_train, n_test): main = the reproduction split; then rolling
WINDOWS = {"main": (374, 600, 500), "roll0": (374, 600, 300),
           "roll1": (474, 600, 300), "roll2": (574, 600, 300)}

_W = {}            # worker-global data (set by _init)


def _init(weather_norm):
    _W["w"] = weather_norm


def _r2(y, p):
    ss = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss > 0 else float("nan")


def _targets(w, idx):
    """(len(idx), n_targets) matrix of y[k+h, var] for all (h, var)."""
    return np.stack([w[idx + h, j] for h in HORIZONS for j in range(len(VARS))], axis=1)


def _dual_ridge(Xf, Yf, Xs, alphas):
    """Ridge in dual form (n_fit < n_features): predictions on Xs for each alpha.
    Returns array (n_alpha, len(Xs), n_targets)."""
    K = Xf @ Xf.T
    s, U = np.linalg.eigh(K)
    UtY = U.T @ Yf
    Ks = Xs @ Xf.T
    out = []
    for a in alphas:
        coef = U @ (UtY / (s + a)[:, None])
        out.append(Ks @ coef)
    return np.stack(out)


def esn_states(w, n_steps, m, sr, leak, ins, init):
    rng = np.random.default_rng(10_000 * init + m)
    W = rng.standard_normal((m, m))
    W /= np.max(np.abs(np.linalg.eigvals(W))) if m <= 1000 else _power_radius(W, rng)
    Win = rng.uniform(-1, 1, (m, 1 + w.shape[1]))
    W *= sr
    Win *= ins
    x = np.zeros(m)
    X = np.empty((n_steps, m))
    for k in range(n_steps):
        x = (1 - leak) * x + leak * np.tanh(Win[:, 0] + Win[:, 1:] @ w[k] + W @ x)
        X[k] = x
    return np.hstack([X, np.ones((n_steps, 1))])


def _power_radius(W, rng, iters=200):
    v = rng.standard_normal(W.shape[0])
    lam = 1.0
    for _ in range(iters):
        u = W @ v
        lam = np.linalg.norm(u)
        v = u / lam
    return lam


def _score_windows(X, w):
    """For each window: val R² (fit 450 / val 150) and test R² (fit 600 / test)
    for every alpha and target. Returns {window: (val[a,t], test[a,t])}."""
    res = {}
    for name, (w0, ntr, nte) in WINDOWS.items():
        fit = np.arange(w0, w0 + ntr - N_VAL)
        val = np.arange(w0 + ntr - N_VAL, w0 + ntr)
        trn = np.arange(w0, w0 + ntr)
        tst = np.arange(w0 + ntr, w0 + ntr + nte)
        Yv = _targets(w, val)
        Yt = _targets(w, tst)
        pv = _dual_ridge(X[fit], _targets(w, fit), X[val], ALPHA)
        pt = _dual_ridge(X[trn], _targets(w, trn), X[tst], ALPHA)
        vr = np.array([[_r2(Yv[:, t], pv[a, :, t]) for t in range(Yv.shape[1])] for a in range(len(ALPHA))])
        tr = np.array([[_r2(Yt[:, t], pt[a, :, t]) for t in range(Yt.shape[1])] for a in range(len(ALPHA))])
        res[name] = (vr, tr)
    return res


def _job(args):
    m, init, sr = args
    w = _W["w"]
    n_steps = max(w0 + a + b for (w0, a, b) in WINDOWS.values())
    out = {}
    for leak, ins in itertools.product(LEAK, INSCALE):
        X = esn_states(w, n_steps, m, sr, leak, ins, init)
        out[(m, sr, leak, ins)] = _score_windows(X, w)
    return init, out


def qrc_features(fids):
    spec = np.abs(np.fft.fft(fids, axis=1))
    bins = np.sort(np.argsort(spec[0])[::-1][:653])
    return spec[:, bins]


def qrc_predictions(Xq, w, window, cv_lambda=True):
    """Original QRC readout protocol: ridge, lambda by CV on the train block."""
    from app.qrc.config import TrainingConfig
    from app.qrc.training import train_readout

    w0, ntr, nte = window
    trn = np.arange(w0, w0 + ntr)
    tst = np.arange(w0 + ntr, w0 + ntr + nte)
    cfg = TrainingConfig(device="cpu")
    Yt = _targets(w, trn)
    preds = []
    for t in range(Yt.shape[1]):
        model = train_readout(Xq[trn], Yt[:, t], cfg)
        preds.append(np.asarray(model.predict(Xq[tst])).ravel())
    return np.stack(preds, axis=1), _targets(w, tst)


def select(results, window, keep):
    """Best config(s) by validation R² (mean over inits). ``keep`` filters configs.
    Returns (single, flexible) where single = {cfg, alpha_idx per target, test per
    target (mean, sd over inits)}, flexible = per-target best."""
    cfgs = [c for c in results[INITS[0]] if keep(c)]
    n_t = len(HORIZONS) * len(VARS)

    def val_test(c):
        v = np.mean([results[i][c][window][0] for i in INITS], axis=0)     # [a, t]
        te = np.array([results[i][c][window][1] for i in INITS])           # [init, a, t]
        return v, te

    best_single, best_score = None, -np.inf
    flex = [None] * n_t
    for c in cfgs:
        v, te = val_test(c)
        a_best = np.nanargmax(v, axis=0)                                     # alpha per target
        score = float(np.nanmean(v[a_best, np.arange(n_t)]))
        if score > best_score:
            best_score = score
            best_single = {"cfg": c, "alpha_idx": a_best.tolist(),
                           "test": [(float(np.mean(te[:, a_best[t], t])), float(np.std(te[:, a_best[t], t])))
                                    for t in range(n_t)]}
        for t in range(n_t):
            vt = v[a_best[t], t]
            if flex[t] is None or vt > flex[t]["val"]:
                flex[t] = {"cfg": c, "alpha_idx": int(a_best[t]), "val": float(vt),
                           "test": (float(np.mean(te[:, a_best[t], t])), float(np.std(te[:, a_best[t], t])))}
    return best_single, flex


def block_bootstrap_delta(y, p_q, p_e_list, n_boot=2000, block=30, seed=0):
    """CI of R²(QRC) - mean_init R²(ESN) under a moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y)
    starts = np.arange(0, n - block + 1)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n // block + 1)])[:n]
        d = _r2(y[idx], p_q[idx]) - np.mean([_r2(y[idx], pe[idx]) for pe in p_e_list])
        deltas.append(d)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.array(deltas) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    ap.add_argument("--saved", default=None, help="saved reproduction json (qrc_paper4_weather_v2.json)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.trace, allow_pickle=True)
    w = np.asarray(z["weather_norm"], float)
    Xq = qrc_features(np.asarray(z["fids"]))
    print(f"trace: fids {z['fids'].shape}, weather {w.shape}, split {z['split'].tolist()}; "
          f"QRC features {Xq.shape}", flush=True)

    # ---- QRC (original protocol) ----
    qrc = {}
    for name, win in WINDOWS.items():
        p, y = qrc_predictions(Xq, w, win)
        qrc[name] = {"pred": p, "y": y,
                     "r2": [_r2(y[:, t], p[:, t]) for t in range(y.shape[1])]}
        print(f"QRC {name}: " + " ".join(f"{_r2(y[:, t], p[:, t]):.3f}" for t in range(y.shape[1])), flush=True)

    # ---- ESN grid ----
    jobs = [(m, i, sr) for m in SIZES for i in INITS for sr in SR]
    results = {i: {} for i in INITS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(w,)) as ex:
        for k, (init, out) in enumerate(ex.map(_job, jobs)):
            results[init].update(out)
            print(f"ESN job {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    targets = [f"{v}_h{h}" for h in HORIZONS for v in VARS]
    variants = {"single": lambda c: True, "matched": lambda c: c[0] <= MATCHED_MAX,
                "no_leak": lambda c: c[2] == 1.0}
    report = {"experiment": "weather_fair_esn", "trace": a.trace, "windows": WINDOWS,
              "horizons": HORIZONS, "vars": VARS, "targets": targets,
              "grid": {"size": SIZES, "spectral_radius": SR, "leak": LEAK, "input_scale": INSCALE,
                       "ridge_alpha": ALPHA, "inits": len(INITS), "n_val": N_VAL},
              "qrc_r2": {n: qrc[n]["r2"] for n in WINDOWS}, "esn": {}}
    for name in WINDOWS:
        report["esn"][name] = {}
        for vname, keep in variants.items():
            single, flex = select(results, name, keep)
            report["esn"][name][vname] = {"cfg": dict(zip(("size", "sr", "leak", "inscale"), single["cfg"])),
                                          "test": single["test"]}
            if vname == "single":
                report["esn"][name]["flexible"] = [{"cfg": dict(zip(("size", "sr", "leak", "inscale"), f["cfg"])),
                                                     "test": f["test"]} for f in flex]
                report["esn"][name]["_single_obj"] = single

    # ---- bootstrap CI on the main split: QRC vs ESN(single), predictions rebuilt ----
    single = report["esn"]["main"].pop("_single_obj")
    for n in WINDOWS:
        report["esn"][n].pop("_single_obj", None)
    m, sr, leak, ins = single["cfg"]
    w0, ntr, nte = WINDOWS["main"]
    n_steps = w0 + ntr + nte
    trn = np.arange(w0, w0 + ntr)
    tst = np.arange(w0 + ntr, w0 + ntr + nte)
    esn_preds = []
    for i in INITS:
        X = esn_states(w, n_steps, m, sr, leak, ins, i)
        P = _dual_ridge(X[trn], _targets(w, trn), X[tst], ALPHA)              # [a, n, t]
        esn_preds.append(np.stack([P[single["alpha_idx"][t], :, t] for t in range(len(targets))], axis=1))
    y = qrc["main"]["y"]
    report["bootstrap_main"] = {}
    for t, tn in enumerate(targets):
        lo, hi, p_pos = block_bootstrap_delta(y[:, t], qrc["main"]["pred"][:, t], [e[:, t] for e in esn_preds])
        report["bootstrap_main"][tn] = {"delta_ci95": [lo, hi], "frac_qrc_better": p_pos}

    if a.saved and Path(a.saved).exists():
        d = json.loads(Path(a.saved).read_text(encoding="utf-8"))
        report["saved_reproduction"] = {"qrc": {h: {v: d["qrc_weather"][str(h)][v]["r2"] for v in VARS}
                                                for h in HORIZONS},
                                        "esn_best_old": {h: {v: max(d["esn_weather"][s][str(h)][v] for s in d["esn_weather"])
                                                             for v in VARS} for h in HORIZONS}}
    Path(a.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print("\nmain split, test R² (QRC | ESN single | ESN no-leak | ESN matched | 95% CI of QRC-ESN)")
    for t, tn in enumerate(targets):
        e = report["esn"]["main"]
        print(f"  {tn:13s} {qrc['main']['r2'][t]:.3f} | {e['single']['test'][t][0]:.3f} | "
              f"{e['no_leak']['test'][t][0]:.3f} | {e['matched']['test'][t][0]:.3f} | "
              f"[{report['bootstrap_main'][tn]['delta_ci95'][0]:+.3f}, {report['bootstrap_main'][tn]['delta_ci95'][1]:+.3f}]")
    print("ESN single cfg:", report["esn"]["main"]["single"]["cfg"], "| no-leak cfg:", report["esn"]["main"]["no_leak"]["cfg"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
