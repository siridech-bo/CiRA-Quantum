"""Weather benchmark harness: QRC readout/reservoir variants vs a tuned ESN, with
one symmetric, leakage-free selection protocol (SOP §6.5).

Lessons from qrc_weather_fair.py: a single 150-day validation slice cannot rank
configurations on this seasonal series (slow-leak ESNs won validation and failed
test). Here every choice — ESN config, ridge alpha, and which QRC variant — is
made on the POOLED validation score over all rolling windows, and the test
blocks are only read for the final report.

Windows (start, n_train, n_test): main (374, 600, 500) + three rolling origins
(374/474/574, 600, 300). Inside each window: fit on the first 450 train days,
validate on the last 150; refit on all 600 for test. Targets: temp and
humidity at horizons 1..45; metric R². Pooled metric = mean R² over windows x
horizons (per variable and overall).

  esn mode    : run the ESN grid once, save every config's val/test R² (npz)
  report mode : build QRC feature variants (from a trace .npz or extra .npz
                feature files), score them with the same ridge/alpha protocol,
                select on pooled validation, compare against the ESN selected
                the same way (overall and size-matched), block-bootstrap CI of
                the pooled test difference.

  python scripts/qrc_weather_bench.py esn --workers 40 --out esn_grid.npz
  python scripts/qrc_weather_bench.py report --esn esn_grid.npz \
      --trace artifacts/traces/weather_full.npz --out weather_bench.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qrc_weather_fair import (  # noqa: E402
    ALPHA, HORIZONS, INITS, INSCALE, LEAK, N_VAL, SIZES, SR, VARS, WINDOWS,
    _dual_ridge, _r2, _targets, esn_states,
)

WIN_NAMES = list(WINDOWS)
N_T = len(HORIZONS) * len(VARS)
TARGETS = [f"{v}_h{h}" for h in HORIZONS for v in VARS]
_W = {}


def _init(w):
    _W["w"] = w


def score_features(X, w, standardize=False):
    """val/test R² for every (window, alpha, target): arrays [win, alpha, t].
    ``standardize`` z-scores features with the fit (val) / train (test) block
    statistics — used for QRC readouts whose raw scales vary over decades."""
    val = np.zeros((len(WIN_NAMES), len(ALPHA), N_T))
    tst = np.zeros_like(val)
    for wi, name in enumerate(WIN_NAMES):
        w0, ntr, nte = WINDOWS[name]
        fit = np.arange(w0, w0 + ntr - N_VAL)
        vl = np.arange(w0 + ntr - N_VAL, w0 + ntr)
        trn = np.arange(w0, w0 + ntr)
        te = np.arange(w0 + ntr, w0 + ntr + nte)
        for rows_fit, rows_eval, out in ((fit, vl, val), (trn, te, tst)):
            A, B = X[rows_fit], X[rows_eval]
            if standardize:
                mu, sd = A.mean(0), A.std(0) + 1e-12
                A, B = (A - mu) / sd, (B - mu) / sd
                A = np.hstack([A, np.ones((len(A), 1))])
                B = np.hstack([B, np.ones((len(B), 1))])
            P = _dual_ridge(A, _targets(w, rows_fit), B, ALPHA)
            Y = _targets(w, rows_eval)
            out[wi] = [[_r2(Y[:, t], P[a, :, t]) for t in range(N_T)] for a in range(len(ALPHA))]
    return val, tst


def _esn_job(args):
    m, init, sr = args
    w = _W["w"]
    n_steps = max(a + b + c for (a, b, c) in WINDOWS.values())
    out = []
    for leak in LEAK:
        for ins in INSCALE:
            X = esn_states(w, n_steps, m, sr, leak, ins, init)
            for std in (0, 1):            # same readout preprocessing options as the QRC
                v, t = score_features(X, w, standardize=bool(std))
                out.append(((m, sr, leak, ins, std), v, t))
    return init, out


def run_esn(w, workers, out_path):
    jobs = [(m, i, sr) for m in SIZES for i in INITS for sr in SR]
    store = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(w,)) as ex:
        for k, (init, res) in enumerate(ex.map(_esn_job, jobs)):
            for cfg, v, t in res:
                store[(cfg, init)] = (v, t)
            print(f"ESN job {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    cfgs = sorted({c for (c, _) in store})
    val = np.array([[store[(c, i)][0] for i in INITS] for c in cfgs])     # [cfg, init, win, a, t]
    tst = np.array([[store[(c, i)][1] for i in INITS] for c in cfgs])
    np.savez(out_path, cfgs=np.array(cfgs, float), val=val, test=tst,
             alpha=np.array(ALPHA), windows=np.array(WIN_NAMES), targets=np.array(TARGETS))
    print(f"wrote {out_path}")


def selection_windows():
    """Windows whose validation days all precede the FIRST test day of any window.
    Selection (configs, encoder training) may only see those days — the rolling
    windows overlap in time, so a later window's validation block contains
    earlier windows' test days (leakage, found 2026-10-21)."""
    first_test = min(w0 + ntr for (w0, ntr, _) in WINDOWS.values())
    return [i for i, n in enumerate(WIN_NAMES) if WINDOWS[n][0] + WINDOWS[n][1] <= first_test]


def pick(val, tst):
    """Given val/test [.., win, a, t] for ONE candidate (init-averaged already),
    choose alpha per (win, t) on val; return the pooled val over the leak-free
    selection windows and the test matrix [win, t] at those alphas."""
    a_best = np.nanargmax(val, axis=1)                                      # [win, t]
    wi, ti = np.meshgrid(np.arange(val.shape[0]), np.arange(val.shape[2]), indexing="ij")
    v = val[wi, a_best, ti]
    t = tst[wi, a_best, ti]
    return float(np.nanmean(v[selection_windows()])), t, a_best


def select_esn(esn, max_size=None):
    cfgs = esn["cfgs"]
    best = None
    for k, c in enumerate(cfgs):
        if max_size is not None and c[0] > max_size:
            continue
        v_mean = esn["val"][k].mean(0)                                       # avg over inits
        t_inits = esn["test"][k]
        pv, _, a_best = pick(v_mean, t_inits.mean(0))
        if best is None or pv > best["pooled_val"]:
            wi, ti = np.meshgrid(np.arange(v_mean.shape[0]), np.arange(v_mean.shape[2]), indexing="ij")
            per_init = np.array([ti_[wi, a_best, ti] for ti_ in t_inits])  # [init, win, t]
            best = {"cfg": dict(zip(("size", "sr", "leak", "inscale", "std"), c.tolist())), "pooled_val": pv,
                    "test": per_init.mean(0), "test_init_sd": per_init.std(0), "alpha_idx": a_best}
    return best


def qrc_variants(fids, dwell, system_name="crotonic9_paper4", readout_qubits=(4, 5, 6, 7, 8)):
    """Readout variants built from the saved complex FIDs (no new simulation)."""
    spec = np.fft.fft(fids, axis=1)
    mag = np.abs(spec)
    order = np.argsort(mag[0])[::-1]
    b653, b1306 = np.sort(order[:653]), np.sort(order[:1306])
    v = {
        "mag653 (baseline)": mag[:, b653],
        "mag1306": mag[:, b1306],
        "mag2048 (all bins)": mag,
        "reim653": np.hstack([spec.real[:, b653], spec.imag[:, b653]]),
        "mag+reim653": np.hstack([mag[:, b653], spec.real[:, b653], spec.imag[:, b653]]),
    }
    try:                                     # physics-informed line projection (SOP 3.2)
        from app.qrc.spectral_lines import physics_informed_lines
        from qrc_gen_traces import _resolve_system
        ls = physics_informed_lines(_resolve_system(system_name), tuple(readout_qubits))
        phi = np.asarray(ls.projection(fids.shape[1], dwell))                   # (M, D)
        Xl = fids @ phi
        v[f"lines{ls.d_eff} re/im"] = np.hstack([Xl.real, Xl.imag])
        v[f"lines{ls.d_eff} re/im+mag"] = np.hstack([Xl.real, Xl.imag, np.abs(Xl)])
    except Exception as e:  # noqa: BLE001
        print(f"line projection skipped: {type(e).__name__}: {e}")
    return v


def pooled_bootstrap(y_by_win, pq_by_win, pe_by_win, n_boot=2000, block=30, seed=0):
    """CI of pooled (mean over windows, targets) R² difference QRC - ESN, block
    bootstrap within each window's test block."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ds = []
        for y, pq, pes in zip(y_by_win, pq_by_win, pe_by_win):
            n = len(y)
            starts = rng.choice(np.arange(n - block + 1), n // block + 1)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            for t in range(y.shape[1]):
                rq = _r2(y[idx, t], pq[idx, t])
                re = np.mean([_r2(y[idx, t], pe[idx, t]) for pe in pes])
                ds.append(rq - re)
        diffs.append(np.nanmean(ds))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.array(diffs) > 0))


def predictions(X, w, a_best, standardize):
    """Test predictions per window at the chosen alphas: list of [n_test, t]."""
    out, ys = [], []
    for wi, name in enumerate(WIN_NAMES):
        w0, ntr, nte = WINDOWS[name]
        trn = np.arange(w0, w0 + ntr)
        te = np.arange(w0 + ntr, w0 + ntr + nte)
        A, B = X[trn], X[te]
        if standardize:
            mu, sd = A.mean(0), A.std(0) + 1e-12
            A = np.hstack([(A - mu) / sd, np.ones((len(A), 1))])
            B = np.hstack([(B - mu) / sd, np.ones((len(B), 1))])
        P = _dual_ridge(A, _targets(w, trn), B, ALPHA)
        out.append(np.stack([P[a_best[wi, t], :, t] for t in range(N_T)], axis=1))
        ys.append(_targets(w, te))
    return ys, out


def report(args):
    z = np.load(args.trace, allow_pickle=True)
    w = np.asarray(z["weather_norm"], float)
    esn = dict(np.load(args.esn, allow_pickle=True))
    variants = qrc_variants(np.asarray(z["fids"]), float(z["fid_dwell"]))
    for extra in args.extra or []:                    # extra feature files: name=path.npz (key X)
        name, path = extra.split("=", 1)
        variants[name] = np.asarray(np.load(path)["X"], float)

    rows = {}
    for name, X in variants.items():
        for std in (False, True):
            v, t = score_features(np.asarray(X, float), w, standardize=std)
            pv, tm, a_best = pick(v, t)
            rows[f"{name}{' [z]' if std else ''}"] = {"X": X, "std": std, "pooled_val": pv, "test": tm,
                                                       "a_best": a_best, "n_feat": int(X.shape[1])}
    best_name = max(rows, key=lambda k: rows[k]["pooled_val"])
    esn_all = select_esn(esn)
    best = rows[best_name]
    esn_matched = select_esn(esn, max_size=max(best["n_feat"], 300))

    def pooled(tm):
        tm = np.asarray(tm)
        return {"all": float(np.nanmean(tm)), "temp": float(np.nanmean(tm[:, 0::2])),
                "humidity": float(np.nanmean(tm[:, 1::2]))}

    # bootstrap: selected QRC variant vs selected ESN (all sizes, and matched)
    ys, pq = predictions(np.asarray(best["X"], float), w, best["a_best"], best["std"])
    out_cmp = {}
    for label, sel in (("esn_all", esn_all), ("esn_matched", esn_matched)):
        c = sel["cfg"]
        n_steps = max(a + b + cc for (a, b, cc) in WINDOWS.values())
        pe_inits = []
        for i in INITS:
            Xe = esn_states(w, n_steps, int(c["size"]), c["sr"], c["leak"], c["inscale"], i)
            pe_inits.append(predictions(Xe, w, sel["alpha_idx"], bool(c.get("std", 0)))[1])
        pe_by_win = [[pe_inits[i][wi] for i in range(len(INITS))] for wi in range(len(WIN_NAMES))]
        lo, hi, frac = pooled_bootstrap(ys, pq, pe_by_win)
        out_cmp[label] = {"cfg": c, "pooled_val": sel["pooled_val"], "pooled_test": pooled(sel["test"]),
                          "qrc_minus_esn_ci95": [lo, hi], "frac_qrc_better": frac}

    res = {"n_val": N_VAL, "selection_windows": [WIN_NAMES[i] for i in selection_windows()], "protocol": "select on pooled validation (4 windows x last n_val train days); alpha per "
                       "(window,target) on val; test read once", "windows": WINDOWS, "targets": TARGETS,
           "qrc_variants": {k: {"n_feat": r["n_feat"], "standardized": r["std"], "pooled_val": r["pooled_val"],
                                "pooled_test": pooled(r["test"]), "test_by_window": np.asarray(r["test"]).tolist()}
                            for k, r in rows.items()},
           "qrc_selected": best_name, "comparison": out_cmp}
    Path(args.out).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    print(f"\n{'variant':34s} {'n_feat':>6s} {'val':>7s} {'test(all)':>9s} {'temp':>7s} {'humid':>7s}")
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["pooled_val"]):
        p = pooled(r["test"])
        print(f"{k:34s} {r['n_feat']:6d} {r['pooled_val']:7.3f} {p['all']:9.3f} {p['temp']:7.3f} {p['humidity']:7.3f}"
              + ("   <- selected" if k == best_name else ""))
    for label, c in out_cmp.items():
        p = c["pooled_test"]
        print(f"{label:34s} {c['cfg']}  val {c['pooled_val']:.3f}  test {p['all']:.3f} "
              f"(temp {p['temp']:.3f}, humid {p['humidity']:.3f})  QRC-ESN 95% CI "
              f"[{c['qrc_minus_esn_ci95'][0]:+.3f}, {c['qrc_minus_esn_ci95'][1]:+.3f}]")
    print(f"wrote {args.out}")


def main():
    global N_VAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=150,
                    help="validation days at the end of each 600-day train block (fit on the rest)")
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("esn")
    e.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    r.add_argument("--esn", required=True)
    r.add_argument("--extra", nargs="*", help="name=path.npz feature files (key X, rows = days)")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    N_VAL = a.n_val                       # validation slice length (days, end of train block)
    if a.mode == "esn":
        w = np.asarray(np.load(a.trace, allow_pickle=True)["weather_norm"], float)
        run_esn(w, a.workers, a.out)
    else:
        report(a)


if __name__ == "__main__":
    main()
