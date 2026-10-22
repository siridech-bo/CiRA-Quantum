"""Pooled, seed-aware comparison of weather models (QRC vs classical).

Every model is a GROUP of replicates (QRC encoder seeds, ESN reservoir seeds).
Each replicate's feature matrix goes through the identical readout protocol of
qrc_weather_bench (standardize or not, ridge alpha per (window, target), both
chosen on validation; selection only on the leak-free windows), then test
predictions per window. Reported per group: pooled test R² (mean over windows
x horizons x variables), mean ± s.d. over replicates; and per comparison:
  * pooled block-bootstrap 95% CI of  mean_rep R²(A) - mean_rep R²(B)
    (block 30 days, resampled within each window's test block) — the
    significance test using all seeds together;
  * the same for the replicate ENSEMBLES (predictions averaged within a group).

  python scripts/qrc_weather_pool.py --out pool.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qrc_weather_bench import (  # noqa: E402
    INITS, N_T, TARGETS, WIN_NAMES, WINDOWS, pick, predictions, qrc_variants, score_features, select_esn,
)
from qrc_weather_fair import esn_states  # noqa: E402


def replicate_preds(X, w):
    """Choose standardization + alpha on validation; return per-window test preds."""
    best = None
    for std in (False, True):
        v, t = score_features(np.asarray(X, float), w, standardize=std)
        pv, _, a_best = pick(v, t)
        if best is None or pv > best[0]:
            best = (pv, std, a_best)
    ys, preds = predictions(np.asarray(X, float), w, best[2], best[1])
    return ys, preds, {"standardized": best[1], "pooled_val": best[0]}


def r2_rows(y, p):
    """R² per target for arrays y (n, T), p (..., n, T) -> (..., T)."""
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ((p - y) ** 2).sum(-2) / ss


def pooled_r2(ys, preds):                      # preds: list over windows of (n, T)
    return float(np.mean([r2_rows(y, p) for y, p in zip(ys, preds)]))


def bootstrap(ys, A, B, n_boot=2000, block=30, seed=0):
    """A, B: [rep][window] -> (n, T). CI of mean_rep pooled R²(A) - mean_rep pooled R²(B)."""
    rng = np.random.default_rng(seed)
    Aw = [np.stack([a[wi] for a in A]) for wi in range(len(ys))]           # per window (reps, n, T)
    Bw = [np.stack([b[wi] for b in B]) for wi in range(len(ys))]
    d = []
    for _ in range(n_boot):
        da = []
        for y, pa, pb in zip(ys, Aw, Bw):
            n = len(y)
            idx = np.concatenate([np.arange(s, s + block)
                                  for s in rng.choice(np.arange(n - block + 1), n // block + 1)])[:n]
            ra = r2_rows(y[idx], pa[:, idx]).mean(0)                           # mean over reps -> (T,)
            rb = r2_rows(y[idx], pb[:, idx]).mean(0)
            da.append(ra - rb)
        d.append(np.mean(da))
    d = np.asarray(d)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float(np.mean(d > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    ap.add_argument("--esn-grid", default="artifacts/qrc_runs/weather-bench-20261021/esn_grid_v300.npz")
    ap.add_argument("--learned", default="artifacts/qrc_runs/weather-learn-leakfree-20261021/seed{s}/X_learned.npz")
    ap.add_argument("--esn-learned", default="artifacts/qrc_runs/weather-esn-learn-20261022/X_esn_{v}_init{i}.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.trace, allow_pickle=True)
    w = np.asarray(z["weather_norm"], float)

    groups = {"qrc_arcsin": [qrc_variants(np.asarray(z["fids"]), float(z["fid_dwell"]))["mag653 (baseline)"]],
              "qrc_learned": [np.load(a.learned.format(s=s))["X"] for s in (0, 1, 2)]}
    cfg = select_esn(dict(np.load(a.esn_grid, allow_pickle=True)))["cfg"]
    n_steps = max(x + y + c for (x, y, c) in WINDOWS.values())
    groups["esn_tuned"] = [esn_states(w, n_steps, int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], i)
                           for i in INITS]
    for v in ("win", "mlp"):
        files = [a.esn_learned.format(v=v, i=i) for i in INITS]
        if all(Path(f).exists() for f in files):
            groups[f"esn_learned_{v}"] = [np.load(f)["X"] for f in files]

    res, preds, ys = {"esn_cfg": cfg, "groups": {}, "comparisons": {}}, {}, None
    for g, Xs in groups.items():
        reps = []
        for X in Xs:
            ys, p, info = replicate_preds(X, w)
            reps.append((p, info))
        preds[g] = [p for p, _ in reps]
        per_rep = [pooled_r2(ys, p) for p in preds[g]]
        ens = [np.mean([p[wi] for p in preds[g]], axis=0) for wi in range(len(ys))]
        res["groups"][g] = {"n_reps": len(reps), "pooled_test_r2": per_rep,
                            "mean": float(np.mean(per_rep)), "sd": float(np.std(per_rep)),
                            "ensemble": pooled_r2(ys, ens),
                            "temp": float(np.mean([r2_rows(y, np.mean([p[wi] for p in preds[g]], 0))[0::2].mean()
                                                   for wi, y in enumerate(ys)])),
                            "humidity": float(np.mean([r2_rows(y, np.mean([p[wi] for p in preds[g]], 0))[1::2].mean()
                                                       for wi, y in enumerate(ys)])),
                            "readout": [i for _, i in reps]}
        print(f"{g:18s} reps={len(reps)} pooled test R2 {np.mean(per_rep):.4f} ± {np.std(per_rep):.4f} "
              f"(per rep {', '.join(f'{x:.3f}' for x in per_rep)}); ensemble {res['groups'][g]['ensemble']:.4f}",
              flush=True)

    for b in [g for g in groups if g != "qrc_learned"]:
        ci, frac = bootstrap(ys, preds["qrc_learned"], preds[b])
        ens_a = [[np.mean([p[wi] for p in preds["qrc_learned"]], 0) for wi in range(len(ys))]]
        ens_b = [[np.mean([p[wi] for p in preds[b]], 0) for wi in range(len(ys))]]
        ci_e, frac_e = bootstrap(ys, ens_a, ens_b)
        res["comparisons"][f"qrc_learned - {b}"] = {"pooled_ci95": ci, "frac_positive": frac,
                                                    "ensemble_ci95": ci_e, "ensemble_frac_positive": frac_e}
        sig = "SIGNIFICANT" if ci[0] > 0 else ("worse" if ci[1] < 0 else "n.s.")
        print(f"qrc_learned - {b:16s} pooled 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] P(>0)={frac:.3f} {sig} | "
              f"ensembles [{ci_e[0]:+.4f}, {ci_e[1]:+.4f}] P(>0)={frac_e:.3f}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
