"""Why does the QRC win / stay comparable / lose against the tuned ESN and LSTM?

Offline analysis of the real-world benchmark (no retraining of the benchmark
models): for each dataset, characteristics of the DATA and of each model's
READOUT FEATURES are measured and set against the outcome, at two levels:
  * dataset level (9 points): outcome category vs each characteristic
  * target level (~90 points: variable x horizon): R² differences vs horizon and
    per-target predictability, from the saved test predictions.

Characteristics
  n_train            training length (steps)
  predictability     best pooled test R² among all models (signal vs noise)
  linear_r2          tuned linear autoregressive baseline (ridge on lagged inputs,
                     lag count + alpha chosen on validation) — added as a model
  nonlin_gain        best nonlinear model R² - linear AR R²
  shift_sd           max over inputs of |mean_test - mean_train| / sd_train
  qrc_ood / esn_ood  fraction of test readout features beyond 4 sd of their
                     training distribution (QRC arcsin FFT features; tuned-ESN states)
  memory             input decorrelation time (first lag with ACF < 1/e), steps
  spectral_conc      fraction of input power in the top 1% of frequencies

Outcome: comparable (95% CI of learned QRC - model contains 0), better, worse.

  python backend/scripts/qrc_realworld_factors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qrc_realworld_bench import (  # noqa: E402
    esn_states, load_dataset, make_targets, select_esn, target_names,
)
from qrc_era5_weather import r2_cols, ridge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/qrc_runs/realworld"
DATA = ROOT.parent / "docs/QRC/data"
ORDER = ["kaggle_delhi", "era5_delhi", "era5_mumbai", "era5_bangkok", "era5_beijing", "era5_berlin",
         "beijing_pm25", "etth1", "santafe"]
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)


def lagged(w, L):
    """Rows k: [w[k], w[k-1], ..., w[k-L+1]] (zero before the start)."""
    n, d = w.shape
    X = np.zeros((n, L * d))
    for l in range(L):
        X[l:, l * d:(l + 1) * d] = w[:n - l]
    return X


def linear_ar(ds):
    """Tuned linear AR: lag count and alpha chosen on VAL, refit TRAIN+VAL."""
    sp, w = ds["split"], ds["w"]
    Yt, Yv = make_targets(ds, sp["train"]), make_targets(ds, sp["val"])
    best = None
    for L in (1, 3, 7, 14, 30, 60, 120):
        X = lagged(w, L)
        P = ridge(X[sp["train"]], Yt, [X[sp["val"]]], ALPHAS, True)
        r = np.array([r2_cols(Yv, p[0]) for p in P])                # [alpha, t]
        score = float(r.max(0).mean())
        if best is None or score > best[0]:
            best = (score, L, r.argmax(0))
    _, L, a_idx = best
    X = lagged(w, L)
    fit = np.concatenate([sp["train"], sp["val"]])
    P = ridge(X[fit], make_targets(ds, fit), [X[sp["test"]]], ALPHAS, True)
    pred = np.stack([P[a_idx[t]][0][:, t] for t in range(len(ds["targets"]))], axis=1)
    return pred, L


def ood_fraction(X, sp, thr=4.0):
    mu, sd = X[sp["train"]].mean(0), X[sp["train"]].std(0) + 1e-12
    z = np.abs((X[sp["test"]] - mu) / sd)
    return float((z > thr).mean()), float(np.median(z))


def decorrelation(x):
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[:len(x)]
    ac /= ac[0]
    below = np.where(ac < np.exp(-1))[0]
    return int(below[0]) if len(below) else len(x)


def spectral_concentration(x):
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    p = np.sort(p[1:])[::-1]
    k = max(1, int(0.01 * len(p)))
    return float(p[:k].sum() / p.sum())


def outcome(ci):
    return "better" if ci[0] > 0 else ("worse" if ci[1] < 0 else "comparable")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows, tgt_rows = [], []
    for name in ORDER:
        ds = load_dataset(name)
        sp, w = ds["split"], ds["w"]
        run = RUNS / name
        ev = json.loads((DATA / f"qrc_realworld_{name}.json").read_text(encoding="utf-8"))
        z = np.load(run / "predictions.npz")
        Y = z["Y_test"]
        r2 = {g[5:]: r2_cols(Y, z[g]).mean(0) if z[g].ndim == 3 else r2_cols(Y, z[g])
              for g in z.files if g.startswith("pred_")}                 # per-target R², mean over reps
        lin_pred, L = linear_ar(ds)
        r2["linear_ar"] = r2_cols(Y, lin_pred)
        pooled = {k: float(v.mean()) for k, v in r2.items()}
        best_nonlin = max(pooled[k] for k in ("qrc_learned", "esn_tuned", "lstm_tuned"))
        cfg, _ = select_esn(run / "esn_grid.npz")
        Xe = esn_states(w, int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], 0)
        q_ood, q_med = ood_fraction(np.load(run / "X_arcsin.npz")["X"], sp)
        e_ood, e_med = ood_fraction(Xe, sp)
        tr = sp["train"]
        shift = max(abs(w[sp["test"], j].mean() - w[tr, j].mean()) / (w[tr, j].std() + 1e-12) for j in range(w.shape[1]))
        rows.append({
            "dataset": name, "n_train": int(len(tr)), "n_test": int(len(sp["test"])),
            "vs_esn": outcome(ev["comparisons"]["qrc_learned - esn_tuned"]["ci95"]),
            "vs_lstm": outcome(ev["comparisons"]["qrc_learned - lstm_tuned"]["ci95"]),
            "d_esn": pooled["qrc_learned"] - pooled["esn_tuned"], "d_lstm": pooled["qrc_learned"] - pooled["lstm_tuned"],
            "r2": pooled, "predictability": max(pooled.values()), "linear_ar_lags": L,
            "nonlin_gain": best_nonlin - pooled["linear_ar"], "shift_sd": float(shift),
            "qrc_ood": q_ood, "qrc_median_z": q_med, "esn_ood": e_ood, "esn_median_z": e_med,
            "memory": int(np.median([decorrelation(w[tr, j]) for j in range(w.shape[1])])),
            "spectral_conc": float(np.mean([spectral_concentration(w[tr, j]) for j in range(w.shape[1])])),
        })
        hmax = max(h for _, h in ds["targets"])
        for t, (v, h) in enumerate(ds["targets"]):
            tgt_rows.append({"dataset": name, "target": target_names(ds)[t], "h_frac": h / hmax,
                             "best_r2": float(max(r2[k][t] for k in r2)),
                             "d_esn": float(r2["qrc_learned"][t] - r2["esn_tuned"][t]),
                             "d_lstm": float(r2["qrc_learned"][t] - r2["lstm_tuned"][t]),
                             "d_lin": float(r2["qrc_learned"][t] - r2["linear_ar"][t])})
        print(f"{name:13s} vs ESN {rows[-1]['vs_esn']:10s} vs LSTM {rows[-1]['vs_lstm']:10s} "
              f"pred {rows[-1]['predictability']:.3f} lin {pooled['linear_ar']:.3f} nonlin_gain {rows[-1]['nonlin_gain']:+.3f} "
              f"shift {shift:.2f} qrc_ood {q_ood:.3f} esn_ood {e_ood:.3f} mem {rows[-1]['memory']} "
              f"spec {rows[-1]['spectral_conc']:.2f} n_train {len(tr)}", flush=True)

    keys = ["n_train", "predictability", "nonlin_gain", "shift_sd", "qrc_ood", "memory", "spectral_conc"]
    corr = {}
    for target in ("d_esn", "d_lstm"):
        y = [r[target] for r in rows if r["dataset"] != "etth1"]           # ETTh1 is an outlier failure
        corr[target] = {k: spearman([r[k] for r in rows if r["dataset"] != "etth1"], y) for k in keys}
    T = [r for r in tgt_rows if r["dataset"] != "etth1"]
    tcorr = {target: {k: spearman([r[k] for r in T], [r[target] for r in T]) for k in ("h_frac", "best_r2")}
             for target in ("d_esn", "d_lstm", "d_lin")}
    out = {"datasets": rows, "targets": tgt_rows, "spearman_dataset_level_excl_etth1": corr,
           "spearman_target_level_excl_etth1": tcorr}
    (DATA / "qrc_realworld_factors.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nSpearman (dataset level, n=8, excl. ETTh1):")
    for t, c in corr.items():
        print(f"  {t}: " + ", ".join(f"{k} {v:+.2f}" for k, v in c.items()))
    print(f"Spearman (target level, n={len(T)}, excl. ETTh1):")
    for t, c in tcorr.items():
        print(f"  {t}: " + ", ".join(f"{k} {v:+.2f}" for k, v in c.items()))


if __name__ == "__main__":
    main()
