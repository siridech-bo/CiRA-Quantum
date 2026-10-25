"""Design step 5 — readout-side variants R1-R5 (pre-registered, QRC_Design_Concepts.md §11,
commit a7cb3e6). Offline on the saved frozen-design features (g = 0.35, tau = 30 ms, 653 bins);
every variant is applied identically to the validation-selected tuned ESN (3 inits).

  dev      VALIDATION-only scores for every variant, QRC and ESN, on one dataset
  freeze   aggregate the 8 development datasets, apply the freeze rule (P1), write the decision
  posthoc  POST-HOC test-set drift check (ETTh1 / ETTh2, test sets already seen), labelled

Variants: base | delta (last-value re-centring) | lags (raw input lags appended) | ar_residual
(ridge on the tuned linear AR's residual) | rff (RBF random-Fourier-feature readout) | delay
(PCA32 of past features at t-7/30/90 appended — adds classical memory, SOP §6) | delta_lags.

  python scripts/qrc_design_r.py dev --dataset D --run-root R
  python scripts/qrc_design_r.py freeze
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.qrc.provenance import provenance  # noqa: E402
from qrc_era5_weather import ALPHA, r2_cols, ridge  # noqa: E402
from qrc_realworld_bench import INITS, esn_states, load_dataset, make_targets, select_esn  # noqa: E402
from qrc_realworld_factors import lagged  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "docs/QRC/data"
DEV = ["era5_delhi", "era5_mumbai", "era5_bangkok", "era5_beijing", "era5_berlin", "beijing_pm25", "etth1", "santafe"]
VARIANTS = ("base", "delta", "lags", "ar_residual", "rff", "delay", "delta_lags")
AR_LAGS = (1, 3, 7, 14, 30, 60, 120)
LAGS, LAG_SCALE = (1, 7, 30), (1.0, 3.0, 10.0)
RFF_M, RFF_GAMMA = 2000, (0.3, 1.0, 3.0)
DELAYS, DELAY_PCS = (7, 30, 90), 32


# ---------------------------------------------------------------- readout protocol (as the benchmark)

def _std(X, idx):
    mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-12
    return (X - mu) / sd


def offsets(ds, idx):
    """w_v(t) for every target (the last observed value of the target's variable)."""
    return np.stack([ds["w"][idx, v] for (v, _h) in ds["targets"]], axis=1)


def fit_predict(X, ds, fit, pred, off_fit, off_pred):
    """ridge over ALPHA x std; returns [std][alpha] -> prediction on `pred` (original scale)."""
    Y = make_targets(ds, fit) - off_fit
    out = []
    for std in (False, True):
        P = ridge(X[fit], Y, [X[pred]], ALPHA, std)
        out.append([p[0] + off_pred for p in P])
    return out


def val_pooled(X, ds, off=None):
    """pooled VAL R2 with the benchmark's selection (std global, alpha per target)."""
    sp = ds["split"]
    o_t = off(sp["train"], "val") if off else 0.0
    o_v = off(sp["val"], "val") if off else 0.0
    Yv = make_targets(ds, sp["val"])
    P = fit_predict(X, ds, sp["train"], sp["val"], o_t, o_v)
    return float(max(np.nanmean(np.max([r2_cols(Yv, p) for p in P[s]], axis=0)) for s in (0, 1)))


# ---------------------------------------------------------------- tuned linear AR (as qrc_realworld_factors)

class AR:
    """lag count L and alpha per target chosen on VAL; predictions for a fit/predict split."""

    def __init__(self, ds):
        sp, self.ds = ds["split"], ds
        Yt, Yv = make_targets(ds, sp["train"]), make_targets(ds, sp["val"])
        best = None
        for L in AR_LAGS:
            X = lagged(ds["w"], L)
            P = ridge(X[sp["train"]], Yt, [X[sp["val"]]], ALPHA, True)
            r = np.array([r2_cols(Yv, p[0]) for p in P])
            if best is None or r.max(0).mean() > best[0]:
                best = (float(r.max(0).mean()), L, r.argmax(0))
        _, self.L, self.a_idx = best
        self.X = lagged(ds["w"], self.L)

    def predict(self, fit, idx):
        P = ridge(self.X[fit], make_targets(self.ds, fit), [self.X[idx]], ALPHA, True)
        return np.stack([P[self.a_idx[t]][0][:, t] for t in range(len(self.ds["targets"]))], axis=1)


# ---------------------------------------------------------------- variants

def rff(F, tr, gamma):
    Z = _std(F, tr)
    rng = np.random.default_rng(0)
    g = gamma / Z.shape[1]
    Om = rng.normal(0.0, np.sqrt(2 * g), (Z.shape[1], RFF_M))
    b = rng.uniform(0, 2 * np.pi, RFF_M)
    return np.sqrt(2.0 / RFF_M) * np.cos(Z @ Om + b)


def delay(F, tr):
    Z = _std(F, tr)
    _, _, Vt = np.linalg.svd(Z[tr], full_matrices=False)
    P = Z @ Vt[:DELAY_PCS].T
    parts = [F]
    for k in DELAYS:
        S = np.zeros_like(P)
        S[k:] = P[:-k]
        parts.append(S)
    return np.hstack(parts)


def with_lags(F, ds, L, s):
    tr = ds["split"]["train"]
    return np.hstack([_std(F, tr), s * _std(lagged(ds["w"], L), tr)])


def candidates(name, F, ds, ar, test=False):
    """[(hp, X, offset_fn)] for a variant; offset_fn(idx, mode) -> per-target offset."""
    tr = ds["split"]["train"]
    delta = lambda idx, _m: offsets(ds, idx)  # noqa: E731
    cands = []
    if name == "base":
        cands = [(None, F, None)]
    elif name == "delta":
        cands = [(None, F, delta)]
    elif name in ("lags", "delta_lags"):
        cands = [((L, s), with_lags(F, ds, L, s), delta if name == "delta_lags" else None)
                 for L, s in itertools.product(LAGS, LAG_SCALE)]
    elif name == "ar_residual":
        sp = ds["split"]
        a_tr, a_va = ar.predict(sp["train"], sp["train"]), ar.predict(sp["train"], sp["val"])
        cache = {id(sp["train"]): a_tr, id(sp["val"]): a_va}
        cands = [(None, F, lambda idx, _m: cache[id(idx)])]
        if test:
            fit = np.concatenate([sp["train"], sp["val"]])
            cache.update({id(fit): ar.predict(fit, fit), id(sp["test"]): ar.predict(fit, sp["test"])})
            cands = [(None, F, lambda idx, _m: cache[id(idx)], fit)]
    elif name == "rff":
        cands = [(g, rff(F, tr, g), None) for g in RFF_GAMMA]
    elif name == "delay":
        cands = [(None, delay(F, tr), None)]
    return cands


def variant_score(name, F, ds, ar):
    """max over the variant's hyper-parameters of pooled VAL R2; returns (score, best_hp)."""
    best = None
    cands = candidates(name, F, ds, ar)
    for hp, X, off in cands:
        sc = val_pooled(X, ds, off)
        if best is None or sc > best[0]:
            best = (sc, hp)
    return best


def test_pooled(name, F, ds, ar):
    """hp, std and alpha (per target) on VAL; refit TRAIN+VAL; pooled TEST R2."""
    sp = ds["split"]
    cands = candidates(name, F, ds, ar, test=True)
    fit = cands[0][3] if len(cands[0]) > 3 else np.concatenate([sp["train"], sp["val"]])
    Yv, Yte = make_targets(ds, sp["val"]), make_targets(ds, sp["test"])
    best = None
    for c in (candidates(name, F, ds, ar) if name == "ar_residual" else cands):
        hp, X, off = c[:3]
        o_t, o_v = (off(sp["train"], "val"), off(sp["val"], "val")) if off else (0.0, 0.0)
        P = fit_predict(X, ds, sp["train"], sp["val"], o_t, o_v)
        r = np.array([[r2_cols(Yv, p) for p in P[s]] for s in (0, 1)])         # [std, alpha, t]
        s = int(np.argmax(np.nanmean(r.max(1), axis=1)))
        sc = float(np.nanmean(r[s].max(0)))
        if best is None or sc > best[0]:
            best = (sc, hp, s, r[s].argmax(0))
    _, hp, s, a_idx = best
    hp_c = [c for c in cands if c[0] == hp][0]
    X, off = hp_c[1], hp_c[2]
    o_f, o_te = (off(fit, "test"), off(sp["test"], "test")) if off else (0.0, 0.0)
    P = fit_predict(X, ds, fit, sp["test"], o_f, o_te)[s]
    pred = np.stack([P[a_idx[k]][:, k] for k in range(len(ds["targets"]))], axis=1)
    return float(r2_cols(Yte, pred).mean()), hp


def cmd_posthoc(a, ds):
    """POST-HOC (test sets already seen): drift check on ETTh1 / ETTh2, labelled as such."""
    root = Path(a.run_root)
    ar = AR(ds)
    cfg, _ = select_esn(root / "esn_grid.npz")
    models = {"qrc": [feats_qrc(root)],
              "esn": [esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], i) for i in INITS]}
    res = {"dataset": a.dataset, "label": "POST-HOC: test set already seen; not confirmatory", "test": {}}
    for m, Fs in models.items():
        res["test"][m] = {}
        for v in VARIANTS:
            out = [test_pooled(v, F, ds, ar) for F in Fs]
            res["test"][m][v] = float(np.mean([o[0] for o in out]))
            print(f"{a.dataset:8s} {m} {v:12s} TEST R2 {res['test'][m][v]:+.4f}", flush=True)
    res["provenance"] = provenance(dataset=a.dataset, step="R1-R5 post-hoc drift check")
    (DATA / f"qrc_design_r_posthoc_{a.dataset}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


def feats_qrc(root):
    return np.load(Path(root) / "design_c1" / "g0.35" / "X.npz")["X"]


def cmd_dev(a, ds):
    root = Path(a.run_root)
    ar = AR(ds)
    cfg, _ = select_esn(root / "esn_grid.npz")
    models = {"qrc": [feats_qrc(root)]}
    models["esn"] = [esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], i) for i in INITS]
    res = {"dataset": a.dataset, "gain": 0.35, "readout": "fid bins653 (D=653) + variant", "esn_cfg": cfg,
           "ar_lags": ar.L, "scores": {}, "hp": {}}
    for m, Fs in models.items():
        res["scores"][m], res["hp"][m] = {}, {}
        for v in VARIANTS:
            t0 = time.time()
            out = [variant_score(v, F, ds, ar) for F in Fs]
            res["scores"][m][v] = float(np.mean([o[0] for o in out]))
            res["hp"][m][v] = [o[1] for o in out]
            print(f"{a.dataset:13s} {m} {v:12s} val R2 {res['scores'][m][v]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    res["provenance"] = provenance(dataset=a.dataset, step="R1-R5 dev (validation only)")
    out = DATA / f"qrc_design_r_dev_{a.dataset}.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")


def cmd_freeze(_a, _ds=None):
    S = {n: json.loads((DATA / f"qrc_design_r_dev_{n}.json").read_text(encoding="utf-8"))["scores"] for n in DEV}
    q = {v: [S[n]["qrc"][v] for n in DEV] for v in VARIANTS}
    e = {v: [S[n]["esn"][v] for n in DEV] for v in VARIANTS}
    mean = {v: float(np.mean(q[v])) for v in VARIANTS}
    worse = {v: int(sum(x < b - 0.005 for x, b in zip(q[v], q["base"]))) for v in VARIANTS}
    ok = [v for v in VARIANTS if v != "base" and mean[v] >= mean["base"] + 0.005 and worse[v] <= 2]
    adopted = max(ok, key=mean.get) if ok else "base"
    gap = {v: {n: S[n]["qrc"][v] - S[n]["esn"][v] for n in DEV} for v in VARIANTS}
    res = {"rule": "P1: best mean QRC VAL R2 over 8 dev datasets; adopt if >= base + 0.005 and worse than base "
                   "by > 0.005 on <= 2 datasets (QRC_Design_Concepts.md §11, a7cb3e6)",
           "mean_val_qrc": mean, "mean_val_esn": {v: float(np.mean(e[v])) for v in VARIANTS},
           "datasets_worse_than_base": worse, "eligible": ok, "adopted": adopted,
           "P2_qrc_minus_esn": gap, "provenance": provenance(step="R1-R5 freeze")}
    (DATA / "qrc_design_r_frozen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{'variant':12s} {'QRC mean':>9s} {'ESN mean':>9s} {'worse':>5s}  " + " ".join(f"{n[:8]:>8s}" for n in DEV))
    for v in VARIANTS:
        print(f"{v:12s} {mean[v]:9.4f} {res['mean_val_esn'][v]:9.4f} {worse[v]:5d}  "
              + " ".join(f"{x:8.4f}" for x in q[v]))
    print(f"ADOPTED: {adopted}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "freeze", "posthoc"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--run-root", default=None)
    a = ap.parse_args()
    if a.cmd == "freeze":
        return cmd_freeze(a)
    (cmd_posthoc if a.cmd == "posthoc" else cmd_dev)(a, load_dataset(a.dataset))


if __name__ == "__main__":
    main()
