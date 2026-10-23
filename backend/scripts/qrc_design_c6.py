"""Design step 2 — C6 bounded readout features (pre-registered, QRC_Design_Concepts.md §7,
commit 53d8e6a).

Transforms of the QRC readout features (653 FFT magnitudes), fitted on the TRAIN block
only: raw | clipz (z clipped to ±3) | tanhz (tanh(z/3)) | rank (training-CDF quantile).

  dev      freeze rule — mean pooled VALIDATION R² over the 8 development datasets at the
           frozen gain g = 0.35; adopt the best bounded transform unless it is below raw by
           more than 0.005. Writes the frozen choice.
  posthoc  (labelled post-hoc; test sets already seen) raw vs frozen transform on the 9
           development datasets' test sets — criterion R2.
  confirm  untouched ETTh2: QRC g=1 raw, g=0.35 raw, g=0.35 + frozen transform, tuned ESN
           (transform choice on VAL), tuned LSTM, tuned linear AR — criteria R1, R3.

  python scripts/qrc_design_c6.py dev
  python scripts/qrc_design_c6.py posthoc
  python scripts/qrc_design_c6.py confirm --dataset etth2 --run-dir DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.qrc.provenance import provenance  # noqa: E402
from qrc_era5_weather import r2_cols  # noqa: E402
from qrc_realworld_bench import (  # noqa: E402
    INITS, esn_states, load_dataset, make_targets, select_esn, test_preds, val_scores,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/qrc_runs/realworld"
DATA = ROOT.parent / "docs/QRC/data"
DEV = ["era5_delhi", "era5_mumbai", "era5_bangkok", "era5_beijing", "era5_berlin", "beijing_pm25", "etth1", "santafe"]
BOUNDED = ("clipz", "tanhz", "rank")
TRANSFORMS = ("raw",) + BOUNDED
FROZEN_FILE = DATA / "qrc_design_c6_frozen.json"


def transform(name, X, train_idx):
    if name == "raw":
        return X
    A = X[train_idx]
    if name in ("clipz", "tanhz"):
        z = (X - A.mean(0)) / (A.std(0) + 1e-12)
        return np.clip(z, -3, 3) if name == "clipz" else np.tanh(z / 3)
    if name == "rank":
        S = np.sort(A, axis=0)
        return np.stack([np.searchsorted(S[:, j], X[:, j], side="right") / len(S) for j in range(X.shape[1])], axis=1)
    raise KeyError(name)


def val_pooled(X, ds):
    vs = val_scores(X, ds)
    return float(max(np.nanmean(vs[s].max(0)) for s in (0, 1)))


def test_r2(X, ds):
    Y = make_targets(ds, ds["split"]["test"])
    P = test_preds(X, ds)
    return P, float(r2_cols(Y, P).mean())


def boot(Y, A, B):
    rng = np.random.default_rng(0)
    n, block = len(Y), 30
    d = []
    for _ in range(2000):
        i = np.concatenate([np.arange(s, s + block) for s in rng.choice(n - block + 1, n // block + 1)])[:n]
        d.append(np.mean([r2_cols(Y[i], a[i]).mean() for a in A]) - np.mean([r2_cols(Y[i], b[i]).mean() for b in B]))
    point = float(np.mean([r2_cols(Y, a).mean() for a in A]) - np.mean([r2_cols(Y, b).mean() for b in B]))
    return {"point": point, "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "frac_positive": float((np.array(d) > 0).mean())}


def cmd_dev(_):
    res = {"step": "C6 dev (validation only)", "gain": 0.35, "datasets": {}}
    for name in DEV:
        ds = load_dataset(name)
        X = np.load(RUNS / name / "design_c1" / "g0.35" / "X.npz")["X"]
        res["datasets"][name] = {t: val_pooled(transform(t, X, ds["split"]["train"]), ds) for t in TRANSFORMS}
        print(f"{name:13s} " + "  ".join(f"{t} {v:.4f}" for t, v in res["datasets"][name].items()), flush=True)
    mean = {t: float(np.mean([res["datasets"][n][t] for n in DEV])) for t in TRANSFORMS}
    best = max(BOUNDED, key=lambda t: mean[t])
    adopted = best if mean[best] >= mean["raw"] - 0.005 else "raw"
    res.update({"mean_val": mean, "best_bounded": best, "frozen": adopted,
                "rule": "best bounded transform by mean pooled VAL R2 over 8 dev datasets, unless < raw - 0.005",
                "provenance": provenance(step="C6 dev")})
    FROZEN_FILE.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("mean val: " + "  ".join(f"{t} {v:.4f}" for t, v in mean.items()))
    print(f"FROZEN transform: {adopted}")


def cmd_posthoc(_):
    fz = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))["frozen"]
    res = {"step": "C6 post-hoc on development test sets (already seen; labelled post-hoc)", "frozen": fz,
           "gain": 0.35, "datasets": {}}
    for name in DEV + ["kaggle_delhi"]:
        ds = load_dataset(name)
        Y = make_targets(ds, ds["split"]["test"])
        X = np.load(RUNS / name / "design_c1" / "g0.35" / "X.npz")["X"]
        Pr, r_raw = test_r2(X, ds)
        Pt, r_t = test_r2(transform(fz, X, ds["split"]["train"]), ds)
        b = boot(Y, Pt[None], Pr[None])
        res["datasets"][name] = {"raw": r_raw, fz: r_t, "diff": b}
        print(f"{name:13s} raw {r_raw:+.4f}  {fz} {r_t:+.4f}  diff {b['point']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]",
              flush=True)
    neg = [n for n, v in res["datasets"].items() if v[fz] < 0]
    loss = [n for n, v in res["datasets"].items() if v["diff"]["point"] < -0.005]
    res["R2_pass"] = not neg and not loss
    res["R2_detail"] = {"negative_r2": neg, "loss_gt_0.005": loss}
    res["provenance"] = provenance(step="C6 posthoc")
    (DATA / "qrc_design_c6_posthoc.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"R2 pass: {res['R2_pass']}  {res['R2_detail']}")


def cmd_confirm(a):
    fz = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))["frozen"]
    ds = load_dataset(a.dataset)
    run = Path(a.run_dir)
    sp = ds["split"]
    Y = make_targets(ds, sp["test"])
    X1 = np.load(run / "design_c1" / "g1" / "X.npz")["X"]
    X035 = np.load(run / "design_c1" / "g0.35" / "X.npz")["X"]
    preds = {"qrc_g1_raw": test_preds(X1, ds)[None], "qrc_g035_raw": test_preds(X035, ds)[None],
             f"qrc_g035_{fz}": test_preds(transform(fz, X035, sp["train"]), ds)[None]}
    cfg, _ = select_esn(run / "esn_grid.npz")
    esn_states_list = [esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], i) for i in INITS]
    esn_choice = max(("raw", fz), key=lambda t: np.mean([val_pooled(transform(t, Xe, sp["train"]), ds)
                                                       for Xe in esn_states_list]))
    preds["esn_tuned"] = np.stack([test_preds(transform(esn_choice, Xe, sp["train"]), ds) for Xe in esn_states_list])
    preds["lstm_tuned"] = np.load(run / "lstm.npz")["preds"]
    from qrc_realworld_factors import linear_ar
    preds["linear_ar"] = linear_ar(ds)[0][None]
    res = {"dataset": a.dataset, "preregistration_commit": "53d8e6a", "frozen_transform": fz,
           "esn_transform_choice": esn_choice, "esn_cfg": cfg, "n_test": int(len(Y)), "groups": {}, "tests": {}}
    for g, P in preds.items():
        per = [float(r2_cols(Y, p).mean()) for p in P]
        res["groups"][g] = {"pooled": float(np.mean(per)), "sd": float(np.std(per))}
    ref = f"qrc_g035_{fz}"
    for b in [g for g in preds if g != ref]:
        res["tests"][f"{ref} - {b}"] = boot(Y, preds[ref], preds[b])
    r_t, r_raw = res["groups"][ref]["pooled"], res["groups"]["qrc_g035_raw"]["pooled"]
    d = res["tests"][f"{ref} - qrc_g035_raw"]
    res["R1_pass"] = bool(r_t >= 0 and (d["ci95"][0] > 0 if r_raw < 0 else d["ci95"][0] > -0.005))
    res["provenance"] = provenance(dataset=a.dataset, frozen=fz)
    np.savez(run / "c6_predictions.npz", Y_test=Y, **{f"pred_{k}": v for k, v in preds.items()})
    (run / "c6_confirm.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for g, v in res["groups"].items():
        print(f"  {g:22s} pooled test R2 {v['pooled']:+.4f}")
    for k, v in res["tests"].items():
        print(f"  {k:40s} {v['point']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print(f"  ESN transform chosen on VAL: {esn_choice}   R1 pass: {res['R1_pass']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "posthoc", "confirm"])
    ap.add_argument("--dataset", default="etth2")
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()
    {"dev": cmd_dev, "posthoc": cmd_posthoc, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
