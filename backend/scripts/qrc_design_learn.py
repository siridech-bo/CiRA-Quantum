"""Design step 4 — learned encoding on top of the frozen gain g = 0.35 (pre-registered,
QRC_Design_Concepts.md §9).

Arm: theta = 0.35*arcsin(sqrt s) + 0.5*tanh(MLP(s)) (zero-initialised, so training starts
exactly at the current design), trained by qrc_realworld_bench.py learn --gain 0.35 on a
train-period slice only; 3 seeds; readout = the g = 0.35 pass's 653 bins.

  dev      VALIDATION-only scores on one dataset: g0.35 fixed, learned g0.35 (per seed),
           learned g1 (existing, descriptive)
  freeze   aggregate the 8 development datasets, apply the freeze rule, write the decision
  confirm  untouched city: learned g0.35 vs g0.35 fixed, g1, tuned ESN / LSTM / linear AR

  python scripts/qrc_design_learn.py dev --dataset D --run-root R
  python scripts/qrc_design_learn.py freeze
  python scripts/qrc_design_learn.py confirm --dataset D --run-root R
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
DATA = ROOT.parent / "docs/QRC/data"
SEEDS = (0, 1, 2)
DEV = ["era5_delhi", "era5_mumbai", "era5_bangkok", "era5_beijing", "era5_berlin", "beijing_pm25", "etth1", "santafe"]


def val_pooled(X, ds):
    vs = val_scores(X, ds)
    return float(max(np.nanmean(vs[s].max(0)) for s in (0, 1)))


def learned_X(root, seed, gain="0.35"):
    p = Path(root) / (f"design_learn/g{gain}_seed{seed}" if gain != "1" else f"learned_seed{seed}")
    return np.load(p / "X_learned.npz")["X"]


def cmd_dev(a, ds):
    root = Path(a.run_root)
    res = {"dataset": a.dataset, "scores": {"g0.35_fixed": val_pooled(np.load(root / "design_c1/g0.35/X.npz")["X"], ds)}}
    res["scores"]["learned_g0.35"] = [val_pooled(learned_X(root, s), ds) for s in SEEDS]
    if all((root / f"learned_seed{s}/X_learned.npz").exists() for s in SEEDS):
        res["scores"]["learned_g1"] = [val_pooled(learned_X(root, s, "1"), ds) for s in SEEDS]
    res["inner_val_nmse"] = {s: json.loads(str(np.load(root / f"design_learn/g0.35_seed{s}/trace_learned.npz")["meta"]))
                             for s in SEEDS}
    res["provenance"] = provenance(dataset=a.dataset, step="learned encoding at g=0.35, dev (validation only)")
    (root / "learn_dev.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    sc = res["scores"]
    print(f"{a.dataset:13s} g0.35 {sc['g0.35_fixed']:.4f}  learned g0.35 {np.mean(sc['learned_g0.35']):.4f} "
          f"{['%.4f' % v for v in sc['learned_g0.35']]}"
          + (f"  learned g1 {np.mean(sc['learned_g1']):.4f}" if "learned_g1" in sc else ""), flush=True)


def cmd_freeze(_a, _ds=None):
    S = {n: json.loads((DATA / f"qrc_design_learn_dev_{n}.json").read_text(encoding="utf-8"))["scores"] for n in DEV}
    fixed = {n: S[n]["g0.35_fixed"] for n in DEV}
    lrn = {n: float(np.mean(S[n]["learned_g0.35"])) for n in DEV}
    m_fixed, m_lrn = float(np.mean(list(fixed.values()))), float(np.mean(list(lrn.values())))
    adopted = m_lrn >= m_fixed + 0.005
    res = {"rule": "adopt learned g=0.35 (seed-mean) if mean VAL R2 over 8 dev datasets >= g0.35 fixed + 0.005",
           "per_dataset": {n: {"g0.35_fixed": fixed[n], "learned_g0.35": lrn[n]} for n in DEV},
           "mean_fixed": m_fixed, "mean_learned": m_lrn, "wins": sum(lrn[n] > fixed[n] for n in DEV),
           "adopted": bool(adopted), "provenance": provenance(step="learned g=0.35 freeze")}
    if all("learned_g1" in S[n] for n in DEV):
        res["mean_learned_g1"] = float(np.mean([np.mean(S[n]["learned_g1"]) for n in DEV]))
    (DATA / "qrc_design_learn_frozen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for n in DEV:
        print(f"  {n:13s} fixed {fixed[n]:.4f}  learned {lrn[n]:.4f}  {lrn[n] - fixed[n]:+.4f}")
    print(f"mean fixed {m_fixed:.4f} learned {m_lrn:.4f} (wins {res['wins']}/8) -> ADOPTED {adopted}")


def boot(Y, A, B):
    rng = np.random.default_rng(0)
    n, block = len(Y), 30
    d = []
    for _ in range(2000):
        i = np.concatenate([np.arange(s, s + block) for s in rng.choice(n - block + 1, n // block + 1)])[:n]
        d.append(np.mean([r2_cols(Y[i], x[i]).mean() for x in A]) - np.mean([r2_cols(Y[i], x[i]).mean() for x in B]))
    pt = float(np.mean([r2_cols(Y, x).mean() for x in A]) - np.mean([r2_cols(Y, x).mean() for x in B]))
    return {"point": pt, "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "frac_positive": float((np.array(d) > 0).mean())}


def cmd_confirm(a, ds):
    root = Path(a.run_root)
    sp = ds["split"]
    Y = make_targets(ds, sp["test"])
    preds = {"qrc_learned_g035": np.stack([test_preds(learned_X(root, s), ds) for s in SEEDS]),
             "qrc_g035": test_preds(np.load(root / "design_c1/g0.35/X.npz")["X"], ds)[None],
             "qrc_g1": test_preds(np.load(root / "design_c1/g1/X.npz")["X"], ds)[None]}
    cfg, _ = select_esn(root / "esn_grid.npz")
    preds["esn_tuned"] = np.stack([test_preds(esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"],
                                                         cfg["inscale"], i), ds) for i in INITS])
    preds["lstm_tuned"] = np.load(root / "lstm.npz")["preds"]
    from qrc_realworld_factors import linear_ar
    preds["linear_ar"] = linear_ar(ds)[0][None]
    res = {"dataset": a.dataset, "preregistration": "QRC_Design_Concepts.md §9", "n_test": int(len(Y)),
           "esn_cfg": cfg, "groups": {}, "tests": {}}
    for g, P in preds.items():
        per = [float(r2_cols(Y, p).mean()) for p in P]
        res["groups"][g] = {"pooled": float(np.mean(per)), "sd": float(np.std(per))}
    for b in [g for g in preds if g != "qrc_learned_g035"]:
        res["tests"][f"qrc_learned_g035 - {b}"] = boot(Y, preds["qrc_learned_g035"], preds[b])
    res["L1_pass"] = bool(res["tests"]["qrc_learned_g035 - qrc_g035"]["ci95"][0] > 0)
    res["provenance"] = provenance(dataset=a.dataset, step="learned g=0.35 confirmation")
    np.savez(root / "learn_predictions.npz", Y_test=Y, **{f"pred_{k}": v for k, v in preds.items()})
    (root / "learn_confirm.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for g, v in res["groups"].items():
        print(f"  {g:18s} pooled test R2 {v['pooled']:+.4f}")
    for k, v in res["tests"].items():
        print(f"  {k:36s} {v['point']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print(f"  L1 pass: {res['L1_pass']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "freeze", "confirm"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--run-root", default=None)
    a = ap.parse_args()
    ds = load_dataset(a.dataset) if a.dataset else None
    {"dev": cmd_dev, "freeze": cmd_freeze, "confirm": cmd_confirm}[a.cmd](a, ds)


if __name__ == "__main__":
    main()
