"""Confirmation 1 evaluation (pre-registered in QRC_Design_Concepts.md §6, commit 29152bf).

Frozen rule: QRC encoding gain g = 0.35 (primary); per-city validation-selected g from
{0.2, 0.35, 0.5, 0.75, 1} (secondary). Untouched ERA5 cities. Protocol identical to
the real-world benchmark (readout chosen on VAL, refit TRAIN+VAL, TEST read once).

Requires in --run-dir: design_c1/g{g}/X.npz for g in the grid (qrc_design_c1.py,
whose VAL scores drive the secondary rule), esn_grid.npz and lstm.npz
(qrc_realworld_bench.py esn / lstm).

Criteria
  S1  pooled test R2(g=0.35) - R2(g=1) > 0 with 95% CI excluding 0 (per city)
  S2  long-horizon (h >= 30) gain and short-horizon (h <= 10) change, per city
  S3  descriptive: QRC g=0.35 (and selected g) vs tuned ESN / LSTM / linear AR

  python scripts/qrc_design_confirm.py --dataset era5_chennai --run-dir DIR
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
    INITS, esn_states, load_dataset, make_targets, select_esn, target_names, test_preds,
)
from qrc_realworld_factors import linear_ar  # noqa: E402

GRID = ("0.2", "0.35", "0.5", "0.75", "1")
PRIMARY = "0.35"


def boot(Y, A, B, idx_sets):
    """A, B: (reps, n, T) — CI of mean_rep pooled R²(A) - mean_rep pooled R²(B), and
    the same restricted to target columns (dict name -> column indices)."""
    rng = np.random.default_rng(0)
    n, block = len(Y), 30
    boots = [np.concatenate([np.arange(s, s + block) for s in rng.choice(n - block + 1, n // block + 1)])[:n]
             for _ in range(2000)]
    out = {}
    for name, cols in idx_sets.items():
        d = np.array([r2_cols(Y[i][:, cols], A[:, i][..., cols]).mean() - r2_cols(Y[i][:, cols], B[:, i][..., cols]).mean()
                      for i in boots])
        point = float(np.mean([r2_cols(Y[:, cols], a[:, cols]).mean() for a in A])
                      - np.mean([r2_cols(Y[:, cols], b[:, cols]).mean() for b in B]))
        out[name] = {"point": point, "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "frac_positive": float((d > 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    ds = load_dataset(a.dataset)
    run = Path(a.run_dir)
    sp = ds["split"]
    Y = make_targets(ds, sp["test"])
    H = np.array([h for _, h in ds["targets"]])
    cols = {"all": np.arange(len(H)), "long": np.where(H >= 30)[0], "short": np.where(H <= 10)[0]}

    c1 = json.loads((run / "design_c1" / "c1_results.json").read_text(encoding="utf-8"))
    g_sel = max(GRID, key=lambda g: c1["gains"][g]["val"]["pooled"])          # secondary rule (VAL only)
    preds = {}
    for label, g in (("qrc_g1", "1"), ("qrc_g035", PRIMARY), ("qrc_gsel", g_sel)):
        preds[label] = np.stack([test_preds(np.load(run / "design_c1" / f"g{g}" / "X.npz")["X"], ds)])
    for label, mx in (("esn_tuned", None), ("esn_matched", 653)):
        cfg, _ = select_esn(run / "esn_grid.npz", mx)
        preds[label] = np.stack([test_preds(esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"],
                                                       cfg["inscale"], i), ds) for i in INITS])
    preds["lstm_tuned"] = np.load(run / "lstm.npz")["preds"]
    lin, L = linear_ar(ds)
    preds["linear_ar"] = lin[None]

    res = {"dataset": a.dataset, "preregistration_commit": "29152bf", "rule_primary_g": float(PRIMARY),
           "rule_secondary_g": float(g_sel), "n_test": int(len(Y)), "groups": {}, "tests": {}}
    for g, P in preds.items():
        per = [float(r2_cols(Y, p).mean()) for p in P]
        res["groups"][g] = {"pooled": float(np.mean(per)), "sd": float(np.std(per)),
                            "long": float(np.mean([r2_cols(Y[:, cols["long"]], p[:, cols["long"]]).mean() for p in P])),
                            "short": float(np.mean([r2_cols(Y[:, cols["short"]], p[:, cols["short"]]).mean() for p in P])),
                            "per_target": dict(zip(target_names(ds), np.mean([r2_cols(Y, p) for p in P], 0).tolist()))}
    res["tests"]["S1_S2 g0.35 - g1"] = boot(Y, preds["qrc_g035"], preds["qrc_g1"], cols)
    res["tests"][f"secondary g{g_sel} - g1"] = boot(Y, preds["qrc_gsel"], preds["qrc_g1"], cols)
    for b in ("esn_tuned", "esn_matched", "lstm_tuned", "linear_ar"):
        res["tests"][f"S3 g0.35 - {b}"] = boot(Y, preds["qrc_g035"], preds[b], {"all": cols["all"]})
        res["tests"][f"S3 g1 - {b}"] = boot(Y, preds["qrc_g1"], preds[b], {"all": cols["all"]})
    s1 = res["tests"]["S1_S2 g0.35 - g1"]["all"]
    res["S1_pass"] = bool(s1["ci95"][0] > 0)
    res["provenance"] = provenance(dataset=a.dataset, grid=list(GRID), primary=PRIMARY, secondary=g_sel,
                                   linear_ar_lags=int(L))
    np.savez(run / "confirm_predictions.npz", Y_test=Y, **{f"pred_{k}": v for k, v in preds.items()})
    (run / "confirm.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    print(f"{a.dataset}: secondary rule picked g={g_sel}")
    for g, v in res["groups"].items():
        print(f"  {g:12s} pooled {v['pooled']:.4f}  long {v['long']:.4f}  short {v['short']:.4f}")
    for k, v in res["tests"].items():
        print(f"  {k:28s} " + "  ".join(f"{n}: {x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
                                         for n, x in v.items()))
    print(f"  S1 pass: {res['S1_pass']}")


if __name__ == "__main__":
    main()
