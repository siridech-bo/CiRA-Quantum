"""Design concept C1 — encoding gain (QRC "input scale"), tests hypothesis H1.

H1 (QRC_Design_Concepts.md): the QRC's memory is limited by input overwrite —
every step rotates each spin by theta = arcsin(sqrt(s)) (up to pi/2). Scaling the
pulse, theta = g * arcsin(sqrt(s)), g in (0, 1], should lengthen memory, at the cost
of weaker input injection.

Pre-registered protocol: VALIDATION ONLY — no test data are read here.
  * readout: ridge on standardised 653-bin FFT magnitudes; alpha per target chosen
    on an INNER split of the training period (first 80 % fit / last 20 % select),
    then refit on the whole training period and scored on the validation period;
  * metrics on the validation period: pooled R² (all targets), short-horizon
    (h <= 25 % of max) and long-horizon (h >= 60 % of max) R²; memory function =
    R² of reconstructing the input L steps back from the current features;
  * every pass saves its raw FID trace (persist rule) + features; results carry
    provenance.

  python scripts/qrc_design_c1.py --dataset era5_delhi --run-root DIR [--gains ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.qrc.provenance import provenance  # noqa: E402
from qrc_era5_weather import qrc_machine, r2_cols, ridge, run_pass  # noqa: E402
from qrc_realworld_bench import load_dataset, make_targets, spin_inputs, target_names  # noqa: E402

ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)
LAGS = {"day": (1, 2, 5, 10, 20, 30, 45, 60, 90), "hour": (1, 3, 6, 12, 24, 48, 96, 168),
        "step": (1, 2, 3, 5, 10, 20, 40)}


def val_eval(X, ds):
    """alpha per target on an inner split of TRAIN; refit TRAIN; score VAL."""
    sp = ds["split"]
    tr = sp["train"]
    cut = int(0.8 * len(tr))
    a_fit, a_sel = tr[:cut], tr[cut:]
    P = ridge(X[a_fit], make_targets(ds, a_fit), [X[a_sel]], ALPHAS, True)
    Ysel = make_targets(ds, a_sel)
    a_idx = np.argmax(np.array([r2_cols(Ysel, p[0]) for p in P]), axis=0)
    P = ridge(X[tr], make_targets(ds, tr), [X[sp["val"]]], ALPHAS, True)
    pred = np.stack([P[a_idx[t]][0][:, t] for t in range(len(ds["targets"]))], axis=1)
    r2 = r2_cols(make_targets(ds, sp["val"]), pred)
    hmax = max(h for _, h in ds["targets"])
    hf = np.array([h / hmax for _, h in ds["targets"]])
    return {"pooled": float(r2.mean()), "short": float(r2[hf <= 0.25].mean()),
            "long": float(r2[hf >= 0.6].mean()), "per_target": dict(zip(target_names(ds), r2.tolist()))}


def memory_function(X, ds):
    """R² of reconstructing the input L steps back from current features (VAL)."""
    sp, w = ds["split"], ds["w"]
    tr, va = sp["train"], sp["val"]
    out = {}
    for L in LAGS[ds["unit"]]:
        P = ridge(X[tr], w[tr - L], [X[va]], (1e-4, 1e-2, 1.0, 1e2), True)
        out[L] = max(float(r2_cols(w[va - L], p[0]).mean()) for p in P)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-root", required=True, help="realworld run dir of this dataset")
    ap.add_argument("--gains", default="0.1,0.2,0.35,0.5,0.75,1.0")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    ds = load_dataset(a.dataset)
    root = Path(a.run_root)
    out_dir = root / "design_c1"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = qrc_machine(a.device, torch.complex128)
    base = np.arcsin(np.sqrt(spin_inputs(ds, m)))
    results = {"concept": "C1 encoding gain", "hypothesis": "H1 input overwrite limits memory",
               "dataset": a.dataset, "protocol": "validation only; alpha on inner train split", "gains": {}}
    for g in [float(x) for x in a.gains.split(",")]:
        t0 = time.time()
        gd = out_dir / f"g{g:g}"
        gd.mkdir(exist_ok=True)
        if (gd / "X.npz").exists():
            X = np.load(gd / "X.npz")["X"]
        else:
            th = torch.tensor(g * base, dtype=torch.float64, device=a.device)
            F = run_pass(m, th, log_every=0)
            mag = torch.abs(torch.fft.fft(F, dim=1))
            bins = torch.sort(torch.argsort(mag[0], descending=True)[:653]).values
            X = mag[:, bins].cpu().numpy()
            np.savez(gd / "trace.npz", fids_H=F.cpu().numpy().astype(np.complex64),
                     meta=json.dumps({"dataset": a.dataset, "gain": g, "encoding": "g*arcsin_sqrt"}))
            np.savez(gd / "X.npz", X=X, bins=bins.cpu().numpy())
        ev = val_eval(X, ds)
        mf = memory_function(X, ds)
        results["gains"][f"{g:g}"] = {"val": ev, "memory": mf}
        print(f"{a.dataset:13s} g={g:<5g} val pooled {ev['pooled']:.4f} short {ev['short']:.4f} long {ev['long']:.4f} | "
              f"memory " + " ".join(f"L{L}:{v:.2f}" for L, v in mf.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    results["provenance"] = provenance(dataset=a.dataset, gains=a.gains, alphas=list(ALPHAS),
                                       lags=list(LAGS[ds["unit"]]))
    (out_dir / "c1_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
