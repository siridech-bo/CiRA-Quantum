"""Design step 3 — C5 step time tau + C7 compact readout (pre-registered,
QRC_Design_Concepts.md §8, commit ed96b17). All at the frozen encoding gain g = 0.35.

  pass     QRC passes at extra tau values (saves FID trace + 653-bin features per tau;
           tau = 30 ms reuses design_c1/g0.35)
  dev      VALIDATION-only scores for every (tau, readout) on one dataset
  freeze   aggregate the 8 development datasets, apply the freeze rule, write the decision
  confirm  untouched city: current vs adopted design, tuned ESN / LSTM / linear AR, test once

Readouts: bins653 (current) | bins200 | bins100 (top-K by first-step magnitude) |
pca128 | pca64 (PCA fitted on the training block) | lines (FID projected onto the
physics-informed proton lines, re + im).

  python scripts/qrc_design_c5c7.py pass --dataset D --run-root R --taus 0.015,0.06,0.1
  python scripts/qrc_design_c5c7.py dev  --dataset D --run-root R
  python scripts/qrc_design_c5c7.py freeze
  python scripts/qrc_design_c5c7.py confirm --dataset D --run-root R
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
from qrc_era5_weather import r2_cols, run_pass  # noqa: E402
from qrc_realworld_bench import (  # noqa: E402
    INITS, esn_states, load_dataset, make_targets, select_esn, spin_inputs, test_preds, val_scores,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "docs/QRC/data"
G = 0.35
TAUS = ("0.015", "0.03", "0.06", "0.1")
READOUTS = ("bins653", "bins200", "bins100", "pca128", "pca64", "lines")
DEV = ["era5_delhi", "era5_mumbai", "era5_bangkok", "era5_beijing", "era5_berlin", "beijing_pm25", "etth1", "santafe"]
DWELL, NPTS = 3e-4, 2048


def machine(tau, device):
    """qrc_era5_weather.qrc_machine with a configurable tau."""
    import torch
    from app.qrc.block_propagator import BlockPropagator
    from app.qrc.config import SimConfig
    from app.qrc.system import QRCSystem
    from qrc_gen_traces import _resolve_system
    sysc = _resolve_system("crotonic9_paper4")
    sim = SimConfig(tau=tau, n_virtual=1, evolution_mode="action")
    sysm = QRCSystem(sysc, sim)
    cdt = torch.complex128
    bp = BlockPropagator(sysc, sim, device=device, cdtype=cdt, which=("z",))
    lab = [x.upper() for x in sysc.labels]
    H = [i for i, x in enumerate(lab) if x.startswith("H")]
    C = [i for i, x in enumerate(lab) if x.startswith("C")]
    vec0 = torch.tensor(sysm.qt.operator_to_vector(sysm.rho0).full().ravel(), dtype=cdt, device=device)
    return {"bp": bp, "ro": bp.fid_readout(H, NPTS, DWELL), "vec0": vec0, "H": H, "C": C, "n": sysm.n,
            "torch": torch, "cdt": cdt}


def tau_dir(root, tau):
    return Path(root) / ("design_c1/g0.35" if tau == "0.03" else f"design_c5/tau{tau}")


def cmd_pass(a, ds):
    import torch
    for tau in a.taus.split(","):
        d = tau_dir(a.run_root, tau)
        if (d / "X.npz").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        m = machine(float(tau), a.device)
        t0 = time.time()
        th = torch.tensor(G * np.arcsin(np.sqrt(spin_inputs(ds, m))), dtype=torch.float64, device=a.device)
        F = run_pass(m, th, log_every=0)
        mag = torch.abs(torch.fft.fft(F, dim=1))
        bins = torch.sort(torch.argsort(mag[0], descending=True)[:653]).values
        np.savez(d / "trace.npz", fids_H=F.cpu().numpy().astype(np.complex64),
                 meta=json.dumps({"dataset": a.dataset, "gain": G, "tau": float(tau), "fid": f"H {NPTS} x {DWELL}"}))
        np.savez(d / "X.npz", X=mag[:, bins].cpu().numpy(), bins=bins.cpu().numpy())
        print(f"{a.dataset} tau={tau}: pass {time.time() - t0:.0f}s", flush=True)


_LINES = {}


def readout(name, root, tau, train_idx):
    d = tau_dir(root, tau)
    X = np.load(d / "X.npz")["X"]
    if name == "bins653":
        return X
    if name in ("bins200", "bins100"):
        k = int(name[4:])
        return X[:, np.argsort(X[0])[::-1][:k]]
    if name in ("pca128", "pca64"):
        k = int(name[3:])
        A = X[train_idx]
        mu, sd = A.mean(0), A.std(0) + 1e-12
        Z = (X - mu) / sd
        _, _, Vt = np.linalg.svd(Z[train_idx], full_matrices=False)
        return Z @ Vt[:k].T
    if name == "lines":
        if "phi" not in _LINES:
            from app.qrc.spectral_lines import physics_informed_lines
            from qrc_gen_traces import _resolve_system
            ls = physics_informed_lines(_resolve_system("crotonic9_paper4"), (4, 5, 6, 7, 8))
            _LINES["phi"] = np.asarray(ls.projection(NPTS, DWELL))
        F = np.load(d / "trace.npz")["fids_H"]
        P = F @ _LINES["phi"]
        return np.hstack([P.real, P.imag])
    raise KeyError(name)


def val_pooled(X, ds):
    vs = val_scores(X, ds)
    return float(max(np.nanmean(vs[s].max(0)) for s in (0, 1)))


def cmd_dev(a, ds):
    tr = ds["split"]["train"]
    res = {"dataset": a.dataset, "gain": G, "scores": {}}
    for tau in TAUS:
        res["scores"][tau] = {}
        for r in READOUTS:
            res["scores"][tau][r] = val_pooled(readout(r, a.run_root, tau, tr), ds)
        print(f"{a.dataset:13s} tau={tau:5s} " + "  ".join(f"{r} {v:.4f}" for r, v in res["scores"][tau].items()), flush=True)
    res["provenance"] = provenance(dataset=a.dataset, step="C5/C7 dev (validation only)")
    out = Path(a.run_root) / "c5c7_dev.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_freeze(_a, _ds=None):
    S = {n: json.loads((DATA / f"qrc_design_c5c7_dev_{n}.json").read_text(encoding="utf-8"))["scores"] for n in DEV}
    mean = {f"{t}|{r}": float(np.mean([S[n][t][r] for n in DEV])) for t in TAUS for r in READOUTS}
    best = max(mean, key=mean.get)
    cur = mean["0.03|bins653"]
    adopted = best if mean[best] >= cur + 0.005 else "0.03|bins653"
    res = {"rule": "best mean VAL R2 over 8 dev datasets; adopt only if >= current (0.03|bins653) + 0.005",
           "mean_val": mean, "best": best, "current": cur, "adopted": adopted,
           "changed": adopted != "0.03|bins653", "provenance": provenance(step="C5/C7 freeze")}
    (DATA / "qrc_design_c5c7_frozen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("mean validation R2 (tau | readout):")
    for t in TAUS:
        print(f"  tau={t:5s} " + "  ".join(f"{r} {mean[f'{t}|{r}']:.4f}" for r in READOUTS))
    print(f"best {best} {mean[best]:.4f} vs current {cur:.4f} -> ADOPTED {adopted}")


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
    fz = json.loads((DATA / "qrc_design_c5c7_frozen.json").read_text(encoding="utf-8"))
    tau_a, ro_a = fz["adopted"].split("|")
    sp = ds["split"]
    Y = make_targets(ds, sp["test"])
    preds = {"qrc_current": test_preds(readout("bins653", a.run_root, "0.03", sp["train"]), ds)[None],
             "qrc_adopted": test_preds(readout(ro_a, a.run_root, tau_a, sp["train"]), ds)[None]}
    cfg, _ = select_esn(Path(a.run_root) / "esn_grid.npz")
    preds["esn_tuned"] = np.stack([test_preds(esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"],
                                                         cfg["inscale"], i), ds) for i in INITS])
    preds["lstm_tuned"] = np.load(Path(a.run_root) / "lstm.npz")["preds"]
    from qrc_realworld_factors import linear_ar
    preds["linear_ar"] = linear_ar(ds)[0][None]
    res = {"dataset": a.dataset, "preregistration_commit": "ed96b17", "adopted": fz["adopted"], "groups": {},
           "tests": {}}
    for g, P in preds.items():
        res["groups"][g] = {"pooled": float(np.mean([r2_cols(Y, p).mean() for p in P]))}
    res["tests"]["T1 adopted - current"] = boot(Y, preds["qrc_adopted"], preds["qrc_current"])
    for b in ("esn_tuned", "lstm_tuned", "linear_ar"):
        res["tests"][f"T3 adopted - {b}"] = boot(Y, preds["qrc_adopted"], preds[b])
        res["tests"][f"T3 current - {b}"] = boot(Y, preds["qrc_current"], preds[b])
    res["provenance"] = provenance(dataset=a.dataset, adopted=fz["adopted"])
    np.savez(Path(a.run_root) / "c5c7_predictions.npz", Y_test=Y, **{f"pred_{k}": v for k, v in preds.items()})
    (Path(a.run_root) / "c5c7_confirm.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for g, v in res["groups"].items():
        print(f"  {g:12s} pooled test R2 {v['pooled']:+.4f}")
    for k, v in res["tests"].items():
        print(f"  {k:28s} {v['point']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pass", "dev", "freeze", "confirm"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--taus", default="0.015,0.06,0.1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ds = load_dataset(a.dataset) if a.dataset else None
    {"pass": cmd_pass, "dev": cmd_dev, "freeze": cmd_freeze, "confirm": cmd_confirm}[a.cmd](a, ds)


if __name__ == "__main__":
    main()
