"""Design step 6 (pre-registered, QRC_Design_Concepts.md §13, commit d022533).

Part A — encoding control: `paper` theta = arcsin(s) (Hou et al. PRL 136, 120602), `std` arcsin(sqrt s)
(g = 1) and `final` 0.35 arcsin(sqrt s); 653-bin proton FID readout (D = 653).
Part B — readout channels at g = 0.35 from ONE pass: H (proton FID, 653 bins, the standard), C (13C FID,
653 bins), HC, HZ (+ 9 single-spin <sz>), HP (+ 512 populations diag(rho), idealised).

  pass    GPU: reservoir pass(es) for a dataset; raw traces saved BEFORE any metric (persist rule)
  dev     VALIDATION-only scores (Step 5 protocol, val_pooled) + memory function, per dataset
  freeze  aggregate the 8 dev datasets, apply Rules A and B, write the decision
  confirm Confirmation 4 (untouched city): adopted vs frozen design and tuned baselines (K1/K2/K3)

  python scripts/qrc_design_p.py pass --part A|B --dataset D --run-root R
  python scripts/qrc_design_p.py dev --part A|B --dataset D --run-root R
  python scripts/qrc_design_p.py freeze
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.qrc.provenance import provenance  # noqa: E402
from qrc_design_c1 import memory_function  # noqa: E402
from qrc_design_r import AR, DEV, _std, boot, offsets, test_pooled, val_pooled  # noqa: E402
from qrc_era5_weather import qrc_machine, r2_cols  # noqa: E402
from qrc_realworld_bench import INITS, esn_states, load_dataset, make_targets, select_esn, spin_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "docs/QRC/data"
ENC = {"paper": lambda s: np.arcsin(s), "std": lambda s: np.arcsin(np.sqrt(s)),
       "final": lambda s: 0.35 * np.arcsin(np.sqrt(s))}
CHANNELS = ("H", "C", "HC", "HZ", "HP")
DIMS = {"H": 653, "C": 653, "HC": 1306, "HZ": 662, "HP": 1165}
C_FID = (2048, 5e-5)
PREREG = "d022533"


def _top_bins(F, k=653):
    import torch
    mag = torch.abs(torch.fft.fft(F, dim=1))
    bins = torch.sort(torch.argsort(mag[0], descending=True)[:k]).values
    return mag[:, bins].cpu().numpy(), bins.cpu().numpy()


def _pass(m, theta, channels):
    """one pass; returns dict of per-step arrays: fids_H always, + fids_C / z / pops if channels."""
    import torch
    from qrc_weather_learn import _rx
    bp, d = m["bp"], 1 << m["n"]
    ro_c = bp.fid_readout(m["C"], *C_FID) if channels else None
    out = {k: [] for k in (("fids_H", "fids_C", "z", "pops") if channels else ("fids_H",))}
    vec, t0 = m["vec0"], time.time()
    th = torch.tensor(theta, dtype=torch.float64, device=m["vec0"].device)
    with torch.no_grad():
        for k in range(th.shape[0]):
            z, vec = bp.step(vec, [_rx(th[k, q], m["cdt"]) for q in range(m["n"])])
            out["fids_H"].append(bp.fid(vec, m["ro"]))
            if channels:
                out["fids_C"].append(bp.fid(vec, ro_c))
                out["z"].append(z)
                out["pops"].append(vec.reshape(d, d).diagonal().real)
            if k % 5000 == 0:
                print(f"  step {k}/{th.shape[0]} ({time.time() - t0:.0f}s)", flush=True)
    return {k: torch.stack(v) for k, v in out.items()}


def _save_trace(path, ds, name, meta, arrs):
    sp = ds["split"]
    np.savez(path, **{k: (v.cpu().numpy().astype(np.complex64) if v.is_complex() else v.cpu().numpy().astype(np.float32))
                      for k, v in arrs.items()},
             weather_norm=ds["w"], split_train=sp["train"], split_val=sp["val"], split_test=sp["test"],
             horizons=np.array(sorted({h for _, h in ds["targets"]})), fid_dwell=np.array([3e-4, C_FID[1]]),
             seed=np.array([0]), meta=json.dumps({"dataset": name, "prereg": PREREG, **meta}))


def cmd_pass(a, ds):
    import torch
    root = Path(a.run_root) / "design_p"
    root.mkdir(parents=True, exist_ok=True)
    m = qrc_machine(a.device, torch.complex128)
    s = np.clip(spin_inputs(ds, m), 0, 1)
    encs = ("paper", "std", "final") if a.part == "A" else ("final",)
    for e in encs:
        tag = f"A_{e}" if a.part == "A" else "B_final"
        if (root / f"{tag}_X.npz").exists():
            continue
        t0 = time.time()
        arrs = _pass(m, ENC[e](s), channels=a.part == "B")
        _save_trace(root / f"{tag}_trace.npz", ds, a.dataset,
                    {"encoding": e, "part": a.part, "readout_H": "proton FID 2048 x 0.3 ms",
                     "readout_C": f"13C FID {C_FID[0]} x {C_FID[1] * 1e6:.0f} us" if a.part == "B" else None}, arrs)
        XH, bH = _top_bins(arrs["fids_H"])
        X = {"H": XH, "bins_H": bH}
        if a.part == "B":
            XC, bC = _top_bins(arrs["fids_C"])
            X.update(C=XC, bins_C=bC, Z=arrs["z"].cpu().numpy(), P=arrs["pops"].cpu().numpy())
        np.savez(root / f"{tag}_X.npz", **X)
        print(f"{a.dataset} {tag}: pass + trace in {time.time() - t0:.0f}s", flush=True)


def channel_feats(Xd, ch, tr):
    """H / C raw (as the standard readout); combined channels: each block standardised on TRAIN."""
    if ch in ("H", "C"):
        return Xd[ch]
    return np.hstack([_std(Xd["H"], tr), _std(Xd[{"HC": "C", "HZ": "Z", "HP": "P"}[ch]], tr)])


def cmd_dev(a, ds):
    root = Path(a.run_root) / "design_p"
    tr = ds["split"]["train"]
    res = {"dataset": a.dataset, "part": a.part, "prereg": PREREG, "protocol": "val_pooled (Step 5), VALIDATION only",
           "scores": {}, "memory": {}, "D": {}}
    if a.part == "A":
        conds = {e: np.load(root / f"A_{e}_X.npz")["H"] for e in ENC}
    else:
        Xd = dict(np.load(root / "B_final_X.npz"))
        conds = {ch: channel_feats(Xd, ch, tr) for ch in CHANNELS}
        saved = Path(a.run_root) / "design_c1" / "g0.35" / "X.npz"
        if saved.exists():
            Xs = np.load(saved)["X"]
            res["reproduce_saved_g035"] = float(np.abs(Xs - Xd["H"]).max() / (np.abs(Xs).max() + 1e-30))
    for k, X in conds.items():
        t0 = time.time()
        res["scores"][k] = val_pooled(X, ds)
        res["memory"][k] = memory_function(X, ds)
        res["D"][k] = int(X.shape[1])
        mem = res["memory"][k]
        print(f"{a.dataset:13s} {a.part} {k:6s} D={X.shape[1]:5d} val R2 {res['scores'][k]:.4f} | memory "
              + " ".join(f"L{L}:{v:.2f}" for L, v in mem.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    res["provenance"] = provenance(dataset=a.dataset, step=f"Step 6 part {a.part} dev (validation only)")
    out = DATA / f"qrc_design_p{a.part}_dev_{a.dataset}.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")


def _rule(q, base, guard):
    mean = {k: float(np.mean(v)) for k, v in q.items()}
    worse = {k: int(sum(x < b - 0.005 for x, b in zip(v, q[base]))) for k, v in q.items()}
    drop1 = {}
    for k, v in q.items():
        d = np.array(v) - np.array(q[base])
        drop1[k] = float(np.mean(np.delete(np.array(v), int(np.argmax(d))))) - float(
            np.mean(np.delete(np.array(q[base]), int(np.argmax(d)))))
    ok = [k for k in q if k != base and mean[k] >= mean[base] + 0.005 and worse[k] <= 2
          and (not guard or drop1[k] >= 0.0025)]
    return mean, worse, drop1, ok, (max(ok, key=mean.get) if ok else None)


def cmd_freeze(_a, _ds=None):
    res = {"prereg": PREREG}
    for part, base, guard, keys in (("A", "final", False, tuple(ENC)), ("B", "H", True, CHANNELS)):
        S = {n: json.loads((DATA / f"qrc_design_p{part}_dev_{n}.json").read_text(encoding="utf-8")) for n in DEV}
        q = {k: [S[n]["scores"][k] for n in DEV] for k in keys}
        mean, worse, drop1, ok, adopted = _rule(q, base, guard)
        if part == "A":   # Rule A: only `paper` may replace the frozen encoding
            ok = [k for k in ok if k == "paper"]
            adopted = ok[0] if ok else None
        lastlag = {n: max(S[n]["memory"][base], key=lambda L: int(L)) for n in DEV}
        res[part] = {"per_dataset": {k: dict(zip(DEV, v)) for k, v in q.items()}, "mean": mean,
                     "worse_than_base": worse, "mean_change_without_best_dataset": drop1, "eligible": ok,
                     "adopted": adopted,
                     "memory_longest_lag": {k: {n: S[n]["memory"][k][lastlag[n]] for n in DEV} for k in keys},
                     "D": S[DEV[0]]["D"]}
        if part == "B":
            res[part]["reproduce_saved_g035"] = {n: S[n].get("reproduce_saved_g035") for n in DEV}
        print(f"Part {part}: " + "  ".join(f"{k} {mean[k]:.4f} (worse {worse[k]}, w/o best {drop1[k]:+.4f})"
                                           for k in keys) + f"  -> adopted: {adopted}")
    res["provenance"] = provenance(step="Step 6 freeze")
    (DATA / "qrc_design_p_frozen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_confirm(a, ds):
    """Confirmation 4 (§13): adopted condition vs frozen design (K1), tuned ESN (K2), descriptive K3."""
    fz = json.loads((DATA / "qrc_design_p_frozen.json").read_text(encoding="utf-8"))
    root = Path(a.run_root)
    sp = ds["split"]
    tr = sp["train"]
    Xd = dict(np.load(root / "design_p" / "B_final_X.npz"))
    base = Xd["H"]
    if fz["B"]["adopted"]:
        new, label = channel_feats(Xd, fz["B"]["adopted"], tr), f"channel {fz['B']['adopted']}"
    else:
        new, label = np.load(root / "design_p" / "A_paper_X.npz")["H"], "encoding paper"
    ar = AR(ds)
    cfg, _ = select_esn(root / "esn_grid.npz")
    E = [esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], i) for i in INITS]
    Y = make_targets(ds, sp["test"])
    P = {"qrc_adopted": np.stack([test_pooled("base", new, ds, ar)[2]]),
         "qrc_final": np.stack([test_pooled("base", base, ds, ar)[2]]),
         "esn_base": np.stack([test_pooled("base", X, ds, ar)[2] for X in E]),
         "lstm_tuned": np.load(root / "lstm.npz")["preds"],
         "linear_ar": np.stack([ar.predict(np.concatenate([tr, sp["val"]]), sp["test"])]),
         "persistence": np.stack([offsets(ds, sp["test"])])}
    hs = np.array([h for _, h in ds["targets"]])
    res = {"dataset": a.dataset, "preregistration_commit": PREREG, "adopted": label,
           "esn_cfg": cfg, "ar_lags": ar.L, "block": 30,
           "test_input_shift_sd": ((ds["w"][sp["test"]].mean(0) - ds["w"][tr].mean(0)) / ds["w"][tr].std(0)).tolist(),
           "groups": {k: {"pooled": float(np.mean([r2_cols(Y, p).mean() for p in v])),
                          "long": float(np.mean([r2_cols(Y, p)[hs >= 30].mean() for p in v]))} for k, v in P.items()},
           "tests": {"K1 adopted - final": boot(Y, P["qrc_adopted"], P["qrc_final"], 30),
                     "K2 adopted - esn_base": boot(Y, P["qrc_adopted"], P["esn_base"], 30)}}
    for b in ("lstm_tuned", "linear_ar", "persistence"):
        res["tests"][f"K3 adopted - {b}"] = boot(Y, P["qrc_adopted"], P[b], 30)
    res["tests"]["K3 final - esn_base"] = boot(Y, P["qrc_final"], P["esn_base"], 30)
    res["provenance"] = provenance(dataset=a.dataset, step="confirm4")
    np.savez(root / "confirm4_predictions.npz", Y_test=Y, **{f"pred_{k}": v for k, v in P.items()})
    for out in (root / "confirm4.json", DATA / f"qrc_confirm4_{a.dataset.replace('era5_', '')}.json"):
        out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    for k, v in res["tests"].items():
        print(f"  {k:28s} {v['point']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pass", "dev", "freeze", "confirm"])
    ap.add_argument("--part", choices=["A", "B"], default="B")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "freeze":
        return cmd_freeze(a)
    {"pass": cmd_pass, "dev": cmd_dev, "confirm": cmd_confirm}[a.cmd](a, load_dataset(a.dataset))


if __name__ == "__main__":
    main()
