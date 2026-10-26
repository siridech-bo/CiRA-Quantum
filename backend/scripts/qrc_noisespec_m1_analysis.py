"""Milestone 1 analysis (pre-registered 62005a8, QRC_NoiseSpec_Milestone1.md sec. 3.4-3.8).

Budget axis: B runs per noise sample. M1 R3 reservoir (best of ridge / MLP, dev CV), M2 S1 ridge,
M3 relaxation dispersion (classical, S1 offset sweep), M4 S1 MLP, M5 R3 MLP. Greedy run selection
on dev; confirmation scored once.

  python scripts/qrc_noisespec_m1_analysis.py --task b --jobs 48 [--smoke]
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

import qrc_noisespec_m1 as m  # noqa: E402
from qrc_noisespec_m0_analysis import (ALPHAS, MLP_GRID, _mlp, _std, boot_diff,  # noqa: E402
                                       r2_cols, ridge_cv, ridge_fit_predict)

ETAS = [1e-3, 1e-2, 3e-2]
ETA_PRIMARY = 1e-2
BUDGETS = [1, 2, 3, 6, 12]
B_PRIMARY, B_SECONDARY = 3, 1
FAMILIES = ("lines-lin", "lines-log", "time-log")
T_IDX = np.arange(0, m.N_ACQ, 16)
NP = len(m.PROBES)
RD_ORDER = [(k, 0) for k in (0, 5, 2, 3, 1, 4)] + [(k, 2) for k in (0, 5, 2, 3, 1, 4)]  # (offset, probe)


def load(task, split):
    z = np.load(m.TRACES / f"qrc_noisespec_m1{task}_{split}.npz", allow_pickle=True)
    n = len(z["targets"])
    return {"R3": z["fids_R3"].astype(np.complex128),
            "S1": z["fids_S1"].astype(np.complex128).reshape(n, -1, m.N_ACQ),   # run = k*14 + j
            "y": z["targets"], "P": z["noise_params"],
            "A": {"R3": float(z["A_R3"]), "S1": float(z["A_S1"])}}


def noisy(f, eta, A, seed):
    rng = np.random.default_rng(seed)
    s = eta * A / np.sqrt(2)
    return f + s * (rng.standard_normal(f.shape) + 1j * rng.standard_normal(f.shape))


def seed_of(task, split, system, ie):
    return {"a": 1, "b": 2}[task] * 100000 + {"dev": 1, "confirm": 2}[split] * 10000 \
        + {"R3": 1, "S1": 2}[system] * 1000 + ie


def line_freqs(system, run):
    if system == "R3":
        return m.r3_clusters()
    return np.array([m.S1_OFFSETS_HZ[run // NP]])


def run_features(f, system, A, sign):
    """f (n, runs, N_ACQ) -> {family: (n, runs, d_run)}."""
    t = np.arange(m.N_ACQ) * m.DWELL
    n, R, _ = f.shape
    out = {}
    if system == "R3":
        Phi = np.exp(sign * 1j * 2 * np.pi * np.outer(t, m.r3_clusters()))
        X = np.einsum("nrt,tc->nrc", f, Phi) / (A * m.N_ACQ)
    else:
        X = np.zeros((n, R, 1), complex)
        for r in range(R):
            ph = np.exp(sign * 1j * 2 * np.pi * t * m.S1_OFFSETS_HZ[r // NP])
            X[:, r, 0] = f[:, r] @ ph / (A * m.N_ACQ)
    out["lines-lin"] = np.concatenate([X.real, X.imag], 2)
    out["lines-log"] = np.log(np.abs(X) + 1e-6)
    out["time-log"] = np.log(np.abs(f[:, :, T_IDX]) / A + 1e-6)
    return out


def stack(F, runs):
    return F[:, runs, :].reshape(F.shape[0], -1)


def best_cv(X, y):
    sc = ridge_cv(X, y)
    i = int(np.argmax(sc))
    return float(sc[i]), float(ALPHAS[i])


def greedy(F, y, bmax, jobs):
    from joblib import Parallel, delayed
    sel, hist = [], []
    cand = list(range(F.shape[1]))
    for _ in range(bmax):
        rest = [r for r in cand if r not in sel]
        sc = Parallel(n_jobs=jobs)(delayed(best_cv)(stack(F, sel + [r]), y) for r in rest)
        i = int(np.argmax([s[0] for s in sc]))
        sel.append(rest[i])
        hist.append({"run": rest[i], "cv": sc[i][0], "alpha": sc[i][1]})
    return sel, hist


def cv3(X, y, fn):
    perm = np.random.default_rng(0).permutation(len(y))
    parts = np.array_split(perm, 3)
    s = 0.0
    for i in range(3):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(3) if j != i])
        s += r2_cols(y[te], fn(X[tr], y[tr], X[te])).mean() / 3
    return s


def learned(Fd, Fc, yd, orders, B, jobs, nets=True):
    """Ridge (best family) and MLP (best family x grid) on each family's greedy prefix of length B
    (or all runs when B is None). Returns dict with preds + selection info."""
    from joblib import Parallel, delayed
    res = {}
    rid = []
    for fam in FAMILIES:
        runs = orders[fam][:B] if B else list(range(Fd[fam].shape[1]))
        Xd, Xc = stack(Fd[fam], runs), stack(Fc[fam], runs)
        cv, al = best_cv(Xd, yd)
        rid.append((cv, fam, al, runs, Xd, Xc))
    cv, fam, al, runs, Xd, Xc = max(rid, key=lambda r: r[0])
    res["ridge"] = {"pred": ridge_fit_predict(Xd, yd, Xc, al), "family": fam, "alpha": al, "cv5": cv,
                    "cv3": cv3(Xd, yd, lambda a, b, c: ridge_fit_predict(a, b, c, al)),
                    "D": int(Xd.shape[1]), "runs": [int(r) for r in runs]}
    if nets:
        cfgs = [(fam, h, l2) for fam in FAMILIES for h, l2 in MLP_GRID]
        data = {f: (stack(Fd[f], orders[f][:B] if B else list(range(Fd[f].shape[1]))),
                    stack(Fc[f], orders[f][:B] if B else list(range(Fc[f].shape[1])))) for f in FAMILIES}
        sc = Parallel(n_jobs=jobs)(delayed(cv3)(data[f][0], yd, lambda a, b, c, h=h, l2=l2: _mlp(a, b, c, h, l2))
                                   for f, h, l2 in cfgs)
        i = int(np.argmax(sc))
        f, h, l2 = cfgs[i]
        res["mlp"] = {"pred": _mlp(data[f][0], yd, data[f][1], h, l2), "family": f, "hidden": list(h),
                      "l2": l2, "cv3": float(sc[i]), "D": int(data[f][0].shape[1])}
    return res


# --- relaxation dispersion (M3) ---------------------------------------------------
def rd_rates(f, A, eta):
    """Decay rate per (sample) from a weighted log-linear fit over the acquisition."""
    t = np.arange(m.N_ACQ) * m.DWELL
    y = np.abs(f)
    thr = 3 * max(eta, 1e-4) * A
    out = np.zeros(len(f))
    for i in range(len(f)):
        ok = y[i] > thr
        if ok.sum() < 5:
            ok = np.zeros_like(ok)
            ok[:5] = True
        w = y[i, ok] ** 2
        out[i] = -np.polyfit(t[ok], np.log(np.maximum(y[i, ok], 1e-12)), 1, w=np.sqrt(w))[0]
    return out


def rd_predict(fS_noisy, A, eta, B, task):
    T2 = m.make_s1(0.0).t2[0]
    runs = RD_ORDER[:B]
    by_k = {}
    for k, j in runs:
        by_k.setdefault(k, []).append(rd_rates(fS_noisy[:, k * NP + j], A, eta))
    ks = sorted(by_k)
    S = np.array([4 * np.maximum(np.mean(by_k[k], 0) - 1 / T2, 1e-3) for k in ks]).T     # (n, n_offsets)
    w = m.CENTRES[ks]
    if task == "b":
        lw = np.log10(w)
        return np.array([np.interp(np.log10(m.CENTRES), lw, np.log10(s)) for s in S])
    cfg = m.TASKS["a"]
    ltau = np.linspace(np.log10(cfg["tau"][0]), np.log10(cfg["tau"][1]), 400)
    tau = 10 ** ltau
    shape = np.log10(2 * tau[:, None] / (1 + (w[None, :] * tau[:, None]) ** 2))          # (ntau, n_off)
    centre = np.mean(ltau)
    out = []
    for s in np.log10(S):
        ls2 = (s[None, :] - shape).mean(1)                                                # log10 sigma^2
        resid = ((s[None, :] - shape - ls2[:, None]) ** 2).sum(1)
        best = resid.min()
        cand = np.where(resid <= best + 1e-9)[0]
        i = cand[np.argmin(np.abs(ltau[cand] - centre))]
        out.append([np.clip(0.5 * ls2[i], *np.log10(cfg["sig"])), ltau[i]])
    return np.array(out)


# --- main ---------------------------------------------------------------------------
def run(task, jobs, smoke):
    t0 = time.time()
    if smoke:
        d = load(task, "dev")
        cut = 800
        dev = {k: (v[:cut] if k not in ("A",) else v) for k, v in d.items()}
        con = {k: (v[cut:] if k not in ("A",) else v) for k, v in d.items()}
    else:
        dev, con = load(task, "dev"), load(task, "confirm")
    A = dev["A"]
    yd, yc = dev["y"], con["y"]
    t = np.arange(m.N_ACQ) * m.DWELL
    sign = max((1, -1), key=lambda s: float(np.abs(dev["R3"][:50, 0] @ np.exp(
        s * 1j * 2 * np.pi * np.outer(t, m.r3_clusters()))).sum()))
    res = {"task": task, "prereg": "62005a8", "A": A, "line_sign": sign, "levels": {}}
    keep = {}
    for ie, eta in enumerate(ETAS):
        bud = BUDGETS if eta == ETA_PRIMARY else [B_PRIMARY]
        bmax = max(bud)
        fd = {s: noisy(dev[s], eta, A[s], seed_of(task, "dev", s, ie)) for s in ("R3", "S1")}
        fc = {s: noisy(con[s], eta, A[s], seed_of(task, "confirm", s, ie)) for s in ("R3", "S1")}
        Fd = {s: run_features(fd[s], s, A[s], sign) for s in fd}
        Fc = {s: run_features(fc[s], s, A[s], sign) for s in fc}
        orders = {s: {} for s in ("R3", "S1")}
        ghist = {s: {} for s in ("R3", "S1")}
        for s in ("R3", "S1"):
            for fam in FAMILIES:
                orders[s][fam], ghist[s][fam] = greedy(Fd[s][fam], yd, min(bmax, Fd[s][fam].shape[1]), jobs)
        print(f"[{task}] eta={eta:g} greedy done ({time.time() - t0:.0f}s)", flush=True)
        lv = {"eta": eta, "budgets": {}, "greedy": ghist}
        for B in bud + ["all"]:
            if B == "all" and eta != ETA_PRIMARY:
                continue
            Bv = None if B == "all" else B
            r3 = learned(Fd["R3"], Fc["R3"], yd, orders["R3"], Bv, jobs)
            s1 = learned(Fd["S1"], Fc["S1"], yd, orders["S1"], Bv, jobs)
            m1 = r3["ridge"] if r3["ridge"]["cv3"] >= r3["mlp"]["cv3"] else r3["mlp"]
            m1_kind = "ridge" if m1 is r3["ridge"] else "mlp"
            P = {"M1": m1["pred"], "M2": s1["ridge"]["pred"], "M4": s1["mlp"]["pred"], "M5": r3["mlp"]["pred"],
                 "M1ridge": r3["ridge"]["pred"]}
            if Bv is not None:
                P["M3"] = rd_predict(fc["S1"], A["S1"], eta, Bv, task)
            info = {"M1": {"readout": m1_kind, **{k: v for k, v in m1.items() if k != "pred"}},
                    "M1ridge": {k: v for k, v in r3["ridge"].items() if k != "pred"},
                    "M2": {k: v for k, v in s1["ridge"].items() if k != "pred"},
                    "M4": {k: v for k, v in s1["mlp"].items() if k != "pred"},
                    "M5": {k: v for k, v in r3["mlp"].items() if k != "pred"},
                    "M3": {"runs": [list(r) for r in RD_ORDER[:Bv]] if Bv else None}}
            row = {"methods": {}}
            for k, p in P.items():
                r = r2_cols(yc, p)
                row["methods"][k] = {"r2_per_target": r.tolist(), "r2_mean": float(r.mean()), **info.get(k, {})}
            for k in ("M2", "M3", "M4"):
                if k in P:
                    row[f"M1_minus_{k}"] = boot_diff(yc, P["M1"], P[k])
            lv["budgets"][str(B)] = row
            print(f"[{task}] eta={eta:g} B={B}: " + "  ".join(f"{k}={row['methods'][k]['r2_mean']:.3f}"
                                                         for k in row["methods"]) + f"  ({time.time() - t0:.0f}s)",
                  flush=True)
            if eta == ETA_PRIMARY and B in (B_PRIMARY, "all"):
                keep[str(B)] = P
        res["levels"][str(eta)] = lv

    lv = res["levels"][str(ETA_PRIMARY)]["budgets"]

    def rule(B):
        row = lv[str(B)]
        conds = {k: bool(row[f"M1_minus_{k}"][0] >= 0.05 and row[f"M1_minus_{k}"][1] > 0) for k in ("M2", "M3", "M4")}
        return {"diffs": {k: row[f"M1_minus_{k}"] for k in ("M2", "M3", "M4")}, "conditions": conds,
                "pass": all(conds.values())}
    if task == "a":
        r = lv["all"]["methods"]["M1"]["r2_per_target"]
        res["verdict"] = {"M1a_rule": "M1 (all 14 runs) R2 >= 0.90 on log sigma and log tau_c at eta=1e-2",
                          "M1_r2": r, "M1a_pass": bool(min(r) >= 0.90),
                          "efficiency_B3_descriptive": rule(B_PRIMARY), "efficiency_B1_descriptive": rule(B_SECONDARY)}
    else:
        res["verdict"] = {"primary_B3": rule(B_PRIMARY), "secondary_B1": rule(B_SECONDARY)}
    print("VERDICT", json.dumps(res["verdict"]), flush=True)
    res["wall_s"] = time.time() - t0
    if smoke:
        print("smoke run: nothing written")
        return
    (m.DATA / f"qrc_noisespec_m1{task}_results.json").write_text(json.dumps(res, indent=1))
    np.savez(m.DATA / f"qrc_noisespec_m1{task}_preds.npz", y=yc, P=con["P"],
             **{f"B{B}_{k}": v for B, P in keep.items() for k, v in P.items()})
    print("done", task, f"{time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.task, a.jobs, a.smoke)


if __name__ == "__main__":
    main()
