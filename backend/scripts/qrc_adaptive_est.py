"""ML-adaptive quantum estimation on the NMR engine (QRC_AdaptiveEst.md).

Stage 1 - yardstick: do the existing (Milestone-0) estimators reach the Cramer-Rao bound
for one parameter, known model, fixed 49-probe bank?

  theta = log10 sigma of one OU dephasing component (tau_c = 1 ms known), M0 noise model
  (exact SLE, commuting sigma_z regime), systems S1 / R3 of Milestone 0.

  python scripts/qrc_adaptive_est.py stage1-gen --system S1     # noiseless FIDs -> traces .npz
  python scripts/qrc_adaptive_est.py stage1-gen --system R3
  python scripts/qrc_adaptive_est.py stage1-analyze --jobs 8    # CRB, MLE, ridge, MLP
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

import qrc_noisespec_m0 as m  # noqa: E402
import qrc_noisespec_m0_analysis as ma  # noqa: E402
from app.qrc import estimation as est  # noqa: E402

# --- Stage 1 design (pre-registered in QRC_AdaptiveEst.md sec. 2) ---------------
S1_TAU = 1e-3                                   # s, known
S1_LO, S1_HI = np.log10(30.0), np.log10(300.0)  # prior on theta = log10 sigma
S1_N_LEVELS = 20
S1_GRID = np.linspace(S1_LO, S1_HI, 801)        # MLE model grid
S1_TEST = np.linspace(S1_LO + 0.15, S1_HI - 0.15, 7)
S1_H = 1e-3                                     # FD step in theta
S1_STENCIL = np.array([-2, -1, 1, 2])
S1_N_TRAIN, S1_TRAIN_SEED = 1200, 501
S1_M = 2000                                     # noise realisations per test point
S1_ETAS = [1e-3, 1e-2, 3e-2, 1e-1]
S1_ETA_PRIMARY = 1e-2
S1_LAGRANGE = 6                                 # points for the MLE interpolant
CHUNK = 250                                     # realisations per noise draw


def s1_trace(system):
    return m.TRACES / f"qrc_adaptive_stage1_{system}.npz"


def fids_at(sp, thetas, batch=25):
    """Noiseless complex128 FIDs (n, 49, N_ACQ) at theta = log10 sigma, tau_c fixed."""
    out = np.zeros((len(thetas), len(m.PROBES), m.N_ACQ), np.complex128)
    for b0 in range(0, len(thetas), batch):
        B = m.Batch(sp, [[(10.0 ** t, S1_TAU)] for t in thetas[b0:b0 + batch]], S1_N_LEVELS)
        for j, (n_pi, t_ms) in enumerate(m.PROBES):
            out[b0:b0 + batch, j] = B.probe(n_pi, t_ms * 1e-3).cpu().numpy()
    return out


def cmd_stage1_gen(a):
    sysc = m.make_systems()[a.system]
    sp = m.Spins(sysc, a.device)
    assert sp.comm < 1e-10
    A = float(abs((sp.rho0 @ sp.P90.T) @ sp.o))
    th_train = np.random.default_rng(S1_TRAIN_SEED).uniform(S1_LO, S1_HI, S1_N_TRAIN)
    sten = (S1_TEST[:, None] + S1_H * S1_STENCIL[None, :]).ravel()
    t0 = time.time()
    f_test = fids_at(sp, S1_TEST)
    f_sten = fids_at(sp, sten).reshape(len(S1_TEST), len(S1_STENCIL), len(m.PROBES), m.N_ACQ)
    print(f"{a.system}: test + stencil {time.time() - t0:.0f}s", flush=True)
    f_grid = fids_at(sp, S1_GRID)
    print(f"{a.system}: grid {time.time() - t0:.0f}s", flush=True)
    f_train = fids_at(sp, th_train).astype(np.complex64)
    print(f"{a.system}: train {time.time() - t0:.0f}s", flush=True)
    m.TRACES.mkdir(parents=True, exist_ok=True)
    path = s1_trace(a.system)
    np.savez(path, fids_test=f_test, theta_test=S1_TEST, fids_stencil=f_sten, stencil=S1_STENCIL,
             h=S1_H, fids_grid=f_grid, theta_grid=S1_GRID, fids_train=f_train, theta_train=th_train,
             seed=S1_TRAIN_SEED, fid_dwell=m.DWELL, probes=np.array(m.PROBES), A=A,
             meta=json.dumps({"stage": 1, "system": a.system, "theta": "log10 sigma (rad/s)",
                              "tau_c": S1_TAU, "N_levels": S1_N_LEVELS,
                              "noise": "H=(xi/2) sum sz, one OU comp (collapsed telegraph), SLE exact",
                              "nu": sysc.chemical_shifts, "J": sysc.j_coupling, "t1": sysc.t1,
                              "t2": sysc.t2, "n_acq": m.N_ACQ, "dwell": m.DWELL,
                              "sequences": "app.qrc.sequences.cpmg (M0 bank)",
                              "commutator_rel": sp.comm}))
    print("wrote", path, f"({time.time() - t0:.0f}s)")


# --- analysis ------------------------------------------------------------------
def noise_seed(system, kind, ie, j=0):
    """Fixed seeds: kind 1 = train, 2 = test (j = test point)."""
    return 600000 + {"S1": 1, "R3": 2}[system] * 10000 + kind * 1000 + ie * 100 + j


def lagrange(theta, grid, k=S1_LAGRANGE):
    """Indices, weights and derivative weights of the k-point Lagrange interpolant."""
    h = grid[1] - grid[0]
    c = int(np.floor((theta - grid[0]) / h)) - k // 2 + 1
    c = min(max(c, 0), len(grid) - k)
    idx = np.arange(c, c + k)
    x = grid[idx]
    w = np.ones(k)
    dw = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        den = np.prod([x[i] - x[j] for j in others])
        w[i] = np.prod([theta - x[j] for j in others]) / den
        dw[i] = sum(np.prod([theta - x[j] for j in others if j != l]) for l in others) / den
    return idx, w, dw


class MLE:
    """Known-model maximum likelihood on the raw complex FID (Gaussian, iid): grid argmin,
    then Gauss-Newton on the 6-point Lagrange interpolant of the model."""

    def __init__(self, grid_fids, grid):
        self.G = grid_fids.reshape(len(grid), -1)
        self.g2 = (np.abs(self.G) ** 2).sum(1)
        self.grid = grid

    def interp(self, th):
        idx, w, dw = lagrange(th, self.grid)
        return w @ self.G[idx], dw @ self.G[idx]

    def fit(self, Y):
        Y = Y.reshape(len(Y), -1)
        cost = self.g2[None, :] - 2 * (Y @ self.G.conj().T).real
        th = self.grid[np.argmin(cost, 1)]
        out = np.empty(len(Y))
        for i, y in enumerate(Y):
            t = th[i]
            for _ in range(30):
                mu, d = self.interp(t)
                step = float((d.conj() @ (y - mu)).real / (np.abs(d) ** 2).sum())
                t = float(np.clip(t + step, self.grid[0], self.grid[-1]))
                if abs(step) < 1e-12:
                    break
            out[i] = t
        return out


def jac_from_stencil(f_sten, h):
    """4th-order central difference along the stencil axis (-2,-1,1,2) -> (n_test, 49, N_ACQ)."""
    return (-f_sten[:, 3] + 8 * f_sten[:, 2] - 8 * f_sten[:, 1] + f_sten[:, 0]) / (12 * h)


def readout_maps(system, A, sign):
    """Linear feature maps (per probe) of the lin families, as real matrices on realify(fid_probe)."""
    phi = ma.phi(system)[sign] / (A * m.N_ACQ)                         # (N_ACQ, L)
    S = np.zeros((len(ma.T_IDX), m.N_ACQ))
    S[np.arange(len(ma.T_IDX)), ma.T_IDX] = 1.0 / A

    def real_map(C):            # y -> C^T y, complex C (N_ACQ, L), on [Re y; Im y]
        Ct = C.T
        return np.block([[Ct.real, -Ct.imag], [Ct.imag, Ct.real]])
    return {"lines-lin": real_map(phi), "time-lin": real_map(S.T.astype(complex))}


def fisher_raw(J, sigma):
    """Total FI over the probe bank of the raw FID (n_test,)."""
    return (np.abs(J) ** 2).reshape(len(J), -1).sum(1) / sigma ** 2


def fisher_readout(J, Mr, sigma):
    out = np.zeros(len(J))
    for i in range(len(J)):
        for j in range(J.shape[1]):
            Jr = np.concatenate([J[i, j].real, J[i, j].imag])[:, None]
            out[i] += est.linear_readout_fisher(Jr, Mr, sigma)[0, 0]
    return out


def cmd_stage1_analyze(a):
    t0 = time.time()
    res = {"stage": 1, "prereg": a.prereg, "test_theta": S1_TEST.tolist(), "systems": {}}
    for system in ("S1", "R3"):
        z = np.load(s1_trace(system), allow_pickle=True)
        A = float(z["A"])
        f_test, f_grid = z["fids_test"], z["fids_grid"]
        f_train = z["fids_train"].astype(np.complex128)
        y_train = z["theta_train"][:, None]
        J = jac_from_stencil(z["fids_stencil"], float(z["h"]))
        sign = ma.line_sign(f_train, system)[0] if system == "R3" else 1
        mle = MLE(f_grid, z["theta_grid"])
        # interpolation check (gate Y3): noiseless MLE on the true test FIDs
        th_clean = mle.fit(f_test)
        maps = readout_maps(system, A, sign)
        R = {"A": A, "line_sign": sign, "D_raw": 2 * len(m.PROBES) * m.N_ACQ,
             "interp_offset": (th_clean - S1_TEST).tolist(), "levels": {}}
        for ie, eta in enumerate(S1_ETAS):
            if eta not in a.etas:
                continue
            s = est.m0_sigma(eta, A)
            crb = 1.0 / fisher_raw(J, s)
            crb_ro = {k: 1.0 / fisher_readout(J, Mr, s) for k, Mr in maps.items()}
            ftr = ma.add_noise(f_train, eta, A, noise_seed(system, 1, ie))
            Ftr = {f: ma.features(ftr, f, A, sign, system) for f in ma.FAMILIES}
            preds = {"MLE": [], "ridge": [], "MLP": []}
            Fte_all = {f: [] for f in ma.FAMILIES}
            for j in range(len(S1_TEST)):
                pm_j = []
                for c in range(0, S1_M, CHUNK):
                    rep = np.repeat(f_test[j:j + 1], CHUNK, 0)
                    Y = ma.add_noise(rep, eta, A, [noise_seed(system, 2, ie, j), c])
                    pm_j.append(mle.fit(Y))
                    for f in ma.FAMILIES:
                        Fte_all[f].append(ma.features(Y, f, A, sign, system))
                preds["MLE"].append(np.concatenate(pm_j))
            Fte = {f: np.concatenate(v) for f, v in Fte_all.items()}
            pr, info_r = ma.ridge_method(Ftr, Fte, y_train)
            pm, info_m = ma.mlp_method(Ftr, Fte, y_train, a.jobs)
            preds["ridge"] = np.split(pr[:, 0], len(S1_TEST))
            preds["MLP"] = np.split(pm[:, 0], len(S1_TEST))
            lv = {"eta": eta, "sigma": s, "crb": crb.tolist(),
                  "crb_readout": {k: v.tolist() for k, v in crb_ro.items()},
                  "ridge_info": info_r, "mlp_info": info_m, "estimators": {}}
            for k, P in preds.items():
                reps = [est.report_vs_bound(P[j][:, None], S1_TEST[j], np.array([[crb[j]]]))
                        for j in range(len(S1_TEST))]
                for r in reps:
                    r.pop("cov")
                eff = np.array([r["efficiency_mse"][0] for r in reps])
                lv["estimators"][k] = {"per_point": reps, "median_eff_mse": float(np.median(eff)),
                                       "median_eff_var": float(np.median([r["efficiency_var"][0] for r in reps]))}
            R["levels"][str(eta)] = lv
            print(f"[{system}] eta={eta:g}  sqrtCRB(mid)={np.sqrt(crb[3]):.2e}  " + "  ".join(
                f"{k}: eff_mse={v['median_eff_mse']:.3g}" for k, v in lv["estimators"].items())
                + f"  ({time.time() - t0:.0f}s)", flush=True)
        res["systems"][system] = R
    # gate Y and the primary question
    gate = {}
    for system, R in res["systems"].items():
        for e in (1e-3, 1e-2):
            lv = R["levels"].get(str(e))
            if lv is None:
                continue
            pts = lv["estimators"]["MLE"]["per_point"]
            ok_eff = all(0.9 <= p["efficiency_var"][0] <= 1.1 for p in pts)
            ok_bias = all(abs(p["bias"][0]) <= 0.2 * np.sqrt(p["crb"][0]) for p in pts)
            off = np.abs(R["interp_offset"]) / np.sqrt(np.array(lv["crb"]))
            gate[f"{system}_eta{e:g}"] = {"eff_in_range": ok_eff, "bias_ok": ok_bias,
                                          "interp_offset_over_sqrtcrb_max": float(off.max()),
                                          "interp_ok": bool(off.max() <= 0.1)}
    res["gate_Y"] = {"cells": gate, "pass": all(c["eff_in_range"] and c["bias_ok"] and c["interp_ok"]
                                               for c in gate.values())}
    q = {}
    for system, R in res["systems"].items():
        lv = R["levels"].get(str(S1_ETA_PRIMARY))
        if lv:
            for k in ("ridge", "MLP"):
                e = lv["estimators"][k]["median_eff_mse"]
                q[f"{system}_{k}"] = {"median_eff_mse": e, "reaches_crb": bool(e >= 0.8)}
    res["Q1"] = {"rule": "median over test points of CRB/MSE >= 0.8 at eta = 1e-2", "cells": q}
    res["wall_s"] = time.time() - t0
    print("GATE Y", json.dumps(res["gate_Y"]))
    print("Q1", json.dumps(res["Q1"]))
    if a.dry:
        return
    m.DATA.mkdir(parents=True, exist_ok=True)
    (m.DATA / "qrc_adaptive_stage1_results.json").write_text(json.dumps(res, indent=1))
    print("wrote results", f"{time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("stage1-gen")
    g.add_argument("--system", choices=["S1", "R3"], required=True)
    g.add_argument("--device", default="cpu")
    an = sub.add_parser("stage1-analyze")
    an.add_argument("--jobs", type=int, default=8)
    an.add_argument("--etas", type=float, nargs="*", default=S1_ETAS)
    an.add_argument("--prereg", default="")
    an.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"stage1-gen": cmd_stage1_gen, "stage1-analyze": cmd_stage1_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
