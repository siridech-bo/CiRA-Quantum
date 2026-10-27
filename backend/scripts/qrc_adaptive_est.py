"""ML-adaptive quantum estimation on the NMR engine (QRC_AdaptiveEst.md).

Stage 1 - yardstick: do the existing (Milestone-0) estimators reach the Cramer-Rao bound
for one parameter, known model, fixed 49-probe bank?

  theta = log10 sigma of one OU dephasing component (tau_c = 1 ms known), M0 noise model
  (exact SLE, commuting sigma_z regime), systems S1 / R3 of Milestone 0.

  python scripts/qrc_adaptive_est.py stage1-gen --system S1     # noiseless FIDs -> traces .npz
  python scripts/qrc_adaptive_est.py stage1-gen --system R3
  python scripts/qrc_adaptive_est.py stage1-analyze --jobs 8    # CRB, MLE, ridge, MLP

Stage 2 - adaptive, known model: does Bayesian-adaptive probe choice reach the CRB in fewer
shots than cycling the fixed bank? (reuses the Stage-1 noiseless traces as the model)

  python scripts/qrc_adaptive_est.py stage2-run --jobs 18       # episodes -> traces .npz
  python scripts/qrc_adaptive_est.py stage2-analyze --prereg <commit>
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


# --- Stage 2 (pre-registered in QRC_AdaptiveEst.md sec. 3) -----------------------------
S2_ETAS = [1e-2, 1e-1]
S2_ETA_PRIMARY = 1e-2
S2_R = 200                          # episodes per (system, eta, theta0, policy)
S2_SHOTS = 196                      # 4 bank cycles
S2_NGRID = 201
S2_POLICIES = ("fixed-cycle", "greedy-fisher", "static-best")
S2_TARGET_SHOTS = 49                # target = CRB of one full bank record


def s2_trace(system, eta, tag=""):
    return m.TRACES / f"qrc_adaptive_stage2_{system}_eta{eta:g}{tag}.npz"


def lagrange_vec(theta, grid, k=S1_LAGRANGE):
    """Vectorised ``lagrange``: idx (n,k), weights (n,k), derivative weights (n,k)."""
    theta = np.asarray(theta, float)
    h = grid[1] - grid[0]
    c = np.clip(np.floor((theta - grid[0]) / h).astype(int) - k // 2 + 1, 0, len(grid) - k)
    idx = c[:, None] + np.arange(k)[None, :]
    x = grid[idx]
    d = theta[:, None] - x
    w = np.ones_like(d)
    dw = np.zeros_like(d)
    for i in range(k):
        den = np.ones(len(theta))
        for j in range(k):
            if j != i:
                den = den * (x[:, i] - x[:, j])
                w[:, i] *= d[:, j]
        for l in range(k):
            if l == i:
                continue
            t = np.ones(len(theta))
            for j in range(k):
                if j not in (i, l):
                    t = t * d[:, j]
            dw[:, i] += t
        w[:, i] /= den
        dw[:, i] /= den
    return idx, w, dw


def s2_cache(system):
    """Projected Stage-1 grid + test signals (small; lets stage2-run go to a remote host
    without the 0.6 GB Stage-1 trace). Written by ``stage2-export``."""
    return m.TRACES / f"qrc_adaptive_stage2_{system}_reduced.npz"


def line_basis(system):
    Q, _ = np.linalg.qr(ma.phi(system)[1])                           # Stage-1 line_sign = +1 (both)
    return Q


def reduced_model(system):
    """Stage-1 model grid on an orthonormal basis Q of the line span: z = y @ Q (white noise
    preserved, same sigma). Returns model(thetas) -> (mu, dmu) each (n, 49, L), A, grid and
    the projected test signals (n_test, 49, L)."""
    if s2_cache(system).exists():
        c = np.load(s2_cache(system))
        G, grid, A, T = c["G"], c["grid"], float(c["A"]), c["T"]
    else:
        z = np.load(s1_trace(system))
        Q = line_basis(system)
        G, grid, A, T = z["fids_grid"] @ Q, z["theta_grid"], float(z["A"]), z["fids_test"] @ Q

    def model(th):
        idx, w, dw = lagrange_vec(th, grid)
        Gi = G[idx]                                                   # (n, k, 49, L)
        return np.einsum("nk,nkpl->npl", w, Gi), np.einsum("nk,nkpl->npl", dw, Gi)
    return model, A, grid, T


def cmd_stage2_export(a):
    for system in a.systems:
        z = np.load(s1_trace(system))
        Q = line_basis(system)
        np.savez(s2_cache(system), G=z["fids_grid"] @ Q, grid=z["theta_grid"], A=z["A"],
                 T=z["fids_test"] @ Q, Q=Q)
        print(f"wrote {s2_cache(system).name}", flush=True)


def _s2_cell(system, eta, j, pol_name, r0, r1):
    from app.qrc import adaptive as ad
    model, A, grid, T = reduced_model(system)
    sig = est.m0_sigma(eta, A)
    true = T[j]                                                       # (49, L)
    if pol_name == "fixed-cycle":
        pol = ad.FixedCycle(true.shape[0])
    elif pol_name == "greedy-fisher":
        pol = ad.GreedyFisher()
    else:
        pol = StaticBest(model, grid)
    out = np.zeros((3, r1 - r0, S2_SHOTS))
    ie = S2_ETAS.index(eta)
    for k, r in enumerate(range(r0, r1)):
        rng = np.random.default_rng([700000, {"S1": 1, "R3": 2}[system], ie, j, r])
        post = ad.GridPosterior(model, sig, S1_LO, S1_HI, n_grid=S2_NGRID)
        out[0, k], out[1, k], out[2, k] = ad.run_episode(pol, post, true, S2_SHOTS, rng)
    return out


class StaticBest:
    """Non-adaptive design: always the probe with the largest prior-averaged one-shot FI."""
    name = "static-best"

    def __init__(self, model, grid):
        _, dmu = model(grid)
        self.j = int(np.argmax((np.abs(dmu) ** 2).sum(2).mean(0)))

    def choose(self, post, stats, t):
        return self.j


def s2_noise(system, eta, j, r0, r1, L):
    """Unit complex shot noise (r1-r0, S2_SHOTS, L) drawn exactly as ``adaptive.run_episode``
    draws it from the per-episode seed (re then im, per shot): GPU runs stay paired with CPU."""
    ie = S2_ETAS.index(eta)
    out = np.empty((r1 - r0, S2_SHOTS, L), complex)
    for k, r in enumerate(range(r0, r1)):
        rng = np.random.default_rng([700000, {"S1": 1, "R3": 2}[system], ie, j, r])
        for t in range(S2_SHOTS):
            out[k, t] = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    return out


def _s2_cell_gpu(system, eta, j, pol_name, r0, r1, device):
    from app.qrc.adaptive_batch import BatchedEpisodes
    c = np.load(s2_cache(system))
    model, A, grid, T = reduced_model(system)
    be = BatchedEpisodes(c["G"], c["grid"], est.m0_sigma(eta, A), S1_LO, S1_HI,
                         n_grid=S2_NGRID, k=S1_LAGRANGE, device=device)
    sj = StaticBest(model, grid).j if pol_name == "static-best" else None
    mn, sd, ch = be.run(T[j], s2_noise(system, eta, j, r0, r1, T.shape[-1]), pol_name, sj)
    return np.stack([mn, sd, ch.astype(float)])


def cmd_stage2_run(a):
    from joblib import Parallel, delayed
    t0 = time.time()
    for system in a.systems:
        for eta in a.etas:
            # tasks = (theta0, policy, episode chunk); seeds depend on r only, so chunking is exact
            tasks = [(j, p, r0, min(r0 + a.chunk, a.R)) for j in range(len(S1_TEST))
                     for p in S2_POLICIES for r0 in range(0, a.R, a.chunk)]
            if a.device:          # batched torch: one task per (theta0, policy), all episodes at once
                tasks = [(j, p, 0, a.R) for j in range(len(S1_TEST)) for p in S2_POLICIES]
                res = [_s2_cell_gpu(system, eta, *t, a.device) for t in tasks]
            else:
                res = Parallel(n_jobs=a.jobs)(delayed(_s2_cell)(system, eta, *t) for t in tasks)
            arr = np.zeros((len(S1_TEST), len(S2_POLICIES), 3, a.R, S2_SHOTS))
            for (j, p, r0, r1), o in zip(tasks, res):
                arr[j, S2_POLICIES.index(p), :, r0:r1] = o
            m.TRACES.mkdir(parents=True, exist_ok=True)
            np.savez(s2_trace(system, eta, a.tag), means=arr[:, :, 0], stds=arr[:, :, 1],
                     chosen=arr[:, :, 2].astype(np.int16), theta_test=S1_TEST,
                     policies=np.array(S2_POLICIES), eta=eta,
                     meta=json.dumps({"stage": 2, "system": system, "eta": eta, "R": a.R,
                                      "shots": S2_SHOTS, "n_grid": S2_NGRID,
                                      "readout": "z = y @ Q, Q orthonormal basis of the line span "
                                                 "(sign +1); white noise, sigma = eta A / sqrt 2",
                                      "model": "Stage-1 grid, 6-point Lagrange",
                                      "noise_rng": "[700000, sys, ie, j, r], common across policies",
                                      "tag": a.tag, "posterior": "GridPosterior with zoom-on-underresolved (v2)" if a.tag else "v1",
                                      "runner": f"torch batched ({a.device})" if a.device else "numpy joblib"}))
            print(f"[{system}] eta={eta:g} wrote {s2_trace(system, eta, a.tag).name} ({time.time() - t0:.0f}s)",
                  flush=True)


def s2_fisher(system):
    """Per-probe one-shot FI at sigma = 1 in the reduced readout, (n_test, 49); and A."""
    z = np.load(s1_trace(system))
    J = jac_from_stencil(z["fids_stencil"], float(z["h"])) @ line_basis(system)
    return (np.abs(J) ** 2).sum(2), float(z["A"])


def ratio_or_inf(a, b):
    return float(a / b) if b > 0 else float("inf")


def first_stable(mse, target):
    """Smallest n with MSE(n') <= target for every n' >= n (None if never)."""
    bad = np.where(mse > target)[0]
    if len(bad) == 0:
        return 1
    n = int(bad[-1]) + 2
    return n if n <= len(mse) else None


def cmd_stage2_analyze(a):
    res = {"stage": 2, "prereg": a.prereg, "tag": a.tag, "systems": {}}
    n = np.arange(1, S2_SHOTS + 1)
    for system in ("S1", "R3"):
        fi1, A = s2_fisher(system)
        res["systems"][system] = {}
        for eta in S2_ETAS:
            z = np.load(s2_trace(system, eta, a.tag))
            fi = fi1 / est.m0_sigma(eta, A) ** 2
            pts = []
            for j, th0 in enumerate(S1_TEST):
                b_bank = 1.0 / (n * fi[j].mean())
                b_opt = 1.0 / (n * fi[j].max())
                target = b_bank[S2_TARGET_SHOTS - 1]
                pt = {"theta0": float(th0), "best_probe": list(m.PROBES[int(fi[j].argmax())]),
                      "fi_max_over_mean": float(fi[j].max() / fi[j].mean()),
                      "b_opt_196": float(b_opt[-1]), "b_bank_196": float(b_bank[-1]),
                      "target": float(target), "policies": {}}
                for ip, p in enumerate(S2_POLICIES):
                    mse = ((z["means"][j, ip] - th0) ** 2).mean(0)
                    pt["policies"][p] = {
                        "mse_196": float(mse[-1]), "e_opt_196": ratio_or_inf(b_opt[-1], mse[-1]),
                        "e_bank_196": ratio_or_inf(b_bank[-1], mse[-1]),
                        "n_star": first_stable(mse, target),
                        "mse_at": {str(k): float(mse[k - 1]) for k in (1, 5, 10, 20, 49, 98, 196)}}
                pts.append(pt)
            pol = {p: [x["policies"][p] for x in pts] for p in S2_POLICIES}
            nstar = {p: [x["n_star"] for x in pol[p]] for p in S2_POLICIES}
            inf = lambda v: np.inf if v is None else v  # noqa: E731
            ratio = [inf(f) / inf(g) for f, g in zip(nstar["fixed-cycle"], nstar["greedy-fisher"])]
            cell = {"points": pts, "n_star": nstar,
                    "median_e_opt_greedy": float(np.median([x["e_opt_196"] for x in pol["greedy-fisher"]])),
                    "median_e_bank_fixed": float(np.median([x["e_bank_196"] for x in pol["fixed-cycle"]])),
                    "shot_ratio_fixed_over_greedy": ratio,
                    "median_shot_ratio": float(np.median(ratio)),
                    "static_over_greedy_mse196": [ratio_or_inf(s["mse_196"], g["mse_196"]) for s, g in
                                                  zip(pol["static-best"], pol["greedy-fisher"])]}
            res["systems"][system][str(eta)] = cell
            print(f"[{system}] eta={eta:g}  e_opt(greedy)={cell['median_e_opt_greedy']:.3f}  "
                  f"e_bank(fixed)={cell['median_e_bank_fixed']:.3f}  n*={nstar}  "
                  f"ratio={cell['median_shot_ratio']:.2f}  static/greedy="
                  f"{np.round(cell['static_over_greedy_mse196'], 2).tolist()}", flush=True)
    v = {}
    for system in ("S1", "R3"):
        c = res["systems"][system][str(S2_ETA_PRIMARY)]
        v[system] = {"gate_fixed_efficient": bool(0.8 <= c["median_e_bank_fixed"] <= 1.25),
                     "Q2a_greedy_reaches_bound": bool(c["median_e_opt_greedy"] >= 0.8),
                     "Q2b_fewer_shots": bool(c["median_shot_ratio"] >= 1.5),
                     "Q2c_beats_static_at_edge": bool(c["static_over_greedy_mse196"][-1] >= 1.5)}
    res["verdict"] = v
    print("VERDICT", json.dumps(v))
    if a.prereg:
        (m.DATA / f"qrc_adaptive_stage2{a.tag}_results.json").write_text(json.dumps(res, indent=1))


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
    r2 = sub.add_parser("stage2-run")
    r2.add_argument("--jobs", type=int, default=8)
    r2.add_argument("--R", type=int, default=S2_R)
    r2.add_argument("--chunk", type=int, default=S2_R, help="episodes per joblib task")
    r2.add_argument("--systems", nargs="*", choices=["S1", "R3"], default=["S1", "R3"])
    r2.add_argument("--etas", type=float, nargs="*", default=S2_ETAS)
    r2.add_argument("--tag", default="", help="trace-name suffix, e.g. _v2")
    r2.add_argument("--device", default="", help="e.g. cuda: batched torch runner (no CPU fan-out)")
    e2 = sub.add_parser("stage2-export")
    e2.add_argument("--systems", nargs="*", choices=["S1", "R3"], default=["S1", "R3"])
    a2 = sub.add_parser("stage2-analyze")
    a2.add_argument("--prereg", default="")
    a2.add_argument("--tag", default="")
    a = ap.parse_args()
    {"stage1-gen": cmd_stage1_gen, "stage1-analyze": cmd_stage1_analyze,
     "stage2-run": cmd_stage2_run, "stage2-export": cmd_stage2_export,
     "stage2-analyze": cmd_stage2_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
