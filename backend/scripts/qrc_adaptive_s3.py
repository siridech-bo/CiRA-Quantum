"""ML-adaptive quantum estimation, Stage 3 (QRC_AdaptiveEst.md sec. 4): a model-free neural
policy with a hidden nuisance parameter.

  theta = log10 sigma (estimated), nu = log10 tau_c (hidden nuisance, never given to the
  policy), M0 noise model, commuting sigma_z regime (exact SLE fast path), systems S1 / R3.
  Per-shot readout z = y @ Q (line span, as Stage 2). The simulator is a 2-D table of
  noiseless engine FIDs on a (theta, nu) grid + tensor Lagrange interpolation.

  python scripts/qrc_adaptive_s3.py pilot-interp --device cuda   # table-density pilot
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

import qrc_adaptive_est as s12  # noqa: E402
import qrc_noisespec_m0 as m  # noqa: E402
from app.qrc import estimation as est  # noqa: E402

TH_LO, TH_HI = s12.S1_LO, s12.S1_HI               # theta = log10 sigma, as Stages 1-2
NU_LO, NU_HI = np.log10(2e-4), np.log10(5e-3)     # nu = log10 tau_c (s), hidden nuisance
N_LEVELS = s12.S1_N_LEVELS
LAG = 6


def fids_at(sp, th, nu, batch=1000):
    """Noiseless complex128 FIDs (n, 49, N_ACQ) at (theta, nu) pairs."""
    th, nu = np.atleast_1d(th), np.atleast_1d(nu)
    out = np.zeros((len(th), len(m.PROBES), m.N_ACQ), np.complex128)
    for b0 in range(0, len(th), batch):
        sl = slice(b0, b0 + batch)
        B = m.Batch(sp, [[(10.0 ** a, 10.0 ** b)] for a, b in zip(th[sl], nu[sl])], N_LEVELS)
        for j, (n_pi, t_ms) in enumerate(m.PROBES):
            out[sl, j] = B.probe(n_pi, t_ms * 1e-3).cpu().numpy()
    return out


def lagrange_1d(x, grid, k=LAG):
    """Weights of the k-point Lagrange interpolant on a uniform grid: idx (n,k), w (n,k)."""
    idx, w, _ = s12.lagrange_vec(np.atleast_1d(x), grid, k)
    return idx, w


def cmd_pilot_interp(a):
    """Interpolation error of the tensor Lagrange table vs grid density, in units of the
    per-shot noise sigma (eta = 1e-2) after projection: fine 1-D lines through 3 anchors."""
    out = {}
    for system in ("S1", "R3"):
        sp = m.Spins(m.make_systems()[system], a.device)
        A = float(abs((sp.rho0 @ sp.P90.T) @ sp.o))
        sig = est.m0_sigma(1e-2, A)
        Q = s12.line_basis(system)
        anchors = [(1.7, -3.3), (2.0, -3.0), (2.3, -2.6)]
        res = {}
        for axis, (lo, hi), ns in (("theta", (TH_LO, TH_HI), (101, 201, 401)),
                                   ("nu", (NU_LO, NU_HI), (17, 33, 65, 129))):
            errs = {n: [] for n in ns}
            for th0, nu0 in anchors:
                for n in ns:
                    g = np.linspace(lo, hi, n)
                    mid = (g[1:] + g[:-1]) / 2
                    mid = mid[(mid > g[LAG]) & (mid < g[-LAG - 1])][::max(1, len(mid) // 24)]
                    pts = np.concatenate([g, mid])
                    if axis == "theta":
                        f = fids_at(sp, pts, np.full(len(pts), nu0)) @ Q
                    else:
                        f = fids_at(sp, np.full(len(pts), th0), pts) @ Q
                    G, F = f[:n], f[n:]
                    idx, w = lagrange_1d(mid, g)
                    I = np.einsum("nk,nkpl->npl", w, G[idx])
                    errs[n].append(float(np.abs(I - F).max() / sig))
            res[axis] = {str(n): max(v) for n, v in errs.items()}
            print(system, axis, "max |interp err| / sigma(eta=1e-2):",
                  {n: f"{max(v):.2e}" for n, v in errs.items()}, flush=True)
        out[system] = res
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


# --- simulator table + test points (design fixed by the pilot above) ----------------------
TH_GRID = np.linspace(TH_LO, TH_HI, 201)          # pilot: interp err ~3e-9 sigma (R3)
NU_GRID = np.linspace(NU_LO, NU_HI, 65)           # pilot: interp err ~3e-7 sigma (R3)
TEST_TH = np.linspace(TH_LO + 0.15, TH_HI - 0.15, 5)
TEST_NU = np.linspace(NU_LO + 0.15, NU_HI - 0.15, 5)
# primary misspecification sweep (pre-registration sec. 4): theta0 = the low-theta edge test value,
# nu at the 5 grid values + the 4 midpoints -> 9 values, |nu - nu0| in {0, .137, .274, .412, .549}
SWEEP_TH = float(TEST_TH[0])
SWEEP_NU_EXTRA = (TEST_NU[1:] + TEST_NU[:-1]) / 2
FD_H = 1e-3
FD_ST = np.array([-2, -1, 1, 2])


def s3_trace(system, raw=False):
    return m.TRACES / f"qrc_adaptive_stage3_{system}{'_raw' if raw else ''}.npz"


def cmd_gen(a):
    """Noiseless engine FIDs: the (theta, nu) table and the 25 test points + FD stencils.
    Raw FIDs (complex64) -> *_raw.npz; projected z-space (complex128) -> main trace."""
    sysc = m.make_systems()[a.system]
    sp = m.Spins(sysc, a.device)
    assert sp.comm < 1e-10
    A = float(abs((sp.rho0 @ sp.P90.T) @ sp.o))
    Q = s12.line_basis(a.system)
    t0 = time.time()
    TT, NN = np.meshgrid(TH_GRID, NU_GRID, indexing="ij")
    raw = fids_at(sp, TT.ravel(), NN.ravel()).reshape(len(TH_GRID), len(NU_GRID), len(m.PROBES), m.N_ACQ)
    print(f"{a.system}: table {time.time() - t0:.0f}s", flush=True)
    tt, nn = np.meshgrid(TEST_TH, TEST_NU, indexing="ij")
    tt, nn = tt.ravel(), nn.ravel()
    f_test = fids_at(sp, tt, nn)
    s = FD_H * FD_ST
    f_dth = fids_at(sp, (tt[:, None] + s).ravel(), np.repeat(nn, len(s))).reshape(len(tt), len(s), len(m.PROBES), m.N_ACQ)
    f_dnu = fids_at(sp, np.repeat(tt, len(s)), (nn[:, None] + s).ravel()).reshape(len(tt), len(s), len(m.PROBES), m.N_ACQ)
    print(f"{a.system}: test + stencils {time.time() - t0:.0f}s", flush=True)
    meta = json.dumps({"stage": 3, "system": a.system, "theta": "log10 sigma (rad/s)",
                       "nu": "log10 tau_c (s), hidden nuisance", "N_levels": N_LEVELS,
                       "noise": "H=(xi/2) sum sz, one OU comp (collapsed telegraph), SLE exact",
                       "n_acq": m.N_ACQ, "dwell": m.DWELL, "sequences": "app.qrc.sequences.cpmg (M0 bank)",
                       "readout": "z = y @ Q, Q orthonormal basis of the line span (sign +1)",
                       "commutator_rel": sp.comm})
    m.TRACES.mkdir(parents=True, exist_ok=True)
    np.savez(s3_trace(a.system, raw=True), fids_table=raw.astype(np.complex64), fids_test=f_test,
             fids_dth=f_dth, fids_dnu=f_dnu, th_grid=TH_GRID, nu_grid=NU_GRID, test_th=tt, test_nu=nn,
             stencil=FD_ST, h=FD_H, fid_dwell=m.DWELL, probes=np.array(m.PROBES), A=A, meta=meta)
    np.savez(s3_trace(a.system), table=raw @ Q, test=f_test @ Q, dth=f_dth @ Q, dnu=f_dnu @ Q, Q=Q,
             th_grid=TH_GRID, nu_grid=NU_GRID, test_th=tt, test_nu=nn, stencil=FD_ST, h=FD_H,
             probes=np.array(m.PROBES), A=A, meta=meta)
    print("wrote", s3_trace(a.system), f"({time.time() - t0:.0f}s)", flush=True)


def s3_sweep(system):
    return m.TRACES / f"qrc_adaptive_stage3_{system}_sweep.npz"


def cmd_gen_sweep(a):
    """Engine FIDs + FD stencils at the 4 extra sweep points (SWEEP_TH, SWEEP_NU_EXTRA)."""
    sysc = m.make_systems()[a.system]
    sp = m.Spins(sysc, a.device)
    assert sp.comm < 1e-10
    Q = np.load(s3_trace(a.system))["Q"]
    tt, nn = np.full(len(SWEEP_NU_EXTRA), SWEEP_TH), SWEEP_NU_EXTRA.copy()
    s = FD_H * FD_ST
    f_test = fids_at(sp, tt, nn)
    f_dth = fids_at(sp, (tt[:, None] + s).ravel(), np.repeat(nn, len(s))).reshape(len(tt), len(s), len(m.PROBES), m.N_ACQ)
    f_dnu = fids_at(sp, np.repeat(tt, len(s)), (nn[:, None] + s).ravel()).reshape(len(tt), len(s), len(m.PROBES), m.N_ACQ)
    np.savez(s3_sweep(a.system), fids_test=f_test, fids_dth=f_dth, fids_dnu=f_dnu,
             test=f_test @ Q, dth=f_dth @ Q, dnu=f_dnu @ Q, test_th=tt, test_nu=nn, stencil=FD_ST, h=FD_H,
             fid_dwell=m.DWELL, probes=np.array(m.PROBES),
             meta=json.dumps({"stage": 3, "system": a.system, "what": "extra misspecification-sweep points",
                              "readout": "z = y @ Q (line span)", "commutator_rel": sp.comm}))
    print("wrote", s3_sweep(a.system), flush=True)


def points(system):
    """All evaluation points: the 25-point grid (k = 0..24) + the 4 sweep extras (k = 25..28).
    Returns dict th, nu, test (K, 49, L), dth, dnu (K, 4, 49, L), h."""
    z = np.load(s3_trace(system))
    out = {k: z[k] for k in ("test_th", "test_nu", "test", "dth", "dnu")}
    if s3_sweep(system).exists():
        w = np.load(s3_sweep(system))
        out = {k: np.concatenate([out[k], w[k]]) for k in out}
    out["h"] = float(z["h"])
    return out


def fisher_2x2(system):
    """Per-probe one-shot 2x2 FI in (theta, nu) at every evaluation point, sigma = 1, z readout:
    (K, 49, 2, 2); and A."""
    z = points(system)
    A = float(np.load(s3_trace(system))["A"])
    Jt = s12.jac_from_stencil(z["dth"], z["h"])                        # (K, 49, L)
    Jn = s12.jac_from_stencil(z["dnu"], z["h"])
    F = np.stack([np.stack([(Jt.conj() * Jt).real.sum(-1), (Jt.conj() * Jn).real.sum(-1)], -1),
                  np.stack([(Jn.conj() * Jt).real.sum(-1), (Jn.conj() * Jn).real.sum(-1)], -1)], -2)
    return F, A


def c_optimal(F, n_w=1001):
    """min over designs xi of [(sum xi_j F_j)^-1]_theta,theta (per shot), F (P, 2, 2).
    c-optimal designs for 2 parameters need <= 2 support points: brute force over pairs."""
    w = np.linspace(0, 1, n_w)[:, None, None, None, None]
    M = w * F[None, :, None] + (1 - w) * F[None, None, :]              # (n_w, P, P, 2, 2)
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    v = np.where(det > 1e-12 * (M[..., 0, 0] * M[..., 1, 1]).clip(min=1e-300), M[..., 1, 1] / det, np.inf)
    i = np.unravel_index(np.argmin(v), v.shape)
    return float(v[i]), (int(i[1]), int(i[2]), float(w[i[0], 0, 0, 0, 0]))


def bounds(system, eta):
    """Per-shot bounds on theta at every evaluation point (multiply by 1/N for N shots)."""
    F1, A = fisher_2x2(system)
    F = F1 / est.m0_sigma(eta, A) ** 2
    out = []
    for k in range(F.shape[0]):
        Fm = F[k].mean(0)
        b_nopt, design = c_optimal(F[k])
        out.append({"known_opt": float(1 / F[k][:, 0, 0].max()), "known_bank": float(1 / Fm[0, 0]),
                    "nuis_bank": float(np.linalg.inv(Fm)[0, 0]), "nuis_opt": b_nopt,
                    "nuis_opt_design": design})
    return out


def cmd_bounds(a):
    z = np.load(s3_trace("R3"))
    for system in ("S1", "R3"):
        for eta in (1e-2, 1e-1):
            b = bounds(system, eta)
            k = {key: np.array([x[key] for x in b]) for key in ("known_opt", "known_bank", "nuis_bank", "nuis_opt")}
            print(f"{system} eta={eta:g}  RMSE at N=49 (median over 25): "
                  + "  ".join(f"{key}={np.median(np.sqrt(v / 49)):.2e}" for key, v in k.items())
                  + f"  | nuis_bank/known_bank={np.median(k['nuis_bank'] / k['known_bank']):.1f}"
                  + f" (range {np.min(k['nuis_bank'] / k['known_bank']):.1f}-{np.max(k['nuis_bank'] / k['known_bank']):.1f})"
                  + f"  nuis_bank/nuis_opt={np.median(k['nuis_bank'] / k['nuis_opt']):.2f}"
                  + f" (range {np.min(k['nuis_bank'] / k['nuis_opt']):.2f}-{np.max(k['nuis_bank'] / k['nuis_opt']):.2f})",
                  flush=True)
    print("test points (theta, nu):", np.round(np.c_[z["test_th"], z["test_nu"]], 3).tolist())


MODELS = m.TRACES.parent / "models"


def model_path(system, eta, mode, tag=""):
    return MODELS / f"qrc_adaptive_s3_{system}_eta{eta:g}_{mode}{tag}.pt"


def make_sim(system, eta, device):
    from app.qrc.neural_policy import TableSim
    z = np.load(s3_trace(system))
    return TableSim(z["table"], z["th_grid"], z["nu_grid"], est.m0_sigma(eta, float(z["A"])),
                    float(z["A"]), device=device), z


def cmd_train(a):
    import torch
    from app.qrc import neural_policy as npol
    torch.set_num_threads(2)
    sim, _ = make_sim(a.system, a.eta, a.device)
    t0 = time.time()
    log = lambda h: print(json.dumps({**h, "t": round(time.time() - t0, 1)}), flush=True)  # noqa: E731
    est_, pol, hist = npol.train(sim, a.N, a.steps, a.B, a.seed, mode=a.mode, lr=a.lr, log_every=a.log_every, log=log)
    MODELS.mkdir(parents=True, exist_ok=True)
    path = model_path(a.system, a.eta, a.mode, a.tag)
    torch.save({"est": est_.state_dict(), "pol": pol.state_dict(), "hist": hist,
                "args": vars(a), "train_seconds": time.time() - t0}, path)
    print("wrote", path, f"({time.time() - t0:.0f}s)", flush=True)


POLICIES = ("nn-fixed", "nn-adaptive", "bayes-misspec", "bayes-oracle", "bayes-joint-fixed", "bayes-joint")
NU_MISSPEC = -3.0                                  # tau_c = 1 ms (the Stage 1-2 model)


def s3_eval_trace(system, eta, tag=""):
    return m.TRACES / f"qrc_adaptive_stage3_eval_{system}_eta{eta:g}{tag}.npz"


def s3_noise(system, eta, k, R, N, L):
    """Unit complex shot noise (R, N, L), common to every policy (paired)."""
    out = np.empty((R, N, L), complex)
    for r in range(R):
        rng = np.random.default_rng([800000, {"S1": 1, "R3": 2}[system], int(round(-np.log10(eta) * 10)), k, r])
        for t in range(N):
            out[r, t] = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    return out


def _nn_eval(sim, ckpt, mode, mu, th, noise, N):
    import torch
    from app.qrc import neural_policy as npol
    d_in = sim.P * (2 * sim.L + 2) + 1
    est_ = npol.Estimator(d_in, (sim.th_lo, sim.th_hi), (sim.nu_lo, sim.nu_hi)).to(sim.dev)
    pol = npol.PolicyNet(d_in, sim.P).to(sim.dev)
    est_.load_state_dict(ckpt["est"])
    pol.load_state_dict(ckpt["pol"])
    est_.eval(); pol.eval()
    B = noise.shape[0]
    with torch.no_grad():
        out = npol.rollout(sim, est_, pol, torch.full((B,), th, dtype=torch.float64, device=sim.dev), None,
                           N, None, "greedy" if mode == "adaptive" else "fixed-cycle",
                           mu=torch.as_tensor(mu, dtype=torch.complex64, device=sim.dev).expand(B, -1, -1),
                           noise=torch.as_tensor(noise, dtype=torch.complex64, device=sim.dev))
    return out["est"][..., 0].double().cpu().numpy(), out["chosen"].cpu().numpy()


def cmd_eval(a):
    """Episodes at every evaluation point (25-point grid + 4 sweep extras) for every policy,
    paired noise -> eval trace .npz (rewritten after each point, before any metric).
    a.R episodes per point for the neural and 1-D Bayesian policies; the 2-D joint policies run
    on the first a.R2d of the same episodes (cost)."""
    import torch
    from app.qrc.adaptive_batch import BatchedEpisodes, BatchedEpisodes2D
    torch.set_num_threads(2)
    sim, z = make_sim(a.system, a.eta, a.device)
    T, th_g, nu_g = z["table"], z["th_grid"], z["nu_grid"]
    sig = sim.sigma
    ck = {mo: torch.load(model_path(a.system, a.eta, mo, a.model_tag), map_location=sim.dev)
          for mo in ("fixed-cycle", "adaptive")}
    pols = [p for p in POLICIES if p in a.policies]
    iv = int(np.argmin(np.abs(nu_g - NU_MISSPEC)))
    assert abs(nu_g[iv] - NU_MISSPEC) < 1e-12
    be_mis = BatchedEpisodes(T[:, iv], th_g, sig, TH_LO, TH_HI, n_grid=201, device=a.device)
    be_2d = BatchedEpisodes2D(T, th_g, nu_g, sig, ((TH_LO, TH_HI), (NU_LO, NU_HI)), device=a.device)
    if a.dev:      # dev points (pilot only; never the test grid): random, table signals
        rd = np.random.default_rng(990)
        test_th = rd.uniform(TH_LO + 0.1, TH_HI - 0.1, a.dev)
        test_nu = rd.uniform(NU_LO + 0.1, NU_HI - 0.1, a.dev)
        sig_mu = sim.mu(torch.as_tensor(test_th, device=sim.dev), torch.as_tensor(test_nu, device=sim.dev)).cpu().numpy().astype(complex)
    else:
        assert s3_sweep(a.system).exists(), "run gen-sweep first"
        pts = points(a.system)
        test_th, test_nu, sig_mu = pts["test_th"], pts["test_nu"], pts["test"]
    K = len(test_th)
    R2d = min(a.R2d, a.R)
    means = np.full((K, len(POLICIES), a.R, a.N), np.nan)
    chosen = np.full((K, len(POLICIES), a.R, a.N), -1, np.int16)
    unres = np.full((K, len(POLICIES), a.N), np.nan)       # 2-D runner: fraction of episodes hitting max_zoom
    out = s3_eval_trace(a.system, a.eta, a.tag + ("_dev" if a.dev else ""))

    def chunked(fn, noise, size):
        rs = [fn(noise[c:c + size]) for c in range(0, len(noise), size)]
        return np.concatenate([r[0] for r in rs]), np.concatenate([r[1] for r in rs])

    t0 = time.time()
    for k in range(K):
        th, nu = float(test_th[k]), float(test_nu[k])
        mu = sig_mu[k]
        noise = s3_noise(a.system, a.eta, k + (1000 if a.dev else 0), a.R, a.N, sim.L)
        for p in pols:
            ip = POLICIES.index(p)
            R_p = R2d if p.startswith("bayes-joint") else a.R
            if p.startswith("nn-"):
                mn, ch = chunked(lambda nz: _nn_eval(sim, ck["fixed-cycle" if p == "nn-fixed" else "adaptive"],
                                                     "fixed" if p == "nn-fixed" else "adaptive", mu, th, nz, a.N),
                                 noise, a.chunk)
            elif p == "bayes-misspec":
                mn, ch = chunked(lambda nz: be_mis.run(mu, nz, "greedy-fisher")[::2], noise, a.chunk)
            elif p == "bayes-oracle":
                # 1-D model at the true nu: Lagrange-interpolate the table along nu
                idx, w = lagrange_1d(np.array([nu]), nu_g)
                be = BatchedEpisodes(np.einsum("k,tkpl->tpl", w[0], T[:, idx[0]]), th_g, sig, TH_LO, TH_HI,
                                     n_grid=201, device=a.device)
                mn, ch = chunked(lambda nz: be.run(mu, nz, "greedy-fisher")[::2], noise, a.chunk)
            else:
                mns, chs, urs = [], [], []
                for c in range(0, R2d, a.chunk2d):
                    o = be_2d.run(mu, noise[c:min(c + a.chunk2d, R2d)], "fixed-cycle" if p == "bayes-joint-fixed" else "greedy-c")
                    mns.append(o[0]); chs.append(o[2]); urs.append(np.array(be_2d.unresolved) * len(o[0]))
                mn, ch = np.concatenate(mns), np.concatenate(chs)
                unres[k, ip] = np.sum(urs, 0) / len(mn)
            means[k, ip, :R_p], chosen[k, ip, :R_p] = mn, ch
            print(f"[{a.system} eta={a.eta:g}] point {k} ({th:.3f},{nu:.3f}) {p}: "
                  f"rmse_N={np.sqrt(np.mean((mn[:, -1] - th) ** 2)):.2e} (M={R_p}, {time.time() - t0:.0f}s)", flush=True)
        np.savez(out, means=means, chosen=chosen, unresolved2d=unres, done=np.arange(K) <= k,
                 test_th=test_th, test_nu=test_nu, policies=np.array(POLICIES), eta=a.eta, N=a.N, R=a.R, R2d=R2d,
                 meta=json.dumps({"stage": 3, "system": a.system, "eta": a.eta, "model_tag": a.model_tag,
                                  "policies_run": pols, "noise_rng": "[800000, sys, round(10 log10 1/eta), k, r]",
                                  "points": "k 0-24: 5x5 grid; k 25-28: sweep extras (theta0, nu midpoints)",
                                  "readout": "z = y @ Q (line span); D = 2L real per shot",
                                  "nu_misspec": NU_MISSPEC, "posterior2d": "v2 (window ~W std, max_zoom 30)"}))
    print("wrote", out, f"({time.time() - t0:.0f}s)", flush=True)


def crossover(dnu, rho):
    """Smallest |nu - nu0| at which rho reaches 1 (linear in log rho between sweep points);
    0 if rho >= 1 already at nu0; None if never reached."""
    o = np.argsort(dnu)
    d, r = np.asarray(dnu)[o], np.log(np.asarray(rho)[o])
    if r[0] >= 0:
        return 0.0
    for i in range(1, len(d)):
        if r[i] >= 0:
            return float(d[i - 1] + (d[i] - d[i - 1]) * (0 - r[i - 1]) / (r[i] - r[i - 1]))
    return None


def cmd_analyze(a):
    """Tier A gates, Tier B (primary) and Tier C from the saved eval traces
    (QRC_AdaptiveEst.md sec. 4.6). Everything in MSE. -> results JSON."""
    res = {"prereg": a.prereg, "tag": a.tag, "units": "MSE (variance)", "cells": {}}
    for system in a.systems:
        tr = s3_eval_trace(system, a.eta, a.tag)
        if not tr.exists():
            print("missing", tr)
            continue
        z = np.load(tr)
        assert bool(z["done"].all()), "incomplete trace"
        P = list(z["policies"])
        th, nu = z["test_th"], z["test_nu"]
        N = int(z["N"])
        b = bounds(system, a.eta)
        assert len(b) == len(th)
        mse = {p: np.nanmean((z["means"][:, P.index(p), :, N - 1] - th[:, None]) ** 2, 1) for p in P}
        kopt = np.array([x["known_opt"] for x in b]) / N            # nu known, best probe
        crb_nu = np.array([x["nuis_opt"] for x in b]) / N           # CRB_nu: nu unknown, best design
        nbank = np.array([x["nuis_bank"] for x in b]) / N
        grid = np.arange(len(th)) < 25
        dnu = np.abs(nu - NU_MISSPEC)
        well = grid & (dnu < 1e-9)                                   # the 5 well-specified grid points
        sweep = np.abs(th - SWEEP_TH) < 1e-9                         # 9 points at the low-theta edge
        far = sweep & (dnu >= 0.4)
        assert well.sum() == 5 and sweep.sum() == 9 and far.sum() == 4
        rho = mse["bayes-misspec"] / mse["nn-adaptive"]
        med = lambda x: float(np.median(x))                         # noqa: E731
        lo_side, hi_side = sweep & (nu <= NU_MISSPEC + 1e-9), sweep & (nu >= NU_MISSPEC - 1e-9)
        x_lo, x_hi = crossover(dnu[lo_side], rho[lo_side]), crossover(dnu[hi_side], rho[hi_side])
        u49 = float(np.nanmax(z["unresolved2d"][:, [P.index("bayes-joint"), P.index("bayes-joint-fixed")], N - 1]))
        c = {"A1a_known_opt_over_misspec_well": med(kopt[well] / mse["bayes-misspec"][well]),
             "A1b_crb_nu_over_joint": med(crb_nu[grid] / mse["bayes-joint"][grid]),
             "A2_unresolved_at_N": u49,
             "A3": "procedural: training draws (theta, nu) continuously from the prior via TableSim (table only); test signals are engine FIDs never loaded by train",
             "Q3M_rho_far_median": med(rho[far]),
             "Q3M2_crossover": {"nu_below_nu0": x_lo, "nu_above_nu0": x_hi},
             "Q3a_e_nu_nn_well": med(crb_nu[well] / mse["nn-adaptive"][well]),
             "Q3b_r_adapt": med(mse["nn-fixed"][grid] / mse["nn-adaptive"][grid]),
             "desc": {"sweep": {"nu": nu[sweep].tolist(), "dnu": dnu[sweep].tolist(), "rho": rho[sweep].tolist()},
                      "rho_grid_offcentre_median": med(rho[grid & (dnu > 1e-9)]),
                      "e_nu_nn_all": med(crb_nu[grid] / mse["nn-adaptive"][grid]),
                      "e_nu_joint_well": med(crb_nu[well] / mse["bayes-joint"][well]),
                      "e_bank_joint_fixed": med(nbank[grid] / mse["bayes-joint-fixed"][grid]),
                      "e_bank_nn_fixed": med(nbank[grid] / mse["nn-fixed"][grid]),
                      "e_known_oracle": med(kopt[grid] / mse["bayes-oracle"][grid]),
                      "joint_over_nn": med(mse["bayes-joint"][grid] / mse["nn-adaptive"][grid])},
             "per_point_mse": {p: mse[p].tolist() for p in P},
             "bounds_mse": {"known_opt": kopt.tolist(), "crb_nu": crb_nu.tolist(), "nuis_bank": nbank.tolist()},
             "test_th": th.tolist(), "test_nu": nu.tolist()}
        c["verdict"] = {"A1a": 0.8 <= c["A1a_known_opt_over_misspec_well"] <= 1.25,
                        "A1b": 0.8 <= c["A1b_crb_nu_over_joint"] <= 1.25,
                        "A2": u49 <= 0.05, "A3": True,
                        "Q3M": c["Q3M_rho_far_median"] >= 2.0,
                        "Q3M2": x_lo is not None and x_hi is not None,
                        "Q3a": c["Q3a_e_nu_nn_well"] >= 0.3, "Q3b": c["Q3b_r_adapt"] >= 1.5}
        c["void"] = not all(c["verdict"][k] for k in ("A1a", "A1b", "A2", "A3"))  # sec. 4.5: any Tier A fail
        res["cells"][f"{system}_eta{a.eta:g}"] = c
        print(system, json.dumps({k: v for k, v in c.items() if k not in ("per_point_mse", "bounds_mse", "test_th", "test_nu", "A3")}), flush=True)
    cells = res["cells"].values()
    if len(res["cells"]) == 2:
        res["overall"] = {k: all(c["verdict"][k] for c in cells) for k in ("A1a", "A1b", "A2", "A3", "Q3M", "Q3M2", "Q3a", "Q3b")}
        res["overall"]["void_systems"] = [s for s, c in res["cells"].items() if c["void"]]
    (m.DATA / f"qrc_adaptive_stage3{a.tag}_results.json").write_text(json.dumps(res, indent=1))


def cmd_figure(a):
    """Fig. AE3 from the results JSON: (a) rho(dnu) on the theta0 sweep, (b) MSE vs nu at theta0 (R3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.loads((m.DATA / f"qrc_adaptive_stage3{a.tag}_results.json").read_text())
    col = {"R3": "#2a78d6", "S1": "#eb6834"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for s in ("R3", "S1"):
        c = res["cells"][f"{s}_eta{a.eta:g}"]
        sw = c["desc"]["sweep"]
        x = np.array(sw["nu"]) - NU_MISSPEC
        o = np.argsort(x)
        void = c["void"]
        ax[0].plot(x[o], np.array(sw["rho"])[o], "-o", lw=2, ms=8, color=col[s],
                   mfc="white" if void else col[s], label=f"{s}" + (" (VOID: gate A2)" if void else ""))
    ax[0].set_yscale("log")
    ax[0].axhline(1, color="#888", lw=1)
    ax[0].axhline(2, color="#888", lw=1, ls="--")
    for xv in (-0.4, 0.4):
        ax[0].axvline(xv, color="#bbb", lw=1, ls=":")
    ax[0].text(-0.58, 2.3, "Q3-M bar ρ = 2", fontsize=8, color="#555")
    ax[0].set_xlabel("ν − ν₀  (log10 τc; ν₀ = 1 ms)")
    ax[0].set_ylabel("ρ = MSE(ν₀-assumed Bayes) / MSE(NN adaptive)")
    ax[0].set_title(f"(a) misspecification ratio, θ₀ = {SWEEP_TH:.3f}, shot {a.N}", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    c = res["cells"][f"R3_eta{a.eta:g}"]
    th, nu = np.array(c["test_th"]), np.array(c["test_nu"])
    sw = np.abs(th - SWEEP_TH) < 1e-9
    o = np.argsort(nu[sw])
    xs = nu[sw][o] - NU_MISSPEC
    pm, bm = c["per_point_mse"], c["bounds_mse"]
    for key, lab, cc in (("bayes-misspec", "Bayes, ν₀ assumed", "#eb6834"),
                         ("nn-adaptive", "NN adaptive (model-free)", "#2a78d6"),
                         ("bayes-joint", "Bayes, joint (θ, ν)", "#1baf7a")):
        ax[1].plot(xs, np.array(pm[key])[sw][o], "-o", lw=2, ms=7, color=cc, label=lab)
    ax[1].plot(xs, np.array(bm["crb_nu"])[sw][o], "--", lw=1.5, color="#555", label="CRB_ν (ν unknown)")
    ax[1].plot(xs, np.array(bm["known_opt"])[sw][o], ":", lw=1.5, color="#555", label="K (ν known)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("ν − ν₀")
    ax[1].set_ylabel("MSE of θ̂ at shot 49")
    ax[1].set_title("(b) R3, same sweep", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
        x.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "figAE3_stage3_misspec.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot-interp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    gs = sub.add_parser("gen-sweep")
    gs.add_argument("--system", choices=["S1", "R3"], required=True)
    gs.add_argument("--device", default="cuda")
    g = sub.add_parser("gen")
    g.add_argument("--system", choices=["S1", "R3"], required=True)
    g.add_argument("--device", default="cuda")
    sub.add_parser("bounds")
    t = sub.add_parser("train")
    t.add_argument("--system", choices=["S1", "R3"], required=True)
    t.add_argument("--eta", type=float, required=True)
    t.add_argument("--mode", choices=["adaptive", "fixed-cycle"], required=True)
    t.add_argument("--N", type=int, default=49)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--B", type=int, default=1024)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=500)
    t.add_argument("--tag", default="")
    t.add_argument("--device", default="cuda")
    e = sub.add_parser("eval")
    e.add_argument("--system", choices=["S1", "R3"], required=True)
    e.add_argument("--eta", type=float, required=True)
    e.add_argument("--N", type=int, default=49)
    e.add_argument("--R", type=int, default=2000)
    e.add_argument("--R2d", type=int, default=200)
    e.add_argument("--chunk", type=int, default=500)
    e.add_argument("--policies", nargs="*", default=list(POLICIES))
    e.add_argument("--model-tag", default="")
    e.add_argument("--tag", default="")
    e.add_argument("--chunk2d", type=int, default=50)
    e.add_argument("--dev", type=int, default=0, help="pilot: N random dev points instead of the test grid")
    e.add_argument("--device", default="cuda")
    y = sub.add_parser("analyze")
    y.add_argument("--systems", nargs="*", default=["R3", "S1"])
    y.add_argument("--eta", type=float, default=0.1)
    y.add_argument("--N", type=int, default=49)
    y.add_argument("--tag", default="")
    y.add_argument("--prereg", default="")
    f = sub.add_parser("figure")
    f.add_argument("--eta", type=float, default=0.1)
    f.add_argument("--N", type=int, default=49)
    f.add_argument("--tag", default="_final")
    a = ap.parse_args()
    {"pilot-interp": cmd_pilot_interp, "gen": cmd_gen, "gen-sweep": cmd_gen_sweep, "bounds": cmd_bounds,
     "train": cmd_train, "eval": cmd_eval, "analyze": cmd_analyze, "figure": cmd_figure}[a.cmd](a)


if __name__ == "__main__":
    main()
