"""De-risking Gate 2, regime b (QRC_Gate2_Richness.md sec. 2.9): train one fresh model-free policy
per system on a broad noise prior (single OU + multi-Lorentzian + discrete-fluctuator 1/f +
strong-coupling telegraph + drift at any tau), then score it on the frozen regime-a configurations
plus the corrected drift family (drift at tau != tau_0).

  python scripts/qrc_gate2b.py gen-bank --system R3        # training bank (engine, raw FIDs kept)
  python scripts/qrc_gate2b.py pilot-bank --system R3      # theta-interpolation accuracy (bank only)
  python scripts/qrc_gate2b.py gen-bound --system R3       # theta stencils -> known-shape bound K
  python scripts/qrc_gate2b.py train --system R3
  python scripts/qrc_gate2b.py eval --system R3
  python scripts/qrc_gate2b.py analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qrc_adaptive_s3 as s3  # noqa: E402
import qrc_gate2_richness as g2  # noqa: E402
import qrc_noisespec_m0 as m  # noqa: E402

SYS_ID = {"S1": 1, "R3": 2}
BANK_TH = np.linspace(s3.TH_LO, s3.TH_HI, 41)
BANK_N = {"ML2": 60, "ML3": 60, "1f": 60, "TEL": 60}
MIX = {"ou": 0.15, "drift": 0.20, "bank": 0.65}     # training-prior weights
DRIFT_DMAX = 0.30
TRAIN = {"steps": 20000, "B": 1024, "lr": 3e-4, "seed": {"S1": 940, "R3": 930}}   # Stage-3 budget
TAG_B = "_gate2b"
FD_H = 1e-3
NEAR_EFF = 2.0                                      # baseline with MSE <= 2 K is near-efficient
BARS = {"rho_median": 2.0, "rho_worst": 1.0, "g_worst": 1.0, "twin_worst": 2.0}


# --- training bank -------------------------------------------------------------------------
def bank_configs(system):
    """Random training configurations (seeds disjoint from families(): 31337)."""
    rng = np.random.default_rng([31337, SYS_ID[system]])
    lo, hi = np.log10(1e-4), np.log10(2e-2)          # tau 0.1 - 20 ms (wider than the test draws)
    out = []
    for C in (2, 3):
        k = 0
        while k < BANK_N[f"ML{C}"]:
            lt = rng.uniform(lo, hi, C)
            if lt.max() - lt.min() < 0.6:
                continue
            f = rng.dirichlet(np.ones(C))
            lv = [(20 if t >= -3 else 7) if C == 2 else (9 if t >= -3 else 4) for t in lt]
            out.append({"name": f"bML{C}-{k}", "family": "multi-lorentzian",
                        "comps": [(float(a), float(10 ** t), n) for a, t, n in zip(f, lt, lv)]})
            k += 1
    for k in range(BANK_N["1f"]):
        alpha = rng.uniform(0.6, 1.4)
        wlo, whi = 10 ** rng.uniform(np.log10(30), np.log10(100)), 10 ** rng.uniform(np.log10(5e3), np.log10(2e4))
        tau = np.logspace(np.log10(1 / whi), np.log10(1 / wlo), 5)
        w = tau ** (alpha - 1.0)
        out.append({"name": f"b1f-{k}", "family": "1/f", "alpha": float(alpha),
                    "comps": [(float(x), float(t), n) for x, t, n in zip(w / w.sum(), tau, (1, 2, 3, 5, 7))]})
    for k in range(BANK_N["TEL"]):
        out.append({"name": f"bTEL-{k}", "family": "telegraph",
                    "comps": [(1.0, float(10 ** rng.uniform(np.log10(2e-4), np.log10(2e-2))), 1)]})
    return out


def nu_aux(c):
    """Auxiliary nu label (estimator head / critic only): log10 power-weighted tau, in the box."""
    return float(np.clip(np.log10(sum(f * t for f, t, _ in c["comps"])), s3.NU_LO, s3.NU_HI))


def bank_trace(system):
    return m.TRACES / f"qrc_gate2b_bank_{system}.npz"


def bound_trace(system):
    return m.TRACES / f"qrc_gate2b_bound_{system}.npz"


def g2b_eval_trace(system):
    return m.TRACES / f"qrc_gate2b_eval_{system}_eta{g2.ETA:g}.npz"


def engine_theta(sp, c, thetas, chunk=41):
    """Engine FIDs (len(thetas), 49, N_ACQ) for one configuration, batched over theta. The engine
    caches one (B, K, K) propagator per segment length and coherence class, so the batch is
    capped at B K^2 <= 9e6 (GPU memory)."""
    K = int(np.prod([n + 1 for *_, n in c["comps"]]))
    chunk = max(1, min(chunk, int(9e6 // K ** 2)))
    out = []
    for c0 in range(0, len(thetas), chunk):
        B = m.Batch(sp, [g2.plist(c, t) for t in thetas[c0:c0 + chunk]], s3.N_LEVELS)
        out.append(np.stack([B.probe(n_pi, t_ms * 1e-3).cpu().numpy() for n_pi, t_ms in m.PROBES], 1))
        del B
        torch.cuda.empty_cache()
    return np.concatenate(out)


def cmd_gen_bank(a):
    sp = m.Spins(m.make_systems()[a.system], a.device)
    assert sp.comm < 1e-10
    Q = np.load(s3.s3_trace(a.system))["Q"]
    cfgs = bank_configs(a.system)
    raw = np.zeros((len(cfgs), len(BANK_TH), len(m.PROBES), m.N_ACQ), np.complex64)
    t0 = time.time()
    for i, c in enumerate(cfgs):
        raw[i] = engine_theta(sp, c, BANK_TH, a.chunk)
        if i % 20 == 0 or i == len(cfgs) - 1:
            print(f"{a.system} bank {i + 1}/{len(cfgs)} {c['name']} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(bank_trace(a.system), fids=raw, sig=raw.astype(np.complex128) @ Q, th_grid=BANK_TH,
             names=np.array([c["name"] for c in cfgs]), families=np.array([c["family"] for c in cfgs]),
             nu_aux=np.array([nu_aux(c) for c in cfgs]), fid_dwell=m.DWELL, probes=np.array(m.PROBES),
             meta=json.dumps({"gate": "2b", "system": a.system, "what": "training bank (regime b)",
                              "configs": cfgs, "seed": [31337, SYS_ID[a.system]], "readout": "z = y @ Q"}))
    print("wrote", bank_trace(a.system), f"({time.time() - t0:.0f}s)", flush=True)


def cmd_pilot_bank(a):
    """Interpolation error of the 41-point theta table (bank configurations only, never a test
    configuration): engine at random off-grid theta vs 6-point Lagrange, in units
    max_j ||dmu_j|| sqrt(49) / sigma (as d)."""
    from app.qrc import estimation as est
    sp = m.Spins(m.make_systems()[a.system], a.device)
    z = np.load(bank_trace(a.system))
    Q, A = np.load(s3.s3_trace(a.system))["Q"], float(np.load(s3.s3_trace(a.system))["A"])
    sig = est.m0_sigma(g2.ETA, A)
    cfgs = bank_configs(a.system)
    rng = np.random.default_rng(77)
    out = []
    for i in rng.choice(len(cfgs), 8, replace=False):
        th = rng.uniform(s3.TH_LO + 0.05, s3.TH_HI - 0.05, 3)
        ex = engine_theta(sp, cfgs[i], th) @ Q
        idx, w = s3.lagrange_1d(th, BANK_TH)
        ip = np.einsum("nk,nkpl->npl", w, z["sig"][i][idx])
        d = float(np.linalg.norm(ex - ip, axis=-1).max() * np.sqrt(g2.N_SHOTS) / sig)
        out.append({"cfg": cfgs[i]["name"], "d_interp": d})
        print(out[-1], flush=True)
    print("max d_interp", max(o["d_interp"] for o in out))


# --- known-shape bound K (theta stencils at every scored static configuration) ------------
def cmd_gen_bound(a):
    sp = m.Spins(m.make_systems()[a.system], a.device)
    Q = np.load(s3.s3_trace(a.system))["Q"]
    cfgs = g2.references() + [c for c in g2.families() if c["family"] != "drift"]
    st = np.zeros((len(cfgs), len(g2.THETAS), 2, len(m.PROBES), m.N_ACQ), np.complex128)
    t0 = time.time()
    for i, c in enumerate(cfgs):
        for k, th in enumerate(g2.THETAS):
            st[i, k] = engine_theta(sp, c, np.array([th - FD_H, th + FD_H]), chunk=1)   # K up to 3528
        print(f"{a.system} {c['name']} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(bound_trace(a.system), fids_pm=st, sig_pm=st @ Q, h=FD_H,
             names=np.array([c["name"] for c in cfgs]), thetas=np.array(g2.THETAS),
             meta=json.dumps({"gate": "2b", "what": "theta -h/+h engine stencils for K", "system": a.system}))
    print("wrote", bound_trace(a.system), f"({time.time() - t0:.0f}s)", flush=True)


# --- corrected drift (tau != tau_0) ----------------------------------------------------------
def drift_c_configs():
    """Drift around the two outer Stage-3 tau values (0.28 ms, 3.55 ms), where the tau_0 baseline
    is misspecified; stationary twin = the reference at the same tau (REF-0 / REF-4)."""
    out = []
    for kn, twin in ((0, "REF-0"), (4, "REF-4")):
        for kind in ("sigma-linear", "sigma-rw", "tau-linear"):
            for d in g2.DRIFT_DELTA:
                out.append({"name": f"DC-{kind}-{int(d * 100)}-nu{kn}", "family": "drift-c", "kind": kind,
                            "delta": d, "nu": float(s3.TEST_NU[kn]), "twin": twin})
    return out


def drift_paths(c, theta, R, seed):
    """As g2.drift_schedule, around nu = c['nu']."""
    th, nu = g2.drift_schedule(c, theta, R, seed)
    return th, nu - np.log10(g2.TAU0) + c.get("nu", np.log10(g2.TAU0))


# --- mixture simulator ---------------------------------------------------------------------
class MixtureSim:
    """Broad training prior. sample_prior -> (theta, nu_aux, per-shot mu (B, N, P, L))."""

    def __init__(self, sim, bank, device):
        import torch
        self.s, self.dev = sim, sim.dev
        for k in ("P", "L", "sigma", "A", "th_lo", "th_hi", "nu_lo", "nu_hi"):
            setattr(self, k, getattr(sim, k))
        self.T = torch.as_tensor(bank["sig"], dtype=torch.complex64, device=self.dev)   # (C, n_th, P, L)
        self.nu_aux = torch.as_tensor(bank["nu_aux"], dtype=torch.float64, device=self.dev)
        self.th0, self.hth, self.n_th = float(BANK_TH[0]), float(BANK_TH[1] - BANK_TH[0]), len(BANK_TH)
        self.N = g2.N_SHOTS

    def mu(self, th, nu):
        return self.s.mu(th, nu)

    def _bank_mu(self, th, ci):
        import torch
        from app.qrc.neural_policy import lagrange_weights
        idx, w, _ = lagrange_weights(th, self.th0, self.hth, self.n_th, 6)
        Ti = self.T[ci[:, None], idx]                                   # (b, 6, P, L)
        return torch.einsum("bk,bkpl->bpl", w.to(torch.float32).to(Ti.dtype), Ti)

    def sample_prior(self, B, gen):
        import torch
        dev, N = self.dev, self.N
        u = lambda n: torch.rand(n, generator=gen, device=dev, dtype=torch.float64)   # noqa: E731
        th = self.th_lo + 0.05 + (self.th_hi - self.th_lo - 0.1) * u(B)
        kind = torch.bucketize(u(B), torch.tensor([MIX["ou"], MIX["ou"] + MIX["drift"]], device=dev, dtype=torch.float64))
        nu = self.nu_lo + (self.nu_hi - self.nu_lo) * u(B)
        mu = torch.empty(B, N, self.P, self.L, dtype=torch.complex64, device=dev)
        o = kind == 0
        if o.any():
            mu[o] = self.s.mu(th[o], nu[o]).to(mu.dtype)[:, None].expand(-1, N, -1, -1)
        b = kind == 2
        if b.any():
            ci = torch.randint(0, self.T.shape[0], (int(b.sum()),), generator=gen, device=dev)
            mu[b] = self._bank_mu(th[b], ci)[:, None].expand(-1, N, -1, -1)
            nu[b] = self.nu_aux[ci]
        d = kind == 1
        if d.any():
            nd = int(d.sum())
            lg = torch.log10(1 + DRIFT_DMAX * u(nd))[:, None]
            t = (torch.arange(N, device=dev, dtype=torch.float64) / (N - 1) - 0.5)[None]
            dk = torch.randint(0, 3, (nd,), generator=gen, device=dev)[:, None]
            rw = torch.cumsum(torch.randn(nd, N, generator=gen, device=dev, dtype=torch.float64), 1) / np.sqrt(N - 1)
            rw = rw - rw[:, :1]
            pth = torch.where(dk == 0, lg * t, torch.where(dk == 1, lg * rw, torch.zeros_like(rw)))
            pth = pth - 0.5 * torch.log10(torch.mean(10.0 ** (2 * pth), 1, keepdim=True)) + th[d][:, None]
            pnu = nu[d][:, None] + torch.where(dk == 2, lg * t, torch.zeros_like(rw))
            pth = pth.clamp(self.th_lo, self.th_hi)
            pnu = pnu.clamp(self.nu_lo, self.nu_hi)
            mu[d] = self.s.mu(pth.reshape(-1), pnu.reshape(-1)).to(mu.dtype).reshape(nd, N, self.P, self.L)
            nu[d] = pnu.mean(1)
        return th, nu, mu


def cmd_train(a):
    import torch
    from app.qrc import neural_policy as npol
    torch.set_num_threads(2)
    sim, _ = s3.make_sim(a.system, g2.ETA, a.device)
    msim = MixtureSim(sim, np.load(bank_trace(a.system)), a.device)
    t0 = time.time()
    log = lambda h: print(json.dumps({**h, "t": round(time.time() - t0, 1)}), flush=True)  # noqa: E731
    steps = a.steps or TRAIN["steps"]
    est_, pol, hist = npol.train(msim, g2.N_SHOTS, steps, TRAIN["B"], TRAIN["seed"][a.system], mode="adaptive",
                                 lr=TRAIN["lr"], log_every=500, log=log)
    path = s3.model_path(a.system, g2.ETA, "adaptive", TAG_B + a.tag)
    torch.save({"est": est_.state_dict(), "pol": pol.state_dict(), "hist": hist, "args": vars(a),
                "train": TRAIN, "mix": MIX, "train_seconds": time.time() - t0}, path)
    print("wrote", path, f"({time.time() - t0:.0f}s)", flush=True)


# --- scoring -------------------------------------------------------------------------------
POLICIES_B = ("nn-b", "nn-adaptive", "bayes-misspec")


def cmd_eval(a):
    """nn-b on every configuration (references, frozen families incl. the VOID drift, corrected
    drift); nn-adaptive (regime a) and the baseline on the corrected drift (their other numbers are
    the regime-a trace, same signals and noise); nn-adaptive re-run on REF-2 as an identity check."""
    import torch
    from app.qrc.adaptive_batch import BatchedEpisodes
    torch.set_num_threads(2)
    sim, z = s3.make_sim(a.system, g2.ETA, a.device)
    T, th_g, nu_g = z["table"], z["th_grid"], z["nu_grid"]
    ck = {"nn-b": {"adaptive": torch.load(s3.model_path(a.system, g2.ETA, "adaptive", TAG_B), map_location=sim.dev)},
          "nn-adaptive": {"adaptive": torch.load(s3.model_path(a.system, g2.ETA, "adaptive", g2.MODEL_TAG), map_location=sim.dev)}}
    iv = int(np.argmin(np.abs(nu_g - np.log10(g2.TAU0))))
    be_mis = BatchedEpisodes(T[:, iv], th_g, sim.sigma, s3.TH_LO, s3.TH_HI, n_grid=201, device=a.device)
    gz = np.load(g2.g2_trace(a.system))
    static = {str(n): gz["sig"][i] for i, n in enumerate(gz["names"])}
    cfgs = g2.references() + g2.families() + drift_c_configs()
    K, R = len(cfgs), a.R
    means = np.full((K, len(g2.THETAS), len(POLICIES_B), R, g2.N_SHOTS), np.nan)
    out = g2b_eval_trace(a.system)
    t0 = time.time()
    for kt, theta in enumerate(g2.THETAS):
        noise = g2.g2_noise(a.system, kt, R, sim.L)                 # identical to regime a
        for i, c in enumerate(cfgs):
            if c["family"] in ("drift", "drift-c"):
                thd, nud = drift_paths(c, theta, R, seed=[SYS_ID[a.system], kt, i])   # i: same index as regime a
            run = ["nn-b"] + (["nn-adaptive", "bayes-misspec"] if c["family"] == "drift-c" or
                              (c["name"] == "REF-2" and kt == 0) else [])
            for p in run:
                ip = POLICIES_B.index(p)
                mns = []
                for c0 in range(0, R, a.chunk):
                    sl = slice(c0, min(c0 + a.chunk, R))
                    if c["family"] in ("drift", "drift-c"):
                        mu = sim.mu(torch.as_tensor(thd[sl].ravel(), device=sim.dev),
                                    torch.as_tensor(nud[sl].ravel(), device=sim.dev)).reshape(sl.stop - sl.start, g2.N_SHOTS, sim.P, sim.L)
                    else:
                        mu = static[c["name"]][kt]
                    if p.startswith("nn"):
                        mn, _ = g2._nn(sim, ck[p], "nn-adaptive", mu, theta, noise[sl])
                    else:
                        mn = be_mis.run(mu.cpu().numpy().astype(complex) if torch.is_tensor(mu) else mu,
                                        noise[sl], "greedy-fisher")[0]
                    mns.append(mn)
                means[i, kt, ip] = np.concatenate(mns)
            print(f"[{a.system}] th={theta:.3f} {c['name']}: " + " ".join(
                f"{p}={np.mean((means[i, kt, POLICIES_B.index(p), :, -1] - theta) ** 2):.2e}" for p in run)
                + f" ({time.time() - t0:.0f}s)", flush=True)
            np.savez(out, means=means, names=np.array([c["name"] for c in cfgs]),
                     families=np.array([c["family"] for c in cfgs]), thetas=np.array(g2.THETAS),
                     policies=np.array(POLICIES_B), eta=g2.ETA, N=g2.N_SHOTS, R=R,
                     meta=json.dumps({"gate": "2b", "system": a.system, "model_b": TAG_B, "model_a": g2.MODEL_TAG,
                                      "noise_rng": "[900000, sys, k_theta, r] (= regime a)",
                                      "drift_c_rng": "[910000, sys, k_theta, i_cfg]"}))
    print("wrote", out, f"({time.time() - t0:.0f}s)", flush=True)


def known_bound(system):
    """Known-shape best-probe bound on MSE at N = 49 for every scored configuration x theta:
    static: sigma^2 / (49 max_j ||dmu_j/dtheta||^2); drift: sigma^2 / max_j sum_t ||dmu_jt/dtheta||^2
    (whole path shifted; random walk: mean over the first 200 episodes)."""
    import torch
    from app.qrc import estimation as est
    from app.qrc.neural_policy import TableSim
    z = np.load(s3.s3_trace(system))
    sig = est.m0_sigma(g2.ETA, float(z["A"]))
    sim = TableSim(z["table"], z["th_grid"], z["nu_grid"], sig, float(z["A"]), device="cpu", dtype=torch.complex128)
    b = np.load(bound_trace(system))
    out = {}
    for i, n in enumerate(b["names"]):
        J = (b["sig_pm"][i, :, 1] - b["sig_pm"][i, :, 0]) / (2 * b["h"])            # (n_theta, 49, L)
        out[str(n)] = sig ** 2 / (g2.N_SHOTS * (np.abs(J) ** 2).sum(-1).max(-1))
    cfgs = g2.references() + g2.families() + drift_c_configs()
    for i, c in enumerate(cfgs):
        if c["family"] not in ("drift", "drift-c"):
            continue
        Ks = []
        for kt, theta in enumerate(g2.THETAS):
            thd, nud = drift_paths(c, theta, 200, [SYS_ID[system], kt, i])
            f = lambda d: sim.mu(torch.as_tensor((thd + d).ravel()), torch.as_tensor(nud.ravel())).numpy().reshape(200, g2.N_SHOTS, sim.P, sim.L)  # noqa: E731
            J = (f(FD_H) - f(-FD_H)) / (2 * FD_H)
            Ks.append(float(np.mean(sig ** 2 / (np.abs(J) ** 2).sum(-1).sum(1).max(-1))))
        out[c["name"]] = np.array(Ks)
    return out


def cmd_analyze(a):
    """Sec. 2.9 rule. Per system, policy nn-b (nn-adaptive reported under the same metrics)."""
    res = {"bars": BARS, "near_efficient": NEAR_EFF, "systems": {}}
    passes = []
    for system in ("S1", "R3"):
        ea, eb = np.load(g2.g2_eval_trace(system)), np.load(g2b_eval_trace(system))
        th = eb["thetas"]
        names_b, fams_b = [str(x) for x in eb["names"]], [str(x) for x in eb["families"]]
        names_a = [str(x) for x in ea["names"]]
        pa, pb = [str(x) for x in ea["policies"]], [str(x) for x in eb["policies"]]
        mse_b = np.mean((eb["means"][..., -1] - th[None, :, None, None]) ** 2, -1)          # (K, n_th, 3)
        mse_a = np.mean((ea["means"][..., -1] - th[None, :, None, None]) ** 2, -1)
        # identity check: regime-a policy on REF-2, theta_0 re-run in regime b == regime a
        i2b, i2a = names_b.index("REF-2"), names_a.index("REF-2")
        np.testing.assert_allclose(eb["means"][i2b, 0, pb.index("nn-adaptive")], ea["means"][i2a, 0, pa.index("nn-adaptive")], rtol=1e-5, atol=1e-7)

        def get(n, pol):
            if pol == "nn-b" or n.startswith("DC-"):
                return mse_b[names_b.index(n), :, pb.index(pol)]
            return mse_a[names_a.index(n), :, pa.index(pol)]
        Kb = known_bound(system)
        ref = [n for n in names_b if n.startswith("REF-")]
        rep = {}
        for pol in ("nn-b", "nn-adaptive"):
            env = np.max([get(n, pol) for n in ref], 0)
            rows = []
            for n, f in zip(names_b, fams_b):
                if f in ("reference", "telegraph-twin"):
                    continue
                for kt in range(len(th)):
                    nn, bl, K = float(get(n, pol)[kt]), float(get(n, "bayes-misspec")[kt]), float(Kb[n][kt])
                    r = {"name": n, "family": f, "theta": float(th[kt]), "mse_nn": nn, "mse_baseline": bl, "K": K,
                         "rho": bl / nn, "g": nn / env[kt], "eff_nn": K / nn, "eff_baseline": K / bl,
                         "rho_eligible": bool(bl > NEAR_EFF * K)}
                    tw = {"telegraph": n.replace("TEL", "TWIN"), "drift-c": next((c["twin"] for c in drift_c_configs() if c["name"] == n), None)}.get(f)
                    if tw:
                        r["twin"] = tw
                        r["twin_ratio"] = (nn / K) / (float(get(tw, pol)[kt]) / float(Kb[tw][kt]))
                    rows.append(r)
            scored = [r for r in rows if r["family"] != "drift"]                       # old drift: VOID, reported
            fam = {}
            for F in ("multi-lorentzian", "telegraph", "1/f", "drift-c"):
                rs = [r for r in scored if r["family"] == F]
                fam[F] = {"rho_median": float(np.median([r["rho"] for r in rs])), "n": len(rs),
                          "rho_worst_eligible": float(min([r["rho"] for r in rs if r["rho_eligible"]], default=np.inf)),
                          "g_worst": float(max(r["g"] for r in rs)),
                          "twin_worst": float(max([r["twin_ratio"] for r in rs if "twin_ratio" in r], default=0.0))}
            crit = {"C1_rho_median": all(v["rho_median"] >= BARS["rho_median"] for v in fam.values()),
                    "C2_rho_worst": all(v["rho_worst_eligible"] >= BARS["rho_worst"] for v in fam.values()),
                    "C3_g_worst": all(v["g_worst"] <= BARS["g_worst"] for v in fam.values()),
                    "C4_twin_worst": all(v["twin_worst"] <= BARS["twin_worst"] for v in fam.values())}
            rep[pol] = {"families": fam, "criteria": crit, "envelope": env.tolist(), "rows": rows}
            if pol == "nn-b":
                passes.append(all(crit.values()))
        res["systems"][system] = rep
    res["verdict"] = "GREEN-equivalent (deployable)" if all(passes) and len(passes) == 2 else "RED"
    outp = m.DATA / "qrc_gate2b_results.json"
    outp.write_text(json.dumps(res, indent=1))
    for system, rep in res["systems"].items():
        for pol, v in rep.items():
            print(system, pol, v["criteria"])
            for F, x in v["families"].items():
                print(f"   {F:17s} rho_med={x['rho_median']:.3g} rho_worst={x['rho_worst_eligible']:.3g} "
                      f"g_worst={x['g_worst']:.3g} twin_worst={x['twin_worst']:.3g}")
    print("regime b verdict:", res["verdict"], "->", outp)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen-bank", "pilot-bank", "gen-bound", "train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--system", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--chunk", type=int, default=41 if name == "gen-bank" else 500)
        p.add_argument("--R", type=int, default=2000)
        p.add_argument("--steps", type=int, default=0)
        p.add_argument("--tag", default="")
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"gen-bank": cmd_gen_bank, "pilot-bank": cmd_pilot_bank, "gen-bound": cmd_gen_bound, "train": cmd_train,
     "eval": cmd_eval, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
