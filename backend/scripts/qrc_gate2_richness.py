"""De-risking Gate 2 (QRC_Gate2_Richness.md): is the frozen Stage-3 model-free policy robust when
the TRUE noise is richer than one OU process, while the model-aware baseline still assumes one OU
at tau_0 = 1 ms?

  estimand theta = log10 sigma_tot, sigma_tot^2 = (1/2pi) int S(w) dw = sum_c sigma_c^2
  (drift: sigma_tot^2 averaged over the 49 shots of the episode)

Families (sec. 2.2): multi-Lorentzian (C = 2, 3), 1/f (log-spaced Lorentzian sum), drift
(sigma linear / random walk, tau_c linear), random telegraph (one fluctuator, Kubo-Anderson).
Static families run through the exact SLE engine (product of collapsed-telegraph chains,
qrc_noisespec_m0.Batch); drift runs through the Stage-3 table (single OU, per-shot signal).

  python scripts/qrc_gate2_richness.py validate --device cuda      # generator fidelity note
  python scripts/qrc_gate2_richness.py gen --system R3 --device cuda   # engine signals -> trace
  python scripts/qrc_gate2_richness.py eval --system R3 --device cuda  # episodes -> eval trace
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

import qrc_adaptive_s3 as s3  # noqa: E402
import qrc_noisespec_m0 as m  # noqa: E402

MODEL_TAG = "_pilot20k"                            # the frozen Stage-3 policies (both systems)
ETA = 0.1
THETAS = (float(s3.TEST_TH[0]), float(s3.TEST_TH[2]))   # low-theta edge 1.627, mid 1.977
TAU0 = 1e-3                                        # baseline's assumed tau_c (= Stage-3 nu0)
N_SHOTS = 49
POLICIES = ("nn-adaptive", "nn-fixed", "bayes-misspec")
OOF_BAND = (50.0, 1e4)                             # 1/f band, rad/s (tau 0.1 - 20 ms)
OOF_COMPS = 5
OOF_LEVELS = (1, 2, 4, 8, 12)                      # per component, short -> long tau (K = 3510)
ML_LEVELS = lambda tau: 20 if tau >= 1e-3 else 7   # noqa: E731  tau >= 1 ms as Gaussian as the Stage-3 table
N_REF_D = 40                                       # level count of the Gaussianity reference for d
DRIFT_DELTA = (0.05, 0.10, 0.20)


def families():
    """Frozen configuration list (drawn once from seed 20261102). Each entry: name, family,
    and either comps [(frac of sigma_tot^2, tau_s, N_levels)] (static) or a drift spec."""
    rng = np.random.default_rng(20261102)
    out = []
    for C in (2, 3):
        k = 0
        while k < 4:
            lt = rng.uniform(np.log10(2e-4), np.log10(5e-3), C)
            if lt.max() - lt.min() < 0.6:          # require a real spread (>= 4x in tau)
                continue
            f = rng.dirichlet(np.ones(C))
            out.append({"name": f"ML{C}-{k}", "family": "multi-lorentzian",
                        "comps": [(float(fi), float(10 ** t), ML_LEVELS(10 ** t)) for fi, t in zip(f, lt)]})
            k += 1
    for alpha in (0.8, 1.0, 1.2):
        tau = np.logspace(np.log10(1 / OOF_BAND[1]), np.log10(1 / OOF_BAND[0]), OOF_COMPS)
        w = tau ** (alpha - 1.0)
        out.append({"name": f"1f-a{alpha:g}", "family": "1/f", "alpha": alpha,     # discrete-fluctuator (McWhorter)
                    "comps": [(float(x), float(t), n) for x, t, n in zip(w / w.sum(), tau, OOF_LEVELS)]})
    for kind in ("sigma-linear", "sigma-rw", "tau-linear"):
        for d in DRIFT_DELTA:
            out.append({"name": f"D-{kind}-{int(d * 100)}", "family": "drift", "kind": kind, "delta": d})
    for tau in (5e-4, 1e-3, 2e-3, 5e-3):
        out.append({"name": f"TEL-{tau * 1e3:g}ms", "family": "telegraph", "comps": [(1.0, tau, 1)]})
        out.append({"name": f"TWIN-{tau * 1e3:g}ms", "family": "telegraph-twin", "comps": [(1.0, tau, s3.N_LEVELS)]})
    return out


def references():
    """Single-OU references at the 5 Stage-3 tau_c test values (the envelope for g, G);
    REF-2 is tau_0 = 1 ms."""
    return [{"name": f"REF-{k}", "family": "reference", "comps": [(1.0, float(10 ** v), s3.N_LEVELS)]}
            for k, v in enumerate(s3.TEST_NU)]


def plist(cfg, theta):
    st2 = (10.0 ** theta) ** 2
    return [(float(np.sqrt(f * st2)), t, n) for f, t, n in cfg["comps"]]


def S_gen(w, comps_abs):
    """Two-sided PSD of the generator (exact for any level count): sum of Lorentzians."""
    return sum(2 * s ** 2 * t / (1 + (w * t) ** 2) for s, t, *_ in comps_abs)


def drift_schedule(cfg, theta, R, seed):
    """Per-episode, per-shot (theta_t, nu_t), each (R, N); the power mean of sigma_t^2 over the
    episode is pinned to sigma_tot = 10^theta (the estimand) for every episode."""
    t = np.arange(N_SHOTS) / (N_SHOTS - 1) - 0.5
    lg = np.log10(1 + cfg["delta"])
    nu0 = np.log10(TAU0)
    if cfg["kind"] == "tau-linear":
        return np.full((R, N_SHOTS), theta), np.broadcast_to(nu0 + lg * t, (R, N_SHOTS)).copy()
    if cfg["kind"] == "sigma-linear":
        th = np.broadcast_to(lg * t, (R, N_SHOTS)).copy()
    else:                                   # random walk, end-to-end std of log10 sigma = lg
        rng = np.random.default_rng([910000, *seed])
        th = np.cumsum(rng.standard_normal((R, N_SHOTS)) * lg / np.sqrt(N_SHOTS - 1), 1)
        th -= th[:, :1]
    th = th - 0.5 * np.log10(np.mean(10.0 ** (2 * th), 1, keepdims=True)) + theta
    return th, np.full((R, N_SHOTS), nu0)


def g2_trace(system):
    return m.TRACES / f"qrc_gate2_{system}.npz"


def g2_eval_trace(system, tag=""):
    return m.TRACES / f"qrc_gate2_eval_{system}_eta{ETA:g}{tag}.npz"


# --- validation (fidelity note; no policy is run) ----------------------------------------
def mc_psd(comps_abs, dt, n, seed):
    """Monte-Carlo PSD of the generator process: each component = (sigma/sqrt N) * sum of N
    independent +-1 telegraphs with flip rate 1/(2 tau). Welch-averaged periodogram."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for s, tau, N in comps_abs:
        q = 0.5 * (1 - np.exp(-dt / tau))           # flip prob per step (exact for the RTN)
        for _ in range(N):
            flips = rng.random(n) < q
            sgn = np.where(rng.random() < 0.5, 1.0, -1.0) * np.cumprod(np.where(flips, -1.0, 1.0))
            x += s / np.sqrt(N) * sgn
    seg = 4096
    xs = x[: n // seg * seg].reshape(-1, seg)
    P = np.abs(np.fft.rfft(xs - xs.mean(1, keepdims=True), axis=1)) ** 2
    w = 2 * np.pi * np.fft.rfftfreq(seg, dt)
    return w[1:], (P.mean(0) * dt / seg)[1:]            # two-sided S(w)


def cmd_validate(a):
    import torch
    from app.qrc import estimation as est
    rep = {"psd": {}, "gauss": {}}
    cfgs = families()
    wv = np.logspace(np.log10(OOF_BAND[0]), np.log10(OOF_BAND[1]), 200)
    # (1) 1/f: generator PSD vs the A / w^alpha target over the inner band (one decade in)
    for c in cfgs:
        if c["family"] == "1/f":
            S = S_gen(wv, plist(c, 0.0))
            inner = (wv > OOF_BAND[0] * 2) & (wv < OOF_BAND[1] / 2)
            A = np.exp(np.mean(np.log(S[inner] * wv[inner] ** c["alpha"])))
            dev = np.log10(S[inner] / (A / wv[inner] ** c["alpha"]))
            slope = np.polyfit(np.log10(wv[inner]), np.log10(S[inner]), 1)[0]
            rep["psd"][c["name"]] = {"max_abs_log10_ripple": float(np.abs(dev).max()), "fit_slope": float(slope)}
    # (2) Monte-Carlo PSD of the generator process vs the analytic sum-of-Lorentzians PSD
    for name in ("ML3-0", "1f-a1", "TEL-1ms"):
        c = next(x for x in cfgs if x["name"] == name)
        ca = plist(c, 2.0)
        w, P = mc_psd(ca, 2e-5, 2 ** 22, 7)
        band = (w > 30) & (w < 2e4)
        r = P[band] / S_gen(w[band], ca)
        rb = np.exp(np.convolve(np.log(r), np.ones(25) / 25, "valid"))     # smoothed ratio
        rep["psd"][name + "-mc"] = {"median_ratio": float(np.median(r)), "smoothed_ratio_range": [float(rb.min()), float(rb.max())]}
    # (3) level-count (Gaussianity) check on engine signals: d = max_j ||mu_j(N) - mu_j(N_ref)||
    #     * sqrt(49) / sigma_shot, i.e. the deviation in shot-noise units if all 49 shots hit probe j
    for system in ("S1", "R3"):
        sp = m.Spins(m.make_systems()[system], a.device)
        z = np.load(s3.s3_trace(system))
        Q, sig = z["Q"], est.m0_sigma(ETA, float(z["A"]))
        out = {}
        for theta in THETAS:
            for tau in (2e-4, 1e-3, 5e-3):
                st = 10.0 ** theta
                ref = None
                for N in (40, 20, 7, 3, 1):
                    B = m.Batch(sp, [[(st, tau, N)]], N)
                    f = np.stack([B.probe(n_pi, t_ms * 1e-3).cpu().numpy()[0] for n_pi, t_ms in m.PROBES]) @ Q
                    if ref is None:
                        ref = f
                        continue
                    d = float(np.linalg.norm(f - ref, axis=-1).max() * np.sqrt(N_SHOTS) / sig)
                    out[f"th{theta:.3f}_tau{tau * 1e3:g}ms_N{N}"] = d
                    print(system, theta, tau, N, f"d={d:.3g}", flush=True)
        rep["gauss"][system] = out
        del sp
        torch.cuda.empty_cache() if a.device == "cuda" else None
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["psd"], indent=1))


# --- engine signals ----------------------------------------------------------------------
def cmd_gen(a):
    """Noiseless engine FIDs for every static configuration at both theta values + the single-OU
    reference at tau_0 (raw complex FIDs persisted, projected z-space alongside)."""
    sp = m.Spins(m.make_systems()[a.system], a.device)
    assert sp.comm < 1e-10
    Q = np.load(s3.s3_trace(a.system))["Q"]
    from app.qrc import estimation as est
    z3 = np.load(s3.s3_trace(a.system))
    sig = est.m0_sigma(ETA, float(z3["A"]))
    cfgs = references() + [c for c in families() if c["family"] != "drift"]
    raw = np.zeros((len(cfgs), len(THETAS), len(m.PROBES), m.N_ACQ), np.complex128)
    dng = np.zeros((len(cfgs), len(THETAS)))
    t0 = time.time()

    def engine(pl):
        B = m.Batch(sp, [pl], s3.N_LEVELS)
        return np.stack([B.probe(n_pi, t_ms * 1e-3).cpu().numpy()[0] for n_pi, t_ms in m.PROBES])

    cache = {}

    def d_comp(sc, tc, nc):
        """Non-Gaussianity of one component: max_j ||mu_j(N_c) - mu_j(N=40)|| sqrt(49) / sigma."""
        key = (round(sc, 9), round(tc, 9), nc)
        if key not in cache:
            cache[key] = 0.0 if nc >= N_REF_D else float(
                np.linalg.norm((engine([(sc, tc, nc)]) - engine([(sc, tc, N_REF_D)])) @ Q, axis=-1).max()
                * np.sqrt(N_SHOTS) / sig)
        return cache[key]

    for i, c in enumerate(cfgs):
        for k, theta in enumerate(THETAS):
            pl = plist(c, theta)
            raw[i, k] = engine(pl)
            dng[i, k] = sum(d_comp(*x) for x in pl)       # additive (first-order) per-config d
        print(f"{a.system} {c['name']} K={int(np.prod([n + 1 for *_, n in c['comps']]))} d={dng[i].round(3).tolist()} "
              f"({time.time() - t0:.0f}s)", flush=True)
    names = np.array([c["name"] for c in cfgs])
    meta = json.dumps({"gate": 2, "system": a.system, "thetas": THETAS, "configs": cfgs, "tau0": TAU0,
                       "noise": "H=(xi/2) sum sz, xi = sum of collapsed-telegraph chains (levels per comp), SLE exact",
                       "n_acq": m.N_ACQ, "dwell": m.DWELL, "readout": "z = y @ Q (Stage-3 Q)",
                       "commutator_rel": sp.comm})
    np.savez(g2_trace(a.system), fids=raw.astype(np.complex64), sig=raw @ Q, d_nongauss=dng, names=names, thetas=np.array(THETAS),
             Q=Q, fid_dwell=m.DWELL, probes=np.array(m.PROBES), meta=meta)
    print("wrote", g2_trace(a.system), f"({time.time() - t0:.0f}s)", flush=True)


# --- episodes ----------------------------------------------------------------------------
def g2_noise(system, k_theta, R, L):
    """Unit complex shot noise (R, N, L), common to every policy AND every configuration at the
    same theta (paired ratios rho, g, G)."""
    out = np.empty((R, N_SHOTS, L), complex)
    for r in range(R):
        rng = np.random.default_rng([900000, {"S1": 1, "R3": 2}[system], k_theta, r])
        for t in range(N_SHOTS):
            out[r, t] = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    return out


def cmd_eval(a):
    import torch
    from app.qrc.adaptive_batch import BatchedEpisodes
    torch.set_num_threads(2)
    sim, z = s3.make_sim(a.system, ETA, a.device)
    T, th_g, nu_g = z["table"], z["th_grid"], z["nu_grid"]
    ck = {mo: torch.load(s3.model_path(a.system, ETA, mo, MODEL_TAG), map_location=sim.dev)
          for mo in ("fixed-cycle", "adaptive")}
    iv = int(np.argmin(np.abs(nu_g - np.log10(TAU0))))
    assert abs(nu_g[iv] - np.log10(TAU0)) < 1e-12
    be_mis = BatchedEpisodes(T[:, iv], th_g, sim.sigma, s3.TH_LO, s3.TH_HI, n_grid=201, device=a.device)
    g = np.load(g2_trace(a.system))
    static = {str(n): g["sig"][i] for i, n in enumerate(g["names"])}
    cfgs = references() + families()
    if a.only:
        cfgs = [c for c in cfgs if c["name"] in a.only]
    K, R = len(cfgs), a.R
    means = np.full((K, len(THETAS), len(POLICIES), R, N_SHOTS), np.nan)
    chosen = np.full((K, len(THETAS), len(POLICIES), R, N_SHOTS), -1, np.int16)
    out = g2_eval_trace(a.system, a.tag)
    t0 = time.time()
    for kt, theta in enumerate(THETAS):
        noise = g2_noise(a.system, kt, R, sim.L)
        for i, c in enumerate(cfgs):
            if c["family"] == "drift":
                thd, nud = drift_schedule(c, theta, R, seed=[{"S1": 1, "R3": 2}[a.system], kt, i])
            for ip, p in enumerate(POLICIES):
                mns, chs = [], []
                for c0 in range(0, R, a.chunk):
                    sl = slice(c0, min(c0 + a.chunk, R))
                    if c["family"] == "drift":
                        tt = torch.as_tensor(thd[sl].ravel(), device=sim.dev)
                        vv = torch.as_tensor(nud[sl].ravel(), device=sim.dev)
                        mu = sim.mu(tt, vv).reshape(sl.stop - sl.start, N_SHOTS, sim.P, sim.L)
                    else:
                        mu = static[c["name"]][kt]
                    if p.startswith("nn-"):
                        mn, ch = _nn(sim, ck, p, mu, theta, noise[sl])
                    else:
                        o = be_mis.run(mu.cpu().numpy().astype(complex) if torch.is_tensor(mu) else mu,
                                       noise[sl], "greedy-fisher")
                        mn, ch = o[0], o[2]
                    mns.append(mn); chs.append(ch)
                means[i, kt, ip], chosen[i, kt, ip] = np.concatenate(mns), np.concatenate(chs)
            print(f"[{a.system}] th={theta:.3f} {c['name']}: " + " ".join(
                f"{p}={np.mean((means[i, kt, ip, :, -1] - theta) ** 2):.2e}" for ip, p in enumerate(POLICIES))
                + f" ({time.time() - t0:.0f}s)", flush=True)
            np.savez(out, means=means, chosen=chosen, names=np.array([c["name"] for c in cfgs]),
                     families=np.array([c["family"] for c in cfgs]), thetas=np.array(THETAS),
                     policies=np.array(POLICIES), done=np.array([(kt, i)]), eta=ETA, N=N_SHOTS, R=R,
                     meta=json.dumps({"gate": 2, "system": a.system, "model_tag": MODEL_TAG, "tau0": TAU0,
                                      "noise_rng": "[900000, sys, k_theta, r] (shared by all configs)",
                                      "drift_rng": "[910000, sys, k_theta, i_cfg]",
                                      "readout": "z = y @ Q (line span); D = 2L real per shot",
                                      "estimand": "log10 sigma_tot (drift: power mean over the 49 shots)"}))
    print("wrote", out, f"({time.time() - t0:.0f}s)", flush=True)


def _nn(sim, ck, p, mu, theta, noise):
    import torch
    from app.qrc import neural_policy as npol
    d_in = sim.P * (2 * sim.L + 2) + 1
    mode = "fixed-cycle" if p == "nn-fixed" else "adaptive"
    est_ = npol.Estimator(d_in, (sim.th_lo, sim.th_hi), (sim.nu_lo, sim.nu_hi)).to(sim.dev)
    pol = npol.PolicyNet(d_in, sim.P).to(sim.dev)
    est_.load_state_dict(ck[mode]["est"]); pol.load_state_dict(ck[mode]["pol"])
    est_.eval(); pol.eval()
    B = noise.shape[0]
    mu = torch.as_tensor(mu, device=sim.dev).to(torch.complex64)
    if mu.dim() == 2:                                # static (P, L) -> (B, P, L)
        mu = mu[None].expand(B, -1, -1)
    with torch.no_grad():
        out = npol.rollout(sim, est_, pol, torch.full((B,), theta, dtype=torch.float64, device=sim.dev), None,
                           N_SHOTS, None, "greedy" if mode == "adaptive" else "fixed-cycle", mu=mu,
                           noise=torch.as_tensor(noise, dtype=torch.complex64, device=sim.dev))
    return out["est"][..., 0].double().cpu().numpy(), out["chosen"].cpu().numpy()


# --- analysis (frozen with the pre-registration, QRC_Gate2_Richness.md sec. 2.3-2.5) ------
REQUIRED = ("multi-lorentzian", "telegraph")       # median rho >= 2 and median g <= 2
DESIRED = ("1/f", "drift")                         # median rho >= 1 (not worse than the baseline)


def cmd_analyze(a):
    res = {"systems": {}, "rule": "per family and system: median over configs x both theta; GREEN needs "
           "rho >= 2 and g <= 2 on multi-lorentzian + telegraph, rho >= 1 on 1/f + drift, on BOTH systems"}
    passes, complete = [], True
    for system in ("S1", "R3"):
        f = g2_eval_trace(system, a.tag)
        if not f.exists():
            complete = False
            continue
        e = np.load(f)
        names, fams, th = [str(x) for x in e["names"]], [str(x) for x in e["families"]], e["thetas"]
        pol = [str(x) for x in e["policies"]]
        mse = np.mean((e["means"][..., -1] - th[None, :, None, None]) ** 2, -1)       # (K, n_theta, n_pol)
        assert np.isfinite(mse).all(), "eval incomplete"
        nn, bm = mse[..., pol.index("nn-adaptive")], mse[..., pol.index("bayes-misspec")]
        ref = [i for i, x in enumerate(fams) if x == "reference"]
        env_nn, env_bm = nn[ref].max(0), bm[ref].max(0)                                # per theta
        i0 = names.index("REF-2")                                                      # tau_0 = 1 ms
        dng = dict(zip([str(x) for x in np.load(g2_trace(system))["names"]], np.load(g2_trace(system))["d_nongauss"]))
        rows = []
        for i, n in enumerate(names):
            for kt in range(len(th)):
                r = {"name": n, "family": fams[i], "theta": float(th[kt]), "mse_nn": float(nn[i, kt]),
                     "mse_baseline": float(bm[i, kt]), "mse_nn_fixed": float(mse[i, kt, pol.index("nn-fixed")]),
                     "rho": float(bm[i, kt] / nn[i, kt]), "g": float(nn[i, kt] / env_nn[kt]),
                     "G": float(bm[i, kt] / env_bm[kt]), "g_tau0": float(nn[i, kt] / nn[i0, kt]),
                     "G_tau0": float(bm[i, kt] / bm[i0, kt]),
                     "d": float(dng[n][kt]) if n in dng else None}
                if fams[i] == "telegraph":                 # distribution effect vs the same-tau OU twin
                    j = names.index(n.replace("TEL", "TWIN"))
                    r["dist_nn"] = float(nn[i, kt] / nn[j, kt]); r["dist_baseline"] = float(bm[i, kt] / bm[j, kt])
                    r["rho_twin"] = float(bm[j, kt] / nn[j, kt])
                rows.append(r)
        fam = {}
        for F in REQUIRED + DESIRED:
            rs = [r for r in rows if r["family"] == F]
            md = {k: float(np.median([r[k] for r in rs])) for k in ("rho", "g", "G", "g_tau0", "G_tau0")}
            md["pass"] = bool(md["rho"] >= 2 and md["g"] <= 2) if F in REQUIRED else bool(md["rho"] >= 1)
            md["n"] = len(rs)
            fam[F] = md
            passes.append(md["pass"])
        res["systems"][system] = {"families": fam, "rows": rows, "envelope_nn": env_nn.tolist(),
                                  "envelope_baseline": env_bm.tolist()}
    res["regime_a"] = ("incomplete" if not complete else
                       "GREEN" if all(passes) else "not GREEN -> regime b decides AMBER / RED")
    out = m.DATA / f"qrc_gate2_results{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    for system, v in res["systems"].items():
        for F, md in v["families"].items():
            print(f"{system} {F:17s} rho={md['rho']:.3g} g={md['g']:.3g} G={md['G']:.3g} pass={md['pass']}")
    print("regime a:", res["regime_a"], "->", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--device", default="cuda")
    v.add_argument("--out", default=str(m.DATA / "qrc_gate2_validation.json"))
    gn = sub.add_parser("gen"); gn.add_argument("--system", required=True); gn.add_argument("--device", default="cuda")
    e = sub.add_parser("eval"); e.add_argument("--system", required=True); e.add_argument("--device", default="cuda")
    e.add_argument("--R", type=int, default=2000); e.add_argument("--chunk", type=int, default=500)
    e.add_argument("--tag", default=""); e.add_argument("--only", nargs="*")
    an = sub.add_parser("analyze"); an.add_argument("--tag", default="")
    a = ap.parse_args()
    {"validate": cmd_validate, "gen": cmd_gen, "eval": cmd_eval, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
