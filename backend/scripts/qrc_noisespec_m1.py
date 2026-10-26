"""Milestone 1: transverse-noise spectroscopy with a multi-spin comb (QRC_NoiseSpec_Milestone1.md).

Injected noise H_noise = (xi(t)/2) sum_i sigma_x_i (rotating frame), xi = sum of OU components,
each an (N+1)-level collapsed telegraph sum (exact Lorentzian PSD). Exact ensemble average by
the stochastic Liouville equation (Kubo-Anderson):

    G = I_K (x) L0 + diag(xi) (x) Lx + W (x) I_{d^2}

Lx does NOT commute with L0, so G is propagated in full: P = expm(G * 25 us) on the GPU, binary
powers P^(2^k), states advanced by matrix-vector products. All probe timings are multiples of
25 us. Redfield (TCL2) theory for the check: d rho/dt = L0 rho - [A, [X, rho]], A = sum sx/2,
X_mn = A_mn Gamma(omega_mn), Gamma(w) = sum_j sigma_j^2 tau_j / (1 + i w tau_j).

  python scripts/qrc_noisespec_m1.py check                      # SLE vs brute force, vs Monte-Carlo
  python scripts/qrc_noisespec_m1.py mech --device cuda         # mechanism gate (comb + Redfield)
  python scripts/qrc_noisespec_m1.py gen --task a --split dev --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.qrc.config import SimConfig, SystemConfig, crotonic_acid_paper4  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / "backend/artifacts/traces"
DATA = REPO / "docs/QRC/data"
CDT = torch.complex128

BASE = 25e-6                       # s; every probe timing is a multiple
N_ACQ, DWELL_STEPS = 512, 2        # 512 points at 50 us
DWELL = DWELL_STEPS * BASE
PROBES = ([("fid", 0, T) for T in (0, 5, 20, 50, 100)]
          + [("ir", 0, T) for T in (5, 20, 50, 100, 200)]
          + [("cpmg", n, T) for n, T in ((4, 20), (4, 100), (16, 20), (16, 100))])   # 14
EDGES = np.logspace(np.log10(2e3), np.log10(8e4), 7)          # 6 bins, rad/s
CENTRES = np.sqrt(EDGES[:-1] * EDGES[1:])
S1_OFFSETS_HZ = CENTRES / (2 * np.pi)                           # single-spin sweep set
TASKS = {
    "a": {"comps": 1, "N": 12, "sig": (100.0, 1000.0), "tau": (1e-5, 1e-3),
          "seeds": {"dev": 301, "confirm": 302}},
    "b": {"comps": 2, "N": 6, "sig": (100.0, 1000.0), "tau": (1e-5, 1e-3),
          "seeds": {"dev": 401, "confirm": 402}},
}
N_SAMPLES = 1200
MECH_SEED, MECH_N = 399, 40


# --- systems -----------------------------------------------------------------
def make_r3(j_scale=1.0):
    """R3 = crotonic C1, C3, H2 (paper offsets, each relative to its own carrier).
    ``j_scale`` != 1 is a POST-HOC diagnostic only (not the pre-registered system)."""
    c = crotonic_acid_paper4()
    idx = [0, 2, 5]
    return SystemConfig(n_qubits=3, chemical_shifts=[c.chemical_shifts[i] for i in idx],
                        j_coupling=(j_scale * np.asarray(c.j_coupling)[np.ix_(idx, idx)]).tolist(),
                        t1=[c.t1[i] for i in idx], t2=[c.t2[i] for i in idx],
                        labels=["C1", "C3", "H2"])


def make_s1(offset_hz):
    """S1 = a single 13C with C3's T1/T2, carrier set so its offset is ``offset_hz``."""
    c = crotonic_acid_paper4()
    return SystemConfig(n_qubits=1, chemical_shifts=[float(offset_hz)], j_coupling=[[0.0]],
                        t1=[c.t1[2]], t2=[c.t2[2]], labels=["C3"])


def r3_clusters(window=N_ACQ * DWELL):
    """Analytic single-quantum lines of R3, merged within the acquisition resolution 1/window."""
    s = make_r3()
    nu, J = np.asarray(s.chemical_shifts), np.asarray(s.j_coupling)
    lines = []
    for k in range(3):
        sp = [j for j in range(3) if j != k]
        for z in itertools.product((1, -1), repeat=2):
            lines.append(nu[k] + 0.5 * sum(zz * J[k, j] for zz, j in zip(z, sp)))
    lines = np.sort(lines)
    groups = [[lines[0]]]
    for f in lines[1:]:
        (groups[-1].append(f) if f - groups[-1][-1] < 1.0 / window else groups.append([f]))
    return np.array([np.mean(g) for g in groups])


class Spins:
    def __init__(self, sysc: SystemConfig):
        sim = SimConfig(tau=0.03, n_virtual=1, evolution_mode="action")
        self.sysm = s = QRCSystem(sysc, sim)
        qt = s.qt
        self.n, self.d = s.n, s.dim
        self.L0 = np.asarray(s._ensure_liouvillian().todense(), dtype=np.complex128)
        self.A = sum(0.5 * x for x in s.sx)
        self.Lx = np.asarray(qt.liouvillian(self.A, []).full(), dtype=np.complex128)
        self.comm = float(np.abs(self.L0 @ self.Lx - self.Lx @ self.L0).max() / np.abs(self.L0).max())

        def rot(axis, theta):
            op = {"x": qt.sigmax(), "y": qt.sigmay()}[axis]
            r = np.cos(theta / 2) * qt.qeye(2) - 1j * np.sin(theta / 2) * op
            U = qt.tensor([r] * self.n) if self.n > 1 else r
            return np.asarray(qt.sprepost(U, U.dag()).full(), dtype=np.complex128)

        self.pul = {"90x": rot("x", np.pi / 2), "180x": rot("x", np.pi), "180y": rot("y", np.pi)}
        o_op = sum(s.sy[p] + 1j * s.sx[p] for p in range(self.n))
        self.o = qt.operator_to_vector(o_op.trans()).full().ravel().astype(np.complex128)
        self.rho0 = qt.operator_to_vector(s._build_initial_state()).full().ravel().astype(np.complex128)
        self.A_amp = float(abs(self.o @ (self.pul["90x"] @ self.rho0)))
        self.H = np.asarray(s.H.full(), dtype=np.complex128)

    def redfield_L(self, params):
        """L0 + TCL2/Redfield generator for Lorentzian components ``params`` [(sigma, tau)]."""
        E = np.real(np.diag(self.H))                       # H0 is diagonal (weak coupling)
        A = np.asarray(self.A.full(), dtype=np.complex128)
        w = E[:, None] - E[None, :]
        Gam = sum(s ** 2 * t / (1 + 1j * w * t) for s, t in params)
        X = A * Gam
        I = np.eye(self.d)
        # column-stacked vec: vec(M rho) = (I kron M) vec(rho); vec(rho M) = (M^T kron I) vec(rho)
        pre = lambda M: np.kron(I, M)
        post = lambda M: np.kron(M.T, I)
        comA = pre(A) - post(A)
        comX = pre(X) - post(X)
        return self.L0 - comA @ comX


# --- noise -------------------------------------------------------------------
def chain(sig, tau, N):
    k = np.arange(N + 1)
    g = 1.0 / (2.0 * tau)
    W = np.zeros((N + 1, N + 1))
    for j in k:
        if j > 0:
            W[j - 1, j] = j * g
        if j < N:
            W[j + 1, j] = (N - j) * g
        W[j, j] = -N * g
    xi = sig * (2 * k - N) / np.sqrt(N)
    p = np.array([comb(N, j) for j in k], float) / 2.0 ** N
    return W, xi, p


def noise_generator(params, N):
    W, xi, p = chain(*params[0], N)
    for sig, tau in params[1:]:
        W2, xi2, p2 = chain(sig, tau, N)
        W = np.kron(W, np.eye(len(xi2))) + np.kron(np.eye(len(xi)), W2)
        xi = (xi[:, None] + xi2[None, :]).ravel()
        p = np.kron(p, p2)
    return W, xi, p


def sample_params(task, n, seed):
    cfg = TASKS[task]
    rng = np.random.default_rng(seed)
    ls, lt = np.log10(cfg["sig"]), np.log10(cfg["tau"])
    return [[(10 ** rng.uniform(*ls), 10 ** rng.uniform(*lt)) for _ in range(cfg["comps"])]
            for _ in range(n)]


def bin_targets(params):
    out = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        v = sum(2 * s ** 2 * (np.arctan(b * t) - np.arctan(a * t)) / (b - a) for s, t in params)
        out.append(np.log10(v))
    return np.array(out)


def targets(task, P):
    if task == "a":
        return np.array([[np.log10(p[0][0]), np.log10(p[0][1])] for p in P])
    return np.array([bin_targets(p) for p in P])


# --- exact propagation ---------------------------------------------------------
class Propagator:
    """Dense generator G (dim x dim) -> FIDs for every probe. ``G`` already includes everything."""

    def __init__(self, G, sp: Spins, p_stat, device):
        self.sp, self.dev = sp, device
        self.K = len(p_stat)
        Gt = torch.tensor(G, dtype=CDT, device=device)
        P = torch.linalg.matrix_exp(Gt * BASE)
        self.pow = [P]
        for _ in range(14):
            self.pow.append(self.pow[-1] @ self.pow[-1])
        d2 = sp.d ** 2
        self.v0 = torch.tensor(np.kron(p_stat, sp.rho0), dtype=CDT, device=device)
        self.pul = {k: torch.tensor(v, dtype=CDT, device=device) for k, v in sp.pul.items()}
        self.o = torch.tensor(sp.o, dtype=CDT, device=device)
        self.d2 = d2

    def free(self, v, steps):
        k = 0
        while steps:
            if steps & 1:
                v = self.pow[k] @ v
            steps >>= 1
            k += 1
        return v

    def pulse(self, v, name):
        return (v.reshape(self.K, self.d2) @ self.pul[name].T).reshape(-1)

    def sequence(self, kind, n, T_ms):
        st = int(round(T_ms * 1e-3 / BASE))
        v = self.v0
        if kind == "fid":
            v = self.free(self.pulse(v, "90x"), st)
        elif kind == "ir":
            v = self.pulse(self.free(self.pulse(v, "180x"), st), "90x")
        else:
            half = st // (2 * n)
            assert 2 * n * half == st
            v = self.free(self.pulse(v, "90x"), half)
            for i in range(n):
                v = self.free(self.pulse(v, "180y"), 2 * half if i < n - 1 else half)
        return v

    def run(self, probes=PROBES):
        V = torch.stack([self.sequence(*p) for p in probes], 1)          # (dim, n_probes)
        Pd = self.pow[1]                                                  # 2 steps = 50 us
        out = []
        for m in range(N_ACQ):
            if m:
                V = Pd @ V
            out.append((V.reshape(self.K, self.d2, -1).sum(0) * self.o[:, None]).sum(0))
        return torch.stack(out, 1)                                        # (n_probes, N_ACQ)


def sle_fids(sp, params, N, device, probes=PROBES):
    W, xi, p = noise_generator(params, N)
    K = len(xi)
    G = np.kron(np.eye(K), sp.L0) + np.kron(np.diag(xi), sp.Lx) + np.kron(W, np.eye(sp.d ** 2))
    return Propagator(G, sp, p, device).run(probes).cpu().numpy()


def redfield_fids(sp, params, device, probes=PROBES):
    return Propagator(sp.redfield_L(params), sp, np.ones(1), device).run(probes).cpu().numpy()


# --- generation --------------------------------------------------------------
def cmd_gen(a):
    cfg = TASKS[a.task]
    seed = cfg["seeds"][a.split]
    P = sample_params(a.task, a.n or N_SAMPLES, seed)
    r3 = Spins(make_r3())
    s1 = [Spins(make_s1(f)) for f in S1_OFFSETS_HZ]
    fR = np.zeros((len(P), len(PROBES), N_ACQ), np.complex64)
    fS = np.zeros((len(P), len(s1), len(PROBES), N_ACQ), np.complex64)
    t0 = time.time()
    for i, par in enumerate(P):
        fR[i] = sle_fids(r3, par, cfg["N"], a.device)
        for k, sp in enumerate(s1):
            fS[i, k] = sle_fids(sp, par, cfg["N"], a.device)
        if i % 100 == 0:
            print(f"{i}/{len(P)} {time.time() - t0:.0f}s", flush=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    path = TRACES / f"qrc_noisespec_m1{a.task}_{a.split}.npz"
    np.savez(path, fids_R3=fR, fids_S1=fS, fid_dwell=DWELL, split=a.split, seed=seed,
             noise_params=np.array(P), targets=targets(a.task, P),
             probes=np.array([f"{k}:{n}:{T}" for k, n, T in PROBES]),
             s1_offsets_hz=S1_OFFSETS_HZ, A_R3=r3.A_amp, A_S1=s1[0].A_amp,
             meta=json.dumps({"task": a.task, "N_levels": cfg["N"], "comps": cfg["comps"],
                              "noise": "H=(xi/2) sum sx (rotating frame), xi = sum of OU (collapsed telegraph), SLE exact",
                              "R3": {"nu": make_r3().chemical_shifts, "J": make_r3().j_coupling,
                                     "t1": make_r3().t1, "t2": make_r3().t2},
                              "S1_offsets_hz": S1_OFFSETS_HZ.tolist(), "base_s": BASE,
                              "n_acq": N_ACQ, "dwell": DWELL}))
    print("wrote", path, f"{time.time() - t0:.0f}s")


# --- validation ----------------------------------------------------------------
def cmd_check(a):
    from scipy.sparse import csr_matrix, identity, kron, diags
    from scipy.sparse.linalg import expm_multiply
    res = {}
    r3 = Spins(make_r3())
    res["R3_commutator_rel"] = r3.comm
    print(f"R3 ||[L0,Lx]||/||L0|| = {r3.comm:.2e} (nonzero: transverse noise does not commute)")
    # (1) squaring propagator vs brute-force expm_multiply on the full sparse SLE generator
    params = [(600.0, 1e-4), (300.0, 2e-5)]
    W, xi, p = noise_generator(params, 2)
    K, d2 = len(xi), r3.d ** 2
    G = (kron(identity(K), csr_matrix(r3.L0)) + kron(diags(xi), csr_matrix(r3.Lx))
         + kron(csr_matrix(W), identity(d2))).tocsc()
    fast = sle_fids(r3, params, 2, a.device)
    errs = []
    for j in (0, 6, 12):
        kind, n, T = PROBES[j]
        st = int(round(T * 1e-3 / BASE))
        v = np.kron(p, r3.rho0)

        def pul(v, nm):
            return (v.reshape(K, d2) @ r3.pul[nm].T).ravel()

        def fr(v, s):
            return expm_multiply(G * (s * BASE), v) if s else v
        if kind == "fid":
            v = fr(pul(v, "90x"), st)
        elif kind == "ir":
            v = pul(fr(pul(v, "180x"), st), "90x")
        else:
            h = st // (2 * n)
            v = fr(pul(v, "90x"), h)
            for i in range(n):
                v = fr(pul(v, "180y"), 2 * h if i < n - 1 else h)
        sl = []
        for m in (0, 10, 100):
            vm = fr(v, m * DWELL_STEPS)
            sl.append((vm.reshape(K, d2).sum(0) * r3.o).sum())
        e = float(np.abs(fast[j, [0, 10, 100]] - np.array(sl)).max() / np.abs(sl).max())
        errs.append(e)
        print(f"  squaring vs brute-force SLE, probe {PROBES[j]}: rel.err {e:.1e}")
    res["sle_vs_bruteforce_max_relerr"] = max(errs)
    # (2) single spin, intrinsic relaxation off: SLE vs Monte-Carlo (same telegraph process; Gaussian OU)
    sysc = SystemConfig(n_qubits=1, chemical_shifts=[600.0], j_coupling=[[0.0]], t1=[1e6], t2=[1e6],
                        labels=["X"])
    sp = Spins(sysc)
    sig, tau, N = 700.0, 1e-4, 12
    sle = sle_fids(sp, [(sig, tau)], N, a.device, probes=[("fid", 0, 0)])[0]
    rng = np.random.default_rng(0)
    M, dt = 10000, 1e-6
    Om = 2 * np.pi * 600.0
    ts_idx = [40, 200, 400]                         # 2, 10, 20 ms
    nsteps = ts_idx[-1] * int(round(DWELL / dt))
    rho0 = sp.pul["90x"] @ sp.rho0
    r0 = rho0.reshape(2, 2, order="F")
    Ot = sp.o.reshape(2, 2, order="F")             # = O^T, so sum(rho * O^T) = Tr(O rho)
    for proc in ("telegraph", "gaussian_ou"):
        if proc == "telegraph":
            up = rng.random((M, N)) < 0.5
        else:
            x = rng.normal(0, sig, M)
        R = np.broadcast_to(r0, (M, 2, 2)).copy()
        out = {}
        a_ = np.exp(-dt / tau)
        for step in range(1, nsteps + 1):
            xi = sig * (2 * up.sum(1) - N) / np.sqrt(N) if proc == "telegraph" else x
            hx, hz = xi / 2, Om / 2 * np.ones(M)
            h = np.sqrt(hx ** 2 + hz ** 2)
            c, s_ = np.cos(h * dt), np.sin(h * dt) / h
            U = np.empty((M, 2, 2), complex)
            U[:, 0, 0] = c - 1j * s_ * hz
            U[:, 1, 1] = c + 1j * s_ * hz
            U[:, 0, 1] = -1j * s_ * hx
            U[:, 1, 0] = -1j * s_ * hx
            R = U @ R @ np.conj(np.transpose(U, (0, 2, 1)))
            if proc == "telegraph":
                flip = rng.random((M, N)) < dt / (2 * tau)
                up ^= flip
            else:
                x = a_ * x + np.sqrt(1 - a_ ** 2) * rng.normal(0, sig, M)
            for m in ts_idx:
                if step == m * int(round(DWELL / dt)):
                    out[m] = complex(np.mean((R * Ot[None]).sum((1, 2))))
        cmp = [(m, abs(sle[m]), abs(out[m])) for m in ts_idx]
        res[f"sle_vs_mc_{proc}"] = cmp
        for m, x1, x2 in cmp:
            print(f"  SLE(N={N}) vs MC {proc} ({M} traj) t={m * DWELL * 1e3:.0f} ms: |S| {x1:.4f} vs {x2:.4f}"
                  f" (+/- {sp.A_amp / np.sqrt(M):.4f})")
    res["R3_clusters_hz"] = r3_clusters().tolist()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "qrc_noisespec_m1_check.json").write_text(json.dumps(res, indent=1))


# --- mechanism gate --------------------------------------------------------------
MECH_PROBES = [p for p in PROBES if p[0] == "fid"]


def cluster_rates(fids, freqs, sign):
    """Per-cluster decay rate from the FID-delay series (T = 0, 5, 20, 50, 100 ms)."""
    t = np.arange(N_ACQ) * DWELL
    Phi = np.exp(sign * 1j * 2 * np.pi * np.outer(t, freqs))
    X = np.abs(fids @ Phi)                                 # (5 delays, n_clusters)
    T = np.array([p[2] for p in MECH_PROBES]) * 1e-3
    rates = []
    for c in range(len(freqs)):
        y = X[:, c] / X[0, c]
        ok = y > 1e-3
        rates.append(float(-np.polyfit(T[ok], np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else np.nan)
    return np.array(rates), X


def cmd_mech(a):
    r3 = Spins(make_r3(a.j_scale))
    freqs = r3_clusters()
    P = sample_params("a", MECH_N, MECH_SEED)
    f0 = sle_fids(r3, [(1e-9, 1e-4)], 2, a.device, MECH_PROBES)
    t = np.arange(N_ACQ) * DWELL
    sign = max((1, -1), key=lambda s: float(np.abs(f0[0] @ np.exp(s * 1j * 2 * np.pi * np.outer(t, freqs))).sum()))
    r0, _ = cluster_rates(f0, freqs, sign)
    rows = []
    for par in P:
        fs = sle_fids(r3, par, TASKS["a"]["N"], a.device, MECH_PROBES)
        fr = redfield_fids(r3, par, a.device, MECH_PROBES)
        rs, _ = cluster_rates(fs, freqs, sign)
        rr, _ = cluster_rates(fr, freqs, sign)
        s, tc = par[0]
        own = [2 * s ** 2 * tc / (1 + (2 * np.pi * f * tc) ** 2) / 4 for f in freqs]
        rows.append({"sigma": s, "tau": tc, "sigma_tau": s * tc,
                     "excess_sle": (rs - r0).tolist(), "excess_redfield": (rr - r0).tolist(),
                     "own_S_over_4": own})
    weak = [r for r in rows if r["sigma_tau"] <= 0.3]
    rel, ratio, corr = [], [], []
    for r in weak:
        es, er = np.array(r["excess_sle"]), np.array(r["excess_redfield"])
        ok = es > 2.0
        rel += list(np.abs(es[ok] - er[ok]) / np.abs(es[ok]))
        if ok.sum() >= 2:
            ratio.append(float(es[ok].max() / es[ok].min()))
            if np.std(es) > 0 and np.std(er) > 0:
                corr.append(float(np.corrcoef(es, er)[0, 1]))
    g1 = float(np.median(rel)) if rel else float("nan")
    g2a = float(np.median(ratio)) if ratio else float("nan")
    g2b = float(np.median(corr)) if corr else float("nan")
    out = {"clusters_hz": freqs.tolist(), "sign": sign, "intrinsic_rates": r0.tolist(), "draws": rows,
           "n_weak": len(weak),
           "G1_median_relerr_sle_vs_redfield": g1, "G1_pass": bool(g1 <= 0.20),
           "G2_median_max_over_min_excess": g2a, "G2_median_corr_sle_redfield": g2b,
           "G2_pass": bool(g2a >= 1.5 and g2b >= 0.9)}
    out["gate_pass"] = out["G1_pass"] and out["G2_pass"]
    DATA.mkdir(parents=True, exist_ok=True)
    out["j_scale"] = a.j_scale
    name = "qrc_noisespec_m1_mech.json" if a.j_scale == 1.0 else f"qrc_noisespec_m1_mech_posthoc_j{a.j_scale:g}.json"
    (DATA / name).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "draws"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("check", "mech"):
        s = sub.add_parser(nm)
        s.add_argument("--device", default="cpu")
        s.add_argument("--j-scale", type=float, default=1.0)
    g = sub.add_parser("gen")
    g.add_argument("--task", choices=["a", "b"], required=True)
    g.add_argument("--split", choices=["dev", "confirm"], required=True)
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--device", default="cpu")
    a = ap.parse_args()
    {"check": cmd_check, "mech": cmd_mech, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
