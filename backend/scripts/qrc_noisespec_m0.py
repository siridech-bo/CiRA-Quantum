"""Milestone 0: noise-spectrum recovery by a simulated NMR reservoir (QRC_NoiseSpec_Milestone0.md).

Injected noise H_noise = (xi(t)/2) sum_i sigma_z_i, xi = sum of OU components, each an
(N+1)-level collapsed telegraph sum (exact Lorentzian PSD). The ensemble average is the
stochastic Liouville equation (Kubo-Anderson), solved exactly:

    d/dt R_k = L0 R_k + xi_k Lz R_k + sum_k' W_kk' R_k'

with L0 the engine Liouvillian. [L0, Lz] = 0 is checked numerically, so a free segment
factorises: R <- E0(t) R, then each coherence class p (Lz eigenvalue lambda_p) gets
expm((W + lambda_p diag xi) t) on the noise index. Pulses mix classes; they act as
superoperators on every k.

  python scripts/qrc_noisespec_m0.py check                       # validation (sec. 4.1)
  python scripts/qrc_noisespec_m0.py gen --task a --split dev    # raw FIDs -> traces .npz
  python scripts/qrc_noisespec_m0.py analyze --task a            # dev selection + confirm
  python scripts/qrc_noisespec_m0.py figure
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

from app.qrc import sequences  # noqa: E402
from app.qrc.config import SimConfig, SystemConfig, crotonic_acid_paper4  # noqa: E402
from app.qrc.spectral_lines import physics_informed_lines  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / "backend/artifacts/traces"
DATA = REPO / "docs/QRC/data"
CDT = torch.complex128

N_PI = (0, 1, 2, 4, 8, 16, 32)
T_MS = (2, 5, 10, 20, 50, 100, 200)
PROBES = [(n, t) for n in N_PI for t in T_MS]            # 49
N_ACQ, DWELL = 512, 50e-6
TASKS = {
    "a": {"comps": 1, "N": 20, "sig": (30.0, 300.0), "tau": (0.1e-3, 30e-3),
          "seeds": {"dev": 101, "confirm": 102}},
    "b": {"comps": 2, "N": 10, "sig": (20.0, 200.0), "tau": (0.05e-3, 50e-3),
          "seeds": {"dev": 201, "confirm": 202}},
}
N_SAMPLES = 1200
BINS = np.logspace(1, np.log10(3e4), 7)                   # 6 bins, rad/s


# --- systems -----------------------------------------------------------------
def make_systems():
    """R3 = crotonic C1-C3 (13C, protons decoupled, carrier at mean shift); S1 = C2 alone."""
    c = crotonic_acid_paper4()
    idx = [0, 1, 2]
    nu = np.asarray(c.chemical_shifts)[idx]
    nu = nu - nu.mean()
    J = np.asarray(c.j_coupling)[np.ix_(idx, idx)]
    r3 = SystemConfig(n_qubits=3, chemical_shifts=nu.tolist(), j_coupling=J.tolist(),
                      t1=[c.t1[i] for i in idx], t2=[c.t2[i] for i in idx],
                      labels=["C1", "C2", "C3"])
    s1 = SystemConfig(n_qubits=1, chemical_shifts=[0.0], j_coupling=[[0.0]],
                      t1=[c.t1[1]], t2=[c.t2[1]], labels=["C2"])
    return {"R3": r3, "S1": s1}


class Spins:
    """Engine-derived superoperators for one system (column-stacked vec, as the engine)."""

    def __init__(self, sysc: SystemConfig, device="cpu"):
        sim = SimConfig(tau=0.03, n_virtual=1, evolution_mode="action")
        self.sysm = s = QRCSystem(sysc, sim)
        qt = s.qt
        self.n, self.d = s.n, s.dim
        self.dev = device
        L0 = np.asarray(s._ensure_liouvillian().todense(), dtype=np.complex128)
        Hz = sum(0.5 * z for z in s.sz)
        Lz = np.asarray(qt.liouvillian(Hz, []).full(), dtype=np.complex128)
        assert np.abs(Lz - np.diag(np.diag(Lz))).max() < 1e-12, "Lz not diagonal"
        self.comm = float(np.abs(L0 @ Lz - Lz @ L0).max() / np.abs(L0).max())
        lam = np.round(np.diag(Lz).imag, 9)
        self.classes = [(complex(0, v), np.where(lam == v)[0]) for v in np.unique(lam)]
        self.L0 = L0
        self.Lz = Lz

        def rot(axis, theta):
            op = {"x": qt.sigmax(), "y": qt.sigmay()}[axis]
            r = np.cos(theta / 2) * qt.qeye(2) - 1j * np.sin(theta / 2) * op
            U = qt.tensor([r] * self.n) if self.n > 1 else r
            return np.asarray(qt.sprepost(U, U.dag()).full(), dtype=np.complex128)

        self._rot = rot
        self._P = {}
        self.P90 = self.rot("x", np.pi / 2)
        self.P180 = self.rot("y", np.pi)
        o_op = sum(s.sy[p] + 1j * s.sx[p] for p in range(self.n))
        self.o = torch.tensor(qt.operator_to_vector(o_op.trans()).full().ravel(),
                              dtype=CDT, device=device)
        self.rho0 = torch.tensor(qt.operator_to_vector(s._build_initial_state()).full().ravel(),
                                 dtype=CDT, device=device)
        self._E0 = {}

    def rot(self, axis, theta):
        """Hard-pulse superoperator (cached). Axis 'x' | 'y' | '-x' | '-y'."""
        key = (axis, round(theta, 12))
        if key not in self._P:
            sgn = -1.0 if axis.startswith("-") else 1.0
            self._P[key] = torch.tensor(self._rot(axis[-1], sgn * theta), dtype=CDT, device=self.dev)
        return self._P[key]

    def E0(self, t):
        key = round(t, 12)
        if key not in self._E0:
            from scipy.linalg import expm
            self._E0[key] = torch.tensor(expm(self.L0 * t), dtype=CDT, device=self.dev)
        return self._E0[key]


# --- noise chains ------------------------------------------------------------
def chain(sig, tau, N):
    """(N+1)-level collapsed telegraph sum: generator W (column-stochastic rates), levels, p_stat."""
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
    from math import comb
    p = np.array([comb(N, j) for j in k], float) / 2.0 ** N
    return W, xi, p


def noise_generator(params, N):
    """params: list of (sigma, tau) components -> (W, xi, p) on the product chain."""
    W, xi, p = chain(*params[0], N)
    for sig, tau in params[1:]:
        W2, xi2, p2 = chain(sig, tau, N)
        I1, I2 = np.eye(len(xi)), np.eye(len(xi2))
        W = np.kron(W, I2) + np.kron(I1, W2)
        xi = (xi[:, None] + xi2[None, :]).ravel()
        p = np.kron(p, p2)
    return W, xi, p


class Batch:
    """Exact SLE propagation for a batch of noise samples on one spin system."""

    def __init__(self, spins: Spins, plist, N):
        self.s = spins
        dev = spins.dev
        Ws, xis, ps = zip(*[noise_generator(p, N) for p in plist])
        self.W = torch.tensor(np.stack(Ws), dtype=CDT, device=dev)          # (B,K,K)
        self.xi = torch.tensor(np.stack(xis), dtype=CDT, device=dev)        # (B,K)
        self.p = torch.tensor(np.stack(ps), dtype=CDT, device=dev)
        self._E = {}

    def init(self):
        return self.p[:, :, None] * self.s.rho0[None, None, :]              # (B,K,d2)

    def Enoise(self, t):
        key = round(t, 12)
        if key not in self._E:
            out = []
            for lam, _ in self.s.classes:
                A = self.W + lam * torch.diag_embed(self.xi)
                out.append(torch.linalg.matrix_exp(A * t))
            self._E[key] = out
        return self._E[key]

    def free(self, R, t, cls=None):
        if t <= 0:
            return R
        R = R @ self.s.E0(t).T
        E = self.Enoise(t)
        R = R.clone()
        for ci, (lam, ix) in enumerate(self.s.classes):
            if cls is not None and ci not in cls:
                continue
            R[:, :, ix] = E[ci] @ R[:, :, ix]
        return R

    def pulse(self, R, axis, angle):
        return R @ self.s.rot(axis, angle).T

    def acquire(self, R, n, dwell):
        """Sample the detected coherence classes at m * dwell, m = 0..n-1 -> (B, n) complex."""
        det = [ci for ci, (_, ix) in enumerate(self.s.classes)
               if torch.abs(self.s.o[ix]).max() > 0]
        keep = np.concatenate([self.s.classes[ci][1] for ci in det])
        out = []
        for k in range(n):
            if k:
                R = self.free(R, dwell, cls=det)
            out.append(R[:, :, keep].sum(1) @ self.s.o[keep])
        return torch.stack(out, 1)

    def run(self, seq):
        return sequences.run(seq, self)

    def probe(self, n_pi, T):
        """pi/2_x, CPMG(n_pi pi_y over T), then N_ACQ samples at DWELL -> (B, N_ACQ) complex."""
        return self.run(sequences.cpmg(n_pi, T, N_ACQ, DWELL))


def sample_params(task, n, seed):
    cfg = TASKS[task]
    rng = np.random.default_rng(seed)
    lo_s, hi_s = np.log10(cfg["sig"])
    lo_t, hi_t = np.log10(cfg["tau"])
    P = []
    for _ in range(n):
        P.append([(10 ** rng.uniform(lo_s, hi_s), 10 ** rng.uniform(lo_t, hi_t))
                  for _ in range(cfg["comps"])])
    return P


def S_two_sided(w, params):
    return sum(2 * s ** 2 * t / (1 + (w * t) ** 2) for s, t in params)


def bin_targets(params):
    out = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        v = sum(2 * s ** 2 * (np.arctan(b * t) - np.arctan(a * t)) / (b - a) for s, t in params)
        out.append(np.log10(v))
    return np.array(out)


def targets(task, P):
    if task == "a":
        return np.array([[np.log10(p[0][0]), np.log10(p[0][1])] for p in P])
    return np.array([bin_targets(p) for p in P])


# --- generation --------------------------------------------------------------
def simulate(spins, P, N, batch=50):
    out = np.zeros((len(P), len(PROBES), N_ACQ), np.complex64)
    for b0 in range(0, len(P), batch):
        B = Batch(spins, P[b0:b0 + batch], N)
        for j, (n_pi, t_ms) in enumerate(PROBES):
            out[b0:b0 + batch, j] = B.probe(n_pi, t_ms * 1e-3).cpu().numpy()
    return out


def cmd_gen(a):
    cfg = TASKS[a.task]
    seed = cfg["seeds"][a.split]
    P = sample_params(a.task, a.n or N_SAMPLES, seed)
    systems = make_systems()
    fids, comm = {}, {}
    for name, sysc in systems.items():
        sp = Spins(sysc, a.device)
        comm[name] = sp.comm
        assert sp.comm < 1e-10, f"[L0, Lz] != 0 for {name}: {sp.comm}"
        t0 = time.time()
        fids[name] = simulate(sp, P, cfg["N"], a.batch)
        print(f"{name}: {len(P)} samples x {len(PROBES)} probes in {time.time() - t0:.0f}s", flush=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    path = TRACES / f"qrc_noisespec_m0{a.task}_{a.split}.npz"
    np.savez(path, fids_R3=fids["R3"], fids_S1=fids["S1"], fid_dwell=DWELL, split=a.split,
             seed=seed, noise_params=np.array(P), targets=targets(a.task, P),
             probes=np.array(PROBES),
             meta=json.dumps({"task": a.task, "N_levels": cfg["N"], "comps": cfg["comps"],
                              "noise": "H=(xi/2) sum sz, xi = sum of OU (collapsed telegraph), SLE exact",
                              "systems": {k: {"nu": v.chemical_shifts, "J": v.j_coupling,
                                              "t1": v.t1, "t2": v.t2} for k, v in systems.items()},
                              "n_acq": N_ACQ, "dwell": DWELL, "commutator_rel": comm}))
    print("wrote", path)


# --- validation (sec. 4.1) ---------------------------------------------------
def cmd_check(a):
    from scipy.sparse import csr_matrix, identity, kron, diags
    from scipy.sparse.linalg import expm_multiply
    systems = make_systems()
    res = {}
    for name, sysc in systems.items():
        sp = Spins(sysc)
        res[f"{name}_commutator_rel"] = sp.comm
        print(f"{name}: d={sp.d}  ||[L0,Lz]||/||L0|| = {sp.comm:.1e}  classes={len(sp.classes)}")
    # (1) factorised exact vs brute-force expm_multiply on the full SLE generator (R3, task-b chain)
    sp = Spins(systems["R3"])
    params = [(120.0, 2e-3), (60.0, 0.2e-3)]
    W, xi, p = noise_generator(params, 4)
    K, d2 = len(xi), sp.d ** 2
    G = (kron(identity(K), csr_matrix(sp.L0)) + kron(diags(xi), csr_matrix(sp.Lz))
         + kron(csr_matrix(W), identity(d2))).tocsc()
    P90, P180 = sp.P90.numpy(), sp.P180.numpy()
    o = sp.o.numpy()

    def brute(n_pi, T, ts):
        r = (p[:, None] * sp.rho0.numpy()[None, :])
        r = (r @ P90.T).ravel()
        segs = [T] if n_pi == 0 else [T / n_pi / 2] + [T / n_pi] * (n_pi - 1) + [T / n_pi / 2]
        for i, s in enumerate(segs):
            r = expm_multiply(G * s, r)
            if n_pi and i < n_pi:
                r = (r.reshape(K, d2) @ P180.T).ravel()
        out = []
        for t in ts:
            rt = expm_multiply(G * t, r) if t else r
            out.append(rt.reshape(K, d2).sum(0) @ o)
        return np.array(out)

    B = Batch(sp, [params], 4)
    errs = []
    for n_pi, T in [(0, 5e-3), (2, 10e-3), (8, 20e-3)]:
        fast = B.probe(n_pi, T).numpy()[0]
        ts = [0.0, 10 * DWELL, 100 * DWELL]
        slow = brute(n_pi, T, ts)
        e = np.abs(fast[[0, 10, 100]] - slow).max() / np.abs(slow).max()
        errs.append(e)
        print(f"  factorised vs brute-force full SLE, probe (N={n_pi}, T={T*1e3:.0f} ms): rel.err {e:.1e}")
    res["factorised_vs_bruteforce_max_relerr"] = float(max(errs))
    # (2) SLE (N=20) vs Monte-Carlo OU trajectories, single spin, FID and CPMG-4 echo decay
    s1 = Spins(systems["S1"])
    sig, tau = 150.0, 1e-3
    Bs = Batch(s1, [[(sig, tau)]], 20)
    rng = np.random.default_rng(0)
    M, dt = 20000, 5e-6
    for n_pi, T in [(0, 10e-3), (4, 20e-3)]:
        sle = Bs.probe(n_pi, T).numpy()[0, 0]
        nsteps = int(round(T / dt))
        x = rng.normal(0, sig, M)
        phi = np.zeros(M)
        a_ = np.exp(-dt / tau)
        for k in range(nsteps):
            t = (k + 0.5) * dt
            f = 1.0 if n_pi == 0 else (-1.0) ** int(np.floor(t / (T / n_pi) + 0.5))
            phi += f * x * dt
            x = a_ * x + np.sqrt(1 - a_ ** 2) * rng.normal(0, sig, M)
        mc = np.abs(np.mean(np.exp(1j * phi)))
        # noiseless amplitude with T2 only
        B0 = Batch(s1, [[(1e-9, tau)]], 2)
        amp0 = abs(B0.probe(n_pi, T).numpy()[0, 0])
        r_sle = abs(sle) / amp0
        print(f"  SLE(N=20) vs MC OU ({M} traj), N_pi={n_pi}, T={T*1e3:.0f} ms: "
              f"coherence {r_sle:.4f} vs {mc:.4f} (+/- {1/np.sqrt(M):.4f})")
        res[f"sle_vs_mc_Npi{n_pi}"] = [float(r_sle), float(mc), float(1 / np.sqrt(M))]
    ls = physics_informed_lines(systems["R3"])
    res["R3_lines_hz"] = ls.freqs_hz.tolist()
    res["R3_D_eff"] = ls.d_eff
    print(f"  R3 lines D_eff={ls.d_eff}: {np.round(ls.freqs_hz, 1).tolist()}")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "qrc_noisespec_m0_check.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    g = sub.add_parser("gen")
    g.add_argument("--task", choices=["a", "b"], required=True)
    g.add_argument("--split", choices=["dev", "confirm"], required=True)
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--batch", type=int, default=50)
    g.add_argument("--device", default="cpu")
    a = ap.parse_args()
    {"check": cmd_check, "gen": cmd_gen}[a.cmd](a)


if __name__ == "__main__":
    main()
