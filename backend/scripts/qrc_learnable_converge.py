"""Convergence + local-optimum test for the learnable encoding (CPU, no GPU).

The prototype (qrc_learnable_proto.py) showed the pipeline trains but, in a short
run, loses to arcsin(sqrt) with the learned theta(s) trending toward it. This
script answers the crux question cheaply, before any GPU: **is arcsin(sqrt) a
local optimum the gradient cannot beat?**

Three conditions on the same task/system:
  (A) baseline: fixed arcsin(sqrt(s))
  (B) learned from RANDOM init (does it catch arcsin given enough steps?)
  (C) learned from an ARCSIN WARM-START (pretrain the encoder to arcsin(sqrt),
      then train on the task) -- if the gradient stays put, arcsin is a local
      optimum; if it escapes and improves, there is headroom beyond arcsin.

Verdict logic:
  * C improves clearly over A  -> arcsin is NOT optimal (headroom -> justify GPU)
  * C stays ~ A (little angle movement) -> arcsin is a local optimum (methods result)

Run: python backend/scripts/qrc_learnable_converge.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.qrc.config import SimConfig, SystemConfig  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402

torch.manual_seed(0)
CDT = torch.complex128
RDT = torch.float64
SX2 = torch.tensor([[0, 1], [1, 0]], dtype=CDT)
I2 = torch.eye(2, dtype=CDT)
OUT_FIG = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig13_learnable_converge.png"


def light_system(n_virtual=3):
    sysc = SystemConfig(
        n_qubits=3, chemical_shifts=[40.0, -25.0, 15.0],
        j_coupling=[[0, 5, 2], [5, 0, 3], [2, 3, 0]],
        t1=[3.0, 3.0, 3.0], t2=[0.6, 0.6, 0.6], labels=["A", "B", "C"],
    )
    return QRCSystem(sysc, SimConfig(tau=0.02, n_virtual=n_virtual, evolution_mode="action"))


def global_rx(theta, n):
    c, s = torch.cos(theta / 2).to(CDT), torch.sin(theta / 2).to(CDT)
    u1 = c * I2 - 1j * s * SX2
    U = u1
    for _ in range(n - 1):
        U = torch.kron(U, u1)
    return U


class Encoder(torch.nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden, dtype=RDT), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1, dtype=RDT))

    def forward(self, s):
        return torch.pi * torch.sigmoid(self.net(s.reshape(-1, 1))).reshape(-1)


def warmstart_arcsin(enc, iters=800):
    """Pretrain the encoder (no reservoir) so theta(s) ~ arcsin(sqrt(s))."""
    opt = torch.optim.Adam(enc.parameters(), lr=0.02)
    sg = torch.linspace(0, 1, 64, dtype=RDT)
    tgt = torch.asin(torch.sqrt(sg))
    for _ in range(iters):
        opt.zero_grad()
        loss = ((enc(sg) - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    return float(loss)


def target(u):
    y = np.zeros_like(u)
    for t in range(2, len(u)):
        y[t] = 0.5 * u[t] + 0.35 * u[t - 1] ** 2 - 0.25 * u[t - 1] * u[t - 2]
    return y


def features(angles, sysm, g):
    d = sysm.dim
    rho0 = sysm.qt.basis(sysm.dim, 0) * sysm.qt.basis(sysm.dim, 0).dag()
    vec = torch.tensor(sysm.qt.operator_to_vector(rho0).full().ravel().astype(np.complex128))
    rows = []
    for th in angles:
        feat, vec = sysm.step_diff(vec, global_rx(th, sysm.n), g)
        rows.append(feat)
    return torch.stack(rows)


def nmse(F, y, n_tr, alpha=1e-3):
    F = torch.cat([F, torch.ones(F.shape[0], 1, dtype=RDT)], dim=1)
    yt = torch.tensor(y, dtype=RDT)
    Ftr, ytr, Fte, yte = F[:n_tr], yt[:n_tr], F[n_tr:], yt[n_tr:]
    w = torch.linalg.solve(Ftr.T @ Ftr + alpha * torch.eye(F.shape[1], dtype=RDT), Ftr.T @ ytr)
    return ((Fte @ w - yte) ** 2).mean() / yte.var()


def train(enc, sysm, g, u_t, y_use, washout, n_tr, steps=90, lr=0.04):
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve = []
    for _ in range(steps):
        opt.zero_grad()
        F = features(enc(u_t), sysm, g)[washout:]
        loss = nmse(F, y_use, n_tr)
        loss.backward()
        opt.step()
        sched.step()
        curve.append(float(loss))
    return curve


def main():
    # V=4, T=240: the working regime where arcsin has real skill (proto NMSE
    # ~0.57). At V=3/T=110 the baseline collapses to ~mean-predictor (NMSE~0.99)
    # and any "win" is noise/degenerate -- do NOT compare encodings there.
    sysm = light_system(n_virtual=4)
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    rng = np.random.default_rng(7)
    T, washout, steps = 240, 20, 100
    u = rng.random(T)
    y = target(u)
    u_t = torch.tensor(u, dtype=RDT)
    y_use = y[washout:]
    n_tr = int(0.6 * len(y_use))
    nfeat = 9 * g["V"] + 1
    print(f"N={sysm.n} V={g['V']} substeps={g['substeps']} | T={T} n_train={n_tr} "
          f"n_feat={nfeat} (n_train>>n_feat: {n_tr > 2*nfeat})")

    # (A) baseline arcsin
    with torch.no_grad():
        Fb = features([torch.tensor(float(np.arcsin(np.sqrt(s))), dtype=RDT) for s in u], sysm, g)[washout:]
        base = float(nmse(Fb, y_use, n_tr))
    print(f"\n(A) arcsin(sqrt) baseline:      test NMSE = {base:.4f}")

    t0 = time.time()
    # (B) learned from random init
    torch.manual_seed(1)
    encR = Encoder()
    curveR = train(encR, sysm, g, u_t, y_use, washout, n_tr, steps)
    with torch.no_grad():
        finR = float(nmse(features(encR(u_t), sysm, g)[washout:], y_use, n_tr))
    print(f"(B) learned (random init):      test NMSE = {finR:.4f}  ({steps} steps)")

    # (C) learned from arcsin warm-start
    torch.manual_seed(2)
    encW = Encoder()
    pre = warmstart_arcsin(encW)
    with torch.no_grad():
        sg = torch.linspace(0, 1, 21, dtype=RDT)
        theta_ws0 = encW(sg).clone()                    # angle map right after warm-start
        Fw0 = features(encW(u_t), sysm, g)[washout:]
        ws_start = float(nmse(Fw0, y_use, n_tr))
    curveW = train(encW, sysm, g, u_t, y_use, washout, n_tr, steps)
    with torch.no_grad():
        finW = float(nmse(features(encW(u_t), sysm, g)[washout:], y_use, n_tr))
        theta_ws1 = encW(sg).clone()
    dt = time.time() - t0
    print(f"(C) warm-start@arcsin -> trained: start NMSE = {ws_start:.4f} "
          f"(pretrain MSE {pre:.1e}) -> final = {finW:.4f}")
    moved = float((theta_ws1 - theta_ws0).abs().mean())
    with torch.no_grad():
        stdR = float(encR(sg).std())          # angle-map spread (degeneracy flag)
    stdW = float(theta_ws1.std())
    arcs_std = float(torch.asin(torch.sqrt(sg)).std())
    print(f"    angle map moved from arcsin by mean |Δθ| = {moved:.3f} rad")
    print(f"    angle-map std: arcsin={arcs_std:.2f}  learned-random={stdR:.2f}  "
          f"warm-start-final={stdW:.2f}  (std≈0 = degenerate constant encoding)")
    print(f"\ntrained both in {dt:.0f}s")

    # verdict, with guardrails against (i) an under-powered baseline and
    # (ii) a degenerate near-constant 'win'.
    tol, DEGEN = 0.02, 0.10
    best = min(finR, finW)
    winner_std = stdW if finW <= finR else stdR
    if base > 0.80:
        verdict = (f"UNDER-POWERED: arcsin baseline NMSE {base:.3f} ≈ mean-predictor "
                   "— config too weak to compare encodings; NOT a valid test")
    elif best < base - tol and winner_std > DEGEN:
        verdict = (f"HEADROOM (non-degenerate): a real learned encoding beats arcsin "
                   f"({best:.4f} vs {base:.4f}, angle std {winner_std:.2f}) "
                   "-> 9-spin GPU run justified")
    elif best < base - tol and winner_std <= DEGEN:
        verdict = (f"DEGENERATE WIN — DISCARD: learned 'beats' arcsin ({best:.4f}) only "
                   f"via a near-constant encoding (std {winner_std:.2f}) that ignores the "
                   "input; not a real improvement — arcsin stands")
    elif moved < 0.10 and finW <= base + tol:
        verdict = (f"arcsin(sqrt) is a LOCAL OPTIMUM: warm-start barely moved "
                   f"(|Δθ|={moved:.2f}) and did not beat it ({finW:.4f} vs {base:.4f}) "
                   "-> methods result; GPU unlikely to change the verdict")
    else:
        verdict = (f"TIE / INCONCLUSIVE: learned does not beat arcsin (random {finR:.4f}, "
                   f"warm {finW:.4f} vs base {base:.4f}; warm |Δθ|={moved:.2f})")
    print("VERDICT:", verdict)

    _figure(curveR, curveW, base, sg.numpy(), theta_ws0.numpy(), theta_ws1.numpy(),
            np.arcsin(np.sqrt(sg.numpy())), encR, finR, finW, verdict)


def _figure(curveR, curveW, base, sg, ws0, ws1, arcs, encR, finR, finW, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with torch.no_grad():
        learnedR = encR(torch.tensor(sg, dtype=RDT)).numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    axL.plot(curveR, color="#4da3ff", lw=1.5, label=f"random init -> {finR:.3f}")
    axL.plot(curveW, color="#e07b3c", lw=1.5, label=f"arcsin warm-start -> {finW:.3f}")
    axL.axhline(base, color="#3fb950", ls="--", lw=1.2, label=f"arcsin baseline = {base:.3f}")
    axL.set_xlabel("Adam step"); axL.set_ylabel("test NMSE")
    axL.set_title("A - convergence (random vs arcsin warm-start)")
    axL.legend(fontsize=8); axL.grid(alpha=0.25)

    axR.plot(sg, arcs, color="#3fb950", lw=2.0, label="arcsin(sqrt(s)) (Paper 4)")
    axR.plot(sg, ws1, color="#e07b3c", lw=1.8, ls="-", label="warm-start, after training")
    axR.plot(sg, learnedR, color="#4da3ff", lw=1.6, ls=":", label="learned (random init)")
    axR.set_xlabel("input s"); axR.set_ylabel("pulse angle theta")
    axR.set_title("B - encodings after convergence")
    axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle("Learnable-encoding convergence + local-optimum test (3-spin, CPU)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
