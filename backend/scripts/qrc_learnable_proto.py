"""Learnable-encoding prototype — train an encoding network by gradient descent
through the quantum reservoir, and see what the gradient discovers.

This is the first end-to-end learnable-encoding experiment (plan §8, the "NMR
molecule as a Quantum Neural ODE" direction). It uses the differentiable
reservoir step (``QRCSystem.ensure_diff`` / ``step_diff``) added to system.py,
which the go/no-go checks proved is autograd-correct.

Pipeline (all differentiable):
    input s -> encoder MLP(s;W) -> pulse angle theta -> U=Rx(theta)
    -> reservoir step (Lindblad evolution) -> features
    -> closed-form ridge readout -> test NMSE = loss
Adam updates only the encoder W (the readout is the always-optimal ridge; the
molecule's dynamics are fixed by physics). We compare the trained encoder to the
fixed arcsin(sqrt(s)) baseline and plot the angle map the gradient learns.

3-spin, CPU-only (no GPU) — a proof of concept and a first look at what a learned
encoding does relative to Paper-4's arcsin(sqrt). Scales to CUDA unchanged by
passing device='cuda' to ensure_diff.

Run: python backend/scripts/qrc_learnable_proto.py
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
OUT_FIG = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig12_learnable_proto.png"


def light_system():
    """A light 3-spin system (small shifts/short tau) so the CPU forward is fast
    while keeping the real Lindblad structure (shifts + J-coupling + T2)."""
    n = 3
    sysc = SystemConfig(
        n_qubits=n,
        chemical_shifts=[40.0, -25.0, 15.0],
        j_coupling=[[0, 5, 2], [5, 0, 3], [2, 3, 0]],
        t1=[3.0, 3.0, 3.0], t2=[0.6, 0.6, 0.6],
        labels=["A", "B", "C"],
    )
    sim = SimConfig(tau=0.02, n_virtual=4, evolution_mode="action")
    return QRCSystem(sysc, sim)


def global_rx(theta, n):
    c = torch.cos(theta / 2).to(CDT)
    s = torch.sin(theta / 2).to(CDT)
    u1 = c * I2 - 1j * s * SX2
    U = u1
    for _ in range(n - 1):
        U = torch.kron(U, u1)
    return U


class Encoder(torch.nn.Module):
    """Tiny MLP s -> theta in (0, pi). Random init (does NOT start at arcsin)."""

    def __init__(self, hidden=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden, dtype=RDT), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1, dtype=RDT),
        )

    def forward(self, s):
        raw = self.net(s.reshape(-1, 1))
        return torch.pi * torch.sigmoid(raw).reshape(-1)


def target(u):
    """Fixed short-memory nonlinear target (constant w.r.t. encoder)."""
    y = np.zeros_like(u)
    for t in range(2, len(u)):
        y[t] = 0.5 * u[t] + 0.35 * u[t - 1] ** 2 - 0.25 * u[t - 1] * u[t - 2]
    return y


def reservoir_features(angles, sysm, g):
    """Run the reservoir for a sequence of pulse angles -> feature matrix (T, F)."""
    d = sysm.dim
    rho0 = sysm.qt.basis(sysm.dim, 0) * sysm.qt.basis(sysm.dim, 0).dag()
    vec = torch.tensor(
        sysm.qt.operator_to_vector(rho0).full().ravel().astype(np.complex128)
    )
    rows = []
    for th in angles:
        U = global_rx(th, sysm.n)
        feat, vec = sysm.step_diff(vec, U, g)
        rows.append(feat)
    return torch.stack(rows)


def nmse_split(F, y, n_tr, alpha=1e-3):
    """Fit ridge on train rows, return (test NMSE, train NMSE) — both torch."""
    F = torch.cat([F, torch.ones(F.shape[0], 1, dtype=RDT)], dim=1)
    yt = torch.tensor(y, dtype=RDT)
    Ftr, ytr = F[:n_tr], yt[:n_tr]
    Fte, yte = F[n_tr:], yt[n_tr:]
    A = Ftr.T @ Ftr + alpha * torch.eye(F.shape[1], dtype=RDT)
    w = torch.linalg.solve(A, Ftr.T @ ytr)
    te = ((Fte @ w - yte) ** 2).mean() / yte.var()
    tr = ((Ftr @ w - ytr) ** 2).mean() / ytr.var()
    return te, tr


def main():
    sysm = light_system()
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    print(f"system: N={sysm.n} dim={sysm.dim} | substeps={g['substeps']} "
          f"K={g['K']} V={g['V']} tau={sysm.sim.tau}")

    # Enough steps that n_train >> n_features (9 obs x V + bias) — avoids the
    # p>>n overfitting regime (the Phase-1 pathology) so the comparison is fair.
    rng = np.random.default_rng(7)
    T, washout = 240, 20
    u = rng.random(T)
    y = target(u)
    u_use, y_use = u[washout:], y[washout:]
    n_tr = int(0.6 * len(u_use))

    # --- baseline: fixed arcsin(sqrt(s)) encoding ---
    with torch.no_grad():
        base_ang = [torch.tensor(float(np.arcsin(np.sqrt(s))), dtype=RDT) for s in u]
        Fb = reservoir_features(base_ang, sysm, g)[washout:]
        base_te, base_tr = nmse_split(Fb, y_use, n_tr)
    print(f"\nbaseline arcsin(sqrt): test NMSE={float(base_te):.4f}  "
          f"train NMSE={float(base_tr):.4f}")

    # --- train the encoder ---
    enc = Encoder()
    opt = torch.optim.Adam(enc.parameters(), lr=0.02)
    u_t = torch.tensor(u, dtype=RDT)
    curve = []
    t0 = time.time()
    steps = 45
    for it in range(steps):
        opt.zero_grad()
        angles = enc(u_t)                       # (T,)
        F = reservoir_features(angles, sysm, g)[washout:]
        te, tr = nmse_split(F, y_use, n_tr)
        te.backward()
        opt.step()
        curve.append(float(te))
        if it % 10 == 0 or it == steps - 1:
            print(f"  step {it:3d}: test NMSE={float(te):.4f}  train NMSE={float(tr):.4f}")
    dt = time.time() - t0
    print(f"\ntrained in {dt:.1f}s ({dt/steps*1000:.0f} ms/step)")

    with torch.no_grad():
        angles = enc(u_t)
        F = reservoir_features(angles, sysm, g)[washout:]
        fin_te, fin_tr = nmse_split(F, y_use, n_tr)
    print(f"learned encoder:       test NMSE={float(fin_te):.4f}  "
          f"train NMSE={float(fin_tr):.4f}")
    verdict = ("BEATS" if float(fin_te) < float(base_te) - 1e-3 else
               "TIES" if abs(float(fin_te) - float(base_te)) <= 1e-3 else
               "loses to")
    print(f"\n=> learned encoding {verdict} the arcsin(sqrt) baseline "
          f"({float(fin_te):.4f} vs {float(base_te):.4f})")

    # --- what did it discover? learned theta(s) vs arcsin(sqrt) ---
    with torch.no_grad():
        sg = torch.linspace(0, 1, 21, dtype=RDT)
        learned = enc(sg).numpy()
    arcs = np.arcsin(np.sqrt(sg.numpy()))
    _figure(curve, sg.numpy(), learned, arcs, base_te, fin_te)
    print(f"\nlearned theta(s) vs arcsin(sqrt) (sampled):")
    for i in (0, 5, 10, 15, 20):
        print(f"  s={sg[i]:.2f}: learned={learned[i]:.3f}  arcsin_sqrt={arcs[i]:.3f}")


def _figure(curve, sg, learned, arcs, base_te, fin_te):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    axL.plot(curve, color="#4da3ff", lw=1.6)
    axL.axhline(float(base_te), color="#3fb950", ls="--", lw=1.2,
                label=f"arcsin(sqrt) baseline = {float(base_te):.3f}")
    axL.set_xlabel("Adam step")
    axL.set_ylabel("test NMSE")
    axL.set_title("A - learnable encoding training")
    axL.legend(fontsize=8)
    axL.grid(alpha=0.25)

    axR.plot(sg, arcs, color="#3fb950", lw=1.8, label="arcsin(sqrt(s)) (Paper 4)")
    axR.plot(sg, learned, color="#e07b3c", lw=1.8, label="learned theta(s)")
    axR.set_xlabel("input s")
    axR.set_ylabel("pulse angle theta")
    axR.set_title("B - encoding the gradient discovered")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.25)
    fig.suptitle(
        f"Learnable-encoding prototype (3-spin, CPU): test NMSE "
        f"{float(base_te):.3f} -> {float(fin_te):.3f}", fontsize=11, y=1.02)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
