"""Gradient smoke test — does autograd flow through a Lindblad reservoir to a
learnable encoding? (CPU-only, no GPU.)

Purpose. De-risk the learnable-encoding direction (the "NMR molecule as a
Quantum Neural ODE" idea) by verifying the *one* thing that matters before any
migration or big run: that a gradient computed by torch autograd through a
Taylor-series Lindblad evolution reaches the encoding-network parameters and is
**correct** (matches a finite-difference estimate).

This is a faithful *proof of principle*, not the production stepper: it mirrors
the production mechanism (pulse ρ→UρU†, free evolution as a Taylor series of
Lindblad-superoperator applications, real observables → closed-form ridge
readout → loss) at 3 spins on CPU, so it runs in seconds and costs no GPU. If
the gradient matches finite differences here, the same autograd works on the
existing torch GPU stepper (which uses the identical ops) once its final
``.cpu().numpy()`` detach is lifted and the encoding angle is made a leaf tensor.

Run: python backend/scripts/qrc_grad_smoketest.py
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
CDT = torch.complex128
DEV = torch.device("cpu")

# --- 3-spin NMR-like system -------------------------------------------------
N = 3
DIM = 2 ** N
I2 = torch.eye(2, dtype=CDT)
SX = torch.tensor([[0, 1], [1, 0]], dtype=CDT)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=CDT)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=CDT)


def _kron(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def _op_on(op, i):
    """Single-spin operator ``op`` on spin ``i``, identity elsewhere."""
    return _kron([op if j == i else I2 for j in range(N)])


SXi = [_op_on(SX, i) for i in range(N)]
SYi = [_op_on(SY, i) for i in range(N)]
SZi = [_op_on(SZ, i) for i in range(N)]

# Hamiltonian: chemical shifts (σz) + nearest-neighbour J-coupling (σz σz).
_shifts = [1.0, 0.6, 0.9]          # arbitrary (rad/us-ish)
_J = 0.35
H = sum(_shifts[i] * SZi[i] for i in range(N))
for i in range(N - 1):
    H = H + _J * (SZi[i] @ SZi[i + 1])

# Dissipation: T2 dephasing on every spin (collapse op ∝ σz).
_gamma = 0.15
COLLAPSE = [np.sqrt(_gamma) * SZi[i] for i in range(N)]
_LdL = [c.conj().T @ c for c in COLLAPSE]

TAU = 0.25          # free-evolution time per input step
SUBSTEPS = 2
K_TAYLOR = 10


def lindblad(rho):
    """One application of the Lindblad superoperator L[rho] (torch, differentiable)."""
    out = -1j * (H @ rho - rho @ H)
    for c, ldl in zip(COLLAPSE, _LdL):
        out = out + c @ rho @ c.conj().T - 0.5 * (ldl @ rho + rho @ ldl)
    return out


def free_evolve(rho):
    """exp(TAU * L)[rho] via sub-stepped Taylor series of superoperator applications."""
    h = TAU / SUBSTEPS
    for _ in range(SUBSTEPS):
        term = rho
        acc = rho
        for k in range(1, K_TAYLOR + 1):
            term = (h / k) * lindblad(term)
            acc = acc + term
        rho = acc
    return rho


def rx(theta):
    """Global R_x(theta) on all N spins, built from a torch scalar (differentiable)."""
    c = torch.cos(theta / 2).to(CDT)
    s = torch.sin(theta / 2).to(CDT)
    u1 = c * I2 - 1j * s * SX
    return _kron([u1 for _ in range(N)])


# --- learnable encoding network: s -> theta (tiny MLP 1->Hd->1) -------------
def init_params(hd=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = 0.7
    return {
        "W1": (torch.rand(hd, 1, generator=g) - 0.5) * scale,
        "b1": (torch.rand(hd, generator=g) - 0.5) * scale,
        "W2": (torch.rand(1, hd, generator=g) - 0.5) * scale,
        "b2": (torch.rand(1, generator=g) - 0.5) * scale,
    }


def encode_angle(s, p):
    """theta(s) = pi * sigmoid(MLP(s)) — bounded in (0, pi)."""
    h = torch.tanh(p["W1"] @ s.reshape(1, 1) + p["b1"].reshape(-1, 1))
    raw = (p["W2"] @ h + p["b2"]).reshape(())
    return torch.pi * torch.sigmoid(raw)


# --- reservoir forward pass + ridge readout + loss --------------------------
def narma_like_target(u):
    """A fixed, smooth, short-memory nonlinear target (constant w.r.t. params)."""
    y = np.zeros_like(u)
    for t in range(2, len(u)):
        y[t] = 0.5 * u[t] + 0.3 * u[t - 1] ** 2 - 0.2 * u[t - 1] * u[t - 2]
    return y


def run_loss(p, u, y, washout=10, alpha=1e-3):
    rho = torch.zeros(DIM, DIM, dtype=CDT)
    rho[0, 0] = 1.0                      # start in |0..0><0..0|
    feats = []
    for t in range(len(u)):
        theta = encode_angle(torch.tensor(float(u[t])), p)
        U = rx(theta)
        rho = U @ rho @ U.conj().T       # encode
        rho = free_evolve(rho)           # reservoir evolution
        row = torch.stack(
            [torch.trace(op @ rho).real for op in (SXi + SYi + SZi)]
        )
        feats.append(row)
    F = torch.stack(feats)[washout:]     # (T', 9)
    F = torch.cat([F, torch.ones(F.shape[0], 1)], dim=1)   # + bias column
    yt = torch.tensor(y[washout:])
    # closed-form ridge (differentiable through the normal equations)
    A = F.T @ F + alpha * torch.eye(F.shape[1])
    w = torch.linalg.solve(A, F.T @ yt)
    pred = F @ w
    nmse = ((pred - yt) ** 2).mean() / yt.var()
    return nmse


def main():
    rng = np.random.default_rng(1)
    T = 60
    u = rng.random(T)
    y = narma_like_target(u)
    p = init_params()

    # --- autograd gradient ---
    for k in p:
        p[k].requires_grad_(True)
    loss = run_loss(p, u, y)
    loss.backward()
    grads = {k: p[k].grad.detach().clone() for k in p}
    gnorm = float(torch.sqrt(sum((g ** 2).sum() for g in grads.values())))

    # --- finite-difference check on a sample of parameters ---
    for k in p:
        p[k].requires_grad_(False)
    eps = 1e-5
    checks = [("W1", (0, 0)), ("W1", (3, 0)), ("b1", (2,)),
              ("W2", (0, 1)), ("W2", (0, 5)), ("b2", (0,))]
    print(f"loss (NMSE) = {float(loss):.6f}")
    print(f"||grad|| (autograd) = {gnorm:.6e}")
    print(f"{'param':10s} {'autograd':>14s} {'finite-diff':>14s} {'rel.err':>11s}")
    max_rel = 0.0
    for name, idx in checks:
        base = p[name][idx].item()
        p[name][idx] = base + eps
        lp = float(run_loss(p, u, y))
        p[name][idx] = base - eps
        lm = float(run_loss(p, u, y))
        p[name][idx] = base
        fd = (lp - lm) / (2 * eps)
        ag = float(grads[name][idx])
        denom = max(abs(fd), abs(ag), 1e-12)
        rel = abs(fd - ag) / denom
        max_rel = max(max_rel, rel)
        print(f"{name+str(idx):10s} {ag:14.6e} {fd:14.6e} {rel:11.2e}")

    print(f"\nmax relative error = {max_rel:.2e}")
    ok = gnorm > 1e-8 and max_rel < 1e-4
    print("RESULT:", "PASS — autograd flows through the Lindblad reservoir "
          "and matches finite differences" if ok
          else "FAIL — investigate (grad zero or mismatch)")


if __name__ == "__main__":
    main()
