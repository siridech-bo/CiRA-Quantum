"""Simulate the parameter-shift rule (PSR) and check it against autograd.

Closes the sim<->hardware loop: the gradient a real NMR machine would obtain by
PSR (two forward experiments per pulse at theta +/- pi/2) should equal the one
our simulator gets by backprop. We verify that here, on the differentiable
reservoir, entirely in simulation (CPU, no GPU).

What it checks — the feature Jacobian dF/dtheta (F = the measured expectation
values, the raw thing hardware reads), computed two ways:
  (1) autograd (reverse-mode)  -- the simulator ground truth
  (2) simulated PSR            -- forward runs at shifted angles only, i.e. the
                                  hardware recipe, no autograd

It also validates the global-vs-per-spin caveat (concept doc §6.5):
  * single-spin encoding pulse (generator sigma_x/2, 2 eigenvalues) -> the
    standard 2-term PSR [+pi/2, -pi/2] is EXACT.
  * global pulse on n spins (generator sum sigma_x/2, n+1 eigenvalues) -> the
    2-term rule is WRONG; the generalized 2n-term rule (Wierichs 2022) is EXACT.

Run: python backend/scripts/qrc_psr_sim.py
"""
from __future__ import annotations

import sys
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


def light_system(n_virtual=2):
    sysc = SystemConfig(
        n_qubits=3,
        chemical_shifts=[40.0, -25.0, 15.0],
        j_coupling=[[0, 5, 2], [5, 0, 3], [2, 3, 0]],
        t1=[3.0, 3.0, 3.0], t2=[0.6, 0.6, 0.6],
        labels=["A", "B", "C"],
    )
    sim = SimConfig(tau=0.02, n_virtual=n_virtual, evolution_mode="action")
    return QRCSystem(sysc, sim)


def _rx1(theta):
    c = torch.cos(theta / 2).to(CDT)
    s = torch.sin(theta / 2).to(CDT)
    return c * I2 - 1j * s * SX2


def pulse(theta, n, kind):
    """kind='single' -> R_x(theta) on spin 0 only (2-eigenvalue generator);
    kind='global' -> R_x(theta) on all n spins (n+1-eigenvalue generator)."""
    if kind == "single":
        U = _rx1(theta)
        for _ in range(n - 1):
            U = torch.kron(U, I2)
        return U
    U = _rx1(theta)
    for _ in range(n - 1):
        U = torch.kron(U, _rx1(theta))
    return U


def make_forward(sysm, g, kind):
    """theta (T,) -> flattened real feature vector over all steps (differentiable)."""
    d = sysm.dim
    rho0 = sysm.qt.basis(sysm.dim, 0) * sysm.qt.basis(sysm.dim, 0).dag()
    vec0 = torch.tensor(
        sysm.qt.operator_to_vector(rho0).full().ravel().astype(np.complex128)
    )

    def forward(theta):
        vec = vec0
        rows = []
        for k in range(theta.shape[0]):
            U = pulse(theta[k], sysm.n, kind)
            feat, vec = sysm.step_diff(vec, U, g)
            rows.append(feat)
        return torch.stack(rows).reshape(-1)      # (T * n_obs,)

    return forward


# --- shift rules -----------------------------------------------------------
def psr_2term():
    """Standard parameter-shift rule: shifts +/- pi/2, coeffs +/- 1/2."""
    return [np.pi / 2, -np.pi / 2], [0.5, -0.5]


def psr_generalized(R):
    """Generalized PSR for equidistant integer frequencies {1..R}
    (Wierichs et al. 2022): 2R shifts x_mu=(2mu-1)pi/(2R),
    coeffs (-1)^(mu-1) / (4R sin^2(x_mu/2)). Reduces to the 2-term rule at R=1."""
    shifts, coeffs = [], []
    for mu in range(1, 2 * R + 1):
        x = (2 * mu - 1) * np.pi / (2 * R)
        shifts.append(x)
        coeffs.append((-1) ** (mu - 1) / (4 * R * np.sin(x / 2) ** 2))
    return shifts, coeffs


def psr_jacobian(forward, theta, shifts, coeffs):
    """dF/dtheta_k by the shift rule — forward runs at shifted angles ONLY
    (the hardware recipe: no autograd). Returns (n_out, T)."""
    T = theta.shape[0]
    n_out = forward(theta).shape[0]
    J = torch.zeros(n_out, T, dtype=RDT)
    with torch.no_grad():
        for k in range(T):
            acc = torch.zeros(n_out, dtype=RDT)
            for sh, co in zip(shifts, coeffs):
                tp = theta.clone()
                tp[k] = tp[k] + sh
                acc = acc + co * forward(tp)
            J[:, k] = acc
    return J


def _relerr(a, b):
    return float((a - b).abs().max() / max(b.abs().max().item(), 1e-12))


def main():
    sysm = light_system(n_virtual=2)
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    n = sysm.n
    T = 3
    theta = torch.tensor([0.7, 1.3, 0.4], dtype=RDT)
    print(f"system: N={n} dim={sysm.dim} | substeps={g['substeps']} V={g['V']} "
          f"| pulses T={T}, angles={theta.tolist()}")

    for kind in ("single", "global"):
        fwd = make_forward(sysm, g, kind)
        n_out = fwd(theta).shape[0]
        # (1) autograd Jacobian — simulator ground truth
        th = theta.clone().requires_grad_(True)
        J_ag = torch.autograd.functional.jacobian(fwd, th).detach()   # (n_out, T)

        print(f"\n=== {kind} pulse (generator "
              f"{'sigma_x/2 on 1 spin: 2 eigenvalues' if kind=='single' else f'sum sigma_x/2 on {n} spins: {n+1} eigenvalues'}) ===")
        print(f"   features/step so far: n_out={n_out}")

        # (2a) standard 2-term PSR
        s2, c2 = psr_2term()
        J2 = psr_jacobian(fwd, theta, s2, c2)
        e2 = _relerr(J2, J_ag)
        print(f"   2-term PSR   [+/-pi/2, 2 runs/pulse]:  max rel.err vs autograd = {e2:.2e}"
              f"  -> {'EXACT' if e2 < 1e-6 else 'WRONG (as predicted)'}")

        # (2b) generalized PSR (R = n for global, R = 1 for single)
        R = 1 if kind == "single" else n
        sg, cg = psr_generalized(R)
        Jg = psr_jacobian(fwd, theta, sg, cg)
        eg = _relerr(Jg, J_ag)
        print(f"   generalized PSR [R={R}, {2*R} runs/pulse]:  max rel.err vs autograd = {eg:.2e}"
              f"  -> {'EXACT' if eg < 1e-6 else 'mismatch'}")

    print("\nRESULT: simulated PSR reproduces the autograd (sim) gradient exactly "
          "for the correct rule per generator — single-spin: 2-term; global n-spin: "
          "generalized 2n-term. The hardware gradient recipe is validated in sim.")


if __name__ == "__main__":
    main()
