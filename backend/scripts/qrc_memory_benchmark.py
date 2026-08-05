"""Memory-capacity benchmark with time-multiplexing (virtual nodes) on the NMR spin reservoir.

Reproduces the Das-Giorgi-Zambrini (PRR 2026) STM vs Parity-Check protocol on our
coupled-spin reservoir, sweeping the number of virtual nodes V. Time-multiplexing =
sampling the reservoir observables at V equally-spaced instants within each input
window tau (our n_virtual does exactly this via the dense propagator applied V
times). Feature vector length = (single-qubit observables) x V.

Tasks (fixed arcsin broadcast encoding, single-qubit x/y/z readout, ridge):
  STM (linear memory): input u ~ U(0,1); target y^tau_i = u_{i-tau}.
  PC  (nonlinear):     input u ~ {0,1};  target y^tau_i = (sum_{j=1..tau} u_{i-j}) mod 2.
Capacity C(tau) = squared Pearson correlation on the test split; total capacity =
sum_tau C(tau). We expect (their claim) PC to rise with V while STM stays flat.

Forward-only, CPU, no GPU.
Run: python backend/scripts/qrc_memory_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qrc_learnable_9spin import build_system, forward_features  # noqa: E402

RDT = torch.float64
CDT = torch.complex128
OUT = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig19_memory_multiplex.png"


def arcsin_broadcast(u, n):
    return [torch.full((n,), float(np.arcsin(np.sqrt(min(max(float(s), 0.0), 1.0)))), dtype=RDT)
            for s in u]


def stm_targets(u, taus):
    T = len(u)
    return {t: np.array([u[i - t] if i - t >= 0 else 0.0 for i in range(T)]) for t in taus}


def pc_targets(u, taus):
    T = len(u)
    out = {}
    for t in taus:
        y = np.zeros(T)
        for i in range(T):
            if i >= t:
                y[i] = int(sum(u[i - j] for j in range(1, t + 1))) % 2
        out[t] = y
    return out


def _sqcorr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.std() * b.std())
    return 0.0 if d < 1e-12 else float((a @ b / len(a)) ** 2 / (a.var() * b.var()))


def capacity(F, targets, washout, n_tr, alpha=1e-2):
    F = np.hstack([F, np.ones((len(F), 1))])[washout:]
    total, curve = 0.0, {}
    A = F[:n_tr].T @ F[:n_tr] + alpha * np.eye(F.shape[1])
    for tau, y in targets.items():
        yv = y[washout:]
        w = np.linalg.solve(A, F[:n_tr].T @ yv[:n_tr])
        C = _sqcorr(F[n_tr:] @ w, yv[n_tr:])
        curve[tau] = C; total += C
    return total, curve


def features_for(u, V):
    sysm = build_system(6, n_virtual=V, coupling_scale=1.0)
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    F = forward_features(arcsin_broadcast(u, sysm.n), sysm, g, "cpu", CDT)
    return F.detach().numpy().real


def main():
    T, washout, n_tr = 800, 80, 420
    rng = np.random.default_rng(7)
    u_stm = rng.random(T)                       # uniform for STM
    u_pc = rng.integers(0, 2, T).astype(float)  # binary for PC
    stm_taus = list(range(0, 9)); pc_taus = list(range(1, 7))
    Vs = [1, 2, 5, 10]

    print(f"6-qubit coupled reservoir | T={T} washout={washout} train={n_tr} test={T-washout-n_tr}")
    print(f"{'V':>3s} {'n_feat':>7s} {'STM totMC':>10s} {'PC totCap':>10s}")
    results = {}
    for V in Vs:
        F_stm = features_for(u_stm, V)
        F_pc = features_for(u_pc, V)
        stm_tot, stm_c = capacity(F_stm, stm_targets(u_stm, stm_taus), washout, n_tr)
        pc_tot, pc_c = capacity(F_pc, pc_targets(u_pc, pc_taus), washout, n_tr)
        results[V] = (stm_c, pc_c, stm_tot, pc_tot)
        print(f"{V:3d} {F_stm.shape[1]:7d} {stm_tot:10.3f} {pc_tot:10.3f}")

    _figure(results, stm_taus, pc_taus, Vs)


def _figure(results, stm_taus, pc_taus, Vs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#9aa7b4", 2: "#4da3ff", 5: "#e07b3c", 10: "#cb4b4b"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    for V in Vs:
        stm_c, pc_c, stm_tot, pc_tot = results[V]
        axL.plot(stm_taus, [stm_c[t] for t in stm_taus], marker="o", color=colors[V],
                 label=f"V={V} (totMC {stm_tot:.2f})")
        axR.plot(pc_taus, [pc_c[t] for t in pc_taus], marker="s", color=colors[V],
                 label=f"V={V} (totCap {pc_tot:.2f})")
    axL.set_xlabel("delay τ"); axL.set_ylabel("capacity C (r²)")
    axL.set_title("A — STM (linear memory)"); axL.legend(fontsize=8); axL.grid(alpha=0.25)
    axR.set_xlabel("delay τ"); axR.set_ylabel("capacity C (r²)")
    axR.set_title("B — Parity-Check (nonlinear memory)"); axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle("Time-multiplexing on the NMR spin reservoir: does nonlinear memory rise with V?",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
