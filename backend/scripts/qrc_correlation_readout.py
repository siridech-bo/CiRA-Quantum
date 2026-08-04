"""Do we actually USE the parallel qubits? Correlation readout + entangling drive.

Two quantum resources classical RNN/LSTM lack: many qubits (a 2^n-dim joint state,
the WIDTH) and coherence time (the temporal memory). The catch we measured earlier:
effective dimensionality ~1 and single-qubit readout throw the 2^n space away.

This experiment tests whether (a) reading multi-qubit CORRELATIONS <sigma_i sigma_j>
(which access the joint/entangled state) and (b) a stronger entangling DRIVE
(longer coherent evolution) raise the effective dimensionality and improve task
performance -- i.e. whether the parallel qubits, read properly, start to count.

Fixed arcsin encoding (isolates the readout/drive effect). Forward-only (no
training): effective dim = participation ratio of the feature covariance; task =
NARMA-2 test NMSE with a leakage-free ridge. CPU, no GPU.

Run: python backend/scripts/qrc_correlation_readout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.qrc.config import SimConfig  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402
from qrc_gen_traces import _resolve_system  # noqa: E402
from qrc_learnable_9spin import forward_features, make_task, _three_way  # noqa: E402

RDT = torch.float64
CDT = torch.complex128
OUT = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig16_correlation_readout.png"

CORR_AXES = (("z", "z"), ("x", "x"), ("y", "y"))   # 2-body correlations to add


def build_readout(sysm, with_corr):
    """Dense readout matrix M (n_obs x dim^2), rows = vec(O^T) so M@vec(rho)=<O>.
    Single-qubit always; 2-body correlations appended if with_corr."""
    qt = sysm.qt
    axis = {"x": sysm.sx, "y": sysm.sy, "z": sysm.sz}
    rows = [qt.operator_to_vector(op.trans()).full().ravel()
            for a in ("x", "y", "z") for op in axis[a]]
    n_single = len(rows)
    if with_corr:
        for a, b in CORR_AXES:
            for i in range(sysm.n):
                for j in range(i + 1, sysm.n):
                    O = axis[a][i] * axis[b][j]
                    rows.append(qt.operator_to_vector(O.trans()).full().ravel())
    return torch.tensor(np.asarray(rows), dtype=CDT), len(rows), n_single


def eff_dim(F):
    """Participation ratio of the feature covariance = effective dimensionality."""
    F = F - F.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(F.T @ F / len(F))
    ev = ev[ev > 1e-10]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def narma2_nmse(F, y, alpha=1e-3):
    """Leakage-free 3-way ridge (fit train, report test) NARMA-2 NMSE."""
    F = np.hstack([F, np.ones((len(F), 1))])
    tr, va, te = _three_way(len(F))
    w = np.linalg.solve(F[tr].T @ F[tr] + alpha * np.eye(F.shape[1]), F[tr].T @ y[tr])
    return float(((F[te] @ w - y[te]) ** 2).mean() / y[te].var())


def run(sysm, u, y, washout, with_corr):
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    M, n_obs, n_single = build_readout(sysm, with_corr)
    g = dict(g); g["Mdense"] = M                      # swap in the chosen readout
    ang = [torch.full((sysm.n,), float(np.arcsin(np.sqrt(min(max(s, 0.0), 1.0)))), dtype=RDT)
           for s in u]
    F = forward_features(ang, sysm, g, "cpu", CDT).detach().numpy().real[washout:]
    return eff_dim(F), narma2_nmse(F, y[washout:]), F.shape[1]


def main():
    T, washout = 600, 30
    u, y = make_task("narma2", T, seed=7)
    drives = [("baseline tau=0.03", 0.03), ("long tau=0.12 (more entangling)", 0.12)]
    results = []
    print(f"6-spin, NARMA-2, arcsin encoding, T={T} | measuring effective dim + test NMSE\n")
    print(f"{'drive':32s} {'readout':14s} {'n_feat':>6s} {'eff_dim':>8s} {'test NMSE':>10s}")
    for dname, tau in drives:
        sysm = QRCSystem(_resolve_system(6), SimConfig(tau=tau, n_virtual=2, evolution_mode="action"))
        for with_corr, rname in [(False, "single-qubit"), (True, "single+2body")]:
            ed, nmse, nf = run(sysm, u, y, washout, with_corr)
            results.append((dname, rname, nf, ed, nmse))
            print(f"{dname:32s} {rname:14s} {nf:6d} {ed:8.2f} {nmse:10.4f}")

    _figure(results)


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{d.split()[0]}\n{r}" for d, r, *_ in results]
    edims = [x[3] for x in results]
    nmses = [x[4] for x in results]
    colors = ["#9aa7b4" if "single-qubit" in x[1] else "#e07b3c" for x in results]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    axL.bar(range(len(results)), edims, color=colors)
    axL.set_xticks(range(len(results))); axL.set_xticklabels(labels, fontsize=7)
    axL.set_ylabel("effective dimensionality (participation ratio)")
    axL.set_title("A - are we using the 2^n space?")
    axL.grid(axis="y", alpha=0.25)
    for i, v in enumerate(edims):
        axL.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axR.bar(range(len(results)), nmses, color=colors)
    axR.set_xticks(range(len(results))); axR.set_xticklabels(labels, fontsize=7)
    axR.set_ylabel("NARMA-2 test NMSE (lower better)")
    axR.set_title("B - task performance")
    axR.grid(axis="y", alpha=0.25)
    for i, v in enumerate(nmses):
        axR.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Correlation readout + entangling drive: using the parallel qubits "
                 "(orange = +2-body correlations)", y=1.02, fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
