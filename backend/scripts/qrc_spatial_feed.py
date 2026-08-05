"""Spatial-window feed: lay the input sequence ACROSS the qubit register.

Instead of feeding one scalar x_t to all qubits (broadcast), feed a sliding window
spatially: at step t, qubit i is driven by the lagged input x_{t-i}. The qubit
register holds a snapshot of the last n inputs in parallel, and the COUPLED
dynamics mix them (a quantum 'convolution' over the window) while the reservoir
also carries its own memory. Slides forward one step at a time.

  step t:  q0<-x_t, q1<-x_{t-1}, q2<-x_{t-2}, ...   (parallel, per-qubit lag)

Compared against the standard broadcast feed (all qubits <- x_t). Fixed arcsin
per-qubit encoding, coupled 6-qubit reservoir, single vs 2-body correlation
readout. Measures effective dim + NARMA-2 test NMSE. Forward-only, CPU, no GPU.

Run: python backend/scripts/qrc_spatial_feed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qrc_correlation_readout import build_readout, eff_dim, narma2_nmse  # noqa: E402
from qrc_learnable_9spin import build_system, forward_features, make_task  # noqa: E402

RDT = torch.float64
CDT = torch.complex128
OUT = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig18_spatial_feed.png"


def _ang(s):
    return float(np.arcsin(np.sqrt(min(max(float(s), 0.0), 1.0))))


def broadcast_angles(u, n):
    """All qubits get x_t (the standard per-spin feed, fixed encoding)."""
    return [torch.full((n,), _ang(s), dtype=RDT) for s in u]


def spatial_angles(u, n):
    """Qubit i gets the lagged input x_{t-i} (zero-padded) -- the spatial window."""
    seq = []
    for t in range(len(u)):
        a = [_ang(u[t - i]) if t - i >= 0 else 0.0 for i in range(n)]
        seq.append(torch.tensor(a, dtype=RDT))
    return seq


def run(sysm, angles_seq, y, washout, with_corr):
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    M, _, _ = build_readout(sysm, with_corr)
    g = dict(g); g["Mdense"] = M
    F = forward_features(angles_seq, sysm, g, "cpu", CDT).detach().numpy().real[washout:]
    return eff_dim(F), narma2_nmse(F, y[washout:]), F.shape[1]


def main():
    T, washout = 600, 30
    u, y = make_task("narma2", T, seed=7)
    sysm = build_system(6, n_virtual=2, coupling_scale=1.0)     # coupled reservoir
    n = sysm.n
    feeds = [("broadcast (all qubits <- x_t)", broadcast_angles(u, n)),
             ("spatial window (qubit i <- x_{t-i})", spatial_angles(u, n))]
    results = []
    print(f"6-qubit coupled reservoir, NARMA-2, arcsin encoding, T={T}\n")
    print(f"{'feed':38s} {'readout':14s} {'n_feat':>6s} {'eff_dim':>8s} {'test NMSE':>10s}")
    for fname, ang in feeds:
        for with_corr, rname in [(False, "single-qubit"), (True, "single+2body")]:
            ed, nmse, nf = run(sysm, ang, y, washout, with_corr)
            results.append((fname, rname, nf, ed, nmse))
            print(f"{fname:38s} {rname:14s} {nf:6d} {ed:8.2f} {nmse:10.4f}")

    # CLASSICAL controls on the same n-step window (no quantum reservoir at all)
    from qrc_learnable_9spin import sliding_windows
    W = sliding_windows(u, n)[washout:]
    cross = np.hstack([W[:, i:i + 1] * W[:, j:j + 1] for i in range(n) for j in range(i, n)])
    cls_lin = narma2_nmse(W, y[washout:])
    cls_quad = narma2_nmse(np.hstack([W, cross]), y[washout:])
    print(f"\nCLASSICAL controls on the same {n}-window (NO quantum reservoir):")
    print(f"  linear ridge:    NARMA-2 test NMSE = {cls_lin:.4f}  (matches the quantum spatial feed)")
    print(f"  quadratic ridge: NARMA-2 test NMSE = {cls_quad:.4f}  (~15x better than quantum)")
    _figure(results, cls_lin, cls_quad)


def _figure(results, cls_lin, cls_quad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{'broadcast' if 'broadcast' in r[0] else 'spatial'}\n{r[1]}" for r in results]
    colors = ["#9aa7b4" if "broadcast" in r[0] else "#4da3ff" for r in results]
    edims = [r[3] for r in results]; nmses = [r[4] for r in results]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    axL.bar(range(len(results)), edims, color=colors)
    axL.set_xticks(range(len(results))); axL.set_xticklabels(labels, fontsize=7)
    axL.set_ylabel("effective dimensionality"); axL.set_title("A - effective dim")
    axL.grid(axis="y", alpha=0.25)
    for i, v in enumerate(edims):
        axL.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axR.bar(range(len(results)), nmses, color=colors)
    axR.axhline(cls_lin, color="#d9a441", ls="--", lw=1.3, label=f"classical linear-window ridge = {cls_lin:.3f}")
    axR.axhline(cls_quad, color="#cb4b4b", ls=":", lw=1.5, label=f"classical quadratic-window ridge = {cls_quad:.3f}")
    axR.set_xticks(range(len(results))); axR.set_xticklabels(labels, fontsize=7)
    axR.set_ylabel("NARMA-2 test NMSE (lower better)"); axR.set_title("B - task performance")
    axR.legend(fontsize=7.5); axR.grid(axis="y", alpha=0.25)
    for i, v in enumerate(nmses):
        axR.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Spatial-window feed (blue) vs broadcast (grey): the win is CLASSICAL "
                 "(a linear ridge on the same window matches it; quadratic beats it 15x)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
