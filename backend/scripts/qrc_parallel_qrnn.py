"""Parallel QRNN: does feeding each qubit in parallel work better decoupled
(independent parallel tracks) or coupled (one entangled joint reservoir)?

Input is fed to every qubit in parallel (per-qubit encoding). We sweep the qubit
COUPLING from J=0 (fully parallel / independent single-qubit RNNs) through the
default to 2x (strongly coupled joint reservoir), with single-qubit vs 2-body
correlation readout, and measure effective dimensionality + NARMA-2 test NMSE.

Reading:
- J=0 (parallel/independent): each qubit is its own single-qubit reservoir; the
  joint state is a product state, so correlations <s_i s_j> = <s_i><s_j> add
  nothing -- a clean diagnostic that the parallelism is *not* using a joint space.
- J>0 (coupled): the qubits interact; correlations carry genuine joint information.

Fixed arcsin encoding (isolates the coupling effect), forward-only, CPU, no GPU.

Run: python backend/scripts/qrc_parallel_qrnn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.qrc.config import SimConfig, SystemConfig  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402
from qrc_gen_traces import _resolve_system  # noqa: E402
from qrc_correlation_readout import run as measure  # noqa: E402  (eff_dim + NMSE)
from qrc_learnable_9spin import make_task  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig17_parallel_qrnn.png"


def scaled_system(jscale, tau=0.03, V=2):
    """6-qubit system with the J-couplings scaled (0 = decoupled/parallel)."""
    b = _resolve_system(6)
    sc = SystemConfig(
        n_qubits=b.n_qubits, chemical_shifts=list(b.chemical_shifts),
        j_coupling=(np.asarray(b.j_coupling) * jscale).tolist(),
        t1=list(b.t1), t2=list(b.t2), labels=list(b.labels),
    )
    return QRCSystem(sc, SimConfig(tau=tau, n_virtual=V, evolution_mode="action"))


def main():
    T, washout = 600, 30
    u, y = make_task("narma2", T, seed=7)
    drives = [("J=0 (parallel/decoupled)", 0.0), ("J=1x (default coupled)", 1.0),
              ("J=2x (strong coupled)", 2.0)]
    results = []
    print(f"6-qubit parallel QRNN: per-qubit arcsin feed, NARMA-2, T={T}\n")
    print(f"{'coupling':26s} {'readout':14s} {'n_feat':>6s} {'eff_dim':>8s} {'test NMSE':>10s}")
    for dname, js in drives:
        sysm = scaled_system(js)
        for with_corr, rname in [(False, "single-qubit"), (True, "single+2body")]:
            ed, nmse, nf = measure(sysm, u, y, washout, with_corr)
            results.append((dname, js, rname, nf, ed, nmse))
            print(f"{dname:26s} {rname:14s} {nf:6d} {ed:8.2f} {nmse:10.4f}")

    _figure(results)


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    js_vals = sorted({r[1] for r in results})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    for rname, color, mark in [("single-qubit", "#9aa7b4", "o"), ("single+2body", "#e07b3c", "s")]:
        ed = [next(r[4] for r in results if r[1] == j and r[2] == rname) for j in js_vals]
        nm = [next(r[5] for r in results if r[1] == j and r[2] == rname) for j in js_vals]
        axL.plot(js_vals, ed, marker=mark, color=color, label=rname)
        axR.plot(js_vals, nm, marker=mark, color=color, label=rname)
    axL.set_xlabel("coupling scale (0 = parallel/decoupled)")
    axL.set_ylabel("effective dimensionality")
    axL.set_title("A - does coupling raise the used dimension?")
    axL.legend(fontsize=8); axL.grid(alpha=0.25)
    axR.set_xlabel("coupling scale (0 = parallel/decoupled)")
    axR.set_ylabel("NARMA-2 test NMSE (lower better)")
    axR.set_title("B - task performance vs coupling")
    axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle("Parallel QRNN: per-qubit parallel feed, decoupled vs coupled qubits",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
