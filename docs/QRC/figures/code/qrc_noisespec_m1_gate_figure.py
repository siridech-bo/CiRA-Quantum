"""Fig. NS1-gate: Milestone 1 mechanism gate (pre-registered J) vs post-hoc J = 0 diagnostic.

Reads docs/QRC/data/qrc_noisespec_m1_mech.json, qrc_noisespec_m1_mech_posthoc_j0.json and recomputes
the noiseless delay series (CPU, seconds) -> docs/QRC/figures/figNS1_gate.png

  python docs/QRC/figures/code/qrc_noisespec_m1_gate_figure.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "backend/scripts"))
import qrc_noisespec_m1 as m  # noqa: E402

DATA = REPO / "docs/QRC/data"
FIG = REPO / "docs/QRC/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CL = [("C1 (7750 Hz)", INK2), ("H2 (439 Hz)", BLUE), ("H2 (602 Hz)", AQUA), ("C3 (2618 Hz)", ORANGE),
      ("C3 (2781 Hz)", YELLOW)]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), facecolor=SURF)
    T = np.array([p[2] for p in m.MECH_PROBES])
    t = np.arange(m.N_ACQ) * m.DWELL
    ax = axes[0]
    style(ax)
    for js, ls in ((1.0, "-"), (0.0, "--")):
        sp = m.Spins(m.make_r3(js))
        f0 = m.sle_fids(sp, [(1e-9, 1e-4)], 2, "cpu", m.MECH_PROBES)
        X = np.abs(f0 @ np.exp(1j * 2 * np.pi * np.outer(t, m.r3_clusters())))
        X = X / X[0]
        for c, (lab, col) in enumerate(CL):
            ax.plot(T, X[:, c], ls, color=col, marker="o", ms=3.5, lw=1.6,
                    label=lab if js == 1.0 else None)
    ax.set_xlabel("delay T (ms)", color=INK, fontsize=9.5)
    ax.set_ylabel("cluster amplitude / amplitude at T = 0", color=INK, fontsize=9.5)
    ax.set_title("(a) no noise: J on (solid) vs J = 0 (dashed)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    for ax, fn, title in ((axes[1], "qrc_noisespec_m1_mech.json", "(b) pre-registered system (J on)"),
                          (axes[2], "qrc_noisespec_m1_mech_posthoc_j0.json", "(c) post-hoc: J = 0")):
        style(ax)
        d = json.loads((DATA / fn).read_text())
        for c, (lab, col) in enumerate(CL):
            own = np.array([r["own_S_over_4"][c] for r in d["draws"] if r["sigma_tau"] <= 0.3])
            ex = np.array([r["excess_sle"][c] for r in d["draws"] if r["sigma_tau"] <= 0.3])
            ax.scatter(np.maximum(own, 1e-2), np.maximum(ex, 1e-2), s=14, color=col, label=lab)
        ax.plot([1e-2, 2e2], [1e-2, 2e2], color=INK2, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 2e2)
        ax.set_ylim(1e-2, 2e2)
        ax.set_xlabel("own-line Redfield rate S(|ω|)/4 (1/s)", color=INK, fontsize=9.5)
        ax.set_ylabel("fitted excess decay rate, SLE (1/s)", color=INK, fontsize=9.5)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figNS1_gate.png", dpi=200, facecolor=SURF)
    print("wrote", FIG / "figNS1_gate.png")


if __name__ == "__main__":
    main()
