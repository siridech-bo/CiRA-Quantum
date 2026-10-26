"""Fig. NS0: Milestone 0 noise-spectrum recovery (QRC_NoiseSpec_Milestone0.md).

Reads docs/QRC/data/qrc_noisespec_m0{a,b}_results.json and qrc_noisespec_m0b_preds.npz
-> docs/QRC/figures/figNS0_noisespec_m0.png

  python docs/QRC/figures/code/qrc_noisespec_m0_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "docs/QRC/data"
FIG = REPO / "docs/QRC/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BINS = np.logspace(1, np.log10(3e4), 7)
METHODS = [("M1", "R3 reservoir, ridge", BLUE), ("M2", "S1 single spin, same ridge", YELLOW),
           ("M3", "S1 filter-function inversion", ORANGE), ("M4", "S1 classical net (MLP)", AQUA)]


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
    ra = json.loads((DATA / "qrc_noisespec_m0a_results.json").read_text())
    rb = json.loads((DATA / "qrc_noisespec_m0b_results.json").read_text())
    pz = np.load(DATA / "qrc_noisespec_m0b_preds.npz", allow_pickle=True)
    y, P = pz["y"], pz["P"]

    fig = plt.figure(figsize=(11.5, 7.0), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

    # row 1: true vs recovered spectra, three confirmation samples (low / median / high total power)
    tot = y.mean(1)
    picks = [int(np.argsort(tot)[int(q * (len(tot) - 1))]) for q in (0.15, 0.5, 0.85)]
    w = np.logspace(0.5, 4.8, 300)
    ctr = np.sqrt(BINS[:-1] * BINS[1:])
    for c, i in enumerate(picks):
        ax = fig.add_subplot(gs[0, c])
        style(ax)
        S = sum(2 * s ** 2 * t / (1 + (w * t) ** 2) for s, t in P[i])
        ax.loglog(w, S, color=INK2, lw=1.2, label="true S(ω)")
        ax.plot(ctr, 10 ** y[i], "s", color=INK, ms=6, mfc="none", label="true bin average")
        for k, lab, col in METHODS:
            ax.plot(ctr, 10 ** pz[k][i], "o", color=col, ms=4.5, label=lab)
        for b in BINS:
            ax.axvline(b, color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("ω (rad/s)", color=INK, fontsize=9.5)
        if c == 0:
            ax.set_ylabel("S(ω) (rad²/s)", color=INK, fontsize=9.5)
        ax.set_title(f"({'abc'[c]}) confirmation sample {i}", loc="left", color=INK, fontsize=10)
        if c == 2:
            ax.legend(frameon=False, fontsize=7.2, labelcolor=INK, loc="lower left")

    etas = [float(e) for e in rb["levels"]]
    xs = np.arange(len(etas))
    labels = ["0" if e == 0 else f"{e:g}" for e in etas]

    def vs_eta(ax, res, title, key="r2_mean"):
        style(ax)
        for k, lab, col in METHODS:
            v = [res["levels"][str(e)]["methods"][k][key] for e in etas]
            ax.plot(xs, v, "-o", color=col, lw=2 if k == "M1" else 1.5, ms=4, label=lab)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("measurement noise η (per point, / A)", color=INK, fontsize=9.5)
        ax.set_ylabel("confirmation R²", color=INK, fontsize=9.5)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.axvline(etas.index(0.01), color=GRID, lw=6, zorder=0)

    ax = fig.add_subplot(gs[1, 0])
    vs_eta(ax, ra, "(d) M0a: log σ, log τc (mean)")
    ax.axhline(0.9, color=INK2, lw=1, ls="--")
    ax.text(0.02, 0.9, " M0a threshold 0.90", color=INK2, fontsize=7.5, va="bottom",
            transform=ax.get_yaxis_transform())
    ax = fig.add_subplot(gs[1, 1])
    vs_eta(ax, rb, "(e) M0b: binned log S(ω) (6-bin mean)")
    ax.legend(frameon=False, fontsize=7.2, labelcolor=INK, loc="lower left")

    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    for res, lab, col, off in [(ra, "M0a", INK2, -0.1), (rb, "M0b", BLUE, 0.1)]:
        d = np.array([res["levels"][str(e)]["M1_minus_M2"] for e in etas])
        ax.errorbar(xs + off, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], fmt="o",
                    color=col, ms=4.5, capsize=2.5, lw=1.4, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(0.05, color=INK2, lw=1, ls="--")
    ax.text(0.02, 0.05, " M0b bar +0.05", color=INK2, fontsize=7.5, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xticks(xs, labels)
    ax.set_xlabel("measurement noise η", color=INK, fontsize=9.5)
    ax.set_ylabel("R²(R3 multi-spin) − R²(S1 single spin)", color=INK, fontsize=9.5)
    ax.set_title("(f) multi-spin vs single spin, same ridge (95% CI)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.savefig(FIG / "figNS0_noisespec_m0.png", dpi=200, facecolor=SURF, bbox_inches="tight")
    print("wrote", FIG / "figNS0_noisespec_m0.png")


if __name__ == "__main__":
    main()
