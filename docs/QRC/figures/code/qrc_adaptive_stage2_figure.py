"""Fig. AE2: Stage 2 - adaptive vs fixed vs static probe choice, known model (v2 posterior).

Reads backend/artifacts/traces/qrc_adaptive_stage2_{S1,R3}_eta0.01_v2.npz and
docs/QRC/data/qrc_adaptive_stage2_v2_results.json -> docs/QRC/figures/figAE2_stage2_adaptive.png

  python docs/QRC/figures/code/qrc_adaptive_stage2_figure.py
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
TRACES = REPO / "backend/artifacts/traces"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
POL = [("fixed-cycle", "fixed: cycle the 49-probe bank", ORANGE, "^"),
       ("static-best", "static: best single probe (prior)", BLUE, "s"),
       ("greedy-fisher", "adaptive: greedy posterior FI", AQUA, "o")]
ETA, J = "0.01", 0          # primary eta; theta0 = 1.627 (low edge, largest adaptive gain)


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
    r = json.loads((DATA / "qrc_adaptive_stage2_v2_results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURF)
    n = np.arange(1, 197)
    for ax, (sysn, title) in zip(axes[:2], (("S1", "(a) S1, θ0 = 1.63, η = 1e-2"),
                                            ("R3", "(b) R3, θ0 = 1.63, η = 1e-2"))):
        style(ax)
        z = np.load(TRACES / f"qrc_adaptive_stage2_{sysn}_eta{ETA}_v2.npz")
        pt = r["systems"][sysn][ETA]["points"][J]
        th0 = pt["theta0"]
        ax.plot(n, pt["b_bank_196"] * 196 / n, "--", color=INK2, lw=1.3, label="bound, fixed bank")
        ax.plot(n, pt["b_opt_196"] * 196 / n, "-", color=INK, lw=1.3, label="bound, best probe every shot")
        ax.axhline(pt["target"], color=INK2, lw=0.8, ls=":")
        ax.text(196, pt["target"], " target", color=INK2, fontsize=8, va="center")
        order = [str(x) for x in z["policies"]]
        for p, lab, col, mk in POL:
            mse = ((z["means"][J, order.index(p)] - th0) ** 2).mean(0)
            ax.plot(n, mse, color=col, lw=2, label=lab)
            ns = pt["policies"][p]["n_star"]
            if ns:
                ax.plot(ns, mse[ns - 1], mk, color=col, ms=8, markeredgecolor=SURF, markeredgewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots n", color=INK, fontsize=9.5)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("MSE of log10 σ (200 episodes)", color=INK, fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=7.8, labelcolor=INK, loc="lower left")

    ax = axes[2]
    style(ax)
    for sysn, ls in (("S1", "--"), ("R3", "-")):
        c = r["systems"][sysn][ETA]
        th = [p["theta0"] for p in c["points"]]
        for p, lab, col, mk in POL:
            ax.plot(th, c["n_star"][p], ls, marker=mk, color=col, lw=1.6, ms=6,
                    markeredgecolor=SURF, markeredgewidth=1)
    ax.plot([], [], "-", color=INK2, label="R3")
    ax.plot([], [], "--", color=INK2, label="S1")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    ax.set_xlabel("true θ0 = log10 σ", color=INK, fontsize=9.5)
    ax.set_ylabel("shots to reach the target, n*", color=INK, fontsize=9.5)
    ax.set_title("(c) shots to target, η = 1e-2", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figAE2_stage2_adaptive.png", dpi=200, facecolor=SURF)
    print("wrote", FIG / "figAE2_stage2_adaptive.png")


if __name__ == "__main__":
    main()
