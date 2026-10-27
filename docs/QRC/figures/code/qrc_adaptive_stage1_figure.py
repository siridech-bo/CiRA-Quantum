"""Fig. AE1: Stage 1 yardstick - estimator RMSE vs the Cramer-Rao bound, S1 and R3.

Reads docs/QRC/data/qrc_adaptive_stage1_results.json -> docs/QRC/figures/figAE1_stage1_crb.png

  python docs/QRC/figures/code/qrc_adaptive_stage1_figure.py
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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EST = [("MLE", "known-model MLE (D = 50,176)", AQUA, "o"),
       ("MLP", "M0 MLP", BLUE, "s"),
       ("ridge", "M0 ridge", ORANGE, "^")]


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
    r = json.loads((DATA / "qrc_adaptive_stage1_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURF, sharey=True)
    for ax, (sysn, title) in zip(axes, (("S1", "(a) single spin S1"), ("R3", "(b) 3-spin reservoir R3"))):
        style(ax)
        R = r["systems"][sysn]
        etas = sorted(float(e) for e in R["levels"])
        lv = [R["levels"][str(e) if str(e) in R["levels"] else repr(e)] for e in etas]
        crb = [np.sqrt(np.median(v["crb"])) for v in lv]
        ro = [np.sqrt(np.median(v["crb_readout"]["time-lin"])) for v in lv]
        ax.plot(etas, crb, "--", color=INK, lw=1.6, label="Cramér–Rao bound (raw FID)")
        ax.plot(etas, ro, ":", color=INK2, lw=1.4, label="bound after time-lin readout")
        for k, lab, col, mk in EST:
            y = [np.median([np.sqrt(p["mse"][0]) for p in v["estimators"][k]["per_point"]]) for v in lv]
            ax.plot(etas, y, marker=mk, color=col, lw=2, ms=8, label=lab,
                    markeredgecolor=SURF, markeredgewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("measurement noise η (relative to signal)", color=INK, fontsize=9.5)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("RMSE of log10 σ (median over 7 test points)", color=INK, fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "figAE1_stage1_crb.png", dpi=200, facecolor=SURF)
    print("wrote", FIG / "figAE1_stage1_crb.png")


if __name__ == "__main__":
    main()
