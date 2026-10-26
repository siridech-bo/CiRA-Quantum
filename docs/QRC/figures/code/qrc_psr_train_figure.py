"""Fig. R5: parameter-shift (PSR) training loop vs autograd (qrc_psr_train.py, 4 spins).

Reads docs/QRC/data/qrc_psr_train_4spin.json -> docs/QRC/figures/figR5_psr_train.png

  python docs/QRC/figures/code/qrc_psr_train_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
FIG = REPO / "docs/QRC/figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def main():
    d = json.loads((REPO / "docs/QRC/data/qrc_psr_train_4spin.json").read_text())
    psr, ag = d["val_nmse"]["psr"], d["val_nmse"]["autograd"]
    steps = list(range(len(psr)))
    rel = [g["grad_relerr"] for g in d["gap"]]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.5), facecolor=SURF)
    for ax in (a, b):
        style(ax)
    a.plot(steps, ag, color=BLUE, lw=2, label="autograd (backprop)")
    a.plot(steps, psr, ls="none", marker="o", ms=4.5, mfc="none", mec=ORANGE, mew=1.4,
           label="parameter shift (forward runs only)")
    a.set_xlabel("training step", color=INK, fontsize=10)
    a.set_ylabel("validation NMSE", color=INK, fontsize=10)
    a.set_title("(a) identical training curves", loc="left", color=INK, fontsize=10.5)
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK)

    b.semilogy(steps, rel, color=ORANGE, lw=2)
    b.set_ylim(1e-16, 1e-6)
    b.set_xlabel("training step", color=INK, fontsize=10)
    b.set_ylabel("max relative gradient difference", color=INK, fontsize=10)
    b.set_title("(b) PSR gradient vs autograd", loc="left", color=INK, fontsize=10.5)
    b.text(0.02, 0.92, f"{d['experiments_per_step']:,} shifted experiments per step\n"
           f"{d['params']} encoding parameters, D = {d['D']} observable readout",
           transform=b.transAxes, fontsize=8.5, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "figR5_psr_train.png", dpi=200, facecolor=SURF)
    print("wrote", FIG / "figR5_psr_train.png")


if __name__ == "__main__":
    main()
