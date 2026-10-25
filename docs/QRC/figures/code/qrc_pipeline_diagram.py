"""figR0: schematic of the QRC forecasting pipeline and what this work changed (input and output side).

Neutral boxes: the standard NMR QRC (Hou et al. PRL 136, 120602). Blue: our quantum-side change that was
adopted and confirmed (encoding gain). Orange: our classical readout addition, adopted for drifting inputs
(level handling). Grey dashed: pre-registered changes that were tested and not adopted. No numbers are
plotted; every label restates the frozen design (QRC_Design_Concepts.md, Steps 1-6).

  python docs/QRC/figures/code/qrc_pipeline_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle  # noqa: E402

FIG = Path(__file__).resolve().parents[1]
BLUE, BLUE_BG = "#2a78d6", "#e3eefb"
ORANGE, ORANGE_BG = "#eb6834", "#fde9e0"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984", "#b9b8b2"
SURF, BOX_BG = "#fcfcfb", "#f2f1ed"


def box(ax, x, y, w, h, fc=BOX_BG, ec=LINE, ls="-", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, ls=ls, lw=lw,
                                zorder=2))


def arrow(ax, p, q, color=INK2, lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=8, color=INK, weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, style=style, zorder=5,
            linespacing=1.35)


def molecule(ax, cx, cy):
    """crotonic acid, 9 spins: 4 13C (C1-C4), H1, H2 and the methyl group (H3-H5), strongest J couplings."""
    pos = {"C1": (-5.0, 0.0), "C2": (-1.7, 1.6), "C3": (1.7, 0.0), "C4": (5.0, 1.6),
           "H1": (-1.7, 4.4), "H2": (1.7, -2.8), "Me": (-5.0, -3.4)}
    bonds = [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C2", "H1"), ("C3", "H2"), ("C1", "Me")]
    for a, b in bonds:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.plot([cx + xa, cx + xb], [cy + ya, cy + yb], color=LINE, lw=1.6, zorder=3)
    for k, (dx, dy) in pos.items():
        carbon = k.startswith("C")
        ax.add_patch(Circle((cx + dx, cy + dy), 1.25 if k != "Me" else 1.5, fc=INK2 if carbon else SURF,
                            ec=INK2, lw=1.2, zorder=4))
        text(ax, cx + dx, cy + dy, "¹³C" if carbon else ("H₃" if k == "Me" else "¹H"), size=5.5,
             color=SURF if carbon else INK)


def main():
    fig, ax = plt.subplots(figsize=(12.0, 5.6), facecolor=SURF)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 61)
    ax.axis("off")
    yb, hb = 17, 20                                    # main row
    ym = yb + hb / 2

    # 1 input
    box(ax, 1, yb, 15, hb)
    text(ax, 8.5, yb + hb - 3, "Input series", weight="bold")
    text(ax, 8.5, ym - 1.5, "$s_t$ = (T, RH)\nmin–max to [0, 1]\non pre-test data", size=7.5, color=INK2)

    # 2 encoding (standard box + adopted gain badge)
    box(ax, 20, yb, 23, hb)
    text(ax, 31.5, yb + hb - 3, "Encoding pulses", weight="bold")
    text(ax, 31.5, ym + 0.5, "$R_x(\\theta)$ on every spin\nT → ¹H,  RH → ¹³C", size=7.5, color=INK2)
    box(ax, 21.2, yb + 1.3, 20.6, 7, fc=BLUE_BG, ec=BLUE, lw=1.6)
    text(ax, 31.5, yb + 6.1, "$\\theta = g\\,\\arcsin\\sqrt{s}$,  $g = 0.35$", size=7.5, color=INK, weight="bold")
    text(ax, 31.5, yb + 3.2, "encoding gain: less overwrite", size=6.5, color=BLUE)

    # 3 reservoir
    box(ax, 46.5, yb, 25, hb)
    text(ax, 59, yb + hb - 3, "NMR spin reservoir", weight="bold")
    molecule(ax, 59, ym + 0.3)
    text(ax, 59, yb + 2.9, "crotonic acid, 9 spins, $J$-coupled\n$T_1$, $T_2^*$ relaxation · free evolution $\\tau$ = 30 ms",
         size=6.6, color=INK2)
    arrow(ax, (68.5, yb + hb), (49.5, yb + hb), color=INK2, rad=0.55)
    text(ax, 59, yb + hb + 7.6, "state ρ carried to the next step (memory)", size=7, color=INK2, style="italic")

    # 4 readout
    box(ax, 75, yb, 18, hb)
    text(ax, 84, yb + hb - 3, "FID readout", weight="bold")
    text(ax, 84, ym - 1.5, "π/2 on ¹H → FID\n2048 × 0.3 ms\n|FFT|, 653 bins\n($D$ = 653)", size=7.5,
         color=INK2)

    # 5 ridge + output
    box(ax, 97, yb, 12, hb)
    text(ax, 103, yb + hb - 3, "Linear\nridge", weight="bold")
    text(ax, 103, ym - 3, "α, std\non validation", size=7, color=INK2)
    text(ax, 115, ym, "$\\hat y(t+h)$\nh = 1–45 d", size=8, color=INK)

    for p, q in (((16, ym), (20, ym)), ((43, ym), (46.5, ym)), ((71.5, ym), (75, ym)), ((93, ym), (97, ym)),
                 ((109, ym), (111.5, ym))):
        arrow(ax, p, q)

    # classical branch: level handling (adopted where inputs drift)
    box(ax, 20, 2, 73, 11, fc=ORANGE_BG, ec=ORANGE, lw=1.6)
    text(ax, 56.5, 10.2, "Level handling (classical features, used when inputs drift)", weight="bold", size=8)
    text(ax, 56.5, 5.6, "re-centre each target on the last value $s_t$  ·  append raw input lags "
                        "$s_{t-1}, s_{t-7}, s_{t-30}$\nthe same features are given to the tuned ESN (fair comparison)",
         size=7, color=INK2)
    arrow(ax, (8.5, yb), (8.5, 7.75), color=ORANGE)
    arrow(ax, (8.5, 7.75), (20, 7.75), color=ORANGE)
    arrow(ax, (93, 7.75), (103, 7.75), color=ORANGE)
    arrow(ax, (103, 7.75), (103, yb), color=ORANGE)

    # tested, not adopted (pre-registered)
    def dashed(x, w, lines, anchor_x):
        box(ax, x, 45, w, 9, fc=SURF, ec=MUTED, ls=(0, (3, 2)), lw=1.1)
        text(ax, x + w / 2, 49.5, lines, size=6.6, color=MUTED)
        ax.plot([anchor_x, anchor_x], [45, yb + hb], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)

    dashed(20, 23, "learned per-spin encoding\n$\\theta = \\arcsin s$ (Hou et al.)\n$g = 1$ (standard)", 31.5)
    dashed(75, 18, "¹³C FID · ⟨σ$_z$⟩ · populations\nbounded tanh features\nother τ, compact readouts", 84)
    dashed(96, 14, "kernel (RFF)\nreadout", 103)

    # legend
    lx, y = 1, 59
    for i, (fc, ec, ls, lab) in enumerate(((BOX_BG, LINE, "-", "standard NMR QRC (Hou et al.)"),
                                           (BLUE_BG, BLUE, "-", "ours, quantum side: adopted"),
                                           (ORANGE_BG, ORANGE, "-", "ours, classical: adopted"),
                                           (SURF, MUTED, (0, (3, 2)), "tested, not adopted (pre-registered)"))):
        x0 = lx + i * 29
        ax.add_patch(FancyBboxPatch((x0, y - 0.8), 2.4, 1.6, boxstyle="round,pad=0,rounding_size=0.4", fc=fc,
                                    ec=ec, ls=ls, lw=1.2))
        text(ax, x0 + 3.4, y, lab, size=7, color=INK2, ha="left")

    fig.savefig(FIG / "figR0_pipeline.png", dpi=220, facecolor=SURF, bbox_inches="tight")
    print("wrote figR0_pipeline.png")


if __name__ == "__main__":
    main()
