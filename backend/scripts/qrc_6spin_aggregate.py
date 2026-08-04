"""Aggregate the 6-spin NARMA-2 learnable-encoding results across seeds.

Source: the per-seed runs of qrc_learnable_9spin.py (--system 6 --task narma2,
leakage-free 3-way split). Values transcribed from the run logs
(docs/QRC/qrc_6spin_*.log). Produces mean +/- std, a paired test of per-spin vs
arcsin, and fig15. No GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEEDS = [7, 1, 2, 3, 11]
ARCSIN = [0.3953, 0.5018, 0.3769, 0.4691, 0.3734]
GLOBAL = [0.3460, 0.8623, 1.7457, 0.6380, 0.2843]
PERSPIN = [0.1749, 0.1765, 0.3953, 0.3235, 0.1606]
OUT = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig15_6spin_multiseed.png"


def _stats(x):
    x = np.asarray(x)
    return float(x.mean()), float(x.std(ddof=1))


def main():
    a, g, p = map(np.asarray, (ARCSIN, GLOBAL, PERSPIN))
    for name, x in [("arcsin", a), ("global", g), ("perspin", p)]:
        m, s = _stats(x)
        print(f"{name:8s} test NMSE: mean {m:.3f} +/- {s:.3f}  (per-seed {np.round(x,3).tolist()})")

    # paired per-spin vs arcsin
    d = p - a
    md, sd = _stats(d)
    t = md / (sd / np.sqrt(len(d)))
    wins = int((p < a).sum())
    print(f"\npaired (perspin - arcsin): mean {md:+.3f} +/- {sd:.3f} | t={t:.2f} (dof {len(d)-1}) "
          f"| perspin beats arcsin in {wins}/{len(d)} seeds")
    print(f"mean relative improvement: {100*(1 - p.mean()/a.mean()):.0f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = ["arcsin\n(baseline)", "learned\nglobal", "learned\nper-spin"]
    data = [a, g, p]
    colors = ["#3fb950", "#4da3ff", "#e07b3c"]
    x = np.arange(3)
    means = [d.mean() for d in data]
    stds = [d.std(ddof=1) for d in data]
    ax.bar(x, means, yerr=stds, width=0.6, color=colors, alpha=0.75, capsize=5)
    for xi, d in enumerate(data):        # per-seed points
        ax.scatter(np.full(len(d), xi) + np.linspace(-0.12, 0.12, len(d)), d,
                   color="#1f2328", s=18, zorder=3)
    ax.axhline(1.0, color="#999", ls=":", lw=1, label="mean-predictor (NMSE=1)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("NARMA-2 test NMSE  (lower better)")
    ax.set_title(f"6-spin learnable encoding vs arcsin — {len(SEEDS)} seeds\n"
                 f"per-spin {p.mean():.3f}±{p.std(ddof=1):.3f} vs arcsin {a.mean():.3f}±{a.std(ddof=1):.3f} "
                 f"({wins}/{len(SEEDS)} seeds beat; ~{100*(1-p.mean()/a.mean()):.0f}% mean)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
