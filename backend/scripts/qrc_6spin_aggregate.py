"""Aggregate the 6-spin NARMA-2 learnable-encoding results across seeds and
across two optimizer recipes (loose vs stabilized).

Source: per-seed runs of qrc_learnable_9spin.py (--system 6 --task narma2,
leakage-free 3-way split). Values transcribed from the run logs. No GPU.

Finding: the per-spin advantage over arcsin is ROBUST to the optimizer recipe
(loose: lr 0.04; tight: lr 0.02 + grad-clip + best-val checkpoint) -- 4/5 seeds
beat, ~40% mean, paired p~0.04 either way. Learned *global* stays unreliable
(diverges) regardless. Writes fig15.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEEDS = [7, 1, 2, 3, 11]
ARCSIN = [0.3953, 0.5018, 0.3769, 0.4691, 0.3734]           # optimizer-independent
PERSPIN_LOOSE = [0.1749, 0.1765, 0.3953, 0.3235, 0.1606]    # lr 0.04
PERSPIN_TIGHT = [0.2156, 0.1996, 0.3783, 0.3894, 0.1525]    # lr 0.02 + clip + best-val
GLOBAL_TIGHT = [0.3417, 0.8760, 1.7464, 0.6872, 0.2735]     # unreliable
OUT = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig15_6spin_multiseed.png"


def _ms(x):
    x = np.asarray(x)
    return x.mean(), x.std(ddof=1)


def _paired(p, a):
    p, a = np.asarray(p), np.asarray(a)
    d = p - a
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return d.mean(), t, int((p < a).sum()), 100 * (1 - p.mean() / a.mean())


def main():
    a = np.asarray(ARCSIN)
    for name, x in [("arcsin", ARCSIN), ("perspin (loose)", PERSPIN_LOOSE),
                    ("perspin (tight)", PERSPIN_TIGHT), ("global (tight)", GLOBAL_TIGHT)]:
        m, s = _ms(x)
        print(f"{name:16s} test NMSE: {m:.3f} +/- {s:.3f}  {np.round(x, 3).tolist()}")
    for lbl, p in [("loose", PERSPIN_LOOSE), ("tight", PERSPIN_TIGHT)]:
        md, t, wins, rel = _paired(p, ARCSIN)
        print(f"perspin ({lbl}) vs arcsin: mean d {md:+.3f} | t={t:.2f} (dof 4) | "
              f"{wins}/5 beat | ~{rel:.0f}% mean improvement")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    groups = [("arcsin\n(baseline)", ARCSIN, "#3fb950"),
              ("per-spin\n(lr .04)", PERSPIN_LOOSE, "#e07b3c"),
              ("per-spin\n(lr .02+clip)", PERSPIN_TIGHT, "#c96a2a"),
              ("global\n(unreliable)", GLOBAL_TIGHT, "#4da3ff")]
    x = np.arange(len(groups))
    means = [np.mean(d) for _, d, _ in groups]
    stds = [np.std(d, ddof=1) for _, d, _ in groups]
    ax.bar(x, means, yerr=stds, width=0.62, color=[c for *_, c in groups], alpha=0.75, capsize=5)
    for xi, (_, d, _) in enumerate(groups):
        ax.scatter(np.full(len(d), xi) + np.linspace(-0.13, 0.13, len(d)), d,
                   color="#1f2328", s=18, zorder=3)
    ax.axhline(1.0, color="#999", ls=":", lw=1, label="mean-predictor (NMSE=1)")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("NARMA-2 test NMSE  (lower better)")
    mL, *_ = _ms(PERSPIN_LOOSE); mT, *_ = _ms(PERSPIN_TIGHT); mA, *_ = _ms(ARCSIN)
    ax.set_title("6-spin learnable encoding vs arcsin (5 seeds) — per-spin win is "
                 "optimizer-robust\n"
                 f"per-spin {mL:.2f}/{mT:.2f} vs arcsin {mA:.2f}  (4/5 seeds beat, "
                 "~40% mean, paired p~0.04); global unreliable")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
