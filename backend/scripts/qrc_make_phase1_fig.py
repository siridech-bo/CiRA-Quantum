"""Figure 8 — Phase-1 feature-representation study (the rigorous picture).

Two panels, recomputed from the cached full trace so it's reproducible:

  (A) Standalone weather-R² vs horizon for magnitude653 / multimodal-18 /
      multimodal-136, with blocked-CV mean ± std bands. Shows that richer
      multimodal extraction (18→136) roughly doubles long-horizon skill, but
      that all representations overlap heavily within CV error.
  (B) Paired per-fold Δ (multimodal-136 − magnitude653) at each horizon — the
      honest test. Individual folds + mean; the spread straddling zero is why
      "rich beats baseline" is not supported.

Run from ``backend``::

    python scripts/qrc_make_phase1_fig.py --trace artifacts/traces/weather_full.npz \
        --out docs/QRC/figures/fig8_phase1_features.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.qrc.feature_methods import build_features
from qrc_phase1_rich import rich_multimodal
from qrc_phase1_v2 import blocked_folds, column_masks, load_trace, ridge_r2


def per_fold_r2(Xr, y, folds):
    return np.array([ridge_r2(Xr[a], y[a], Xr[b], y[b]) for a, b in folds])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    ap.add_argument("--out", default="docs/QRC/figures/fig8_phase1_features.png")
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz = load_trace(args.trace)
    fids = np.asarray(npz["fids"])
    n = fids.shape[0]
    wo, ntr, nte = (int(x) for x in np.asarray(npz["split"]).ravel()[:3])
    temp = np.asarray(npz["weather_norm"], dtype=float)[:, 0]
    horizons = [int(h) for h in np.asarray(npz["horizons"]).ravel()]

    X, names = build_features(fids, "multimodal", n_peaks=653, select="first")
    masks = column_masks(names)
    print("building rich multimodal (136)…", flush=True)
    rich, _ = rich_multimodal(fids)
    reps = {
        "magnitude653 (653)": (X[:, masks["magnitude653"]], "#2563eb", "o"),
        "multimodal-18 (18)": (X[:, masks["multimodal_only"]], "#9aa4b2", "s"),
        "multimodal-136 (136)": (rich, "#16a34a", "^"),
    }
    folds = blocked_folds(wo, wo + ntr + nte, args.cv_folds)

    # per-fold R² for every rep × horizon
    r2 = {name: {h: per_fold_r2(Xr, temp[np.arange(n) + h], folds)
                 for h in horizons}
          for name, (Xr, _, _) in reps.items()}
    print("computed per-fold R²", flush=True)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- Panel A: standalone R² vs horizon, mean ± std -------------------
    for name, (_, color, mk) in reps.items():
        means = np.array([r2[name][h].mean() for h in horizons])
        stds = np.array([r2[name][h].std() for h in horizons])
        axA.plot(horizons, means, mk + "-", color=color, lw=2, ms=6, label=name)
        axA.fill_between(horizons, means - stds, means + stds, color=color, alpha=0.13)
    axA.set_xlabel("forecast horizon (days ahead)")
    axA.set_ylabel("weather R²  (temperature, test)")
    axA.set_title("(A) Standalone skill — blocked-CV mean ± std")
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8, loc="lower left")
    axA.set_ylim(0.2, 1.0)

    # ---- Panel B: paired per-fold Δ (rich − magnitude) -------------------
    rich_name, mag_name = "multimodal-136 (136)", "magnitude653 (653)"
    axB.axhline(0, color="#888", lw=1, ls="--")
    means = []
    for i, h in enumerate(horizons):
        d = r2[rich_name][h] - r2[mag_name][h]          # paired, same folds
        xj = np.full(len(d), i) + np.random.default_rng(h).uniform(-0.08, 0.08, len(d))
        axB.scatter(xj, d, s=34, color="#16a34a", alpha=0.7, edgecolor="white", linewidth=0.5, zorder=3)
        axB.scatter([i], [d.mean()], marker="_", s=600, color="#0b4d1f", zorder=4)
        means.append(d.mean())
    axB.plot(range(len(horizons)), means, color="#0b4d1f", lw=1, alpha=0.5, zorder=2)
    axB.set_xticks(range(len(horizons)))
    axB.set_xticklabels([f"h{h}" for h in horizons])
    axB.set_xlabel("forecast horizon")
    axB.set_ylabel("Δ R²  (multimodal-136 − magnitude653)")
    axB.set_title("(B) Paired per-fold difference — no robust winner")
    axB.grid(True, alpha=0.3, axis="y")
    axB.text(0.02, 0.97, "above 0 = rich better\n(folds straddle 0 → wash)",
             transform=axB.transAxes, fontsize=8, va="top", color="#555")

    fig.suptitle("Phase-1: FID feature representations for weather forecasting "
                 "(9-spin crotonic trace, 1474 steps)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
