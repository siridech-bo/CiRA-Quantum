"""Figure 10 — Phase-2.2 phase-amplitude head-to-head.

Compares the memory-capacity spectrum of plain amplitude encoding
(``arcsin_sqrt``, phase-amp OFF) vs. phase-amplitude encoding
(``R_z(2*pi*s) * R_x(theta)``, phase-amp ON), computed with *identical*
settings on the two persisted waveforms so the comparison is apples-to-apples.

No GPU: loads the saved ``.npz`` traces only. Writes
``docs/QRC/figures/fig10_phaseamp.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qrc_memcap import memory_capacity  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "backend" / "artifacts" / "traces"
OUT = ROOT / "docs" / "QRC" / "figures" / "fig10_phaseamp.png"

OFF = "memcap_arcsin_sqrt_30159a4c5e1b22d0.npz"
ON = "memcap_arcsin_sqrt_pa_af6a7ceb7c390eb2.npz"

# identical memcap settings for both sides
MC_KW = dict(kmax=30, washout=10, n_train=None, degrees=(1, 2, 3), n_pca=50, seed=42)


def _load(name: str):
    d = np.load(TRACES / name, allow_pickle=True)
    return d["fids"], d["memcap_input"], json.loads(str(d["meta"]))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fids_off, u_off, _ = _load(OFF)
    fids_on, u_on, _ = _load(ON)
    mc_off = memory_capacity(fids_off, u_off, **MC_KW)
    mc_on = memory_capacity(fids_on, u_on, **MC_KW)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- Panel A: grouped MC bars (linear / nonlinear / total) ---
    cats = ["linear MC", "nonlinear MC", "total MC"]
    off_v = [mc_off["linear_MC"], mc_off["nonlinear_MC"], mc_off["total_MC"]]
    on_v = [mc_on["linear_MC"], mc_on["nonlinear_MC"], mc_on["total_MC"]]
    x = np.arange(len(cats))
    w = 0.38
    axL.bar(x - w / 2, off_v, w, label="amplitude only (arcsin_sqrt)", color="#3fb950")
    axL.bar(x + w / 2, on_v, w, label="+ phase-amp  R_z(2πs)·R_x(θ)", color="#e07b3c")
    for xi, (a, b) in enumerate(zip(off_v, on_v)):
        axL.text(xi - w / 2, a + 0.15, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        axL.text(xi + w / 2, b + 0.15, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    axL.set_xticks(x)
    axL.set_xticklabels(cats)
    axL.set_ylabel("capacity (bits)")
    axL.set_title("A · Memory-capacity spectrum (identical settings)")
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(axis="y", alpha=0.25)

    # --- Panel B: representative FID waveforms (provenance-labelled) ---
    step = fids_off.shape[0] // 2
    axR.plot(np.real(fids_off[step]), color="#3fb950", lw=1.1,
             label=f"amplitude only · {OFF[:26]}…")
    axR.plot(np.real(fids_on[step]), color="#e07b3c", lw=1.1,
             label=f"phase-amp · {ON[:26]}…")
    axR.set_xlabel("FID sample")
    axR.set_ylabel("Re[FID]  (a.u.)")
    axR.set_title(f"B · Representative FID (step {step})")
    axR.legend(fontsize=7, loc="upper right")
    axR.grid(alpha=0.2)

    drop = 100 * (mc_on["total_MC"] - mc_off["total_MC"]) / mc_off["total_MC"]
    fig.suptitle(
        "Phase-2.2 — phase-amplitude encoding vs. plain amplitude "
        f"(total MC {mc_off['total_MC']:.2f} → {mc_on['total_MC']:.2f}, {drop:+.0f}%): "
        "phase-amp HURTS",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"OFF totMC={mc_off['total_MC']:.3f}  ON totMC={mc_on['total_MC']:.3f}  delta={drop:+.1f}%")


if __name__ == "__main__":
    main()
