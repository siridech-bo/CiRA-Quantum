"""Figure 11 — Phase-2.3 protons-only vs. all-spins encoding head-to-head.

Compares the memory-capacity spectrum of encoding the input into all 9 spins
(baseline ``arcsin_sqrt``) vs. into the 5 proton spins only
(``target_qubits=[4..8]``), computed with *identical* settings on the two
persisted waveforms so the comparison is apples-to-apples.

No GPU: loads the saved ``.npz`` traces only. Writes
``docs/QRC/figures/fig11_protons.png``.
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
OUT = ROOT / "docs" / "QRC" / "figures" / "fig11_protons.png"

ALL = "memcap_arcsin_sqrt_30159a4c5e1b22d0.npz"
PRO = "memcap_arcsin_sqrt_protons_0ffd5ab829f265b8.npz"

MC_KW = dict(kmax=30, washout=10, n_train=None, degrees=(1, 2, 3), n_pca=50, seed=42)


def _load(name: str):
    d = np.load(TRACES / name, allow_pickle=True)
    return d["fids"], d["memcap_input"], json.loads(str(d["meta"]))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fids_all, u_all, _ = _load(ALL)
    fids_pro, u_pro, _ = _load(PRO)
    mc_all = memory_capacity(fids_all, u_all, **MC_KW)
    mc_pro = memory_capacity(fids_pro, u_pro, **MC_KW)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- Panel A: grouped MC bars ---
    cats = ["linear MC", "nonlinear MC", "total MC"]
    all_v = [mc_all["linear_MC"], mc_all["nonlinear_MC"], mc_all["total_MC"]]
    pro_v = [mc_pro["linear_MC"], mc_pro["nonlinear_MC"], mc_pro["total_MC"]]
    x = np.arange(len(cats))
    w = 0.38
    axL.bar(x - w / 2, all_v, w, label="all 9 spins (baseline)", color="#3fb950")
    axL.bar(x + w / 2, pro_v, w, label="protons only  [H1–H5]", color="#4da3ff")
    for xi, (a, b) in enumerate(zip(all_v, pro_v)):
        axL.text(xi - w / 2, a + 0.15, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        axL.text(xi + w / 2, b + 0.15, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    axL.set_xticks(x)
    axL.set_xticklabels(cats)
    axL.set_ylabel("capacity (bits)")
    axL.set_title("A · Memory-capacity spectrum (identical settings)")
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(axis="y", alpha=0.25)
    # annotate the key contrast
    axL.annotate("linear MC unchanged\n(protons carry the memory)",
                 xy=(0, all_v[0]), xytext=(0.05, all_v[0] + 3.2),
                 fontsize=7.5, ha="left", color="#59636e",
                 arrowprops=dict(arrowstyle="->", color="#59636e", lw=0.8))
    axL.annotate("nonlinearity lost\n(carbons feed it)",
                 xy=(1 + w / 2, pro_v[1]), xytext=(1.15, pro_v[1] + 2.6),
                 fontsize=7.5, ha="left", color="#bc4c00",
                 arrowprops=dict(arrowstyle="->", color="#bc4c00", lw=0.8))

    # --- Panel B: representative FID waveforms ---
    step = fids_all.shape[0] // 2
    axR.plot(np.real(fids_all[step]), color="#3fb950", lw=1.1,
             label=f"all spins · {ALL[:24]}…")
    axR.plot(np.real(fids_pro[step]), color="#4da3ff", lw=1.1,
             label=f"protons only · {PRO[:24]}…")
    axR.set_xlabel("FID sample")
    axR.set_ylabel("Re[FID]  (a.u.)")
    axR.set_title(f"B · Representative FID (step {step})")
    axR.legend(fontsize=7, loc="upper right")
    axR.grid(alpha=0.2)

    drop = 100 * (mc_pro["total_MC"] - mc_all["total_MC"]) / mc_all["total_MC"]
    fig.suptitle(
        "Phase-2.3 — protons-only vs. all-spins encoding "
        f"(total MC {mc_all['total_MC']:.2f} → {mc_pro['total_MC']:.2f}, {drop:+.0f}%): "
        "linear memory kept, nonlinearity lost",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"ALL totMC={mc_all['total_MC']:.3f}  PRO totMC={mc_pro['total_MC']:.3f}  delta={drop:+.1f}%")


if __name__ == "__main__":
    main()
