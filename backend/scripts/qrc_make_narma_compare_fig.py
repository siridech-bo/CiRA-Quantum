"""Figure 5: our simulated NARMA NMSE vs Paper 4's *experimental* Table I.

Every number is read from the committed data package
``docs/QRC/data/qrc_paper4_narma.json`` (written by qrc_reproduce_paper4.py):
  * this work  -> ``qrc_narma[order].nmse_paper`` (NMSE in Paper 4's definition,
    standard 653-peak FID readout, regime A: fixed encoding, ridge readout only)
  * Hou et al. -> ``paper_table_i[order].best`` (their Table I, as transcribed
    into the same file by the reproduction script)
Nothing is typed in here. CPU only. Run from anywhere:
  python backend/scripts/qrc_make_narma_compare_fig.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "docs" / "QRC" / "data" / "qrc_paper4_narma.json"
OUT = REPO / "docs" / "QRC" / "figures" / "fig5_narma_vs_experiment.png"

d = json.loads(SRC.read_text(encoding="utf-8"))
orders = sorted(int(k) for k in d["qrc_narma"])
ours = [d["qrc_narma"][str(n)]["nmse_paper"] for n in orders]        # this work (sim)
paper = [d["paper_table_i"][str(n)]["best"] for n in orders]          # Hou et al., Table I (expt)
cfg = d["config"]["feature_cfg"]

fig, ax = plt.subplots(figsize=(6.4, 4.3))
ax.plot(orders, paper, "s--", color="#dc2626", lw=2, ms=8,
        label="Hou et al. 2026 (experiment, Table I)")
ax.plot(orders, ours, "o-", color="#2563eb", lw=2, ms=8,
        label=f"This work (simulation, {cfg['readout'].upper()}-{cfg['n_peaks']})")
ax.set_yscale("log")
ax.set_xlabel("NARMA order n")
ax.set_ylabel("NMSE (Paper-4 definition, log scale, lower = better)")
ax.set_title("NARMA: simulation vs Paper-4 experiment")
ax.set_xticks(orders)
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT} (from {SRC.name}): ours={['%.3g' % v for v in ours]}")
