"""Generate publication figures for the QRC scaling report (docs/QRC)."""
from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "../docs/QRC/figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})

rows = [json.loads(x) for x in open("qrc_scaling_progress.jsonl")]
N = [r["n_qubits"] for r in rows]
MC = [r["memory_capacity"] for r in rows]
QRC = [r["narma_nmse"] for r in rows]
ESN = [r["esn_narma_nmse"] for r in rows]
SEC = [r["seconds"] for r in rows]

# --- Fig 1: MC and NARMA vs N (dual panel) --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(N, MC, "o-", color="#2563eb", lw=2, ms=7)
ax[0].set_xlabel("qubit count N"); ax[0].set_ylabel("memory capacity (Σ corr²)")
ax[0].set_title("(a) Memory capacity vs N")
ax[0].axvline(4, ls="--", color="gray", alpha=0.6)
ax[0].annotate("peak at N=4, then saturates ~6–7", (4.1, MC[1]),
               fontsize=9, color="gray")

ax[1].plot(N, QRC, "o-", color="#2563eb", lw=2, ms=7, label="QRC")
ax[1].plot(N, ESN, "s--", color="#dc2626", lw=2, ms=6, label="classical ESN (size-matched)")
ax[1].set_yscale("log")
ax[1].set_xlabel("qubit count N"); ax[1].set_ylabel("NARMA-2 NMSE (log, lower=better)")
ax[1].set_title("(b) NARMA-2 accuracy vs N")
ax[1].legend()
fig.suptitle("QRC N-qubit scaling (fair config: n_train=350 > 3·N·V)", y=1.02)
fig.savefig(f"{OUT}/fig1_scaling.png")
plt.close(fig)

# --- Fig 2: fading-memory curve (corr² vs delay) for a representative N ----
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for idx, color in [(0, "#93c5fd"), (2, "#3b82f6"), (6, "#1e3a8a")]:
    r = rows[idx]
    d = sorted(int(k) for k in r["mc_per_delay"])
    y = [r["mc_per_delay"][str(k)] for k in d]
    ax.plot(d, y, "o-", ms=4, color=color, label=f"N={r['n_qubits']}")
ax.set_xlabel("delay d (steps)"); ax.set_ylabel("corr²(u$_{k-d}$, prediction)")
ax.set_title("Fading memory: smooth corr² decay with delay")
ax.legend()
fig.savefig(f"{OUT}/fig2_fading_memory.png")
plt.close(fig)

# --- Fig 3: compute time per N (GPU) --------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
bars = ax.bar([str(n) for n in N], [s / 60 for s in SEC], color="#2563eb")
ax.set_xlabel("qubit count N"); ax.set_ylabel("wall time per point (min)")
ax.set_title("Compute cost per scaling point (GPU for N≥6)")
for b, s in zip(bars, SEC, strict=True):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
            f"{s/60:.1f}", ha="center", va="bottom", fontsize=8)
fig.savefig(f"{OUT}/fig3_timings.png")
plt.close(fig)

print("figures written to", OUT)
