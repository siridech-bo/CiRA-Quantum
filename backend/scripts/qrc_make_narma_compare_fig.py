"""Figure: our simulated NARMA NMSE vs Paper 4's *experimental* Table I."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

orders = [2, 5, 10, 15, 20]
ours = [5.19e-6, 5.21e-5, 2.46e-5, 1.82e-5, 3.24e-6]        # this work (sim)
paper = [1.74e-7, 4.44e-5, 5.84e-5, 6.37e-5, 4.34e-5]        # Hou et al., Table I (expt)

fig, ax = plt.subplots(figsize=(6.4, 4.3))
ax.plot(orders, paper, "s--", color="#dc2626", lw=2, ms=8,
        label="Hou et al. 2026 (experiment, Table I)")
ax.plot(orders, ours, "o-", color="#2563eb", lw=2, ms=8,
        label="This work (simulation, FID-653)")
ax.set_yscale("log")
ax.set_xlabel("NARMA order n")
ax.set_ylabel("NMSE (log scale, lower = better)")
ax.set_title("NARMA: simulation vs Paper-4 experiment")
ax.set_xticks(orders)
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.savefig("../docs/QRC/figures/fig5_narma_vs_experiment.png", dpi=150,
            bbox_inches="tight")
print("saved ../docs/QRC/figures/fig5_narma_vs_experiment.png")
