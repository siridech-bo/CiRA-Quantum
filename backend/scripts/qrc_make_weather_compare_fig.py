"""Weather: our simulation vs Paper-4 experiment (Fig 4b), same scale.

Paper-4 values are digitized (approximate, +/-~0.03) from Hou et al. Fig 4b,
read at our forecast horizons. Our values are from the committed run
(qrc_paper4_weather.json).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = [1, 5, 10, 15, 20, 30, 45]

# --- Paper 4, Fig 4b (digitized, approximate) ---
paper = {
    "temp": {"QRC":     [0.92, 0.86, 0.82, 0.80, 0.77, 0.72, 0.67],
             "QRC+RBF": [0.92, 0.87, 0.85, 0.85, 0.84, 0.83, 0.82]},
    "humidity": {"QRC":     [0.72, 0.57, 0.53, 0.48, 0.45, 0.42, 0.38],
                 "QRC+RBF": [0.72, 0.58, 0.55, 0.52, 0.50, 0.48, 0.47]},
}

# --- This work (simulation) ---
d = json.load(open("artifacts/qrc_paper4_weather.json"))["qrc_weather"]
ours = {v: {"QRC": [d[str(h)][v]["r2"] for h in H],
            "QRC+RBF": [d[str(h)][v]["rbf_r2"] for h in H]}
        for v in ("temp", "humidity")}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for j, var in enumerate(("temp", "humidity")):
    a = ax[j]
    a.plot(H, ours[var]["QRC+RBF"], "^-", color="#1e3a8a", lw=2, ms=7,
           label="QRC+RBF (this work, sim)")
    a.plot(H, ours[var]["QRC"], "o-", color="#2563eb", lw=2, ms=6,
           label="QRC (this work, sim)")
    a.plot(H, paper[var]["QRC+RBF"], "D--", color="#b91c1c", lw=2, ms=6,
           label="QRC+RBF (Hou et al., expt)")
    a.plot(H, paper[var]["QRC"], "s--", color="#f59e0b", lw=2, ms=6,
           label="QRC (Hou et al., expt)")
    a.set_xlabel("forecast horizon h (days)")
    a.set_ylabel("R²")
    a.set_ylim(0.3, 1.0)
    a.set_title(f"{'Temperature' if j == 0 else 'Humidity'}")
    a.grid(alpha=0.3)
    if j == 0:
        a.legend(fontsize=8)
fig.suptitle("Weather forecasting: this simulation vs Hou et al. 2026 experiment "
             "(Paper-4 values digitized from Fig 4b, approximate)", y=1.02)
fig.savefig("../docs/QRC/figures/fig6_weather_sim_vs_expt.png", dpi=150,
            bbox_inches="tight")
print("saved ../docs/QRC/figures/fig6_weather_sim_vs_expt.png")
