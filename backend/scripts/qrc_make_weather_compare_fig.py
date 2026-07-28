"""Weather: our simulation vs Paper-4 experiment (Fig 4b), same scale,
including the classical-ESN baseline (our exact band + Paper-4 digitized band).

Paper-4 QRC/QRC+RBF/ESN values are digitized (approximate, +/-~0.03) from
Hou et al. Fig 4b at our horizons. Our values are from the committed run.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = [1, 5, 10, 15, 20, 30, 45]

# --- Paper 4, Fig 4b (digitized, approximate) ---
paper = {
    "temp": {
        "QRC":     [0.92, 0.86, 0.82, 0.80, 0.77, 0.72, 0.67],
        "QRC+RBF": [0.92, 0.87, 0.85, 0.85, 0.84, 0.83, 0.82],
        "ESN_lo":  [0.92, 0.84, 0.79, 0.74, 0.64, 0.55, 0.42],
        "ESN_hi":  [0.94, 0.87, 0.83, 0.80, 0.75, 0.68, 0.60]},
    "humidity": {
        "QRC":     [0.72, 0.57, 0.53, 0.48, 0.45, 0.42, 0.38],
        "QRC+RBF": [0.72, 0.58, 0.55, 0.52, 0.50, 0.48, 0.47],
        "ESN_lo":  [0.70, 0.52, 0.44, 0.36, 0.28, 0.18, 0.10],
        "ESN_hi":  [0.74, 0.60, 0.52, 0.48, 0.42, 0.33, 0.32]},
}

# --- This work (simulation) ---
run = json.load(open("artifacts/qrc_paper4_weather_v2.json"))
q = run["qrc_weather"]
e = run["esn_weather"]
sizes = sorted(e, key=int)
ours = {}
for var in ("temp", "humidity"):
    band = np.array([[e[s][str(h)][var] for s in sizes] for h in H])
    ours[var] = {
        "QRC": [q[str(h)][var]["r2"] for h in H],
        "QRC+RBF": [q[str(h)][var]["rbf_r2"] for h in H],
        "ESN_lo": band.min(axis=1), "ESN_hi": band.max(axis=1)}

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
for j, var in enumerate(("temp", "humidity")):
    a = ax[j]
    o, p = ours[var], paper[var]
    # ESN bands
    a.fill_between(H, o["ESN_lo"], o["ESN_hi"], color="#2563eb", alpha=0.13,
                   label="ESN 500–10000 (this work)")
    a.fill_between(H, p["ESN_lo"], p["ESN_hi"], color="#b91c1c", alpha=0.12,
                   label="ESN 500–10000 (expt)")
    # QRC lines
    a.plot(H, o["QRC+RBF"], "^-", color="#1e3a8a", lw=2, ms=7, label="QRC+RBF (sim)")
    a.plot(H, o["QRC"], "o-", color="#2563eb", lw=2, ms=6, label="QRC (sim)")
    a.plot(H, p["QRC+RBF"], "D--", color="#b91c1c", lw=2, ms=6, label="QRC+RBF (expt)")
    a.plot(H, p["QRC"], "s--", color="#f59e0b", lw=2, ms=6, label="QRC (expt)")
    a.set_xlabel("forecast horizon h (days)")
    a.set_ylabel("R²")
    a.set_ylim(0.0, 1.0)
    a.set_title("Temperature" if j == 0 else "Humidity")
    a.grid(alpha=0.3)
    if j == 0:
        a.legend(fontsize=8, ncol=2, loc="lower left")
fig.suptitle("Weather forecasting: this simulation vs Hou et al. 2026 experiment, "
             "with classical-ESN baselines (Paper-4 values digitized from Fig 4b)", y=1.02)
fig.savefig("../docs/QRC/figures/fig6_weather_sim_vs_expt.png", dpi=150,
            bbox_inches="tight")
print("saved fig6 (with ESN bands)")
