"""Effect of FID readout richness (fid_points) on weather forecast skill.

Linear-QRC (ridge) R² vs horizon at fid_points=1024 (v1) and 2048 (v2),
against the Hou et al. experiment. Linear readout isolates the
readout-richness effect (no RBF-tuning confound).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H = [1, 5, 10, 15, 20, 30, 45]
v1 = json.load(open("artifacts/qrc_paper4_weather.json"))["qrc_weather"]
v2 = json.load(open("artifacts/qrc_paper4_weather_v2.json"))["qrc_weather"]
paper = {"temp": [0.92, 0.86, 0.82, 0.80, 0.77, 0.72, 0.67],
         "humidity": [0.72, 0.57, 0.53, 0.48, 0.45, 0.42, 0.38]}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for j, var in enumerate(("temp", "humidity")):
    a = ax[j]
    a.plot(H, [v1[str(h)][var]["r2"] for h in H], "o--", color="#93c5fd", lw=2,
           ms=6, label="QRC, fid_points=1024 (sim)")
    a.plot(H, [v2[str(h)][var]["r2"] for h in H], "o-", color="#1e3a8a", lw=2.5,
           ms=7, label="QRC, fid_points=2048 (sim)")
    a.plot(H, paper[var], "s--", color="#b91c1c", lw=2, ms=6,
           label="QRC (Hou et al., expt)")
    a.set_xlabel("forecast horizon h (days)")
    a.set_ylabel("R²")
    a.set_ylim(0.3, 1.0)
    a.set_title("Temperature" if j == 0 else "Humidity")
    a.grid(alpha=0.3)
    if j == 0:
        a.legend(fontsize=9)
fig.suptitle("Readout richness closes the sim–experiment gap: doubling the FID "
             "samples (1024→2048) lifts long-horizon skill onto the experiment", y=1.02)
fig.savefig("../docs/QRC/figures/fig7_fid_points.png", dpi=150, bbox_inches="tight")
print("saved fig7_fid_points.png")
