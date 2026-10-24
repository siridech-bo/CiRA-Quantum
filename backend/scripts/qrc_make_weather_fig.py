"""Weather QRC-vs-ESN R² figure for the reproduction report (Fig 4).

Reads docs/QRC/data/qrc_paper4_weather_v2.json (nothing typed in). CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "data"
FIG = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures"

d = json.loads((DATA / "qrc_paper4_weather_v2.json").read_text(encoding="utf-8"))
q = d["qrc_weather"]
e = d["esn_weather"]
H = sorted((int(h) for h in q), key=int)
esn_big = str(max(e, key=int))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for j, var in enumerate(("temp", "humidity")):
    a = ax[j]
    a.plot(H, [q[str(h)][var]["r2"] for h in H], "o-", color="#2563eb", lw=2, label="QRC")
    a.plot(H, [q[str(h)][var]["rbf_r2"] for h in H], "^-", color="#1e3a8a", lw=2,
           label="QRC+RBF")
    for s in sorted(e, key=int):
        a.plot(H, [e[s][str(h)][var] for h in H], "s--", lw=1, alpha=0.6,
               label=f"ESN({s})")
    a.set_xlabel("forecast horizon (days)")
    a.set_ylabel("R²")
    a.set_title(f"({'ab'[j]}) {'Temperature' if j==0 else 'Humidity'}")
    a.grid(alpha=0.3)
    if j == 0:
        a.legend(fontsize=8, ncol=2)
fig.suptitle("Weather forecasting: QRC vs classical ESN "
             "(QRC+RBF beats ESN-10000 at long horizons)", y=1.02)
fig.savefig(FIG / "fig4_weather.png", dpi=150, bbox_inches="tight")
print(f"saved {FIG / 'fig4_weather.png'}")
