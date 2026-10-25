"""Post-hoc diagnosis: why is the frozen QRC design behind the tuned ESN in Lagos (Confirmation 3)?

Descriptive only (all test sets already seen). Across the 9 untouched confirmation cities, relates the
QRC(g=0.35) - tuned-ESN pooled test-R2 gap to climate / input statistics computed from the raw ERA5 data
with the benchmark's own normalisation (min-max from pre-test data): input spread in [0, 1] units, share
of variance in the seasonal cycle, anomaly persistence, test-input shift. For the Confirmation 3 cities it
also breaks the gap down per target (variable x horizon) from the saved test predictions, and computes
the base design's descriptive comparisons (F3) against the tuned LSTM / linear AR there with the same
block bootstrap, so the F3 pooled summary can be extended from 6 to 9 cities.

  python scripts/qrc_lagos_diag.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.qrc.provenance import provenance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "docs/QRC/data"
CITIES = {"cairo": "confirm1", "chennai": "confirm1", "moscow": "confirm1",
          "saopaulo": "confirm2", "sydney": "confirm2", "toronto": "confirm2",
          "lagos": "confirm3", "mexicocity": "confirm3", "stockholm": "confirm3"}
H = (1, 5, 10, 15, 20, 30, 45)
VARS = ("temp", "humidity")


def gap(city, tag):
    d = json.loads((DATA / f"qrc_{tag}_{city}.json").read_text(encoding="utf-8"))
    g = d["groups"]
    if tag == "confirm3":
        return g["qrc_base"]["pooled"] - g["esn_base"]["pooled"], d["tests"]["G3 qrc_base - esn_base"]["ci95"]
    return g["qrc_g035"]["pooled"] - g["esn_tuned"]["pooled"], d["tests"]["S3 g0.35 - esn_tuned"]["all"]["ci95"]


def stats(city):
    d = json.loads((ROOT / f"data/weather/era5/{city}_1970_2025.json").read_text(encoding="utf-8"))["daily"]
    dates = [dt.date.fromisoformat(s) for s in d["time"]]
    raw = np.column_stack([np.asarray(d["temperature_2m_mean"], float),
                           np.asarray(d["relative_humidity_2m_mean"], float)])
    ix = lambda y: dates.index(dt.date(y, 1, 1))  # noqa: E731
    i_test = ix(2005)
    lo, hi = raw[:i_test].min(0), raw[:i_test].max(0)
    w = np.clip((raw - lo) / (hi - lo), 0, 1)
    tr = slice(ix(1971), ix(2000))
    doy = np.array([min(x.timetuple().tm_yday, 365) for x in dates]) - 1
    out = {}
    for j, v in enumerate(VARS):
        x = w[:, j]
        clim = np.array([x[tr][doy[tr] == k].mean() for k in range(365)])
        an = x - clim[doy]
        a = an[tr]
        out[v] = {"spread_sd": float(x[tr].std()),
                  "seasonal_share": float(1 - a.var() / x[tr].var()),
                  "anom_sd": float(a.std()),
                  "anom_ac1": float(np.corrcoef(a[:-1], a[1:])[0, 1]),
                  "anom_ac30": float(np.corrcoef(a[:-30], a[30:])[0, 1]),
                  "test_shift_sd": float((x[i_test:].mean() - x[tr].mean()) / x[tr].std())}
    return out


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def per_target(city):
    z = np.load(ROOT / f"artifacts/qrc_runs/step5/era5_{city}/confirm3_predictions.npz")
    Y = z["Y_test"]
    q, e = r2(Y, z["pred_qrc_base"][0]), np.mean([r2(Y, p) for p in z["pred_esn_base"]], 0)
    a = r2(Y, z["pred_linear_ar"][0])
    names = [f"{v}_h{h}" for h in H for v in VARS]
    return {n: {"qrc": float(q[i]), "esn": float(e[i]), "ar": float(a[i]), "gap": float(q[i] - e[i])}
            for i, n in enumerate(names)}


def f3_confirm3(city):
    from qrc_design_r import boot
    z = np.load(ROOT / f"artifacts/qrc_runs/step5/era5_{city}/confirm3_predictions.npz")
    return {b: boot(z["Y_test"], z["pred_qrc_base"], z[f"pred_{b}"], 30) for b in ("esn_base", "lstm_tuned", "linear_ar")}


def f3_earlier(city, tag):
    t = json.loads((DATA / f"qrc_{tag}_{city}.json").read_text(encoding="utf-8"))["tests"]
    return {b: t[f"S3 g0.35 - {k}"]["all"] for b, k in (("esn_base", "esn_tuned"), ("lstm_tuned", "lstm_tuned"),
                                                          ("linear_ar", "linear_ar"))}


def main():
    rows = {}
    for c, tag in CITIES.items():
        gp, ci = gap(c, tag)
        rows[c] = {"tag": tag, "gap": gp, "ci95": ci, **stats(c)}
    feats = [(v, k) for v in VARS for k in ("spread_sd", "seasonal_share", "anom_sd", "anom_ac1", "anom_ac30",
                                              "test_shift_sd")]
    g = np.array([rows[c]["gap"] for c in rows])
    corr = {f"{v}.{k}": float(np.corrcoef(g, [rows[c][v][k] for c in rows])[0, 1]) for v, k in feats}
    pt = {c: per_target(c) for c, t in CITIES.items() if t == "confirm3"}
    f3 = {c: (f3_confirm3(c) if t == "confirm3" else f3_earlier(c, t)) for c, t in CITIES.items()}
    pool = {}
    for b in ("esn_base", "lstm_tuned", "linear_ar"):
        v = [f3[c][b] for c in CITIES]
        pool[b] = {"mean": float(np.mean([x["point"] for x in v])), "qrc_better": sum(x["ci95"][0] > 0 for x in v),
                   "baseline_better": sum(x["ci95"][1] < 0 for x in v), "noninferior_0.02": sum(x["ci95"][0] > -0.02 for x in v)}
    out = {"note": "POST-HOC, descriptive (test sets already seen); n = 9 cities, correlations are indicative only",
           "cities": rows, "corr_with_gap": corr, "per_target_confirm3": pt, "f3_per_city": f3, "f3_pooled_9": pool,
           "provenance": provenance(step="Lagos post-hoc diagnosis")}
    (DATA / "qrc_diag_lagos.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{'city':11s} {'gap':>7s}  " + "  ".join(f"{v[:4]}.{k[:8]:>8s}" for v, k in feats))
    for c in sorted(rows, key=lambda c: rows[c]["gap"]):
        r = rows[c]
        print(f"{c:11s} {r['gap']:+.3f}  " + "  ".join(f"{r[v][k]:13.3f}" for v, k in feats))
    print("corr  " + "  ".join(f"{corr[f'{v}.{k}']:+13.2f}" for v, k in feats))
    for b, r in pool.items():
        print(f"F3 9 cities vs {b}: mean {r['mean']:+.4f}  QRC better {r['qrc_better']}/9  baseline better "
              f"{r['baseline_better']}/9  non-inferior(0.02) {r['noninferior_0.02']}/9")
    for c in ("lagos", "mexicocity", "stockholm"):
        print(c, {b: f"{x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for b, x in f3[c].items()})
    for c, d in pt.items():
        print(c, " ".join(f"{n}:{x['gap']:+.3f}" for n, x in d.items()))


if __name__ == "__main__":
    main()
