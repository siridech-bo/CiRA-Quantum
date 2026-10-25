"""Real-world figures for the manuscript (figR1-figR4). CPU only.

Two stages, so that every plotted number lives in the data package:
  prep   collect / compute the plotted numbers into docs/QRC/data/qrc_realworld_figdata.json
         (confirmation CIs and per-target R2 from the qrc_confirm*_*.json files and, for Confirmation 3,
         the saved test predictions; the tuned ESN's VALIDATION memory function on ERA5 Delhi, computed
         with the same function as the QRC's, qrc_design_c1.memory_function)
  plot   draw figR1-figR4 from that JSON only

  python docs/QRC/figures/code/qrc_realworld_figures.py [--prep]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
BACK = REPO / "backend"
DATA = REPO / "docs/QRC/data"
FIG = REPO / "docs/QRC/figures"
OUT = DATA / "qrc_realworld_figdata.json"
sys.path.insert(0, str(BACK))
sys.path.insert(0, str(BACK / "scripts"))

CITIES = [("chennai", "confirm1", "Chennai"), ("cairo", "confirm1", "Cairo"), ("moscow", "confirm1", "Moscow"),
          ("sydney", "confirm2", "Sydney"), ("saopaulo", "confirm2", "São Paulo"), ("toronto", "confirm2", "Toronto"),
          ("lagos", "confirm3", "Lagos"), ("mexicocity", "confirm3", "Mexico City"), ("stockholm", "confirm3", "Stockholm")]
H = (1, 5, 10, 15, 20, 30, 45)
# validated categorical slots (dataviz reference palette, light): fixed order, never cycled
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def prep():
    from qrc_design_c1 import memory_function
    from qrc_realworld_bench import esn_states, load_dataset, select_esn

    fd = {"forest": {}, "horizon": {}, "memory": {}, "drift": {}}
    diag = _j(DATA / "qrc_diag_lagos.json")
    names = [f"{v}_h{h}" for h in H for v in ("temp", "humidity")]
    for c, tag, _lab in CITIES:
        if tag == "confirm3":
            f3 = diag["f3_per_city"][c]
            fd["forest"][c] = {"g1": None, "esn": f3["esn_base"], "lstm": f3["lstm_tuned"], "ar": f3["linear_ar"]}
            z = np.load(BACK / f"artifacts/qrc_runs/step5/era5_{c}/confirm3_predictions.npz")
            Y = z["Y_test"]
            per = {"qrc": _r2(Y, z["pred_qrc_base"][0]), "esn": np.mean([_r2(Y, p) for p in z["pred_esn_base"]], 0),
                   "lstm": np.mean([_r2(Y, p) for p in z["pred_lstm_tuned"]], 0), "ar": _r2(Y, z["pred_linear_ar"][0])}
            fd["horizon"][c] = {k: dict(zip(names, v.tolist())) for k, v in per.items()}
        else:
            d = _j(DATA / f"qrc_{tag}_{c}.json")
            t = d["tests"]
            fd["forest"][c] = {"g1": t["S1_S2 g0.35 - g1"]["all"], "esn": t["S3 g0.35 - esn_tuned"]["all"],
                               "lstm": t["S3 g0.35 - lstm_tuned"]["all"], "ar": t["S3 g0.35 - linear_ar"]["all"]}
            g = d["groups"]
            fd["horizon"][c] = {k: g[src]["per_target"] for k, src in
                                (("qrc", "qrc_g035"), ("esn", "esn_tuned"), ("lstm", "lstm_tuned"), ("ar", "linear_ar"))}
    c1 = _j(DATA / "qrc_design_c1_era5_delhi.json")["gains"]
    fd["memory"] = {f"qrc_g{g}": c1[g]["memory"] for g in ("0.1", "0.35", "1")}
    ds = load_dataset("era5_delhi")
    cfg, _ = select_esn(BACK / "artifacts/qrc_runs/realworld/era5_delhi/esn_grid.npz")
    fd["memory"]["esn_tuned"] = {str(k): v for k, v in memory_function(
        esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], 0), ds).items()}
    fd["memory_esn_cfg"] = cfg
    for s in ("exchange", "electricity"):
        d = _j(DATA / f"qrc_drift_{s}.json")
        fd["drift"][s] = {"groups": {k: v["pooled"] for k, v in d["groups"].items()},
                          "shift_sd": d["test_input_shift_sd"][0]}
    fd["note"] = ("plotted numbers for figR1-figR4; memory = VALIDATION R2 of reconstructing the input L days back "
                  "(ERA5 Delhi, QRC 653-bin FID readout D=653; tuned ESN init 0); Confirmation 3 per-target R2 "
                  "recomputed from saved predictions")
    OUT.write_text(json.dumps(fd, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT}")


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_forest(fd):
    """figR1: paired differences with 95 % CIs, nine untouched cities, four comparisons."""
    panels = [("g1", "final − standard (g = 1)"), ("esn", "final − tuned ESN"), ("lstm", "final − tuned LSTM"),
              ("ar", "final − tuned linear AR")]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.9), sharey=True, facecolor=SURF)
    ypos = {c: i + (0 if i < 3 else 0.5 if i < 6 else 1.0) for i, (c, _t, _l) in enumerate(CITIES)}
    for ax, (k, title) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        if k != "g1":
            ax.axvline(-0.02, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for c, _t, _l in CITIES:
            r = fd["forest"][c][k]
            if r is None:
                ax.text(0.5, ypos[c], "not run", fontsize=7, color=INK2, va="center", ha="center",
                        transform=ax.get_yaxis_transform())
                continue
            lo, hi = r["ci95"]
            excl = lo > 0 or hi < 0
            ax.plot([lo, hi], [ypos[c]] * 2, color=BLUE, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(r["point"], ypos[c], "o", ms=6, zorder=3, mec=BLUE, mew=1.5,
                    mfc=BLUE if excl else SURF)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("Δ pooled test $R^2$", fontsize=8, color=INK2)
    axes[0].set_yticks([ypos[c] for c, _t, _l in CITIES], [lab for _c, _t, lab in CITIES])
    axes[0].invert_yaxis()
    for i, txt in ((1, "Conf. 1"), (4.5, "Conf. 2"), (8, "Conf. 3")):
        axes[-1].text(1.02, i, txt, transform=axes[-1].get_yaxis_transform(), fontsize=7, color=INK2, va="center")
    axes[1].text(-0.0215, ypos["toronto"] + 0.75, "margin −0.02", fontsize=7, color=INK2, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(FIG / "figR1_confirm_forest.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_memory(fd):
    """figR2: memory function (VALIDATION R2 of the input L days back), ERA5 Delhi."""
    m = fd["memory"]
    lags = sorted(int(L) for L in m["qrc_g0.35"])
    series = [("qrc_g0.35", "QRC, final g = 0.35", BLUE, "o"), ("esn_tuned", "tuned ESN", ORANGE, "s"),
              ("qrc_g1", "QRC, standard g = 1", AQUA, "^"), ("qrc_g0.1", "QRC, g = 0.1", YELLOW, "D")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURF)
    _style(ax)
    for k, lab, col, mk in series:
        y = [m[k][str(L)] for L in lags]
        ax.plot(lags, y, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1, zorder=3)
        dy = {"qrc_g0.35": 7, "qrc_g0.1": -7}.get(k, 0)
        ax.annotate(lab, (lags[-1], y[-1]), xytext=(6, dy), textcoords="offset points", fontsize=8, color=INK,
                    va="center")
    ax.set_xscale("log")
    ax.set_xticks(lags, [str(L) for L in lags])
    ax.set_xlim(0.9, 300)
    ax.set_xlabel("lag L (days)", fontsize=8, color=INK2)
    ax.set_ylabel("memory: validation $R^2$ of $s(t-L)$", fontsize=8, color=INK2)
    ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "figR2_memory_gain.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_horizon(fd):
    """figR3: (a) mean test R2 vs horizon over 9 cities; (b) QRC - ESN per city vs horizon, Lagos marked."""
    hz = fd["horizon"]
    names = lambda h: (f"temp_h{h}", f"humidity_h{h}")  # noqa: E731
    mean_h = lambda c, k, h: np.mean([hz[c][k][n] for n in names(h)])  # noqa: E731
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9), facecolor=SURF)
    _style(a)
    _style(b)
    for k, lab, col, mk in (("qrc", "QRC final", BLUE, "o"), ("esn", "tuned ESN", ORANGE, "s"),
                            ("lstm", "tuned LSTM", AQUA, "^"), ("ar", "tuned linear AR", YELLOW, "D")):
        y = [np.mean([mean_h(c, k, h) for c, _t, _l in CITIES]) for h in H]
        a.plot(H, y, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1, label=lab, zorder=3)
    a.legend(frameon=False, fontsize=8, labelcolor=INK)
    a.set_xlabel("horizon h (days)", fontsize=8, color=INK2)
    a.set_ylabel("test $R^2$ (mean of T, RH; 9 cities)", fontsize=8, color=INK2)
    a.set_title("(a) all models, nine untouched cities", fontsize=9, color=INK, loc="left")
    b.axhline(0, color=INK, lw=0.8)
    for c, _t, lab in CITIES:
        y = [mean_h(c, "qrc", h) - mean_h(c, "esn", h) for h in H]
        hl = c == "lagos"
        b.plot(H, y, color=BLUE if hl else "#b9b8b2", lw=2 if hl else 1.2, zorder=3 if hl else 2,
               marker="o" if hl else None, ms=5, mec=SURF)
        if hl or c in ("saopaulo", "chennai"):
            b.annotate(lab, (H[-1], y[-1]), xytext=(5, 0), textcoords="offset points", fontsize=8,
                       color=INK if hl else INK2, va="center")
    b.set_xlim(0, 52)
    b.set_xlabel("horizon h (days)", fontsize=8, color=INK2)
    b.set_ylabel("QRC final − tuned ESN, test $R^2$", fontsize=8, color=INK2)
    b.set_title("(b) gap to the tuned ESN per city", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "figR3_horizon.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_drift(fd):
    """figR4: drift test, pooled test R2 per model (dot plot), one panel per series (own scale)."""
    rows = [("qrc_base", "QRC, plain readout"), ("qrc_delta_lags", "QRC, level handling"),
            ("esn_base", "tuned ESN"), ("esn_delta_lags", "tuned ESN, level handling"),
            ("lstm_tuned", "tuned LSTM"), ("linear_ar", "tuned linear AR"), ("persistence", "persistence")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True, facecolor=SURF)
    for ax, (s, title, lim) in zip(axes, (("exchange", "exchange rate", (0.90, 0.99)),
                                          ("electricity", "electricity", (0.70, 0.85)))):
        _style(ax)
        ax.grid(axis="y", visible=False)
        g = fd["drift"][s]["groups"]
        for i, (k, _lab) in enumerate(rows):
            q = k.startswith("qrc")
            x = g[k]
            shown = min(max(x, lim[0]), lim[1])
            ax.plot([lim[0], shown], [i, i], color=GRID, lw=1, zorder=1)
            ax.plot(shown, i, "o", ms=7, color=BLUE if q else INK2, mec=SURF, mew=1, zorder=3)
            ax.annotate(f"{x:.3f}" + ("  (off scale)" if x < lim[0] else ""), (shown, i), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, color=INK, va="center")
        ax.set_xlim(lim[0], lim[1] + (lim[1] - lim[0]) * 0.18)
        ax.set_title(f"{title} (test shift {fd['drift'][s]['shift_sd']:+.2f} s.d.)", fontsize=9, color=INK)
        ax.set_xlabel("pooled test $R^2$", fontsize=8, color=INK2)
    axes[0].set_yticks(range(len(rows)), [lab for _k, lab in rows])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "figR4_drift.png", dpi=200, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    a = ap.parse_args()
    if a.prep or not OUT.exists():
        prep()
    fd = _j(OUT)
    fig_forest(fd)
    fig_memory(fd)
    fig_horizon(fd)
    fig_drift(fd)
    print("wrote figR1-figR4")


if __name__ == "__main__":
    main()
