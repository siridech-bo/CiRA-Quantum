"""Publication figures for the reduced-FID learnable-QRC manuscript.

Generates (docs/QRC/figures/):
  figP1_spectrum.png       — physics-informed reduced-FID spectrum + analytic lines
  figP2_quantum_mediated.png — arcsin vs QRC-learned per task, with tau->0 ablation
  figP3_task_dependent.png  — QRC-learned vs tuned-LSTM vs ESN (bars) + learning curves
  figP4_encoding.png        — arcsin vs learned per-spin encoding map (if encoding_detail
                              available from a run; else arcsin-only placeholder)

Numbers are the confirmed results (campaign learnable-campaign-01ce463b, classical
classical-baseline-548dbea6, mackey-h10 learnable-0a45c7b0). CPU, no GPU.
Run: python backend/scripts/qrc_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.qrc.spectral_lines import physics_informed_lines  # noqa: E402
from qrc_learnable_9spin import build_system  # noqa: E402

FIGDIR = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures"
RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "qrc_runs"

# ---- confirmed numbers (test NMSE) ----------------------------------------
ARC = {"NARMA-2": 0.101, "NARMA-10": 0.312, "MG h=10": 0.0235}
QRC = {"NARMA-2": 0.0052, "NARMA-10": 0.120, "MG h=10": 0.0013}
ABL = {"NARMA-2": 0.9994, "NARMA-10": 0.9974}
LSTM = {"NARMA-2": 0.0114, "NARMA-10": 0.234, "MG h=10": 0.0024}
ESN = {"NARMA-2": 0.0046, "NARMA-10": 0.029, "MG h=10": 0.0057}


def _latest(glob_pat, fname):
    hits = sorted(RUNS.glob(f"{glob_pat}/**/{fname}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        hits = sorted(RUNS.glob(f"{glob_pat}/{fname}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def fig_spectrum():
    """Physics-informed reduced-FID spectrum with the analytic D_eff lines marked."""
    trace = _latest("learnable-*", "learned_encoding_fid.npz")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sysm = build_system("6", 137, coupling_scale=2.0)               # exact run system
    ls = physics_informed_lines(sysm.system)
    if trace is not None:
        with np.load(str(trace), allow_pickle=False) as z:
            fids = np.asarray(z["fids"]); dwell = float(z["fid_dwell"])
        fid = fids[fids.shape[0] // 2]                                 # a representative step
        spec = np.abs(np.fft.fftshift(np.fft.fft(fid)))
        freq = np.fft.fftshift(np.fft.fftfreq(fid.size, d=dwell))
        ax.plot(freq, spec / spec.max(), color="#3fb950", lw=1.3, label="reduced FID |spectrum|")
        ax.set_xlim(-1.2 * ls.f_max_hz, 1.2 * ls.f_max_hz)
    for f in ls.freqs_hz:
        ax.axvline(f, color="#cb4b4b", lw=0.4, alpha=0.5)
    ax.axvline(ls.freqs_hz[0], color="#cb4b4b", lw=0.4, alpha=0.5,
               label=f"analytic lines (D_eff={ls.d_eff})")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("normalized |spectrum|")
    ax.set_title(f"Physics-informed reduced-FID readout — 6 spins, coupling x2 "
                 f"(D_eff={ls.d_eff} lines -> {2*ls.d_eff} features)", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    _save(fig, "figP1_spectrum.png")


def fig_quantum_mediated():
    """arcsin vs QRC-learned per task, with the tau->0 ablation collapse annotated."""
    tasks = list(QRC)
    x = np.arange(len(tasks)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - w / 2, [ARC[t] for t in tasks], w, label="arcsin (regime A)", color="#9aa7b4")
    ax.bar(x + w / 2, [QRC[t] for t in tasks], w, label="QRC-learned (regime B)", color="#4da3ff")
    for i, t in enumerate(tasks):
        ax.text(i + w / 2, QRC[t], f" {QRC[t]:.4g}", ha="center", va="bottom", fontsize=7, rotation=90)
        if t in ABL:
            ax.annotate(f"tau->0: {ABL[t]:.2f}\n(mean-predictor)", (i, 1.0),
                        fontsize=6.5, ha="center", va="top", color="#cb4b4b")
    ax.axhline(1.0, color="#cb4b4b", ls=":", lw=1, alpha=0.7)
    ax.set_yscale("log"); ax.set_ylabel("test NMSE (log)"); ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_title("Learned encoding beats arcsin (multi-seed); tau->0 ablation collapses it\n"
                 "-> the gain is carried by the quantum reservoir", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
    _save(fig, "figP2_quantum_mediated.png")


def fig_task_dependent():
    """QRC-learned vs tuned-LSTM vs ESN (bars) + NMSE-vs-n_train learning curves."""
    tasks = list(QRC)
    x = np.arange(len(tasks)); w = 0.26
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axL.bar(x - w, [QRC[t] for t in tasks], w, label="QRC-learned", color="#4da3ff")
    axL.bar(x, [LSTM[t] for t in tasks], w, label="LSTM (tuned)", color="#cb4b4b")
    axL.bar(x + w, [ESN[t] for t in tasks], w, label="ESN (N=1000)", color="#e07b3c")
    axL.set_yscale("log"); axL.set_ylabel("test NMSE (log)"); axL.set_xticks(x); axL.set_xticklabels(tasks)
    axL.set_title("Task-dependent: ESN wins NARMA; QRC-learned wins MG h=10", fontsize=9.5)
    axL.legend(fontsize=8); axL.grid(alpha=0.2, axis="y")

    res = _latest("classical-baseline-*", "results.json")
    if res is not None:
        d = json.loads(Path(res).read_text(encoding="utf-8"))
        # Correct QRC points (the classical run's stored qrc_perspin for mackey used
        # the h=1 campaign value; use the confirmed per-task numbers instead).
        qrc_pt = {"narma10": QRC["NARMA-10"], "mackey_glass_h10": QRC["MG h=10"]}
        colors = {"narma10": "#7bd88f", "mackey_glass_h10": "#c98bdb"}
        for key, c in colors.items():
            r = d["tasks"].get(key, {})
            lc = r.get("learning_curve")
            if not lc:
                continue
            axR.plot(lc["n_trains"], lc["lstm"], "-o", color=c, lw=1.4, label=f"{key} LSTM")
            axR.plot(lc["n_trains"], lc["esn"], "--s", color=c, lw=1.2, alpha=0.7, label=f"{key} ESN")
            axR.scatter([735], [qrc_pt[key]], marker="*", s=140, color=c,
                        edgecolor="k", zorder=5, label=f"{key} QRC-learned")
        axR.set_yscale("log"); axR.set_xlabel("n_train"); axR.set_ylabel("test NMSE (log)")
        axR.set_title("Data-efficiency (learning curves)", fontsize=9.5)
        axR.legend(fontsize=6.5); axR.grid(alpha=0.2)
    _save(fig, "figP3_task_dependent.png")


def fig_encoding():
    """arcsin vs learned per-spin encoding map (from a run's encoding_detail.json)."""
    detail = _latest("learnable-*", "encoding_detail.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sg = np.linspace(0, 0.5, 21)
    ax.plot(sg, np.arcsin(np.sqrt(np.clip(sg, 0, 1))), color="#3fb950", lw=2.2, label="arcsin(sqrt(s))")
    if detail is not None:
        d = json.loads(Path(detail).read_text(encoding="utf-8"))
        if d.get("perspin_angle_map"):
            sgd = np.asarray(d["sgrid"]); amap = np.asarray(d["perspin_angle_map"])
            for q in range(amap.shape[1]):
                ax.plot(sgd, amap[:, q], lw=1.1, alpha=0.8)
            ax.plot([], [], color="#4da3ff", lw=1.1, label="learned per-spin (each spin)")
            ax.set_title(f"Learned per-spin encoding vs arcsin ({d.get('task')})", fontsize=9.5)
        else:
            ax.set_title("Encoding map — learned per-spin pending (run with encoding_detail)", fontsize=9)
    else:
        ax.set_title("Encoding map — arcsin (learned map pending multi-seed run)", fontsize=9)
    ax.set_xlabel("input s"); ax.set_ylabel("pulse angle theta"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    _save(fig, "figP4_encoding.png")


def _save(fig, name):
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_spectrum()
    fig_quantum_mediated()
    fig_task_dependent()
    fig_encoding()
