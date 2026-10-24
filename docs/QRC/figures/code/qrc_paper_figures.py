"""Publication figures for the reduced-FID learnable-QRC manuscript.

Generates (docs/QRC/figures/):
  figP1_spectrum.png       — physics-informed reduced-FID spectrum + analytic lines
  figP2_quantum_mediated.png — arcsin vs QRC-learned per task, with tau->0 ablation
  figP3_task_dependent.png  — QRC-learned vs fairly tuned LSTM / ESN / size-matched ESN
  figP4_encoding.png        — arcsin vs learned per-spin encoding map (if encoding_detail
                              available from a run; else arcsin-only placeholder)

Every number is read from the committed data package docs/QRC/data/ (6-spin campaign,
MG h=10 seed 1 + confirm, fair classical baselines qrc_classical_fair_{esn,lstm}.json);
nothing is typed in. figP1/figP4 read a run's saved trace / encoding_detail.json.
CPU, no GPU. Lives in docs/QRC/figures/code/; writes PNGs to its parent (figures/).
Run: python docs/QRC/figures/code/qrc_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
# This script lives at docs/QRC/figures/code/ ; resolve the repo + backend absolutely
# so it runs from anywhere (imports the physics stack + the run system builder).
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]                      # code -> figures -> QRC -> docs -> repo
_BACKEND = _REPO / "backend"
sys.path.insert(0, str(_BACKEND))             # for `app.qrc...`
sys.path.insert(0, str(_BACKEND / "scripts"))  # for `qrc_learnable_9spin`
from app.qrc.spectral_lines import physics_informed_lines  # noqa: E402
from qrc_learnable_9spin import build_system  # noqa: E402

FIGDIR = _HERE.parents[1]                      # docs/QRC/figures (write PNGs here)
RUNS = _BACKEND / "artifacts" / "qrc_runs"
# Pinned runs (archived with the raw traces): 6-spin MG h=10 seed 1 and seed 2.
SPECTRUM_TRACE = RUNS / "learnable-0a45c7b0" / "learned_encoding_fid.npz"
ENCODING_DETAIL = RUNS / "mackey-h10-confirm-2d21b56f" / "trace_seed2" / "encoding_detail.json"

# ---- numbers: read from the committed data package (docs/QRC/data), never typed in ----
DATA = _HERE.parents[2] / "data"
TASKS = {"NARMA-2": "narma2", "NARMA-10": "narma10", "MG h=10": "mackey_glass_h10"}


def _j(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def numbers():
    """Test NMSE per task: arcsin, QRC-learned (mean, s.d., per-seed), tau->0 ablation,
    fairly tuned LSTM / ESN / size-matched ESN (data seeds 1-3, qrc_classical_fair.py)."""
    jobs = _j("qrc_6spin_campaign.json")["jobs"]
    s1 = _j("qrc_6spin_mackey_h10_seed1.json")["conditions"]
    conf = _j("qrc_6spin_mackey_h10_confirm.json")["seeds"]
    arc, lrn, abl = {}, {}, {}
    for lab, key in (("NARMA-2", "narma2"), ("NARMA-10", "narma10")):
        arc[lab] = [j["arcsin"] for j in jobs if j["task"] == key and not j["ablation"]]
        lrn[lab] = [j["perspin"] for j in jobs if j["task"] == key and not j["ablation"]]
        abl[lab] = [j["perspin"] for j in jobs if j["task"] == key and j["ablation"]][0]
    arc["MG h=10"] = [s1["arcsin"]["test_nmse"]] + [conf[s]["arcsin"] for s in ("2", "3")]
    lrn["MG h=10"] = [s1["perspin"]["test_nmse"]] + [conf[s]["perspin"] for s in ("2", "3")]
    esn, lstm = _j("qrc_classical_fair_esn.json")["results"], _j("qrc_classical_fair_lstm.json")["results"]
    return {"arcsin": {t: float(np.mean(v)) for t, v in arc.items()},
            "qrc": {t: float(np.mean(v)) for t, v in lrn.items()},
            "qrc_std": {t: float(np.std(v, ddof=1)) for t, v in lrn.items()},
            "qrc_seeds": lrn, "ablation": abl,
            "lstm": {t: lstm[k]["lstm_best"]["mean"] for t, k in TASKS.items()},
            "esn": {t: esn[k]["esn_best"]["mean"] for t, k in TASKS.items()},
            "esn_matched": {t: esn[k]["esn_matched"]["mean"] for t, k in TASKS.items()}}


N = numbers()
ARC, QRC, QRC_STD, ABL = N["arcsin"], N["qrc"], N["qrc_std"], N["ablation"]
LSTM, ESN, ESN_M = N["lstm"], N["esn"], N["esn_matched"]


def fig_spectrum():
    """Physics-informed reduced-FID spectrum with the analytic D_eff lines marked."""
    trace = SPECTRUM_TRACE if SPECTRUM_TRACE.exists() else None
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
    ax.bar(x + w / 2, [QRC[t] for t in tasks], w, yerr=[QRC_STD[t] for t in tasks],
           capsize=3, ecolor="#1b3a5c", label="QRC-learned (regime B)", color="#4da3ff")
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
    """QRC-learned vs fairly tuned LSTM, tuned ESN and size-matched ESN (test NMSE, log)."""
    tasks = list(QRC)
    x = np.arange(len(tasks)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - 1.5 * w, [QRC[t] for t in tasks], w, yerr=[QRC_STD[t] for t in tasks],
           capsize=3, ecolor="#1b3a5c", label="QRC-learned (6 spins)", color="#4da3ff")
    ax.bar(x - 0.5 * w, [LSTM[t] for t in tasks], w, label="LSTM (tuned)", color="#cb4b4b")
    ax.bar(x + 0.5 * w, [ESN_M[t] for t in tasks], w, label="ESN (tuned, N<=350)", color="#f0a868")
    ax.bar(x + 1.5 * w, [ESN[t] for t in tasks], w, label="ESN (tuned, N<=1000)", color="#e07b3c")
    ax.set_yscale("log"); ax.set_ylabel("test NMSE (log)"); ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_title("Synthetic benchmarks: the learned QRC beats the tuned LSTM on NARMA-10 and MG h=10;\n"
                 "a fully tuned ESN is 1-3 orders of magnitude better on every task", fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.2, axis="y")
    _save(fig, "figP3_task_dependent.png")


def fig_encoding():
    """arcsin vs learned per-spin encoding map + validation-loss curve (real run)."""
    detail = ENCODING_DETAIL if ENCODING_DETAIL.exists() else None
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    sg = np.linspace(0, 0.5, 21)
    axL.plot(sg, np.arcsin(np.sqrt(np.clip(sg, 0, 1))), color="#3fb950", lw=2.4, label="arcsin(sqrt(s))")
    if detail is not None:
        d = json.loads(Path(detail).read_text(encoding="utf-8"))
        if d.get("perspin_angle_map"):
            sgd = np.asarray(d["sgrid"]); amap = np.asarray(d["perspin_angle_map"])
            for q in range(amap.shape[1]):
                axL.plot(sgd, amap[:, q], lw=1.1, alpha=0.85)
            axL.plot([], [], color="#4da3ff", lw=1.1, label="learned per-spin (each spin)")
            axL.set_title(f"Learned per-spin encoding vs arcsin\n({d.get('task')}, seed {d.get('seed')})",
                          fontsize=9.5)
        if d.get("perspin_val_curve"):
            axR.plot(d["perspin_val_curve"], color="#4da3ff", lw=1.6)
            axR.set_yscale("log"); axR.set_xlabel("Adam step"); axR.set_ylabel("validation NMSE (log)")
            axR.set_title("Encoder training curve", fontsize=9.5); axR.grid(alpha=0.2)
    axL.set_xlabel("input s"); axL.set_ylabel("pulse angle theta"); axL.legend(fontsize=8); axL.grid(alpha=0.2)
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
