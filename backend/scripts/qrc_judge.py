"""Multi-metric encoding judging from saved waveforms (offline — NO GPU).

Replaces the single fidelity-hungry weather-R² with a panel of robust, intrinsic
signals computed *offline* from the saved random-input reservoir traces
(``memcap_*.npz``) — the payoff of persisting the waveform:

  * **linear MC**       — short-term memory
  * **nonlinear IPC**   — degree-2 + degree-3 Legendre capacities (nonlinearity)
  * **NARMA-10 NMSE**   — a real nonlinear task (computed on the SAME random input
                          that drove the reservoir); lower is better
  * **effective dim**   — participation ratio of the reservoir states (richness)

Each is computed under a **fixed** readout (magnitude653) for the clean encoding
comparison, and re-computed under ``multimodal`` as a readout-robustness check.

Run (no GPU)::

    python scripts/qrc_judge.py --glob "artifacts/traces/memcap_*.npz" \
        --out-dir artifacts/qrc_judge
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from app.qrc.feature_methods import build_features
from qrc_memcap import memory_capacity

ALPHAS = np.logspace(-4, 4, 20)


def effective_dim(X: np.ndarray) -> float:
    """Participation ratio of the state covariance eigenvalues — an estimate of
    how many reservoir dimensions are actually used (richness)."""
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    ev = np.linalg.eigvalsh(Xc.T @ Xc / max(1, len(Xc)))
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / np.sum(ev ** 2)) if ev.size else 0.0


def narma_target(u: np.ndarray, order: int = 10) -> np.ndarray:
    """NARMA-``order`` target driven by ``u`` (the same input the reservoir saw)."""
    n = len(u)
    y = np.zeros(n)
    for t in range(order - 1, n - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - order + 1:t + 1])
                    + 1.5 * u[t - order + 1] * u[t] + 0.1)
    return y


def narma_nmse(X, u, tr, te, order=10) -> float:
    """Test NMSE of a ridge readout predicting NARMA-``order`` (lower=better).

    NARMA-10 is only stable for input in ~[0, 0.5]; the reservoir was driven by
    ``u ∈ [0,1]``, so we build the target on ``0.5·u`` — a deterministic function
    of the same input the reservoir remembers, so it remains predictable."""
    y = narma_target(0.5 * np.asarray(u, float), order)
    if not np.all(np.isfinite(y)):
        return float("nan")
    sc = StandardScaler().fit(X[tr])
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
    pred = m.predict(sc.transform(X[te]))
    err = float(np.mean((y[te] - pred) ** 2))
    var = float(np.var(y[te]))
    return err / var if var > 1e-12 else float("nan")


def judge_one(trace_path: str, readout: str) -> dict:
    with np.load(trace_path, allow_pickle=True) as z:
        fids = np.asarray(z["fids"])
        u = np.asarray(z["memcap_input"], float)
        washout, n_train, n_test = (int(x) for x in np.asarray(z["split"]).ravel()[:3])
        fn = json.loads(str(z["meta"])).get("fn", Path(trace_path).stem)
    X, _ = build_features(fids, readout, n_peaks=653, select="first")
    tr = np.arange(washout, washout + n_train)
    te = np.arange(washout + n_train, washout + n_train + n_test)

    mc = memory_capacity(X, 2.0 * u - 1.0, kmax=30, washout=washout,
                         n_train=n_train, degrees=(1, 2, 3))
    caps = mc["caps_by_degree"]
    ipc2 = float(np.sum(caps.get("2", [])))
    ipc3 = float(np.sum(caps.get("3", [])))
    return {
        "fn": fn, "readout": readout,
        "linear_MC": mc["linear_MC"], "ipc2": ipc2, "ipc3": ipc3,
        "nonlinear_IPC": ipc2 + ipc3, "total_capacity": mc["linear_MC"] + ipc2 + ipc3,
        "narma10_nmse": narma_nmse(X, u, tr, te, 10),
        "effective_dim": effective_dim(X[tr]),
        "mem_by_delay": [float(c) for c in caps.get("1", [])],
    }


def _figure(rows, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    prim = [r for r in rows if r["readout"] == "magnitude653"]
    prim.sort(key=lambda r: -r["total_capacity"])
    fns = [r["fn"] for r in prim]
    x = np.arange(len(fns))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(x, [r["linear_MC"] for r in prim], color="#2563eb", label="linear MC")
    axes[0].bar(x, [r["nonlinear_IPC"] for r in prim],
                bottom=[r["linear_MC"] for r in prim], color="#16a34a", label="nonlinear IPC")
    axes[0].set_title("(A) Capacity (memory + nonlinearity)"); axes[0].legend(fontsize=8)
    axes[1].bar(x, [r["narma10_nmse"] for r in prim], color="#e07b3c")
    axes[1].set_title("(B) NARMA-10 NMSE (lower better)")
    axes[2].bar(x, [r["effective_dim"] for r in prim], color="#a78bfa")
    axes[2].set_title("(C) Effective dimensionality")
    for a in axes:
        a.set_xticks(x); a.set_xticklabels(fns, rotation=30, ha="right", fontsize=8)
        a.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Phase-2: encoding judged on a multi-metric panel (magnitude653 readout)", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[judge] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="artifacts/traces/memcap_*.npz")
    ap.add_argument("--out-dir", default="artifacts/qrc_judge")
    ap.add_argument("--readouts", nargs="+", default=["magnitude653", "multimodal"])
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    traces = sorted(glob.glob(args.glob))
    print(f"judging {len(traces)} saved traces × {len(args.readouts)} readouts (offline)", flush=True)
    rows = []
    for tp in traces:
        for ro in args.readouts:
            r = judge_one(tp, ro)
            rows.append(r)
            print(f"[judge] {r['fn']:<14} [{ro:<11}] linMC={r['linear_MC']:.2f} "
                  f"nlIPC={r['nonlinear_IPC']:.2f} narmaNMSE={r['narma10_nmse']:.3f} "
                  f"effDim={r['effective_dim']:.1f}", flush=True)

    def rank(readout, key, reverse=True):
        sub = [r for r in rows if r["readout"] == readout and np.isfinite(r[key])]
        return [r["fn"] for r in sorted(sub, key=lambda r: r[key], reverse=reverse)]

    summary = {
        "generated_utc": datetime.now(UTC).isoformat(), "readouts": args.readouts,
        "rankings_magnitude653": {
            "total_capacity": rank("magnitude653", "total_capacity"),
            "linear_MC": rank("magnitude653", "linear_MC"),
            "nonlinear_IPC": rank("magnitude653", "nonlinear_IPC"),
            "narma10_nmse": rank("magnitude653", "narma10_nmse", reverse=False),
            "effective_dim": rank("magnitude653", "effective_dim"),
        },
        "ranking_multimodal_total": rank("multimodal", "total_capacity"),
        "rows": rows,
    }
    (out / "judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== rankings (magnitude653) ===", flush=True)
    for k, v in summary["rankings_magnitude653"].items():
        print(f"  {k:<16}: {v}", flush=True)
    print(f"  multimodal total : {summary['ranking_multimodal_total']}  (readout-robustness check)", flush=True)
    _figure(rows, out / "judge_panel.png")
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
