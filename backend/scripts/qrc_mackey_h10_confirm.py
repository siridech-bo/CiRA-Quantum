"""Multi-seed confirmation of the Mackey-Glass h=10 QRC-learned win.

Seed 1 already gave learned-perspin 0.0013 (vs tuned-LSTM 0.0024, ESN 0.0057). This
runs the remaining seeds in ONE job (6-spin, coupling x2, reduced-FID readout,
regime B, leakage-free 3-way), aggregates all seeds (mean +/- s.d., win-count vs
arcsin and vs the classical baselines), and writes a summary + per-seed traces /
encoding detail. Wall-clock-budgeted; crash-safe.

Launchable (task 'mackey-h10-confirm') or:
  python backend/scripts/qrc_mackey_h10_confirm.py --run-dir <dir> --out <summary.json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qrc_learnable_9spin import build_system, make_fid_reduced_post, train  # noqa: E402
from qrc_progress import ProgressLogger  # noqa: E402

CDT = torch.complex64
# classical baselines at mackey h=10 (run classical-baseline-548dbea6)
LSTM_H10, ESN_H10 = 0.0024, 0.0057
SEED1_PERSPIN = 0.0013           # from run learnable-0a45c7b0 (seed 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--n-virtual", type=int, default=137)
    ap.add_argument("--coupling-scale", type=float, default=2.0)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--washout", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--budget-hours", type=float, default=10.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("cuda required")
    device = "cuda"
    run_dir = args.run_dir or str(Path(__file__).resolve().parents[1] / "artifacts" / "qrc_runs" / "mackeyh10_local")
    log = ProgressLogger(run_dir=run_dir, title="Mackey-Glass h=10 multi-seed confirmation")
    log.event("config", f"seeds={args.seeds} 6-spin x{args.coupling_scale} reduced-FID "
              f"h={args.horizon} T={args.T} steps={args.steps} budget={args.budget_hours}h")

    summary = {"experiment": "mackey_h10_multiseed", "regime": "B", "readout": "fid_reduced",
               "horizon": args.horizon, "classical": {"lstm": LSTM_H10, "esn": ESN_H10},
               "seeds": {"1": {"perspin": SEED1_PERSPIN, "source": "learnable-0a45c7b0"}}}
    t0 = time.time()
    for i, seed in enumerate(args.seeds):
        if (time.time() - t0) / 3600 > args.budget_hours:
            log.event("budget", f"stop before seed {seed} (budget {args.budget_hours}h)")
            break
        log.status(phase=f"[{i+1}/{len(args.seeds)}] seed {seed}", step=i, total=len(args.seeds))
        log.event("seed", f"seed {seed} start")
        try:
            sysm = build_system("6", args.n_virtual, coupling_scale=args.coupling_scale)
            g = sysm.ensure_diff(device=device, cdtype=CDT)
            post, n_feat, ls = make_fid_reduced_post(sysm, g, device)
            trace_dir = str(Path(run_dir) / f"trace_seed{seed}")
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            results, base, verdict = train(
                sysm, g, device, CDT, task="mackey_glass", T=args.T, washout=args.washout,
                steps=args.steps, lr=args.lr, seed=seed, conditions=("arcsin", "perspin"),
                window=1, no_memory=False, post=post,
                readout_label=f"fid_reduced (D_eff={ls.d_eff})", n_feat_override=n_feat,
                log=log, trace_dir=trace_dir, horizon=args.horizon)
            per = results.get("perspin", (None, None, None, None))
            summary["seeds"][str(seed)] = {
                "arcsin": float(base),
                "perspin": (float(per[0]) if per[0] is not None else None),
                "enc_std": (float(per[2]) if per[2] is not None else None),
                "verdict": verdict, "D_eff": ls.d_eff}
            log.result(f"seed{seed}", {"arcsin": round(base, 4),
                                       "perspin": round(float(per[0]), 5) if per[0] is not None else None})
        except Exception as e:  # noqa: BLE001
            log.event("error", f"seed {seed}: {type(e).__name__}: {e}")
            summary["seeds"][str(seed)] = {"error": f"{type(e).__name__}: {e}"}
        _aggregate(summary)
        _write(summary, args, run_dir, t0)

    _aggregate(summary)
    _write(summary, args, run_dir, t0)
    log.event("done", f"confirmation finished in {(time.time()-t0)/3600:.2f}h")
    log.close("done")


def _aggregate(summary):
    per = [v["perspin"] for v in summary["seeds"].values()
           if isinstance(v, dict) and v.get("perspin") is not None]
    arc = [v["arcsin"] for v in summary["seeds"].values()
           if isinstance(v, dict) and v.get("arcsin") is not None]
    if not per:
        return
    per = np.array(per)
    beats_lstm = int(np.sum(per < LSTM_H10)); beats_esn = int(np.sum(per < ESN_H10))
    summary["aggregate"] = {
        "n_seeds": len(per),
        "perspin_mean": float(per.mean()), "perspin_std": float(per.std()),
        "arcsin_mean": (float(np.mean(arc)) if arc else None),
        "beats_tuned_lstm": f"{beats_lstm}/{len(per)}",
        "beats_esn": f"{beats_esn}/{len(per)}",
        "verdict": ("CONFIRMED: QRC-learned < both classical baselines on all seeds"
                    if beats_lstm == len(per) and beats_esn == len(per)
                    else "MIXED: check per-seed"),
    }


def _write(summary, args, run_dir, t0):
    summary["elapsed_hours"] = round((time.time() - t0) / 3600, 3)
    out = args.out or str(Path(run_dir) / "results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
