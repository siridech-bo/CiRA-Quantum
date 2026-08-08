"""Unattended learnable-encoding CONFIRMATION campaign (multi-dataset, multi-seed,
tau->0 ablation, vs trained classical LSTM) on the physics-informed reduced-FID
readout. One long job (~24-40 h); wall-clock-budgeted so it always finishes and
writes a summary.

For each (task, seed) it measures, on the SAME leakage-free 3-way split:
  - QRC arcsin baseline        (regime A, fixed encoding)         test NMSE
  - QRC learned per-spin       (regime B, trained encoding)       test NMSE
  - classical LSTM             (trained recurrent net on raw u)   test NMSE
and, for flagged (task, seed), the tau->0 ablation of learned per-spin
(`--no-memory`) which MUST collapse if the win is quantum-mediated.

Datasets: narma2 (encoding-sensitive), narma10 (memory-bound — expect QRC tie/loss,
the honest contrast), mackey_glass (chaotic one-step prediction).

Readout: physics-informed reduced FID (SOP 3.2), D = 2*D_eff features. Every result
is labeled (readout, D, regime). Reservoir: 6-spin, coupling x2.

Run (launcher task 'learnable-campaign', GPU) or standalone:
  python backend/scripts/qrc_learnable_campaign.py --run-dir <dir> --out <summary.json>
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

from qrc_learnable_9spin import (  # noqa: E402
    build_system, make_task, make_fid_reduced_post, train, _three_way,
)
from qrc_progress import ProgressLogger  # noqa: E402

RDT = torch.float64
CDT = torch.complex64          # GPU dtype (memory-lean; gradient still fine for encoding)


# ---- classical LSTM baseline (the strong "vs trained RNN" bar) -------------


class LSTMBaseline(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.rnn = torch.nn.LSTM(1, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, u):                       # u: (1, T, 1)
        h, _ = self.rnn(u)
        return self.head(h).reshape(-1)          # (T,)


def lstm_baseline(u, y, device, washout, hidden=64, epochs=400, lr=5e-3, seed=0):
    """Train an LSTM on raw u -> y, leakage-free 3-way split (train/val/test),
    early-stop on val, report TEST NMSE. The strong classical bar."""
    torch.manual_seed(seed)
    y_use = np.asarray(y)[washout:]
    tr, va, te = _three_way(len(y_use))
    U = torch.tensor(np.asarray(u)[washout:], dtype=torch.float32, device=device).reshape(1, -1, 1)
    Y = torch.tensor(y_use, dtype=torch.float32, device=device)
    net = LSTMBaseline(hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    yvar_tr = float(Y[tr].var()) or 1.0
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(U)
        loss = ((pred[tr] - Y[tr]) ** 2).mean() / yvar_tr
        loss.backward(); opt.step()
        with torch.no_grad():
            vnmse = float(((net(U)[va] - Y[va]) ** 2).mean() / (Y[va].var() or 1.0))
        if vnmse < best_val:
            best_val = vnmse
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
    if best_state:
        net.load_state_dict(best_state)
    with torch.no_grad():
        test = float(((net(U)[te] - Y[te]) ** 2).mean() / (Y[te].var() or 1.0))
    return test


# ---- one QRC condition-set for a (task, seed) -----------------------------


def qrc_job(task, seed, device, args, log, trace_dir, no_memory=False):
    """Run QRC arcsin + learned per-spin (regime B, reduced-FID) for one
    (task, seed). Returns {arcsin, perspin, verdict, D, enc_std}."""
    sysm = build_system("6", args.n_virtual, coupling_scale=args.coupling_scale)
    g = sysm.ensure_diff(device=device, cdtype=CDT)
    post, n_feat, ls = make_fid_reduced_post(sysm, g, device)
    label = f"fid_reduced (D_eff={ls.d_eff})"
    # Per-job trace subdir so traces don't overwrite each other across jobs.
    job_trace_dir = None
    if trace_dir and not no_memory:
        job_trace_dir = str(Path(trace_dir) / f"trace_{task}_s{seed}")
        Path(job_trace_dir).mkdir(parents=True, exist_ok=True)
    results, base, verdict = train(
        sysm, g, device, CDT, task=task, T=args.T, washout=args.washout,
        steps=args.steps, lr=args.lr, seed=seed, conditions=("arcsin", "perspin"),
        window=1, no_memory=no_memory, post=post, readout_label=label,
        n_feat_override=n_feat, log=log, trace_dir=job_trace_dir)
    per = results.get("perspin", (None, None, None, None))
    return {"arcsin": float(base), "perspin": (float(per[0]) if per[0] is not None else None),
            "enc_std": (float(per[2]) if per[2] is not None else None),
            "D_eff": ls.d_eff, "D": n_feat, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-virtual", type=int, default=137)
    ap.add_argument("--coupling-scale", type=float, default=2.0)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--washout", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--budget-hours", type=float, default=44.0,
                    help="stop scheduling new QRC jobs past this wall-clock (safety)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("cuda required for the campaign")
    device = "cuda"

    # Job plan (ordered by value; budget guard trims the tail). Each entry:
    # (task, seed, ablation?). ablation runs perspin with tau->0 (--no-memory).
    plan = [
        ("narma2", 1, False), ("narma2", 2, False), ("narma2", 3, False),
        ("narma2", 1, True),                                   # ablation
        ("narma10", 1, False), ("narma10", 2, False), ("narma10", 3, False),
        ("narma10", 1, True),                                  # ablation
        ("mackey_glass", 1, False), ("mackey_glass", 2, False),
    ]

    run_dir = args.run_dir or str(Path(__file__).resolve().parents[1] / "artifacts" / "qrc_runs" / "campaign_local")
    log = ProgressLogger(run_dir=run_dir, title="QRC learnable confirmation campaign")
    log.event("config", f"plan={len(plan)} jobs | 6-spin coupling={args.coupling_scale}x "
              f"reduced-FID T={args.T} steps={args.steps} budget={args.budget_hours}h")

    summary = {"experiment": "learnable_campaign", "regime": "B (trained encoding)",
               "readout": "fid_reduced", "system": "6-spin", "coupling_scale": args.coupling_scale,
               "T": args.T, "steps": args.steps, "jobs": [], "classical_baseline": "LSTM(64)"}
    t0 = time.time()
    trace_root = Path(run_dir)

    for i, (task, seed, ablation) in enumerate(plan):
        elapsed_h = (time.time() - t0) / 3600.0
        if elapsed_h > args.budget_hours:
            log.event("budget", f"stopping: {elapsed_h:.1f}h > budget {args.budget_hours}h "
                      f"({len(plan) - i} jobs skipped)")
            summary["skipped"] = [{"task": t, "seed": s, "ablation": a} for (t, s, a) in plan[i:]]
            break
        kind = "ablation(tau->0)" if ablation else "train"
        log.status(phase=f"[{i+1}/{len(plan)}] {task} seed{seed} {kind}", step=i, total=len(plan))
        log.event("job", f"[{i+1}/{len(plan)}] {task} seed{seed} {kind}")
        try:
            rec = {"task": task, "seed": seed, "ablation": ablation}
            qr = qrc_job(task, seed, device, args, log, str(trace_root), no_memory=ablation)
            rec.update(qr)
            if not ablation:
                # classical LSTM on the SAME task/seed/split (cheap)
                u, y = make_task(task, args.T, seed)
                rec["lstm"] = lstm_baseline(u, y, device, args.washout, seed=seed)
            summary["jobs"].append(rec)
            log.result(f"{task}-s{seed}{'-abl' if ablation else ''}",
                       {"arcsin": round(qr["arcsin"], 4),
                        "perspin": (round(qr["perspin"], 4) if qr["perspin"] is not None else None),
                        "lstm": (round(rec.get("lstm"), 4) if rec.get("lstm") is not None else None)})
        except Exception as e:  # noqa: BLE001 - one job must not kill the campaign
            log.event("error", f"{task} seed{seed} {kind}: {type(e).__name__}: {e}")
            summary["jobs"].append({"task": task, "seed": seed, "ablation": ablation,
                                    "error": f"{type(e).__name__}: {e}"})
        # persist the summary after EVERY job (crash/interrupt-safe)
        _write_summary(summary, args, run_dir, t0)

    summary["aggregate"] = _aggregate(summary["jobs"])
    _write_summary(summary, args, run_dir, t0)
    log.event("done", f"campaign finished: {len(summary['jobs'])} jobs in "
              f"{(time.time()-t0)/3600:.1f}h")
    log.close("done")


def _aggregate(jobs):
    """Per-task: QRC-learned vs arcsin vs LSTM (mean+/-std, win-count), and the
    ablation collapse check."""
    agg = {}
    tasks = sorted({j["task"] for j in jobs if "error" not in j})
    for task in tasks:
        seeds = [j for j in jobs if j["task"] == task and not j.get("ablation") and "error" not in j]
        abls = [j for j in jobs if j["task"] == task and j.get("ablation") and "error" not in j]
        if not seeds:
            continue
        arc = np.array([j["arcsin"] for j in seeds])
        per = np.array([j["perspin"] for j in seeds if j["perspin"] is not None])
        lstm = np.array([j["lstm"] for j in seeds if j.get("lstm") is not None])
        wins = int(np.sum([1 for j in seeds if j["perspin"] is not None and j["perspin"] < j["arcsin"]]))
        entry = {
            "n_seeds": len(seeds),
            "arcsin_mean": float(arc.mean()), "arcsin_std": float(arc.std()),
            "perspin_mean": float(per.mean()) if per.size else None,
            "perspin_std": float(per.std()) if per.size else None,
            "perspin_beats_arcsin": f"{wins}/{len(seeds)}",
            "lstm_mean": float(lstm.mean()) if lstm.size else None,
            "lstm_std": float(lstm.std()) if lstm.size else None,
        }
        if abls:
            abl = np.array([j["perspin"] for j in abls if j["perspin"] is not None])
            entry["ablation_perspin_mean"] = float(abl.mean()) if abl.size else None
            entry["ablation_collapses"] = bool(
                per.size and abl.size and abl.mean() > 2 * per.mean())
        # honest head-to-head verdict
        if per.size and lstm.size:
            entry["qrc_vs_lstm"] = ("QRC-learned wins" if per.mean() < lstm.mean()
                                    else "LSTM wins")
        agg[task] = entry
    return agg


def _write_summary(summary, args, run_dir, t0):
    summary["elapsed_hours"] = round((time.time() - t0) / 3600.0, 3)
    out = args.out or str(Path(run_dir) / "results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
