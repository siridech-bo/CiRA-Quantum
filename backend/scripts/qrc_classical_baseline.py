"""Publication-grade classical baseline for the reduced-FID learnable-QRC result.

Decoupled from the ~40 h QRC campaign: re-scores the QRC operating points from the
saved campaign summary and builds STRONG classical baselines on the SAME tasks /
leakage-free 3-way split (`_three_way`) / test-NMSE metric, so the QRC-vs-classical
comparison is fair.

Baselines:
  * Tuned LSTM  — hyperparameter sweep (hidden x layers x lr) x multiple init seeds,
                  cosine LR schedule + grad clipping + val-based early stopping.
                  Reports the BEST config (mean +/- s.d. over inits).
  * ESN         — leaky Echo State Network, reservoir-size sweep, reports best.
  * Learning curves — NMSE vs n_train for LSTM(best) and ESN(best) (+ QRC overlaid
                  at its known operating point), separating data-efficiency from
                  asymptotic accuracy (the fairness figure).
  * Mackey-Glass at a LONGER horizon (h=10) so the task is discriminative (one-step
    was trivially easy for every model).

Tasks: narma2, narma10, mackey_glass (h=1 and h=10). Wall-clock-budgeted; writes the
summary after every task (crash-safe). Launchable (task 'classical-baseline') or:
  python backend/scripts/qrc_classical_baseline.py --run-dir <dir> --out <summary.json>
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
from qrc_learnable_9spin import make_task, _mackey_glass, _three_way  # noqa: E402
from qrc_progress import ProgressLogger  # noqa: E402


def _task_series(task, T, seed, horizon=1):
    """(u, y) for a task at a given prediction horizon (horizon only affects MG)."""
    if task == "mackey_glass":
        s = _mackey_glass(T + horizon, seed)
        return s[:T], s[horizon:horizon + T]
    return make_task(task, T, seed)


def _nmse(pred, y):
    y = np.asarray(y); pred = np.asarray(pred)
    v = y.var()
    return float(((pred - y) ** 2).mean() / v) if v > 0 else float("nan")


# ---- tuned LSTM -----------------------------------------------------------


class _LSTM(torch.nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.rnn = torch.nn.LSTM(1, hidden, num_layers=layers, batch_first=True,
                                 dropout=0.0)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, u):
        h, _ = self.rnn(u)
        return self.head(h).reshape(-1)


def _train_lstm(u, y, device, washout, hidden, layers, lr, epochs, init_seed,
                n_tr_override=None):
    """One LSTM fit: cosine LR, grad-clip, val early-stop; returns (val, test) NMSE."""
    torch.manual_seed(init_seed)
    y_use = np.asarray(y)[washout:]
    tr, va, te = _three_way(len(y_use))
    if n_tr_override is not None:                       # learning-curve: shrink train only
        tr = slice(0, min(n_tr_override, tr.stop))
    U = torch.tensor(np.asarray(u)[washout:], dtype=torch.float32, device=device).reshape(1, -1, 1)
    Y = torch.tensor(y_use, dtype=torch.float32, device=device)
    net = _LSTM(hidden, layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    yvar_tr = float(Y[tr].var()) or 1.0
    best_val, best_state, patience, bad = float("inf"), None, 40, 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(U)[tr] - Y[tr]) ** 2).mean() / yvar_tr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            v = _nmse(net(U)[va].cpu().numpy(), Y[va].cpu().numpy())
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, {k: t.detach().clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    with torch.no_grad():
        test = _nmse(net(U)[te].cpu().numpy(), Y[te].cpu().numpy())
    return best_val, test


def tuned_lstm(u, y, device, washout, inits, epochs, log=None, tag=""):
    """HP sweep x inits; select config by mean val NMSE; report test mean+/-s.d."""
    grid = [(h, l, lr) for h in (32, 64, 128, 256) for l in (1, 2)
            for lr in (1e-3, 3e-3, 1e-2)]
    best = None
    for (h, l, lr) in grid:
        vals, tests = [], []
        for s in range(inits):
            vv, tt = _train_lstm(u, y, device, washout, h, l, lr, epochs, init_seed=s)
            vals.append(vv); tests.append(tt)
        mval = float(np.mean(vals))
        if best is None or mval < best["val_mean"]:
            best = {"hidden": h, "layers": l, "lr": lr, "val_mean": mval,
                    "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests)),
                    "n_inits": inits}
        if log:
            log.status(phase=f"LSTM {tag} h{h} l{l} lr{lr}", step=grid.index((h, l, lr)) + 1,
                       total=len(grid))
    return best


# ---- ESN ------------------------------------------------------------------


def _esn_run(u, y, washout, n_res, sr=0.9, leak=1.0, seed=0, alpha=1e-3, n_tr_override=None):
    rng = np.random.default_rng(seed)
    u = np.asarray(u); n = len(u)
    Win = rng.uniform(-0.5, 0.5, (n_res, 1))
    W = rng.uniform(-0.5, 0.5, (n_res, n_res))
    rad = np.max(np.abs(np.linalg.eigvals(W))) if n_res <= 800 else _power_radius(W, rng)
    if rad > 0:
        W *= sr / rad
    x = np.zeros(n_res); X = np.zeros((n, n_res))
    for k in range(n):
        x = (1 - leak) * x + leak * np.tanh(Win[:, 0] * u[k] + W @ x)
        X[k] = x
    Xb = np.hstack([X, np.ones((n, 1))])[washout:]
    yu = np.asarray(y)[washout:]
    tr, va, te = _three_way(len(yu))
    if n_tr_override is not None:
        tr = slice(0, min(n_tr_override, tr.stop))
    A = Xb[tr].T @ Xb[tr] + alpha * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb[tr].T @ yu[tr])
    return _nmse(Xb[te] @ w, yu[te])


def _power_radius(W, rng, iters=100):
    v = rng.standard_normal(W.shape[0]); v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        w = W @ v; lam = np.linalg.norm(w)
        if lam == 0:
            return 0.0
        v = w / lam
    return float(lam)


def esn_best(u, y, washout, sizes=(100, 300, 600, 1000)):
    best = None
    for m in sizes:
        vals = [_esn_run(u, y, washout, m, seed=s) for s in range(3)]
        t = float(np.mean(vals))
        if best is None or t < best["test_mean"]:
            best = {"n_reservoir": m, "test_mean": t, "test_std": float(np.std(vals))}
    return best


# ---- learning curves ------------------------------------------------------


def learning_curve(u, y, device, washout, lstm_cfg, esn_size, n_trains, inits=3, epochs=400):
    """NMSE vs n_train for LSTM(best cfg) and ESN(best size)."""
    out = {"n_trains": list(n_trains), "lstm": [], "esn": []}
    for nt in n_trains:
        lt = [_train_lstm(u, y, device, washout, lstm_cfg["hidden"], lstm_cfg["layers"],
                          lstm_cfg["lr"], epochs, init_seed=s, n_tr_override=nt)[1]
              for s in range(inits)]
        out["lstm"].append(float(np.mean(lt)))
        et = [_esn_run(u, y, washout, esn_size, seed=s, n_tr_override=nt) for s in range(3)]
        out["esn"].append(float(np.mean(et)))
    return out


# ---- QRC operating points (from the saved campaign summary) ---------------


def load_qrc_points(campaign_results_path):
    """Per-task QRC learned-perspin test NMSE (mean over seeds) from the campaign."""
    try:
        d = json.loads(Path(campaign_results_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pts = {}
    for j in d.get("jobs", []):
        if j.get("ablation") or "error" in j or j.get("perspin") is None:
            continue
        pts.setdefault(j["task"], []).append(j["perspin"])
    return {t: {"perspin_mean": float(np.mean(v)), "n_train": 735} for t, v in pts.items()}


# ---- main -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--washout", type=int, default=30)
    ap.add_argument("--inits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument("--campaign-results", default=None,
                    help="path to the QRC campaign results.json for operating points")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = args.run_dir or str(Path(__file__).resolve().parents[1] / "artifacts" / "qrc_runs" / "classical_local")
    log = ProgressLogger(run_dir=run_dir, title="QRC strengthened classical baseline")

    tasks = [("narma2", 1), ("narma10", 1), ("mackey_glass", 1), ("mackey_glass", 10)]
    n_trains = [150, 300, 500, 735, 1000]
    # Auto-discover the newest QRC campaign results.json for the operating points.
    camp = args.campaign_results
    if not camp:
        roots = sorted(Path(run_dir).resolve().parents[0].glob("learnable-campaign-*/results.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        camp = str(roots[0]) if roots else None
    qrc_pts = load_qrc_points(camp) if camp else {}
    summary = {"experiment": "classical_baseline", "T": args.T, "inits": args.inits,
               "epochs": args.epochs, "protocol": "tuned-LSTM(sweep,multi-init,cosine,clip,"
               "early-stop) + ESN(size-sweep); leakage-free 3-way; test NMSE",
               "qrc_operating_points": qrc_pts, "tasks": {}}
    log.event("config", f"device={device} tasks={len(tasks)} inits={args.inits} "
              f"epochs={args.epochs} budget={args.budget_hours}h")
    t0 = time.time()

    for i, (task, horizon) in enumerate(tasks):
        if (time.time() - t0) / 3600 > args.budget_hours:
            log.event("budget", f"stopping at task {i} (budget {args.budget_hours}h)")
            break
        key = f"{task}" + (f"_h{horizon}" if task == "mackey_glass" else "")
        log.status(phase=f"[{i+1}/{len(tasks)}] {key}", step=i, total=len(tasks))
        log.event("task", f"[{i+1}/{len(tasks)}] {key}")
        try:
            u, y = _task_series(task, args.T, 1, horizon=horizon)
            lstm = tuned_lstm(u, y, device, args.washout, args.inits, args.epochs, log, key)
            esn = esn_best(u, y, args.washout)
            lc = learning_curve(u, y, device, args.washout, lstm, esn["n_reservoir"],
                                n_trains, inits=max(3, args.inits // 2), epochs=args.epochs)
            rec = {"lstm_best": lstm, "esn_best": esn, "learning_curve": lc,
                   "qrc_perspin": qrc_pts.get(task, {}).get("perspin_mean")}
            summary["tasks"][key] = rec
            log.result(key, {"LSTM": round(lstm["test_mean"], 4), "ESN": round(esn["test_mean"], 4),
                             "QRC": (round(rec["qrc_perspin"], 4) if rec["qrc_perspin"] else "n/a")})
        except Exception as e:  # noqa: BLE001
            log.event("error", f"{key}: {type(e).__name__}: {e}")
            summary["tasks"][key] = {"error": f"{type(e).__name__}: {e}"}
        _write(summary, args, run_dir, t0)

    _figure(summary, run_dir)
    _write(summary, args, run_dir, t0)
    log.event("done", f"classical baseline finished in {(time.time()-t0)/3600:.2f}h")
    log.close("done")


def _write(summary, args, run_dir, t0):
    summary["elapsed_hours"] = round((time.time() - t0) / 3600, 3)
    out = args.out or str(Path(run_dir) / "results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _figure(summary, run_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    tasks = [k for k, v in summary["tasks"].items() if "error" not in v]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), squeeze=False)
    for ax, k in zip(axes[0], tasks):
        rec = summary["tasks"][k]; lc = rec.get("learning_curve", {})
        nt = lc.get("n_trains", [])
        if nt:
            ax.plot(nt, lc["lstm"], "-o", color="#cb4b4b", label="LSTM (tuned)")
            ax.plot(nt, lc["esn"], "-s", color="#e07b3c", label="ESN")
        if rec.get("qrc_perspin"):
            ax.axhline(rec["qrc_perspin"], color="#4da3ff", ls="--", lw=1.6,
                       label="QRC-learned (n_tr=735)")
        ax.set_yscale("log"); ax.set_xlabel("n_train"); ax.set_ylabel("test NMSE")
        ax.set_title(k, fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.suptitle("Data-efficiency: QRC-learned vs tuned LSTM / ESN", y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig20_classical_baseline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
