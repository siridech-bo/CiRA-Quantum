"""Real-world forecasting benchmark: QRC (arcsin + learned encoding) vs tuned ESN
vs tuned LSTM, one protocol for every dataset (SOP §6.5).

Datasets (chronological splits fixed before any result; normalisation from
pre-test data only, later values clipped to [0,1]):
  kaggle_delhi   Kaggle Daily Delhi Climate (the Paper-4 task), the trace's own
                 series/normalisation; train 374-673, val 674-973, test 974-1473
  era5_<city>    Open-Meteo ERA5 daily T + RH 1970-2025 (delhi, mumbai, bangkok,
                 beijing, berlin): train 1971-1999, val 2000-2004, test 2005-
  santafe        Santa Fe laser A (A.dat + A.cont, 10,093 pts), 1 variable
  beijing_pm25   UCI Beijing PM2.5 station, hourly TEMP + DEWP 2010-2014:
                 train 2010-2012, val 2013, test 2014
  etth1          ETTh1 hourly, inputs OT + HUFL, target OT; standard 12/4/4-month split
Targets: each target variable at each horizon (steps). Metric: pooled test R²
(mean over targets); block-bootstrap CIs; replicate groups (QRC encoder seeds,
ESN reservoir seeds, LSTM inits).

Models
  qrc_arcsin   crotonic-9 reservoir, tau 30 ms, exact block propagator + proton FID,
               |FFT| at 653 bins; var0 -> protons, var1 -> carbons (1-variable
               data: the input on every spin); ridge readout (alpha, standardisation
               chosen on VAL, refit TRAIN+VAL)
  qrc_learned  same, per-spin encoding correction trained on a train-period slice
               only (fit = last 3000-1000 train steps, inner val = last 1000)
  esn_tuned    grid size x spectral radius x leak x input scale, 3 inits, selected
               on VAL; esn_matched = same with size <= 653
  lstm_tuned   hidden x layers x lr sweep, 3 inits, multi-output head (all targets),
               early stopping on VAL R²; the selected config is retrained on
               TRAIN+VAL for the selected epoch count and scored on TEST

  python scripts/qrc_realworld_bench.py <cmd> --dataset NAME --run-dir DIR
  cmds: qrc | learn --seed S | esn | lstm | eval
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qrc_era5_weather import (  # noqa: E402
    ALPHA, INITS, INSCALE, LEAK, SIZES, SR, fft_feats, qrc_machine, r2_cols, ridge, run_pass,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- datasets

def _minmax(raw, i_test):
    lo, hi = raw[:i_test].min(0), raw[:i_test].max(0)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def load_dataset(name):
    """-> dict(w (n, d_in) in [0,1], targets [(var, h)], split {train,val,test,washout},
    unit, inputs, n)."""
    if name == "kaggle_delhi":
        z = np.load(ROOT / "artifacts/traces/weather_full.npz", allow_pickle=True)
        w = np.asarray(z["weather_norm"], float)
        H = (1, 5, 10, 15, 20, 30, 45)
        sp = {"washout": 374, "train": np.arange(374, 674), "val": np.arange(674, 974),
              "test": np.arange(974, 1474)}
        return dict(w=w, targets=[(v, h) for h in H for v in (0, 1)], split=sp, unit="day",
                    inputs=["temp", "humidity"])
    if name.startswith("era5_"):
        city = name.split("_", 1)[1]
        d = json.loads((ROOT / f"data/weather/era5/{city}_1970_2025.json").read_text(encoding="utf-8"))["daily"]
        dates = [dt.date.fromisoformat(s) for s in d["time"]]
        raw = np.column_stack([np.asarray(d["temperature_2m_mean"], float),
                               np.asarray(d["relative_humidity_2m_mean"], float)])
        ix = lambda y: dates.index(dt.date(y, 1, 1))  # noqa: E731
        H = (1, 5, 10, 15, 20, 30, 45)
        w = _minmax(raw, ix(2005))
        sp = {"washout": ix(1971), "train": np.arange(ix(1971), ix(2000)), "val": np.arange(ix(2000), ix(2005)),
              "test": np.arange(ix(2005), len(w) - max(H))}
        return dict(w=w, targets=[(v, h) for h in H for v in (0, 1)], split=sp, unit="day",
                    inputs=["temp", "humidity"])
    if name == "santafe":
        x = np.concatenate([np.loadtxt(ROOT / "data/realworld/raw/santafe_A.dat"),
                            np.loadtxt(ROOT / "data/realworld/raw/santafe_A.cont")])[:, None]
        H = (1, 2, 3, 5, 10)
        w = _minmax(x, 7700)
        sp = {"washout": 200, "train": np.arange(200, 6200), "val": np.arange(6200, 7700),
              "test": np.arange(7700, len(w) - max(H))}
        return dict(w=w, targets=[(0, h) for h in H], split=sp, unit="step", inputs=["intensity"])
    if name == "beijing_pm25":
        import pandas as pd
        df = pd.read_csv(ROOT / "data/realworld/raw/PRSA_data_2010.1.1-2014.12.31.csv")
        raw = df[["TEMP", "DEWP"]].to_numpy(float)
        yr = df["year"].to_numpy()
        i_val, i_test = int(np.argmax(yr == 2013)), int(np.argmax(yr == 2014))
        H = (1, 3, 6, 12, 24, 48)
        w = _minmax(raw, i_test)
        sp = {"washout": 500, "train": np.arange(500, i_val), "val": np.arange(i_val, i_test),
              "test": np.arange(i_test, len(w) - max(H))}
        return dict(w=w, targets=[(v, h) for h in H for v in (0, 1)], split=sp, unit="hour",
                    inputs=["temp", "dewpoint"])
    if name == "etth1":
        import pandas as pd
        df = pd.read_csv(ROOT / "data/realworld/raw/ETTh1.csv")
        raw = df[["OT", "HUFL"]].to_numpy(float)
        tr, va, te = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24                  # standard ETT split
        H = (1, 6, 12, 24, 48)
        w = _minmax(raw, tr + va)
        sp = {"washout": 240, "train": np.arange(240, tr), "val": np.arange(tr, tr + va),
              "test": np.arange(tr + va, tr + va + te - max(H))}
        return dict(w=w, targets=[(0, h) for h in H], split=sp, unit="hour", inputs=["OT", "HUFL"])
    raise KeyError(name)


def make_targets(ds, idx):
    w = ds["w"]
    return np.stack([w[idx + h, v] for (v, h) in ds["targets"]], axis=1)


def target_names(ds):
    return [f"{ds['inputs'][v]}_h{h}" for (v, h) in ds["targets"]]


# ---------------------------------------------------------------- readout protocol

def val_scores(X, ds):
    sp = ds["split"]
    Yt, Yv = make_targets(ds, sp["train"]), make_targets(ds, sp["val"])
    res = []
    for std in (False, True):
        P = ridge(X[sp["train"]], Yt, [X[sp["val"]]], ALPHA, std)
        res.append([r2_cols(Yv, p[0]) for p in P])
    return np.asarray(res)                                                    # [std, a, t]


def test_preds(X, ds, vs=None):
    sp = ds["split"]
    vs = val_scores(X, ds) if vs is None else vs
    std = int(np.argmax([np.nanmean(vs[s].max(0)) for s in (0, 1)]))
    a_idx = np.argmax(vs[std], axis=0)
    fit = np.concatenate([sp["train"], sp["val"]])
    P = ridge(X[fit], make_targets(ds, fit), [X[sp["test"]]], ALPHA, bool(std))
    return np.stack([P[a_idx[t]][0][:, t] for t in range(len(ds["targets"]))], axis=1)


# ---------------------------------------------------------------- QRC

def spin_inputs(ds, m):
    """(n, n_spins) input value per spin: var0 -> protons, var1 -> carbons; a
    single-variable series drives every spin."""
    w = ds["w"]
    per = np.zeros((len(w), m["n"]))
    if w.shape[1] == 1:
        per[:] = w[:, 0:1]
    else:
        per[:, m["H"]] = w[:, 0:1]
        per[:, m["C"]] = w[:, 1:2]
    return per


def cmd_qrc(a, ds):
    import torch
    m = qrc_machine(a.device, torch.complex128)
    th = torch.tensor(np.arcsin(np.sqrt(spin_inputs(ds, m))), dtype=torch.float64, device=a.device)
    t0 = time.time()
    F = run_pass(m, th)
    mag = torch.abs(torch.fft.fft(F, dim=1))
    bins = torch.sort(torch.argsort(mag[0], descending=True)[:653]).values
    out = Path(a.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "trace_arcsin.npz", fids_H=F.cpu().numpy().astype(np.complex64), w=ds["w"],
             meta=json.dumps({"dataset": a.dataset, "encoding": "arcsin_sqrt", "fid": "H 2048 x 0.3 ms"}))
    np.savez(out / "X_arcsin.npz", X=mag[:, bins].cpu().numpy(), bins=bins.cpu().numpy())
    print(f"{a.dataset}: arcsin pass {len(ds['w'])} steps in {time.time() - t0:.0f}s", flush=True)


class Encoder:
    pass


def _encoder(d_in, n, hidden=16, scale=0.5):
    import torch

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden, dtype=torch.float64), torch.nn.Tanh(),
                                           torch.nn.Linear(hidden, n, dtype=torch.float64))
            torch.nn.init.zeros_(self.net[2].weight)
            torch.nn.init.zeros_(self.net[2].bias)

        def forward(self, s, base):
            return base + scale * torch.tanh(self.net(s))
    return Enc()


def cmd_learn(a, ds):
    import torch
    m = qrc_machine(a.device, torch.complex64)
    bins = torch.tensor(np.load(Path(a.run_dir) / "X_arcsin.npz")["bins"], device=a.device)
    tr_end = int(ds["split"]["train"][-1]) + 1
    s0 = max(0, tr_end - a.slice - 500)                                     # 500-step washout
    fit = np.arange(tr_end - a.slice, tr_end - a.slice // 3) - s0
    val = np.arange(tr_end - a.slice // 3, tr_end) - s0
    w = ds["w"]
    sub = {"w": w[s0:tr_end + max(h for _, h in ds["targets"])], "targets": ds["targets"]}
    per = spin_inputs(ds, m)
    torch.manual_seed(a.seed)
    enc = _encoder(w.shape[1], m["n"]).to(a.device)
    s_in = torch.tensor(w[s0:tr_end], dtype=torch.float64, device=a.device)
    base = torch.tensor(np.arcsin(np.sqrt(per[s0:tr_end])), dtype=torch.float64, device=a.device)
    Yf = torch.tensor(make_targets(sub, fit), dtype=torch.float64, device=a.device)
    Yv = torch.tensor(make_targets(sub, val), dtype=torch.float64, device=a.device)

    def loss_fn(X, a_idx=None):
        A, B = X[fit], X[val]
        mu, sd = A.mean(0), A.std(0) + 1e-12
        A = torch.cat([(A - mu) / sd, torch.ones(len(A), 1, dtype=A.dtype, device=A.device)], 1)
        B = torch.cat([(B - mu) / sd, torch.ones(len(B), 1, dtype=B.dtype, device=B.device)], 1)
        G, R = A.T @ A, A.T @ Yf
        per_a = []
        for al in ALPHA:
            Wt = torch.linalg.solve(G + al * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), R)
            per_a.append(((B @ Wt - Yv) ** 2).mean(0) / Yv.var(0, unbiased=False))
        per_a = torch.stack(per_a)
        return per_a if a_idx is None else per_a[a_idx, torch.arange(per_a.shape[1])]

    feats = lambda th, g: fft_feats(run_pass(m, th, grad=g, log_every=0), bins).to(torch.float64)  # noqa: E731
    with torch.no_grad():
        grid = loss_fn(feats(enc(s_in, base), False))
    a_idx = torch.argmin(grid, dim=0)
    base_v = float(grid[a_idx, torch.arange(grid.shape[1])].mean())
    print(f"{a.dataset} seed {a.seed}: step-0 val NMSE {base_v:.4f} (slice {s0}-{tr_end})", flush=True)
    opt = torch.optim.Adam(enc.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best = (base_v, {k: v.detach().clone() for k, v in enc.state_dict().items()}, -1)
    for it in range(a.steps):
        t1 = time.time()
        opt.zero_grad()
        loss = loss_fn(feats(enc(s_in, base), True), a_idx).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
        v = loss.item()
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in enc.state_dict().items()}, it)
        print(f"step {it:3d}: val NMSE {v:.4f} (best {best[0]:.4f} @ {best[2]}) {time.time() - t1:.0f}s", flush=True)
    enc.load_state_dict(best[1])
    with torch.no_grad():
        theta = enc(torch.tensor(w, dtype=torch.float64, device=a.device),
                    torch.tensor(np.arcsin(np.sqrt(per)), dtype=torch.float64, device=a.device))
        F = run_pass(m, theta)
        X = fft_feats(F, bins).cpu().numpy()
    out = Path(a.run_dir) / f"learned_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "trace_learned.npz", fids_H=F.cpu().numpy().astype(np.complex64), theta=theta.cpu().numpy(),
             meta=json.dumps({"dataset": a.dataset, "seed": a.seed, "steps": a.steps, "slice": [s0, tr_end],
                              "step0_val_nmse": base_v, "best_val_nmse": best[0], "best_step": best[2]}))
    np.savez(out / "X_learned.npz", X=X)
    print(f"{a.dataset} seed {a.seed}: best val NMSE {best[0]:.4f} vs {base_v:.4f}", flush=True)


# ---------------------------------------------------------------- ESN

_G = {}


def _init(ds):
    _G["ds"] = ds


def esn_states(w, m, sr, leak, ins, init):
    rng = np.random.default_rng(10_000 * init + m)
    W = rng.standard_normal((m, m))
    W *= sr / np.max(np.abs(np.linalg.eigvals(W)))
    Win = ins * rng.uniform(-1, 1, (m, 1 + w.shape[1]))
    x = np.zeros(m)
    X = np.empty((len(w), m))
    for k in range(len(w)):
        x = (1 - leak) * x + leak * np.tanh(Win[:, 0] + Win[:, 1:] @ w[k] + W @ x)
        X[k] = x
    return X


def _esn_job(args):
    m, init, sr = args
    ds = _G["ds"]
    out = {}
    for leak, ins in itertools.product(LEAK, INSCALE):
        out[(m, sr, leak, ins, init)] = val_scores(esn_states(ds["w"], m, sr, leak, ins, init), ds)
    return out


def cmd_esn(a, ds):
    jobs = [(m, i, sr) for m in SIZES for i in INITS for sr in SR]
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(ds,)) as ex:
        for k, out in enumerate(ex.map(_esn_job, jobs)):
            res.update(out)
            print(f"ESN job {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    keys = sorted(res)
    Path(a.run_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.run_dir) / "esn_grid.npz", keys=np.array(keys, float), val=np.array([res[k] for k in keys]))


def select_esn(path, max_size=None):
    z = np.load(path)
    keys, val = z["keys"], z["val"]
    best = None
    for cfg in sorted({tuple(k[:4]) for k in keys}):
        if max_size and cfg[0] > max_size:
            continue
        rows = [i for i, k in enumerate(keys) if tuple(k[:4]) == cfg]
        v = val[rows].mean(0)
        score = max(np.nanmean(v[s].max(0)) for s in (0, 1))
        if best is None or score > best[0]:
            best = (score, cfg)
    return dict(zip(("size", "sr", "leak", "inscale"), best[1])), float(best[0])


# ---------------------------------------------------------------- LSTM

def cmd_lstm(a, ds):
    import torch
    dev = a.device
    sp, w = ds["split"], ds["w"]
    n_t = len(ds["targets"])
    n_all = len(w) - max(h for _, h in ds["targets"])
    Y = make_targets(ds, np.arange(n_all))
    U = torch.tensor(w[:n_all], dtype=torch.float32, device=dev).unsqueeze(0)
    Yt = torch.tensor(Y, dtype=torch.float32, device=dev)

    class Net(torch.nn.Module):
        def __init__(self, hidden, layers):
            super().__init__()
            self.rnn = torch.nn.LSTM(w.shape[1], hidden, num_layers=layers, batch_first=True)
            self.head = torch.nn.Linear(hidden, n_t)

        def forward(self, u):
            return self.head(self.rnn(u)[0])[0]

    L, burn, B = a.chunk, a.burn, a.batch

    def chunks(train_idx):
        """Chunk starts covering the training range; the first ``burn`` steps of a
        chunk only warm the state up and are not scored (standard truncated BPTT)."""
        lo, hi = int(train_idx[0]), int(train_idx[-1]) + 1
        return np.arange(max(0, lo - burn), max(1, hi - L + 1), L - burn)

    def fit(hidden, layers, lr, init, train_idx, eval_idx, max_epochs, fixed_epochs=None):
        torch.manual_seed(init)
        rng = np.random.default_rng(init)
        net = Net(hidden, layers).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        tr = torch.tensor(train_idx, device=dev)
        sd = Yt[tr].std(0) + 1e-8
        in_train = torch.zeros(n_all, dtype=torch.bool, device=dev)
        in_train[tr] = True
        starts = chunks(train_idx)
        best, best_ep, bad, state = -np.inf, 0, 0, None
        epochs = fixed_epochs or max_epochs
        for ep in range(1, epochs + 1):
            net.train()
            perm = rng.permutation(starts)
            for i in range(0, len(perm), B):
                s = perm[i:i + B]
                idx = torch.tensor(np.stack([np.arange(x, x + L) for x in s]), device=dev)   # (b, L)
                P = net.head(net.rnn(U[0][idx])[0])                                          # (b, L, n_t)
                mask = in_train[idx][:, burn:]
                if not mask.any():
                    continue
                err = ((P[:, burn:] - Yt[idx][:, burn:]) / sd) ** 2
                loss = err[mask].mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if fixed_epochs is None and (ep % 5 == 0):
                net.eval()
                with torch.no_grad():
                    r = float(r2_cols(Y[eval_idx], net(U)[eval_idx].cpu().numpy()).mean())
                if r > best:
                    best, best_ep, bad = r, ep, 0
                    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 5
                    if bad >= a.patience:
                        break
        if fixed_epochs is None:
            net.load_state_dict(state)
            return best, best_ep
        with torch.no_grad():
            return net(U)[eval_idx].cpu().numpy()

    grid = [(h, l, lr) for h in (32, 64, 128) for l in (1, 2) for lr in (1e-3, 3e-3, 1e-2)]
    res, t0 = [], time.time()
    for (h, l, lr) in grid:
        runs = [fit(h, l, lr, i, sp["train"], sp["val"], a.max_epochs) for i in INITS]
        res.append({"hidden": h, "layers": l, "lr": lr, "val_r2": [r for r, _ in runs], "epochs": [e for _, e in runs]})
        print(f"LSTM h{h} l{l} lr{lr}: val R2 {np.mean([r for r, _ in runs]):.4f} "
              f"epochs {[e for _, e in runs]} ({time.time() - t0:.0f}s)", flush=True)
    best = max(res, key=lambda r: np.mean(r["val_r2"]))
    fit_idx = np.concatenate([sp["train"], sp["val"]])
    preds = [fit(best["hidden"], best["layers"], best["lr"], i, fit_idx, sp["test"], a.max_epochs,
                 fixed_epochs=max(5, best["epochs"][k])) for k, i in enumerate(INITS)]
    np.savez(Path(a.run_dir) / "lstm.npz", preds=np.stack(preds), meta=json.dumps({"grid": res, "best": best}))
    print(f"LSTM best {best['hidden']}/{best['layers']}/{best['lr']} val {np.mean(best['val_r2']):.4f}; "
          f"test R2 per init {[round(float(r2_cols(make_targets(ds, sp['test']), p).mean()), 4) for p in preds]}",
          flush=True)


# ---------------------------------------------------------------- eval

def cmd_eval(a, ds):
    run = Path(a.run_dir)
    sp = ds["split"]
    Y = make_targets(ds, sp["test"])
    preds, res = {}, {"dataset": a.dataset, "n_test": int(len(sp["test"])), "unit": ds["unit"],
                      "targets": target_names(ds), "groups": {}, "comparisons": {}}
    preds["qrc_arcsin"] = [test_preds(np.load(run / "X_arcsin.npz")["X"], ds)]
    learned = sorted(run.glob("learned_seed*/X_learned.npz"))
    if learned:
        preds["qrc_learned"] = [test_preds(np.load(p)["X"], ds) for p in learned]
    for label, mx in (("esn_tuned", None), ("esn_matched", 653)):
        cfg, sc = select_esn(run / "esn_grid.npz", mx)
        res[f"{label}_cfg"] = {**cfg, "val_pooled": sc}
        preds[label] = [test_preds(esn_states(ds["w"], int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], i), ds)
                        for i in INITS]
    if (run / "lstm.npz").exists():
        z = np.load(run / "lstm.npz")
        preds["lstm_tuned"] = list(z["preds"])
        res["lstm_cfg"] = json.loads(str(z["meta"]))["best"]
    for g, P in preds.items():
        per = [float(r2_cols(Y, p).mean()) for p in P]
        ens = np.mean(P, axis=0)
        res["groups"][g] = {"pooled_test_r2": per, "mean": float(np.mean(per)), "sd": float(np.std(per)),
                            "ensemble": float(r2_cols(Y, ens).mean()),
                            "per_target_ensemble": dict(zip(target_names(ds), r2_cols(Y, ens).tolist()))}
        print(f"{a.dataset:13s} {g:12s} test R2 {np.mean(per):.4f} ± {np.std(per):.4f} "
              f"(reps {', '.join(f'{x:.3f}' for x in per)}) ens {res['groups'][g]['ensemble']:.4f}", flush=True)
    rng = np.random.default_rng(0)
    n, block = len(Y), min(30, max(5, len(Y) // 50))
    boots = [np.concatenate([np.arange(s, s + block) for s in rng.choice(n - block + 1, n // block + 1)])[:n]
             for _ in range(2000)]
    ref = "qrc_learned" if "qrc_learned" in preds else "qrc_arcsin"
    A = np.stack(preds[ref])
    for b in [g for g in preds if g != ref]:
        B = np.stack(preds[b])
        d = np.array([r2_cols(Y[i], A[:, i]).mean() - r2_cols(Y[i], B[:, i]).mean() for i in boots])
        ci = np.percentile(d, [2.5, 97.5]).tolist()
        res["comparisons"][f"{ref} - {b}"] = {"ci95": ci, "frac_positive": float((d > 0).mean()),
                                             "point": float(np.mean([r2_cols(Y, p).mean() for p in A])
                                                            - np.mean([r2_cols(Y, p).mean() for p in B]))}
        sig = "QRC better (sig.)" if ci[0] > 0 else ("QRC worse (sig.)" if ci[1] < 0 else "n.s.")
        print(f"{a.dataset:13s} {ref} - {b:12s} 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] {sig}", flush=True)
    # publication data package: every model's test predictions per replicate,
    # the targets and the test index, plus provenance — figures and statistics
    # can then be drawn from saved data without re-running anything
    from app.qrc.provenance import provenance
    res["provenance"] = provenance(dataset=a.dataset, split={k: [int(v[0]), int(v[-1])] if hasattr(v, "__len__")
                                                            else int(v) for k, v in sp.items()},
                                   bootstrap={"n_boot": 2000, "block": int(block), "seed": 0})
    np.savez(run / "predictions.npz", Y_test=Y, test_idx=sp["test"], targets=np.array(target_names(ds)),
             **{f"pred_{g}": np.stack(P) for g, P in preds.items()})
    (run / "eval.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qrc", "learn", "esn", "lstm", "eval", "info"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--slice", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=256, help="LSTM truncated-BPTT chunk length")
    ap.add_argument("--burn", type=int, default=96, help="unscored warm-up steps per chunk")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    ds = load_dataset(a.dataset)
    if a.cmd == "info":
        sp = ds["split"]
        print(f"{a.dataset}: n={len(ds['w'])} inputs={ds['inputs']} targets={len(ds['targets'])} "
              f"train {sp['train'][0]}-{sp['train'][-1]} ({len(sp['train'])}) val {len(sp['val'])} "
              f"test {sp['test'][0]}-{sp['test'][-1]} ({len(sp['test'])}) unit={ds['unit']}")
        return
    {"qrc": cmd_qrc, "learn": cmd_learn, "esn": cmd_esn, "lstm": cmd_lstm, "eval": cmd_eval}[a.cmd](a, ds)


if __name__ == "__main__":
    main()
