"""ERA5 Delhi weather benchmark (1970-2025, 20,454 days): QRC vs tuned ESN with
enough test data to resolve small differences.

Data: Open-Meteo ERA5 daily temperature_2m_mean + relative_humidity_2m_mean,
Delhi (backend/data/weather/era5/delhi_1970_2025.json, CC BY 4.0). Min-max
normalised with 1970-2004 statistics only; later values clipped to [0,1].

Split (fixed before any result):
  washout 1970 | TRAIN 1971-1999 | VAL 2000-2004 (all selection) |
  TEST 2005-01-01 .. end-45 days (~7,600 days, read once in ``eval``).
Targets: temp and humidity at h = 1,5,10,15,20,30,45 (14). Readout: ridge with
alpha (and feature standardisation) chosen on VAL after fitting on TRAIN, then
refit on TRAIN+VAL for TEST.

Subcommands (run in this order; GPU ones one at a time):
  qrc    — arcsin-encoded QRC pass over all days (exact block propagator + proton
           FID, crotonic-9, tau 30 ms); saves the raw FID trace (persist rule),
           the 653 FFT bins, and features X.
  learn  — train the per-spin encoding correction (qrc_weather_learn.Encoder) on
           a 1984-1999 slice only (fit 1985-1994, validate 1995-1999), then run
           the full pass with it; saves trace + X.
  esn    — ESN grid (size x sr x leak x input scale, 3 inits) scored on VAL.
  eval   — groups (QRC arcsin, QRC learned seeds, tuned ESN inits): pooled test
           R², per-5-year blocks, block-bootstrap CIs, ensembles.
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

HORIZONS = (1, 5, 10, 15, 20, 30, 45)
TARGETS = [f"{v}_h{h}" for h in HORIZONS for v in ("temp", "humidity")]
ALPHA = (1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)
SIZES, SR, LEAK, INSCALE, INITS = (300, 650, 1000), (0.5, 0.8, 0.95, 1.1, 1.3), \
    (0.02, 0.05, 0.1, 0.3, 0.6, 1.0), (0.1, 0.3, 1.0, 3.0), (0, 1, 2)
DATA = "data/weather/era5/delhi_1970_2025.json"


# ---------------------------------------------------------------- data / split

def load(path=DATA):
    d = json.loads(Path(path).read_text(encoding="utf-8"))["daily"]
    dates = [dt.date.fromisoformat(s) for s in d["time"]]
    raw = np.column_stack([np.asarray(d["temperature_2m_mean"], float),
                           np.asarray(d["relative_humidity_2m_mean"], float)])
    i_te = dates.index(dt.date(2005, 1, 1))
    lo, hi = raw[:i_te].min(0), raw[:i_te].max(0)                     # 1970-2004 stats only
    w = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    return dates, w, {"lo": lo.tolist(), "hi": hi.tolist()}


def split(dates):
    ix = lambda y, m=1, d=1: dates.index(dt.date(y, m, d))  # noqa: E731
    n = len(dates) - max(HORIZONS)
    return {"train": np.arange(ix(1971), ix(2000)), "val": np.arange(ix(2000), ix(2005)),
            "test": np.arange(ix(2005), n), "n_steps": len(dates)}


def targets(w, idx):
    return np.stack([w[idx + h, j] for h in HORIZONS for j in range(2)], axis=1)


def r2_cols(Y, P):
    return 1 - ((P - Y) ** 2).sum(-2) / ((Y - Y.mean(0)) ** 2).sum(0)


def ridge(Xa, Ya, Xb_list, alphas, std):
    """Primal ridge (bias column unpenalised-ish via tiny alpha share); returns
    [alpha] -> list of predictions for each Xb."""
    if std:
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        Xa = (Xa - mu) / sd
        Xb_list = [(Xb - mu) / sd for Xb in Xb_list]
    Xa = np.hstack([Xa, np.ones((len(Xa), 1))])
    Xb_list = [np.hstack([Xb, np.ones((len(Xb), 1))]) for Xb in Xb_list]
    G, B = Xa.T @ Xa, Xa.T @ Ya
    s, U = np.linalg.eigh(G)
    UtB = U.T @ B
    out = []
    for a in alphas:
        Wt = U @ (UtB / (s + a)[:, None])
        out.append([Xb @ Wt for Xb in Xb_list])
    return out


def val_scores(X, w, sp):
    """val R² [std, alpha, target] (fit TRAIN, score VAL)."""
    Yt, Yv = targets(w, sp["train"]), targets(w, sp["val"])
    res = []
    for std in (False, True):
        P = ridge(X[sp["train"]], Yt, [X[sp["val"]]], ALPHA, std)
        res.append([r2_cols(Yv, p[0]) for p in P])
    return np.asarray(res)


def test_preds(X, w, sp, vs):
    """Readout chosen on VAL (one std flag, alpha per target), refit TRAIN+VAL,
    predict TEST. ``vs`` = val_scores(X)."""
    std = int(np.argmax([np.nanmean(vs[s].max(0)) for s in (0, 1)]))
    a_idx = np.argmax(vs[std], axis=0)
    fit = np.concatenate([sp["train"], sp["val"]])
    P = ridge(X[fit], targets(w, fit), [X[sp["test"]]], ALPHA, bool(std))
    return np.stack([P[a_idx[t]][0][:, t] for t in range(len(TARGETS))], axis=1), \
        {"std": bool(std), "alpha_idx": a_idx.tolist(), "val_pooled": float(np.mean(vs[std].max(0)))}


# ---------------------------------------------------------------- QRC (block propagator)

def qrc_machine(device, cdt):
    import torch
    from app.qrc.block_propagator import BlockPropagator
    from app.qrc.config import SimConfig
    from app.qrc.system import QRCSystem
    from qrc_gen_traces import _resolve_system
    sysc = _resolve_system("crotonic9_paper4")
    sim = SimConfig(tau=0.03, n_virtual=1, evolution_mode="action")
    sysm = QRCSystem(sysc, sim)
    bp = BlockPropagator(sysc, sim, device=device, cdtype=cdt, which=("z",))
    lab = [x.upper() for x in sysc.labels]
    H = [i for i, x in enumerate(lab) if x.startswith("H")]
    C = [i for i, x in enumerate(lab) if x.startswith("C")]
    ro = bp.fid_readout(H, 2048, 3e-4)
    vec0 = torch.tensor(sysm.qt.operator_to_vector(sysm.rho0).full().ravel(), dtype=cdt, device=device)
    return {"bp": bp, "ro": ro, "vec0": vec0, "H": H, "C": C, "n": sysm.n, "torch": torch, "cdt": cdt}


def base_angles(w, m):
    per = np.zeros((len(w), m["n"]))
    per[:, m["H"]] = w[:, 0:1]
    per[:, m["C"]] = w[:, 1:2]
    return np.arcsin(np.sqrt(np.clip(per, 0, 1)))


def run_pass(m, theta, seg=49, grad=False, keep_fids=False, log_every=2000):
    """theta (days, n) torch -> FIDs (days, 2048) torch (graph kept if grad)."""
    torch = m["torch"]
    from qrc_weather_learn import _rx
    from torch.utils.checkpoint import checkpoint

    def segment(vec, th):
        rows = []
        for k in range(th.shape[0]):
            _, vec = m["bp"].step(vec, [_rx(th[k, q], m["cdt"]) for q in range(m["n"])])
            rows.append(m["bp"].fid(vec, m["ro"]))
        return torch.stack(rows), vec

    vec, out, t0 = m["vec0"], [], time.time()
    for i in range(0, theta.shape[0], seg):
        th = theta[i:i + seg]
        if grad:
            F, vec = checkpoint(segment, vec, th, use_reentrant=False)
        else:
            with torch.no_grad():
                F, vec = segment(vec, th)
        out.append(F)
        if log_every and not grad and (i // seg) % max(1, log_every // seg) == 0:
            print(f"  day {i}/{theta.shape[0]} ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(out)


def fft_feats(F, bins):
    torch = __import__("torch")
    return torch.abs(torch.fft.fft(F, dim=1))[:, bins]


def cmd_qrc(a):
    import torch
    dates, w, sc = load()
    cdt = torch.complex128 if a.dtype == "complex128" else torch.complex64
    m = qrc_machine(a.device, cdt)
    th = torch.tensor(base_angles(w, m), dtype=torch.float64, device=a.device)
    t0 = time.time()
    F = run_pass(m, th)
    mag = torch.abs(torch.fft.fft(F, dim=1))
    bins = torch.sort(torch.argsort(mag[0], descending=True)[:653]).values
    X = mag[:, bins].cpu().numpy()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "trace_arcsin.npz", fids_H=F.cpu().numpy().astype(np.complex64), weather_norm=w,
             scalers=json.dumps(sc), meta=json.dumps({"system": "crotonic9_paper4", "tau": 0.03,
                                                      "encoding": "arcsin_sqrt", "fid": "H 2048 x 0.3 ms"}))
    np.savez(out / "X_arcsin.npz", X=X, bins=bins.cpu().numpy())
    print(f"arcsin pass {len(w)} days in {time.time() - t0:.0f}s; wrote {out}", flush=True)


def cmd_learn(a):
    import torch
    from qrc_weather_learn import Encoder
    dates, w, _ = load()
    cdt = torch.complex64
    m = qrc_machine(a.device, cdt)
    bins = torch.tensor(np.load(Path(a.out_dir) / "X_arcsin.npz")["bins"], device=a.device)
    ix = lambda y: dates.index(dt.date(y, 1, 1))  # noqa: E731
    s0 = ix(1984)                                               # fresh reservoir, 1-year washout
    fit = np.arange(ix(1985), ix(1995)) - s0
    val = np.arange(ix(1995), ix(2000)) - s0
    ws = w[s0:ix(2000) + max(HORIZONS)]
    torch.manual_seed(a.seed)
    enc = Encoder(m["n"]).to(a.device)
    s2 = torch.tensor(ws[:ix(2000) - s0], dtype=torch.float64, device=a.device)
    base = torch.tensor(base_angles(ws[:ix(2000) - s0], m), dtype=torch.float64, device=a.device)
    Yf = torch.tensor(targets(ws, fit), dtype=torch.float64, device=a.device)
    Yv = torch.tensor(targets(ws, val), dtype=torch.float64, device=a.device)

    def val_nmse(X, a_idx=None):
        A, B = X[fit], X[val]
        mu, sd = A.mean(0), A.std(0) + 1e-12
        A = torch.cat([(A - mu) / sd, torch.ones(len(A), 1, dtype=A.dtype, device=A.device)], 1)
        B = torch.cat([(B - mu) / sd, torch.ones(len(B), 1, dtype=B.dtype, device=B.device)], 1)
        G, R = A.T @ A, A.T @ Yf
        per = []
        for al in ALPHA:
            Wt = torch.linalg.solve(G + al * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), R)
            per.append(((B @ Wt - Yv) ** 2).mean(0) / Yv.var(0, unbiased=False))
        per = torch.stack(per)
        return per if a_idx is None else per[a_idx, torch.arange(per.shape[1])]

    feats = lambda th, grad: fft_feats(run_pass(m, th, grad=grad, log_every=0), bins).to(torch.float64)  # noqa: E731
    t0 = time.time()
    with torch.no_grad():
        grid = val_nmse(feats(enc(s2, base), False))
    a_idx = torch.argmin(grid, dim=0)
    base_v = float(grid[a_idx, torch.arange(grid.shape[1])].mean())
    print(f"seed {a.seed}: step-0 (arcsin) val NMSE {base_v:.4f}; forward {time.time() - t0:.0f}s", flush=True)
    opt = torch.optim.Adam(enc.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best = (base_v, {k: v.detach().clone() for k, v in enc.state_dict().items()}, -1)
    for it in range(a.steps):
        t1 = time.time()
        opt.zero_grad()
        loss = val_nmse(feats(enc(s2, base), True), a_idx).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
        v = loss.item()
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in enc.state_dict().items()}, it)
        print(f"step {it:3d}: val NMSE {v:.4f} (best {best[0]:.4f} @ {best[2]}) {time.time() - t1:.0f}s", flush=True)
    enc.load_state_dict(best[1])
    s2f = torch.tensor(w, dtype=torch.float64, device=a.device)
    basef = torch.tensor(base_angles(w, m), dtype=torch.float64, device=a.device)
    with torch.no_grad():
        theta = enc(s2f, basef)
        F = run_pass(m, theta)
        X = fft_feats(F, bins).cpu().numpy()
    out = Path(a.out_dir) / f"learned_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "trace_learned.npz", fids_H=F.cpu().numpy().astype(np.complex64), theta=theta.cpu().numpy(),
             meta=json.dumps({"seed": a.seed, "steps": a.steps, "train_slice": "fit 1985-1994, val 1995-1999",
                              "step0_val_nmse": base_v, "best_val_nmse": best[0], "best_step": best[2]}))
    np.savez(out / "X_learned.npz", X=X)
    torch.save(best[1], out / "encoder.pt")
    print(f"seed {a.seed}: best val NMSE {best[0]:.4f} vs {base_v:.4f}; wrote {out}", flush=True)


# ---------------------------------------------------------------- ESN

_G = {}


def _init(w, sp):
    _G["w"], _G["sp"] = w, sp


def esn_states(w, m, sr, leak, ins, init):
    rng = np.random.default_rng(10_000 * init + m)
    W = rng.standard_normal((m, m))
    W *= sr / np.max(np.abs(np.linalg.eigvals(W)))
    Win = ins * rng.uniform(-1, 1, (m, 3))
    x = np.zeros(m)
    X = np.empty((len(w), m))
    for k in range(len(w)):
        x = (1 - leak) * x + leak * np.tanh(Win[:, 0] + Win[:, 1:] @ w[k] + W @ x)
        X[k] = x
    return X


def _esn_job(args):
    m, init, sr = args
    out = {}
    for leak, ins in itertools.product(LEAK, INSCALE):
        X = esn_states(_G["w"], m, sr, leak, ins, init)
        out[(m, sr, leak, ins, init)] = val_scores(X, _G["w"], _G["sp"])
    return out


def cmd_esn(a):
    dates, w, _ = load()
    sp = split(dates)
    jobs = [(m, i, sr) for m in SIZES for i in INITS for sr in SR]
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(w, sp)) as ex:
        for k, out in enumerate(ex.map(_esn_job, jobs)):
            res.update(out)
            print(f"ESN job {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    keys = sorted(res)
    np.savez(a.out, keys=np.array(keys, float), val=np.array([res[k] for k in keys]))
    print(f"wrote {a.out}")


def select_esn(path, max_size=None):
    z = np.load(path)
    keys, val = z["keys"], z["val"]                                          # [k, std, a, t]
    best = None
    for cfg in {tuple(k[:4]) for k in keys}:
        if max_size and cfg[0] > max_size:
            continue
        rows = [i for i, k in enumerate(keys) if tuple(k[:4]) == cfg]
        v = val[rows].mean(0)                                                # avg over inits
        score = max(np.nanmean(v[s].max(0)) for s in (0, 1))
        if best is None or score > best[0]:
            best = (score, cfg)
    return dict(zip(("size", "sr", "leak", "inscale"), best[1])), best[0]


# ---------------------------------------------------------------- eval

def cmd_eval(a):
    dates, w, _ = load()
    sp = split(dates)
    Y = targets(w, sp["test"])
    run = Path(a.out_dir)
    groups = {"qrc_arcsin": [np.load(run / "X_arcsin.npz")["X"]],
              "qrc_learned": [np.load(p)["X"] for p in sorted(run.glob("learned_seed*/X_learned.npz"))]}
    res = {"split": {k: [int(v[0]), int(v[-1])] for k, v in sp.items() if k != "n_steps"},
           "n_test_days": int(len(sp["test"])), "groups": {}, "comparisons": {}}
    for label, max_size in (("esn_tuned", None), ("esn_matched", 653)):
        cfg, sc = select_esn(a.esn, max_size)
        res[f"{label}_cfg"] = {**cfg, "val_pooled": sc}
        groups[label] = [esn_states(w, int(cfg["size"]), cfg["sr"], cfg["leak"], cfg["inscale"], i) for i in INITS]
    preds = {}
    years = np.array([d.year for d in dates])[sp["test"]]
    blocks = [(y, y + 4) for y in range(2005, 2025, 5)]
    for g, Xs in groups.items():
        preds[g] = [test_preds(X, w, sp, val_scores(X, w, sp))[0] for X in Xs]
        per = [float(r2_cols(Y, p).mean()) for p in preds[g]]
        ens = np.mean(preds[g], axis=0)
        res["groups"][g] = {"pooled_test_r2": per, "mean": float(np.mean(per)), "sd": float(np.std(per)),
                            "ensemble": float(r2_cols(Y, ens).mean()),
                            "temp": float(r2_cols(Y, ens)[0::2].mean()), "humidity": float(r2_cols(Y, ens)[1::2].mean()),
                            "by_5yr": {f"{b0}-{b1}": float(r2_cols(Y[(years >= b0) & (years <= b1)],
                                                                    ens[(years >= b0) & (years <= b1)]).mean())
                                       for b0, b1 in blocks}}
        print(f"{g:12s} pooled test R2 {np.mean(per):.4f} ± {np.std(per):.4f} (reps {', '.join(f'{x:.3f}' for x in per)}) "
              f"ens {res['groups'][g]['ensemble']:.4f} | temp {res['groups'][g]['temp']:.3f} hum {res['groups'][g]['humidity']:.3f}",
              flush=True)

    rng = np.random.default_rng(0)
    n, block = len(Y), 30
    boots = [np.concatenate([np.arange(s, s + block) for s in rng.choice(n - block + 1, n // block + 1)])[:n]
             for _ in range(2000)]
    A = np.stack(preds["qrc_learned"]) if preds["qrc_learned"] else None
    for b in [g for g in groups if g != "qrc_learned"]:
        if A is None:
            break
        B = np.stack(preds[b])
        d = np.array([r2_cols(Y[i], A[:, i]).mean() - r2_cols(Y[i], B[:, i]).mean() for i in boots])
        de = np.array([r2_cols(Y[i], A.mean(0)[i]).mean() - r2_cols(Y[i], B.mean(0)[i]).mean() for i in boots])
        ci, cie = np.percentile(d, [2.5, 97.5]).tolist(), np.percentile(de, [2.5, 97.5]).tolist()
        res["comparisons"][f"qrc_learned - {b}"] = {"pooled_ci95": ci, "frac_positive": float((d > 0).mean()),
                                                    "ensemble_ci95": cie, "ensemble_frac_positive": float((de > 0).mean())}
        sig = "SIGNIFICANT" if ci[0] > 0 else ("worse (significant)" if ci[1] < 0 else "n.s.")
        print(f"qrc_learned - {b:12s} 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] P(>0)={(d > 0).mean():.3f} {sig} | "
              f"ensembles [{cie[0]:+.4f}, {cie[1]:+.4f}]", flush=True)
    print("per 5-year test block (ensembles):")
    for g in groups:
        print(f"  {g:12s} " + " ".join(f"{k}: {v:.3f}" for k, v in res["groups"][g]["by_5yr"].items()))
    Path(a.out).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qrc")
    q.add_argument("--device", default="cuda")
    q.add_argument("--dtype", default="complex128", choices=["complex64", "complex128"])
    q.add_argument("--out-dir", required=True)
    le = sub.add_parser("learn")
    le.add_argument("--device", default="cuda")
    le.add_argument("--seed", type=int, default=0)
    le.add_argument("--steps", type=int, default=40)
    le.add_argument("--lr", type=float, default=0.02)
    le.add_argument("--out-dir", required=True)
    e = sub.add_parser("esn")
    e.add_argument("--workers", type=int, default=32)
    e.add_argument("--out", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--out-dir", required=True)
    ev.add_argument("--esn", required=True)
    ev.add_argument("--out", required=True)
    a = ap.parse_args()
    {"qrc": cmd_qrc, "learn": cmd_learn, "esn": cmd_esn, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
