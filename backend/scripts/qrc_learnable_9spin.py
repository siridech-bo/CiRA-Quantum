"""9-spin learnable-encoding runner + GPU micro-benchmark.

The completeness check: does a learned encoding beat arcsin(sqrt) on the *strong*
9-spin crotonic reservoir (richer dynamics than the 3-spin prototype)? Uses the
differentiable reservoir step (ensure_diff/step_diff), per-spin (frequency-
selective) encoding for maximal expressivity, and closed-form ridge readout.

Two modes:
  --mode benchmark  : run a few forward+backward steps, report ms/step and peak
                      GPU memory. Use this FIRST to get a real wall-clock estimate
                      and confirm backprop memory fits before any full campaign.
  --mode train      : full learnable-encoding training vs arcsin baseline.

Device: --device cuda (real) or cpu (validation). At 9-spin, cpu is impractically
slow; validate the code path on a small system with --system 3 --device cpu.

Run (benchmark): python backend/scripts/qrc_learnable_9spin.py --mode benchmark --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.qrc.config import SimConfig  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402
from qrc_gen_traces import _resolve_system  # noqa: E402

RDT = torch.float64
CDT_MAP = {"complex64": torch.complex64, "complex128": torch.complex128}


def build_system(name, n_virtual):
    sysc = _resolve_system(name)
    sim = SimConfig(tau=0.03, n_virtual=n_virtual, evolution_mode="action")
    return QRCSystem(sysc, sim)


def _rx1(theta, cdt):
    c = torch.cos(theta / 2).to(cdt)
    s = torch.sin(theta / 2).to(cdt)
    sx = torch.tensor([[0, 1], [1, 0]], dtype=cdt, device=theta.device)
    eye = torch.eye(2, dtype=cdt, device=theta.device)
    return c * eye - 1j * s * sx


def per_spin_pulse(thetas, n, cdt):
    """U = tensor product of R_x(theta_i) over spins — per-spin encoding."""
    U = _rx1(thetas[0], cdt)
    for i in range(1, n):
        U = torch.kron(U, _rx1(thetas[i], cdt))
    return U


class EncoderPerSpin(torch.nn.Module):
    """s -> n_out pulse angles in (0, pi). n_out=n gives per-spin (frequency-
    selective) encoding; n_out=1 gives a single global angle (broadcast to all
    spins, same structure as arcsin)."""

    def __init__(self, n_out, hidden=16):
        super().__init__()
        self.n_out = n_out
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden, dtype=RDT), torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_out, dtype=RDT))

    def forward(self, s):
        dev = self.net[0].weight.device
        x = torch.as_tensor(s, dtype=RDT, device=dev).reshape(1, 1)
        return torch.pi * torch.sigmoid(self.net(x)).reshape(-1)


def enc_angles(enc, s, n):
    """Per-spin angles for one input: (n,) for per-spin encoders, broadcast for global."""
    a = enc(s)
    return a if a.shape[0] == n else a.expand(n)


def init_state(sysm, device, cdt):
    rho0 = sysm.qt.basis(sysm.dim, 0) * sysm.qt.basis(sysm.dim, 0).dag()
    np_dt = np.complex64 if cdt == torch.complex64 else np.complex128
    return torch.tensor(sysm.qt.operator_to_vector(rho0).full().ravel().astype(np_dt),
                        device=device)


def forward_features(angles_seq, sysm, g, device, cdt):
    """angles_seq: list of angle-vectors (one per input step) -> feature matrix."""
    vec = init_state(sysm, device, cdt)
    rows = []
    for th in angles_seq:
        U = per_spin_pulse(th, sysm.n, cdt)
        feat, vec = sysm.step_diff(vec, U, g)
        rows.append(feat)
    return torch.stack(rows)


def benchmark(sysm, g, device, cdt, n_steps=3, seq_len=8, baseline_T=None):
    """Time forward+backward and report peak memory — the go/no-go for the campaign.
    If ``baseline_T`` is set, also compute the arcsin NARMA-2 baseline at that T to
    confirm the config is powered (test NMSE < 0.8) before any full training run."""
    enc = EncoderPerSpin(sysm.n).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=0.01)
    u = np.random.default_rng(0).random(seq_len)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # warm-up (build caches / CUDA kernels)
    t_fwd = forward_features([enc(float(s)) for s in u], sysm, g, device, cdt)
    (t_fwd.real.sum()).backward()
    times = []
    for _ in range(n_steps):
        opt.zero_grad()
        t0 = time.time()
        F = forward_features([enc(float(s)) for s in u], sysm, g, device, cdt)
        loss = (F ** 2).mean()
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    ms = 1000 * float(np.median(times))
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    print(f"\n=== BENCHMARK: N={sysm.n} dim={sysm.dim} V={g['V']} "
          f"substeps={g['substeps']} seq_len={seq_len} dtype={cdt} device={device} ===")
    print(f"  forward+backward: {ms:.0f} ms per (seq_len={seq_len}) training step")
    print(f"  ~ {ms/seq_len:.1f} ms per input-step (fwd+bwd)")
    print(f"  peak GPU memory: {peak_gb:.2f} GB")
    print(f"\n  PROJECTED full campaign (per training step scales ~linearly with real seq_len):")
    for T in (240, 300, 400):
        per_step_s = ms / seq_len * T / 1000
        print(f"    T={T}, 100 Adam steps: ~{per_step_s*100/60:.1f} min/condition "
              f"({per_step_s:.1f} s/Adam-step)")

    if baseline_T:
        u, y = make_task("narma2", baseline_T, seed=7)
        wo = 30

        def arc(s):
            a = float(np.arcsin(np.sqrt(min(max(float(s), 0.0), 1.0))))
            return torch.full((sysm.n,), a, dtype=RDT, device=device)

        with torch.no_grad():
            F = forward_features([arc(s) for s in u], sysm, g, device, cdt)[wo:]
            _, base_te = _ridge_fit_eval(F, y[wo:], device)
        b = float(base_te)
        print(f"\n  arcsin NARMA-2 baseline @ T={baseline_T}: test NMSE={b:.4f}  "
              f"({'POWERED (<0.8)' if b < 0.8 else 'UNDER-POWERED -> raise V/T before full run'})")
    return ms, peak_gb


def make_task(task, T, seed):
    """Return (u, y): reservoir driving input u and target y."""
    if task == "narma2":
        from app.qrc.tasks import narma_sequence  # u~U[0,0.5]; cubic input term 0.6 u^3
        return narma_sequence(T, order=2, seed=seed)
    # synthetic memory-2 polynomial (legacy fallback)
    rng = np.random.default_rng(seed)
    u = rng.random(T)
    y = np.zeros_like(u)
    for t in range(2, len(u)):
        y[t] = 0.5 * u[t] + 0.35 * u[t - 1] ** 2 - 0.25 * u[t - 1] * u[t - 2]
    return u, y


def _three_way(n_usable):
    """50/25/25 train/val/test slices over the post-washout sequence."""
    n_tr = int(0.50 * n_usable)
    n_va = int(0.25 * n_usable)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n_usable)


def _ridge_fit_eval(F, y, device, alpha=1e-3):
    """Leakage-free: fit ridge readout on TRAIN, return (val NMSE, test NMSE) with
    the SAME w. The encoder is optimized on val NMSE; test is the held-out verdict."""
    Fb = torch.cat([F, torch.ones(F.shape[0], 1, dtype=RDT, device=device)], dim=1)
    yt = torch.as_tensor(y, dtype=RDT, device=device)
    tr, va, te = _three_way(Fb.shape[0])
    A = Fb[tr].T @ Fb[tr] + alpha * torch.eye(Fb.shape[1], dtype=RDT, device=device)
    w = torch.linalg.solve(A, Fb[tr].T @ yt[tr])
    nmse_val = ((Fb[va] @ w - yt[va]) ** 2).mean() / yt[va].var()
    nmse_test = ((Fb[te] @ w - yt[te]) ** 2).mean() / yt[te].var()
    return nmse_val, nmse_test


def train(sysm, g, device, cdt, task="narma2", T=300, washout=30, steps=100,
          lr=0.04, seed=7, conditions=("arcsin", "perspin")):
    """Learned encoding vs global arcsin(sqrt) baseline on an encoding-sensitive
    task (NARMA-2), leakage-free 3-way split, with under-powered/degenerate
    guardrails. conditions may include 'global' (learned scalar angle) and
    'perspin' (learned per-spin angles)."""
    u, y = make_task(task, T, seed)
    y_use = y[washout:]
    tr, va, te = _three_way(len(y_use))
    n_tr, n_va, n_te = tr.stop - tr.start, va.stop - va.start, te.stop - te.start
    nfeat = 3 * sysm.n * g["V"] + 1
    print(f"task={task} T={T} usable={len(y_use)} tr/va/te={n_tr}/{n_va}/{n_te} "
          f"n_feat={nfeat} (n_train>2*n_feat: {n_tr > 2 * nfeat})")

    def feats(angle_fn):
        return forward_features([angle_fn(s) for s in u], sysm, g, device, cdt)[washout:]

    # (A) global arcsin(sqrt) baseline
    def arcsin_ang(s):
        a = float(np.arcsin(np.sqrt(min(max(float(s), 0.0), 1.0))))
        return torch.full((sysm.n,), a, dtype=RDT, device=device)

    with torch.no_grad():
        _, base_te = _ridge_fit_eval(feats(arcsin_ang), y_use, device)
    base = float(base_te)
    print(f"(A) arcsin baseline: test NMSE = {base:.4f}")

    smax = float(np.max(u))
    sg = np.linspace(0.0, smax, 21)
    results = {"arcsin": (base, None, None, None)}   # (test, val_curve, std, angle_map)

    def run_learned(label, n_out):
        torch.manual_seed(seed)          # vary encoder init with the run seed
        enc = EncoderPerSpin(n_out).to(device)
        opt = torch.optim.Adam(enc.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        curve = []
        t0 = time.time()
        for it in range(steps):
            opt.zero_grad()
            nmse_val, _ = _ridge_fit_eval(feats(lambda s: enc_angles(enc, s, sysm.n)), y_use, device)
            nmse_val.backward(); opt.step(); sched.step()
            curve.append(float(nmse_val))
            if it % 20 == 0 or it == steps - 1:
                print(f"  [{label}] step {it:3d}: val NMSE={float(nmse_val):.4f}")
        dt = time.time() - t0
        with torch.no_grad():
            _, nmse_test = _ridge_fit_eval(feats(lambda s: enc_angles(enc, s, sysm.n)), y_use, device)
            amap = torch.stack([enc_angles(enc, float(s), sysm.n) for s in sg]).cpu().numpy()
        fin, std = float(nmse_test), float(amap.std())
        print(f"({label}) learned: test NMSE={fin:.4f} | angle-map std={std:.2f} "
              f"| {dt:.0f}s ({dt/steps:.1f}s/step)")
        results[label] = (fin, curve, std, amap)

    if "global" in conditions:
        run_learned("global", 1)
    if "perspin" in conditions:
        run_learned("perspin", sysm.n)

    learned = [k for k in ("global", "perspin") if results.get(k, (None, None))[1] is not None]
    best = min(learned, key=lambda k: results[k][0]) if learned else None
    tol = 0.02
    if base > 0.80:
        v = f"UNDER-POWERED (arcsin base {base:.3f} ~ mean-predictor) — invalid"
    elif best and results[best][0] < base - tol and results[best][2] > 0.10:
        v = (f"HEADROOM: learned '{best}' beats arcsin "
             f"({results[best][0]:.4f} vs {base:.4f}, std {results[best][2]:.2f})")
    elif best and results[best][0] < base - tol:
        v = f"DEGENERATE win ('{best}' std {results[best][2]:.2f}) — discard; arcsin stands"
    else:
        b = results[best][0] if best else float("nan")
        v = (f"TIE/LOSE: best learned {b:.4f} vs arcsin {base:.4f} — no beat "
             "(arcsin optimal even on an encoding-favorable task)")
    print("VERDICT:", v)
    _fig14(results, base, sg, task, sysm.n)
    return results, base, v


def _fig14(results, base, sg, task, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"global": "#4da3ff", "perspin": "#e07b3c"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    for k in ("global", "perspin"):
        if results.get(k, (None, None))[1] is not None:
            axL.plot(results[k][1], color=colors[k], lw=1.5, label=f"{k} -> {results[k][0]:.3f}")
    axL.axhline(base, color="#3fb950", ls="--", lw=1.2, label=f"arcsin baseline = {base:.3f}")
    axL.set_xlabel("Adam step"); axL.set_ylabel("val NMSE")
    axL.set_title(f"A - learnable encoding ({task}, {n}-spin)")
    axL.legend(fontsize=8); axL.grid(alpha=0.25)

    axR.plot(sg, np.arcsin(np.sqrt(np.clip(sg, 0, 1))), color="#3fb950", lw=2.0,
             label="arcsin(sqrt(s))")
    for k in ("global", "perspin"):
        if results.get(k, (None, None, None, None))[3] is not None:
            axR.plot(sg, results[k][3].mean(axis=1), color=colors[k], lw=1.6,
                     label=f"learned {k} (mean/spin)")
    axR.set_xlabel("input s"); axR.set_ylabel("pulse angle theta")
    axR.set_title("B - encodings"); axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle(f"{n}-spin learnable encoding vs arcsin ({task}, leakage-free 3-way)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig14_learnable_6spin.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["benchmark", "train"], default="benchmark")
    ap.add_argument("--system", default="crotonic9_paper4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-virtual", type=int, default=4)
    ap.add_argument("--dtype", default="complex64", choices=list(CDT_MAP))
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--task", default="narma2", choices=["narma2", "synthetic"])
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--washout", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conditions", default="arcsin,perspin",
                    help="comma list from: arcsin,global,perspin")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    cdt = CDT_MAP[args.dtype]
    sysm = build_system(args.system, args.n_virtual)
    g = sysm.ensure_diff(device=args.device, cdtype=cdt)
    print(f"system={args.system} N={sysm.n} dim={sysm.dim} | V={g['V']} "
          f"substeps={g['substeps']} K={g['K']} | device={args.device} dtype={args.dtype}")

    if args.mode == "benchmark":
        benchmark(sysm, g, args.device, cdt, seq_len=args.seq_len, baseline_T=args.T)
    else:
        train(sysm, g, args.device, cdt, task=args.task, T=args.T,
              washout=args.washout, steps=args.steps, lr=args.lr, seed=args.seed,
              conditions=tuple(args.conditions.split(",")))


if __name__ == "__main__":
    main()
