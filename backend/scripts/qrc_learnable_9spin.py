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


def build_system(name, n_virtual, coupling_scale=1.0):
    sysc = _resolve_system(name)
    if coupling_scale != 1.0:                       # scale the qubit couplings
        from app.qrc.config import SystemConfig
        sysc = SystemConfig(
            n_qubits=sysc.n_qubits, chemical_shifts=list(sysc.chemical_shifts),
            j_coupling=(np.asarray(sysc.j_coupling) * coupling_scale).tolist(),
            t1=list(sysc.t1), t2=list(sysc.t2), labels=list(sysc.labels))
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
    """input (dim `input_dim`) -> n_out pulse angles in (0, pi). n_out=n gives
    per-spin (frequency-selective) encoding; n_out=1 gives a single global angle
    (broadcast to all spins). input_dim=1 is the scalar-per-step encoder;
    input_dim=k feeds a sliding window of the last k inputs (strategy B) — a
    fully-connected weight over the input window."""

    def __init__(self, n_out, hidden=16, input_dim=1):
        super().__init__()
        self.n_out = n_out
        self.input_dim = input_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden, dtype=RDT), torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_out, dtype=RDT))

    def forward(self, s):
        dev = self.net[0].weight.device
        x = torch.as_tensor(s, dtype=RDT, device=dev).reshape(1, -1)   # (1, input_dim)
        return torch.pi * torch.sigmoid(self.net(x)).reshape(-1)


def enc_angles(enc, s, n):
    """Per-spin angles for one input: (n,) for per-spin encoders, broadcast for global."""
    a = enc(s)
    return a if a.shape[0] == n else a.expand(n)


def sliding_windows(u, k):
    """Sliding windows [u[t], u[t-1], ..., u[t-k+1]] (zero-padded at the start),
    shape (T, k). k=1 reduces to the scalar-per-step input."""
    T = len(u)
    W = np.zeros((T, k), dtype=float)
    for t in range(T):
        for j in range(k):
            if t - j >= 0:
                W[t, j] = u[t - j]
    return W


def init_state(sysm, device, cdt):
    rho0 = sysm.qt.basis(sysm.dim, 0) * sysm.qt.basis(sysm.dim, 0).dag()
    np_dt = np.complex64 if cdt == torch.complex64 else np.complex128
    return torch.tensor(sysm.qt.operator_to_vector(rho0).full().ravel().astype(np_dt),
                        device=device)


def forward_features(angles_seq, sysm, g, device, cdt, reset_each_step=False):
    """angles_seq: list of angle-vectors (one per step) -> feature matrix.

    ``reset_each_step`` re-initializes the reservoir state before each step,
    DISABLING the cross-step quantum memory — the τ→0 ablation control that
    proves the quantum reservoir (not a classical crutch) carries the memory."""
    vec0 = init_state(sysm, device, cdt)
    vec = vec0
    rows = []
    for th in angles_seq:
        U = per_spin_pulse(th, sysm.n, cdt)
        feat, vec = sysm.step_diff(vec, U, g)
        rows.append(feat)
        if reset_each_step:
            vec = vec0
    return torch.stack(rows)


class ClosedLoopController(torch.nn.Module):
    """Memoryless feedback controller for the quantum-memory RNN (strategy 3):
    [s_t, f_{t-1}] -> n per-spin angles in (0, pi), where f_{t-1} is the readout
    of the quantum reservoir at the previous step. The controller has NO recurrent
    state of its own — the intended sequence memory lives in the quantum state ρ.

    NOTE: feeding back f_{t-1} does create a *classical* recurrence through the
    feature vector (f_t = G(f_{t-1}, s_t)); the open-loop and τ→0 ablations isolate
    what the quantum reservoir actually contributes."""

    def __init__(self, n, feat_dim, hidden=32):
        super().__init__()
        self.n = n
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1 + feat_dim, hidden, dtype=RDT), torch.nn.Tanh(),
            torch.nn.Linear(hidden, n, dtype=RDT))

    def forward(self, s_scalar, f_prev):
        dev = self.net[0].weight.device
        x = torch.cat([torch.as_tensor(s_scalar, dtype=RDT, device=dev).reshape(1),
                       f_prev.reshape(-1)]).reshape(1, -1)
        return torch.pi * torch.sigmoid(self.net(x)).reshape(-1)


def forward_features_closedloop(ctrl, u, sysm, g, device, cdt, reset_each_step=False):
    """Closed-loop quantum-memory RNN forward: θ_t = ctrl(s_t, f_{t-1}); the pulse
    drives the reservoir, whose readout f_t is fed back next step. Memory is meant
    to live in ρ; ``reset_each_step`` (τ→0) leaves only the classical f-feedback."""
    vec0 = init_state(sysm, device, cdt)
    vec = vec0
    feat_dim = 3 * sysm.n * g["V"]
    f_prev = torch.zeros(feat_dim, dtype=RDT, device=device)
    rows = []
    for s in u:
        theta = ctrl(float(s), f_prev)
        U = per_spin_pulse(theta, sysm.n, cdt)
        feat, vec = sysm.step_diff(vec, U, g)
        rows.append(feat)
        f_prev = feat                    # feedback (keeps the BPTT graph)
        if reset_each_step:
            vec = vec0
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
    if task in ("narma2", "narma10"):
        from app.qrc.tasks import narma_sequence  # u~U[0,0.5]
        order = 2 if task == "narma2" else 10       # narma10 = memory-bound (strategy-B test)
        return narma_sequence(T, order=order, seed=seed)
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
          lr=0.02, seed=7, conditions=("arcsin", "perspin"), window=1, no_memory=False):
    """Learned encoding vs global arcsin(sqrt) baseline, leakage-free 3-way split,
    with under-powered/degenerate guardrails. conditions: 'global' (learned scalar
    angle) / 'perspin' (learned per-spin angles). window>1 feeds a sliding window
    of the last `window` inputs to a fully-connected encoder (strategy B).
    no_memory disables the cross-step quantum memory (the tau->0 ablation)."""
    u, y = make_task(task, T, seed)
    y_use = y[washout:]
    tr, va, te = _three_way(len(y_use))
    n_tr, n_va, n_te = tr.stop - tr.start, va.stop - va.start, te.stop - te.start
    n_obs = g["Mdense"].shape[0] if g.get("Mdense") is not None else 3 * sysm.n
    nfeat = n_obs * g["V"] + 1
    print(f"task={task} T={T} usable={len(y_use)} tr/va/te={n_tr}/{n_va}/{n_te} "
          f"n_feat={nfeat} (n_train>2*n_feat: {n_tr > 2 * nfeat}) | window={window} "
          f"quantum-memory={'OFF (ablation)' if no_memory else 'ON'}")

    # encoder inputs: scalars (window=1) or sliding windows (window=k)
    Uwin = sliding_windows(u, window) if window > 1 else None
    enc_inputs = [Uwin[t] for t in range(len(u))] if window > 1 else list(u)
    input_dim = window

    def feats(angle_fn, inputs):
        return forward_features([angle_fn(x) for x in inputs], sysm, g, device, cdt,
                                reset_each_step=no_memory)[washout:]

    # (A) arcsin(sqrt) baseline: raw scalar input, standard QRC (with memory)
    def arcsin_ang(s):
        v = float(np.ravel(s)[0])
        a = float(np.arcsin(np.sqrt(min(max(v, 0.0), 1.0))))
        return torch.full((sysm.n,), a, dtype=RDT, device=device)

    with torch.no_grad():
        _, base_te = _ridge_fit_eval(feats(arcsin_ang, list(u)), y_use, device)
    base = float(base_te)
    print(f"(A) arcsin baseline: test NMSE = {base:.4f}")

    smax = float(np.max(u))
    sg = np.linspace(0.0, smax, 21)
    results = {"arcsin": (base, None, None, None)}   # (test, val_curve, std, angle_map)

    def run_learned(label, n_out):
        torch.manual_seed(seed)          # vary encoder init with the run seed
        enc = EncoderPerSpin(n_out, input_dim=input_dim).to(device)
        opt = torch.optim.Adam(enc.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        curve = []
        best_val, best_state = float("inf"), None   # best-validation checkpoint
        t0 = time.time()
        for it in range(steps):
            opt.zero_grad()
            nmse_val, _ = _ridge_fit_eval(feats(lambda x: enc_angles(enc, x, sysm.n), enc_inputs), y_use, device)
            nmse_val.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)   # stabilize
            opt.step(); sched.step()
            v = float(nmse_val)
            curve.append(v)
            if v < best_val:                         # keep the best-val encoder
                best_val = v
                best_state = {k: t.detach().clone() for k, t in enc.state_dict().items()}
            if it % 20 == 0 or it == steps - 1:
                print(f"  [{label}] step {it:3d}: val NMSE={v:.4f}")
        if best_state is not None:                   # report test at best-val, not final
            enc.load_state_dict(best_state)
        dt = time.time() - t0
        with torch.no_grad():
            _, nmse_test = _ridge_fit_eval(feats(lambda x: enc_angles(enc, x, sysm.n), enc_inputs), y_use, device)
            # degeneracy std over the actual encoder inputs (any input dim)
            std = float(torch.stack([enc_angles(enc, x, sysm.n) for x in enc_inputs]).cpu().numpy().std())
            amap = (torch.stack([enc_angles(enc, float(s), sysm.n) for s in sg]).cpu().numpy()
                    if window == 1 else None)
        fin = float(nmse_test)
        print(f"({label}) learned: test NMSE={fin:.4f} (best val {best_val:.4f}) "
              f"| enc std={std:.2f} | {dt:.0f}s ({dt/steps:.1f}s/step)")
        results[label] = (fin, curve, std, amap)

    def run_closedloop(label):
        """Closed-loop quantum-memory RNN: a memoryless feedback controller."""
        torch.manual_seed(seed)
        feat_dim = 3 * sysm.n * g["V"]
        ctrl = ClosedLoopController(sysm.n, feat_dim).to(device)
        opt = torch.optim.Adam(ctrl.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        curve, best_val, best_state = [], float("inf"), None
        t0 = time.time()
        for it in range(steps):
            opt.zero_grad()
            F = forward_features_closedloop(ctrl, u, sysm, g, device, cdt,
                                            reset_each_step=no_memory)[washout:]
            nmse_val, _ = _ridge_fit_eval(F, y_use, device)
            nmse_val.backward()
            torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
            opt.step(); sched.step()
            v = float(nmse_val); curve.append(v)
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in ctrl.state_dict().items()}
            if it % 20 == 0 or it == steps - 1:
                print(f"  [{label}] step {it:3d}: val NMSE={v:.4f}")
        if best_state is not None:
            ctrl.load_state_dict(best_state)
        dt = time.time() - t0
        with torch.no_grad():
            F = forward_features_closedloop(ctrl, u, sysm, g, device, cdt,
                                            reset_each_step=no_memory)[washout:]
            _, nmse_test = _ridge_fit_eval(F, y_use, device)
        fin = float(nmse_test)
        print(f"({label}) closed-loop: test NMSE={fin:.4f} (best val {best_val:.4f}) "
              f"| {dt:.0f}s ({dt/steps:.1f}s/step)")
        results[label] = (fin, curve, 1.0, None)   # std=1.0 placeholder (not degenerate)

    if "global" in conditions:
        run_learned("global", 1)
    if "perspin" in conditions:
        run_learned("perspin", sysm.n)
    if "closedloop" in conditions:
        run_closedloop("closedloop")

    learned = [k for k in ("global", "perspin", "closedloop")
               if results.get(k, (None, None))[1] is not None]
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
    ap.add_argument("--task", default="narma2", choices=["narma2", "narma10", "synthetic"])
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--washout", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02)   # lower + grad-clip + best-val = stabler
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conditions", default="arcsin,perspin",
                    help="comma list from: arcsin,global,perspin")
    ap.add_argument("--window", type=int, default=1,
                    help="encoder input = sliding window of last k inputs (strategy B)")
    ap.add_argument("--no-memory", action="store_true",
                    help="disable cross-step quantum memory (tau->0 ablation)")
    ap.add_argument("--coupling-scale", type=float, default=1.0,
                    help="scale the qubit J-couplings (2.0 = strong coupling)")
    ap.add_argument("--correlation", action="store_true",
                    help="use 2-body correlation readout instead of single-qubit")
    ap.add_argument("--run-dir", default=None, help="progress/events dir (launcher-managed)")
    ap.add_argument("--out", default=None, help="results JSON path (launcher-managed)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    cdt = CDT_MAP[args.dtype]
    sysm = build_system(args.system, args.n_virtual, coupling_scale=args.coupling_scale)
    g = sysm.ensure_diff(device=args.device, cdtype=cdt)
    if args.correlation:                             # swap in 2-body correlation readout
        from qrc_correlation_readout import build_readout
        M, n_obs, _ = build_readout(sysm, True)
        g["Mdense"] = M.to(device=args.device, dtype=cdt)
        print(f"correlation readout: {n_obs} observables (was {3*sysm.n})")
    print(f"system={args.system} N={sysm.n} dim={sysm.dim} | V={g['V']} "
          f"substeps={g['substeps']} K={g['K']} coupling={args.coupling_scale}x "
          f"corr={args.correlation} | device={args.device} dtype={args.dtype}")

    if args.mode == "benchmark":
        benchmark(sysm, g, args.device, cdt, seq_len=args.seq_len, baseline_T=args.T)
    else:
        # Optional launcher plumbing: a run-dir gives live progress + a results
        # JSON so the run is visible/comparable in the QRC dashboard.
        log = None
        if args.run_dir:
            from qrc_progress import ProgressLogger
            log = ProgressLogger(run_dir=args.run_dir,
                                 title=f"QRC learnable encoding ({args.task}, {args.system})")
            log.event("config", f"system={args.system} N={sysm.n} task={args.task} "
                      f"T={args.T} steps={args.steps} conditions={args.conditions} "
                      f"coupling={args.coupling_scale}x correlation={args.correlation}")
        try:
            results, base, verdict = train(
                sysm, g, args.device, cdt, task=args.task, T=args.T,
                washout=args.washout, steps=args.steps, lr=args.lr, seed=args.seed,
                conditions=tuple(args.conditions.split(",")),
                window=args.window, no_memory=args.no_memory)
            if args.run_dir or args.out:
                import json
                payload = {
                    "experiment": "learnable_encoding",
                    "regime": "B (trained readout + trained encoding)",
                    "readout": "correlation" if args.correlation else "observable",
                    "system": args.system, "task": args.task, "n_spins": sysm.n,
                    "coupling_scale": args.coupling_scale, "T": args.T, "steps": args.steps,
                    "quantum_memory": not args.no_memory,
                    "arcsin_baseline_test_nmse": float(base),
                    "verdict": verdict,
                    "conditions": {k: {"test_nmse": float(v[0]), "enc_std": (float(v[2]) if v[2] is not None else None)}
                                   for k, v in results.items()},
                }
                out = args.out or str(Path(args.run_dir) / "results.json")
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
                if log:
                    log.event("results", f"verdict: {verdict}")
                    log.close("done")
        except Exception as e:  # noqa: BLE001
            if log:
                log.event("error", f"{type(e).__name__}: {e}")
                log.close("error")
            raise


if __name__ == "__main__":
    main()
