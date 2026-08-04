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
    """s -> n pulse angles in (0, pi) (frequency-selective encoding)."""

    def __init__(self, n, hidden=16):
        super().__init__()
        self.n = n
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden, dtype=RDT), torch.nn.Tanh(),
            torch.nn.Linear(hidden, n, dtype=RDT))

    def forward(self, s):
        dev = self.net[0].weight.device
        x = torch.as_tensor(s, dtype=RDT, device=dev).reshape(1, 1)
        return torch.pi * torch.sigmoid(self.net(x)).reshape(-1)


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


def benchmark(sysm, g, device, cdt, n_steps=3, seq_len=8):
    """Time forward+backward and report peak memory — the go/no-go for the campaign."""
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
    for T in (120, 240):
        per_step_s = ms / seq_len * T / 1000
        for steps in (100,):
            print(f"    T={T}, {steps} Adam steps: ~{per_step_s*steps/60:.1f} min/condition "
                  f"({per_step_s:.1f} s/Adam-step)")
    return ms, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["benchmark", "train"], default="benchmark")
    ap.add_argument("--system", default="crotonic9_paper4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-virtual", type=int, default=4)
    ap.add_argument("--dtype", default="complex64", choices=list(CDT_MAP))
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    cdt = CDT_MAP[args.dtype]
    sysm = build_system(args.system, args.n_virtual)
    g = sysm.ensure_diff(device=args.device, cdtype=cdt)
    print(f"system={args.system} N={sysm.n} dim={sysm.dim} | V={g['V']} "
          f"substeps={g['substeps']} K={g['K']} | device={args.device} dtype={args.dtype}")

    if args.mode == "benchmark":
        benchmark(sysm, g, args.device, cdt, seq_len=args.seq_len)
    else:
        raise SystemExit("train mode: wire after benchmark confirms feasibility + wall-clock")


if __name__ == "__main__":
    main()
