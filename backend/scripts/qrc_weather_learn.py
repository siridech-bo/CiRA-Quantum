"""Learned input encoding for the weather QRC (improvement plan step 3).

Trains a per-spin encoding correction end-to-end through the quantum reservoir
on the Delhi weather series, then writes the trained encoder's full FID trace
(persist-waveform rule) and features for ``qrc_weather_bench.py report``.

Reservoir and readout = the Paper-4 weather QRC: crotonic-9, tau = 30 ms,
temperature -> protons, humidity -> carbons (R_x pulses), proton FID (2048 pts,
0.3 ms) -> |FFT| at the 653 bins of the baseline run -> standardized -> ridge.
All of it is differentiable on the exact block propagator.

Encoder: theta_spin(day) = arcsin(sqrt(s_spin)) + 0.5 * tanh(MLP([temp, hum])),
final layer zero-initialised, so step 0 IS the arcsin baseline.

Loss (leakage-free): mean validation NMSE over the selection windows (those
whose validation days precede every test block — the rolling windows overlap
in time) x 14 targets — fit on the first ``600 - n_val`` train days, validate on the last
``n_val`` (300); ridge alpha fixed per (window, target) from the baseline's
validation choice. Test blocks are never read here. The best-validation
encoder is kept.

  python scripts/qrc_weather_learn.py --steps 60 --out-dir runs/weather_learn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.qrc.block_propagator import BlockPropagator  # noqa: E402
from app.qrc.config import SimConfig  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402
from qrc_gen_traces import _resolve_system  # noqa: E402
from qrc_weather_bench import N_T, WIN_NAMES, selection_windows  # noqa: E402
from qrc_weather_fair import ALPHA, HORIZONS, WINDOWS  # noqa: E402

RDT = torch.float64


def _rx(theta, cdt):
    c = torch.cos(theta / 2).to(cdt)
    s = torch.sin(theta / 2).to(cdt)
    eye = torch.eye(2, dtype=cdt, device=theta.device)
    sx = torch.tensor([[0, 1], [1, 0]], dtype=cdt, device=theta.device)
    return c * eye - 1j * s * sx


class Encoder(torch.nn.Module):
    def __init__(self, n, hidden=16, scale=0.5):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, hidden, dtype=RDT), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, n, dtype=RDT))
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.zeros_(self.net[2].bias)
        self.scale = scale

    def forward(self, s2, base):                 # s2 (days, 2), base (days, n) arcsin angles
        return base + self.scale * torch.tanh(self.net(s2))


def targets(w, idx):
    return np.stack([w[idx + h, j] for h in HORIZONS for j in range(2)], axis=1)


class WeatherQRC:
    def __init__(self, a, w, device, cdt):
        self.dev, self.cdt = device, cdt
        sysc = _resolve_system(a.system)
        sim = SimConfig(tau=a.tau, n_virtual=1, evolution_mode="action")
        self.sysm = QRCSystem(sysc, sim)
        self.n = self.sysm.n
        self.bp = BlockPropagator(sysc, sim, device=device, cdtype=cdt, which=("z",))
        lab = [x.upper() for x in sysc.labels]
        self.H = [i for i, x in enumerate(lab) if x.startswith("H")]
        self.C = [i for i, x in enumerate(lab) if x.startswith("C")]
        self.ro = self.bp.fid_readout(self.H, a.fid_points, a.dwell)
        self.vec0 = torch.tensor(self.sysm.qt.operator_to_vector(self.sysm.rho0).full().ravel(),
                                 dtype=cdt, device=device)
        self.n_days = int(sum(WINDOWS["main"]))
        s = np.asarray(w[:self.n_days], float)
        per_spin = np.zeros((self.n_days, self.n))
        per_spin[:, self.H] = s[:, 0:1]
        per_spin[:, self.C] = s[:, 1:2]
        self.s2 = torch.tensor(s, dtype=RDT, device=device)
        self.base = torch.tensor(np.arcsin(np.sqrt(np.clip(per_spin, 0, 1))), dtype=RDT, device=device)
        self.bins = None
        self.seg = a.ckpt_every

    def _segment(self, vec, th):
        rows = []
        for k in range(th.shape[0]):
            _, vec = self.bp.step(vec, [_rx(th[k, q], self.cdt) for q in range(self.n)])
            rows.append(self.bp.fid(vec, self.ro))
        return torch.stack(rows), vec

    def fids(self, theta):
        vec, out = self.vec0, []
        for i in range(0, self.n_days, self.seg):
            th = theta[i:i + self.seg]
            if torch.is_grad_enabled():
                F, vec = checkpoint(self._segment, vec, th, use_reentrant=False)
            else:
                F, vec = self._segment(vec, th)
            out.append(F)
        return torch.cat(out)                                               # (days, fid_points)

    def features(self, fids):
        mag = torch.abs(torch.fft.fft(fids, dim=1))
        if self.bins is None:                    # bins fixed from the baseline's first day
            self.bins = torch.sort(torch.argsort(mag[0].detach(), descending=True)[:653]).values
        return mag[:, self.bins].to(RDT)


def window_blocks(n_val):
    """(fit, val) day indices for the leak-free selection windows only: their
    validation days precede every window's test block."""
    out = []
    for name in [WIN_NAMES[i] for i in selection_windows()]:
        w0, ntr, _ = WINDOWS[name]
        out.append((np.arange(w0, w0 + ntr - n_val), np.arange(w0 + ntr - n_val, w0 + ntr)))
    return out


def val_nmse(X, w, blocks, alpha_idx=None):
    """Per (window, target) validation NMSE; ``alpha_idx`` [win, t] fixed, or None
    to return the full [win, alpha, t] grid (for choosing alpha on the baseline)."""
    res = []
    for wi, (fit, vl) in enumerate(blocks):
        A, B = X[fit], X[vl]
        mu, sd = A.mean(0), A.std(0) + 1e-12
        A = torch.cat([(A - mu) / sd, torch.ones(len(fit), 1, dtype=RDT, device=X.device)], 1)
        B = torch.cat([(B - mu) / sd, torch.ones(len(vl), 1, dtype=RDT, device=X.device)], 1)
        Yf = torch.tensor(targets(w, fit), dtype=RDT, device=X.device)
        Yv = torch.tensor(targets(w, vl), dtype=RDT, device=X.device)
        K, Kv = A @ A.T, B @ A.T
        per_a = []
        for a in ALPHA:
            coef = torch.linalg.solve(K + a * torch.eye(len(fit), dtype=RDT, device=X.device), Yf)
            P = Kv @ coef
            per_a.append(((P - Yv) ** 2).mean(0) / Yv.var(0, unbiased=False))
        per_a = torch.stack(per_a)                                          # [alpha, t]
        res.append(per_a if alpha_idx is None else per_a[alpha_idx[wi], torch.arange(N_T)])
    return torch.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="crotonic9_paper4")
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--fid-points", type=int, default=2048)
    ap.add_argument("--dwell", type=float, default=3e-4)
    ap.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=49)
    ap.add_argument("--dtype", default="complex64", choices=["complex64", "complex128"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    cdt = torch.complex64 if a.dtype == "complex64" else torch.complex128
    w = np.asarray(np.load(a.trace, allow_pickle=True)["weather_norm"], float)
    q = WeatherQRC(a, w, a.device, cdt)
    blocks = window_blocks(a.n_val)
    enc = Encoder(q.n, a.hidden).to(a.device)

    # baseline (arcsin): fixes the FFT bins and the ridge alpha per (window, target)
    t0 = time.time()
    with torch.no_grad():
        X0 = q.features(q.fids(enc(q.s2, q.base)))
        grid = val_nmse(X0, w, blocks)                                       # [win, a, t]
    alpha_idx = torch.argmin(grid, dim=1)                                    # [win, t]
    base_val = float(grid.gather(1, alpha_idx.unsqueeze(1)).mean())
    print(f"baseline arcsin: val NMSE {base_val:.4f} (pooled val R2 {1 - base_val:.3f}); "
          f"forward {time.time() - t0:.0f}s on {a.device}", flush=True)

    opt = torch.optim.Adam(enc.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best = (base_val, {k: v.detach().clone() for k, v in enc.state_dict().items()}, -1)
    log = []
    for it in range(a.steps):
        t1 = time.time()
        opt.zero_grad()
        loss = val_nmse(q.features(q.fids(enc(q.s2, q.base))), w, blocks, alpha_idx).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
        v = float(loss)
        log.append(v)
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in enc.state_dict().items()}, it)
        print(f"step {it:3d}: val NMSE {v:.4f} (best {best[0]:.4f} @ {best[2]}) {time.time() - t1:.0f}s",
              flush=True)
        (out / "train_log.json").write_text(json.dumps({"baseline_val_nmse": base_val, "val_nmse": log,
                                                        "best": best[0], "best_step": best[2]}), encoding="utf-8")

    enc.load_state_dict(best[1])
    with torch.no_grad():
        theta = enc(q.s2, q.base)
        F = q.fids(theta)
        X = q.features(F).cpu().numpy()
    meta = {"experiment": "weather_learned_encoding", "system": a.system, "tau": a.tau, "n_val": a.n_val,
            "steps": a.steps, "lr": a.lr, "hidden": a.hidden, "seed": a.seed, "dtype": a.dtype,
            "baseline_val_nmse": base_val, "best_val_nmse": best[0], "best_step": best[2],
            "regime": "B (trained encoding + ridge readout)", "readout": "FID |FFT| 653 bins (H)"}
    np.savez(out / "trace_learned.npz", fids_H=F.cpu().numpy().astype(np.complex64), theta=theta.cpu().numpy(),
             weather_norm=w, split=np.array(WINDOWS["main"]), task="weather", fid_dwell=a.dwell,
             meta=json.dumps(meta))
    np.savez(out / "X_learned.npz", X=X)
    torch.save(best[1], out / "encoder.pt")
    print(f"best val NMSE {best[0]:.4f} vs baseline {base_val:.4f}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
