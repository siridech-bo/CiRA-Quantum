"""Fair classical analogue of the weather QRC's learned encoding: an ESN whose
input stage is trained by gradient descent with the SAME loss, data and budget.

The learned-encoding QRC (qrc_weather_learn.py) trains an input map in front of
a fixed reservoir. A reviewer's fair question: does a classical reservoir with
an equally trainable input stage close the gap? Two variants, both starting
from the tuned ESN chosen on the leak-free validation block (step 0 == that ESN):

  win  — the ESN input weights W_in (m x [bias, temp, hum]) are trained directly
         (3m parameters: more than the QRC encoder, so generous to the ESN)
  mlp  — a 2->16->2 residual MLP on (temp, hum) feeds the fixed W_in, zero-init
         (82 parameters ~ the QRC's 2->16->9 encoder, 201 parameters)

Loss = qrc_weather_learn.val_nmse on the leak-free selection windows (fit days
374-673, validate 674-973; every test day is >= 974), ridge alpha fixed per
target from the step-0 ESN, 40 Adam steps. Writes the trained ESN's states as
``X`` (rows = days) for evaluation with qrc_weather_pool.py.

  python scripts/qrc_weather_esn_learn.py --variant win --init 0 --out X.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qrc_weather_bench import select_esn  # noqa: E402
from qrc_weather_fair import WINDOWS, _power_radius  # noqa: E402
from qrc_weather_learn import val_nmse, window_blocks  # noqa: E402

RDT = torch.float64


def esn_mats(m, init):
    """The exact W / W_in of qrc_weather_fair.esn_states (same RNG stream)."""
    rng = np.random.default_rng(10_000 * init + m)
    W = rng.standard_normal((m, m))
    W /= np.max(np.abs(np.linalg.eigvals(W))) if m <= 1000 else _power_radius(W, rng)
    Win = rng.uniform(-1, 1, (m, 3))
    return W, Win


class ESN(torch.nn.Module):
    def __init__(self, cfg, init, variant, device):
        super().__init__()
        m = int(cfg["size"])
        W, Win = esn_mats(m, init)
        self.W = torch.tensor(cfg["sr"] * W, dtype=RDT, device=device)
        win = torch.tensor(cfg["inscale"] * Win, dtype=RDT, device=device)
        self.leak = float(cfg["leak"])
        self.variant = variant
        if variant == "win":
            self.Win = torch.nn.Parameter(win)
        else:
            self.register_buffer("Win", win)
            self.enc = torch.nn.Sequential(torch.nn.Linear(2, 16, dtype=RDT), torch.nn.Tanh(),
                                           torch.nn.Linear(16, 2, dtype=RDT))
            torch.nn.init.zeros_(self.enc[2].weight)
            torch.nn.init.zeros_(self.enc[2].bias)

    def forward(self, u):                                   # u (days, 2) -> states (days, m)
        if self.variant == "mlp":
            u = u + self.enc(u)
        drive = self.Win[:, 0] + u @ self.Win[:, 1:].T       # (days, m)
        x = torch.zeros(self.W.shape[0], dtype=RDT, device=u.device)
        out = []
        for k in range(u.shape[0]):
            x = (1 - self.leak) * x + self.leak * torch.tanh(drive[k] + self.W @ x)
            out.append(x)
        return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["win", "mlp"], required=True)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--esn-grid", default="artifacts/qrc_runs/weather-bench-20261021/esn_grid_v300.npz")
    ap.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.init)

    w = np.asarray(np.load(a.trace, allow_pickle=True)["weather_norm"], float)
    n_days = int(sum(WINDOWS["main"]))
    cfg = select_esn(dict(np.load(a.esn_grid, allow_pickle=True)))["cfg"]   # leak-free selection
    u = torch.tensor(w[:n_days], dtype=RDT, device=a.device)
    esn = ESN(cfg, a.init, a.variant, a.device)
    blocks = window_blocks(a.n_val)
    n_par = sum(p.numel() for p in esn.parameters() if p.requires_grad)

    with torch.no_grad():
        grid = val_nmse(esn(u), w, blocks)
    alpha_idx = torch.argmin(grid, dim=1)
    base = float(grid.gather(1, alpha_idx.unsqueeze(1)).mean())
    print(f"ESN cfg={cfg} init={a.init} variant={a.variant} trainable={n_par}: "
          f"step-0 val NMSE {base:.4f}", flush=True)

    opt = torch.optim.Adam([p for p in esn.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best = (base, {k: v.detach().clone() for k, v in esn.state_dict().items()}, -1)
    log = []
    for it in range(a.steps):
        t0 = time.time()
        opt.zero_grad()
        loss = val_nmse(esn(u), w, blocks, alpha_idx).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in esn.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        v = loss.item()
        log.append(v)
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in esn.state_dict().items()}, it)
        print(f"step {it:3d}: val NMSE {v:.4f} (best {best[0]:.4f} @ {best[2]}) {time.time() - t0:.1f}s",
              flush=True)
    esn.load_state_dict(best[1])
    with torch.no_grad():
        X = esn(u).cpu().numpy()
    np.savez(a.out, X=X, meta=json.dumps({"cfg": cfg, "init": a.init, "variant": a.variant,
                                           "trainable": n_par, "step0_val_nmse": base,
                                           "best_val_nmse": best[0], "best_step": best[2],
                                           "val_curve": log}))
    print(f"best val NMSE {best[0]:.4f} vs step-0 {base:.4f}; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
