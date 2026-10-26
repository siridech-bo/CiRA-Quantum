"""Parameter-shift-rule (PSR) *training loop* — the hardware recipe, end to end.

qrc_psr_sim.py showed the PSR Jacobian equals autograd. This script closes the
remaining gap: it TRAINS a learnable per-spin encoding with gradients obtained
only from forward runs at shifted pulse angles (what an NMR spectrometer can do),
next to an identical autograd arm, on real data (Delhi weather, next-day T + RH).

Methods demonstration, not a performance claim (no pre-registered comparison).

Encoding (per spin i, day t; p = 0 is exactly the frozen design g = 0.35):
    theta[t, i] = 0.35*arcsin(sqrt(s[t, c(i)])) + p[i,0] + p[i,1]*(T_t-.5) + p[i,2]*(H_t-.5)
    c(i): temperature -> upper half of the spins, humidity -> lower half (gen_traces layout).
Readout: <sigma_x,y,z> per spin at V virtual nodes (NON-STANDARD readout, D = 3*n*V
features; small dense systems only) -> standardised -> ridge (alpha fixed) fit on the
train days -> loss = mean validation NMSE over the 2 targets.

Gradient, PSR arm (no autograd through the quantum part):
    G = dL/dF                                   classical (ridge + loss), autograd OK
    dL/dtheta[t,i] = 1/2 [G.F(theta + pi/2 e_ti) - G.F(theta - pi/2 e_ti)]   (2-term, exact:
                     each R_x(theta_i) has a 2-eigenvalue generator)
    dL/dp = sum_t,i dL/dtheta[t,i] * dtheta[t,i]/dp                         classical
All 2*n*T shifted runs are simulated in one batched lock-step pass; on hardware they are
2*n*T separate experiments per gradient step (reported as `experiments_per_step`).

Persist rule: the reservoir features of the initial and the trained encoders (both
arms) are saved to artifacts/traces/qrc_psr_train_<n>spin.npz before any metric.

  python scripts/qrc_psr_train.py --system 4 --steps 30
  python scripts/qrc_psr_train.py --system 6 --device cuda --steps 20
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

from qrc_gen_traces import _weather_indices  # noqa: E402
from qrc_learnable_9spin import build_system, per_spin_pulse  # noqa: E402

CDT = torch.complex128
RDT = torch.float64
GAIN = 0.35


def load_data(trace, start, n_wash, n_train, n_val, n_test):
    w = np.asarray(np.load(trace, allow_pickle=True)["weather_norm"], float)
    T = n_wash + n_train + n_val + n_test + 1
    w = w[start:start + T]
    s = w[:-1]                                  # inputs, day t
    y = w[1:]                                   # targets, day t+1 (T, RH)
    idx = np.arange(len(s))
    tr = idx[n_wash:n_wash + n_train]
    va = idx[n_wash + n_train:n_wash + n_train + n_val]
    te = idx[n_wash + n_train + n_val:]
    return s, y, tr, va, te


class Reservoir:
    """Dense exact-propagator reservoir, batched over trajectories (≤ 6 spins)."""

    def __init__(self, n, n_virtual, device):
        self.sysm = build_system(str(n), n_virtual, tau=0.03)
        self.g = self.sysm.ensure_diff(device=device, cdtype=CDT, dense=True, block=False)
        assert self.g.get("P") is not None, "dense propagator required"
        self.n, self.d, self.V = self.sysm.n, self.sysm.dim, self.g["V"]
        self.P, self.M = self.g["P"], self.g["Mdense"]
        self.device = device
        rho0 = self.sysm.qt.basis(self.d, 0) * self.sysm.qt.basis(self.d, 0).dag()
        self.vec0 = torch.tensor(self.sysm.qt.operator_to_vector(rho0).full().ravel(),
                                 dtype=CDT, device=device)
        self.nf = self.M.shape[0] * self.V

    def pulse(self, X, U):
        """X (B, d^2) column-stacked states; U (B or 1, d, d)."""
        B, d = X.shape[0], self.d
        rho = X.reshape(B, d, d).transpose(1, 2)
        rho = U @ rho @ U.conj().transpose(-1, -2)
        return rho.transpose(1, 2).reshape(B, -1)

    def evolve(self, X):
        """Free evolution; returns (features (B, nf), X_end). Feature order = step_diff."""
        nodes = []
        for _ in range(self.V):
            X = X @ self.P.T
            nodes.append(X)
        Vm = torch.stack(nodes, dim=1)                      # (B, V, d^2)
        return (Vm @ self.M.T).real.reshape(X.shape[0], -1), X

    def forward(self, theta):
        """theta (T, n) -> features (T, nf); differentiable in theta."""
        x = self.vec0[None]
        rows = []
        for t in range(theta.shape[0]):
            x = self.pulse(x, per_spin_pulse(theta[t], self.n, CDT)[None])
            f, x = self.evolve(x)
            rows.append(f[0])
        return torch.stack(rows)

    @torch.no_grad()
    def psr_vjp(self, theta, G):
        """dL/dtheta (T, n) by the 2-term shift rule — forward runs only.

        Branch (k, i, +/-) equals the base trajectory before step k, so its features
        before k cancel in the +/- difference; only steps >= k are accumulated."""
        T, n = theta.shape
        shifts = torch.tensor([np.pi / 2, -np.pi / 2], dtype=RDT, device=self.device)
        x = self.vec0[None]
        X = torch.zeros(0, self.d ** 2, dtype=CDT, device=self.device)
        acc = torch.zeros(0, dtype=RDT, device=self.device)
        for t in range(T):
            Ub = per_spin_pulse(theta[t], n, CDT)
            Us = []                                         # 2n new branches at step t
            for i in range(n):
                for sh in shifts:
                    th = theta[t].clone()
                    th[i] = th[i] + sh
                    Us.append(per_spin_pulse(th, n, CDT))
            Xnew = self.pulse(x.expand(2 * n, -1), torch.stack(Us))
            Xold = self.pulse(X, Ub[None]) if X.shape[0] else X
            X = torch.cat([Xold, Xnew])
            acc = torch.cat([acc, torch.zeros(2 * n, dtype=RDT, device=self.device)])
            f, X = self.evolve(X)
            acc = acc + f @ G[t]
            _, x = self.evolve(self.pulse(x, Ub[None]))     # advance the base state
        acc = acc.reshape(T, n, 2)
        return 0.5 * (acc[..., 0] - acc[..., 1])


def thetas(p, s, cmap):
    base = GAIN * torch.arcsin(torch.sqrt(s[:, cmap].clamp(0, 1)))       # (T, n)
    lin = p[None, :, 0] + p[None, :, 1] * (s[:, :1] - 0.5) + p[None, :, 2] * (s[:, 1:2] - 0.5)
    return base + lin


def ridge_loss(F, y, tr, ev, alpha):
    """Standardise on train, ridge (+bias) on train, NMSE on `ev`, mean over targets."""
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-8
    Z = (F - mu) / sd
    Z = torch.cat([Z, torch.ones(Z.shape[0], 1, dtype=RDT, device=Z.device)], 1)
    A = Z[tr].T @ Z[tr] + alpha * torch.eye(Z.shape[1], dtype=RDT, device=Z.device)
    W = torch.linalg.solve(A, Z[tr].T @ y[tr])
    err = ((Z[ev] @ W - y[ev]) ** 2).mean(0) / y[ev].var(0)
    return err.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="4")
    ap.add_argument("--n-virtual", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trace", default="artifacts/traces/weather_full.npz")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--splits", type=int, nargs=4, default=[30, 300, 100, 100],
                    metavar=("WASH", "TRAIN", "VAL", "TEST"))
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--out", default="../docs/QRC/data/qrc_psr_train")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = a.device

    res = Reservoir(int(a.system), a.n_virtual, dev)
    n = res.n
    s_np, y_np, tr, va, te = load_data(a.trace, a.start, *a.splits)
    s = torch.tensor(s_np, dtype=RDT, device=dev)
    y = torch.tensor(y_np, dtype=RDT, device=dev)
    temp_q, hum_q = _weather_indices(n)
    cmap = torch.tensor([0 if i in temp_q else 1 for i in range(n)], device=dev)
    T = len(s_np)
    print(f"system N={n} dim={res.d} V={res.V} D={res.nf} (observable readout, non-standard) "
          f"| T={T} days | params={3 * n} | PSR experiments/step = {2 * n * T}")

    def classical_grad(F):
        Fg = F.detach().requires_grad_(True)
        L = ridge_loss(Fg, y, tr, va, a.alpha)
        (G,) = torch.autograd.grad(L, Fg)
        return float(L.detach()), G

    def grad_psr(p):
        th = thetas(p, s, cmap)
        F = res.forward(th.detach())
        L, G = classical_grad(F)
        dth = res.psr_vjp(th.detach(), G)                 # quantum part: shifted runs only
        pg = p.detach().requires_grad_(True)
        (gp,) = torch.autograd.grad(thetas(pg, s, cmap), pg, grad_outputs=dth)  # classical chain
        return L, gp, F.detach()

    def grad_ag(p):
        pg = p.detach().requires_grad_(True)
        F = res.forward(thetas(pg, s, cmap))
        L = ridge_loss(F, y, tr, va, a.alpha)
        (gp,) = torch.autograd.grad(L, pg)
        return float(L.detach()) if torch.is_tensor(L) else L, gp, F.detach()

    arms = {"psr": grad_psr, "autograd": grad_ag}
    p = {k: torch.zeros(n, 3, dtype=RDT, device=dev) for k in arms}
    opt = {k: torch.optim.Adam([p[k].requires_grad_(True)], lr=a.lr) for k in arms}
    hist = {k: [] for k in arms}
    gap, feats0 = [], None
    t0 = time.time()
    for step in range(a.steps + 1):
        out = {}
        for k, fn in arms.items():
            tk = time.time()
            L, gp, F = fn(p[k])
            out[k] = (L, gp, time.time() - tk)
            if step == 0 and k == "psr":
                feats0 = F.cpu().numpy()
            hist[k].append(L)
        rel = float((out["psr"][1] - out["autograd"][1]).abs().max()
                    / out["autograd"][1].abs().max().clamp_min(1e-30))
        dp = float((p["psr"] - p["autograd"]).detach().abs().max())
        gap.append({"step": step, "grad_relerr": rel, "param_maxdiff": dp})
        print(f"step {step:3d}  val NMSE psr={out['psr'][0]:.5f} ag={out['autograd'][0]:.5f}"
              f"  |grad rel.err|={rel:.1e}  |dp|={dp:.1e}"
              f"  t_psr={out['psr'][2]:.1f}s t_ag={out['autograd'][2]:.1f}s", flush=True)
        if step == a.steps:
            break
        for k in arms:
            p[k].grad = out[k][1].clone()
            opt[k].step()

    with torch.no_grad():
        F_fin = {k: res.forward(thetas(p[k], s, cmap)).cpu().numpy() for k in arms}
    trace = Path("artifacts/traces") / f"qrc_psr_train_{n}spin.npz"
    trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez(trace, feats_init=feats0, feats_psr=F_fin["psr"], feats_autograd=F_fin["autograd"],
             p_psr=p["psr"].detach().cpu().numpy(), p_autograd=p["autograd"].detach().cpu().numpy(),
             weather_norm=s_np, target=y_np, split=np.array(a.splits), start=a.start, seed=0,
             horizons=np.array([1]),
             meta=json.dumps({"system": a.system, "n_virtual": a.n_virtual, "tau": 0.03,
                              "gain": GAIN, "readout": f"observables xyz x V (D={res.nf})",
                              "alpha": a.alpha, "lr": a.lr, "steps": a.steps}))

    def score(F, idx):
        return float(ridge_loss(torch.tensor(F, dtype=RDT, device=dev), y, tr, idx, a.alpha))

    summary = {
        "what": "PSR-driven training loop vs autograd, identical init (p=0 = g0.35 design)",
        "system": f"{n}-spin (preset {a.system})", "D": res.nf,
        "readout": "observables <x,y,z> x V nodes (non-standard; not the 653-bin FID)",
        "data": f"Delhi weather_norm days {a.start}..{a.start + T}, splits {a.splits}",
        "experiments_per_step": 2 * n * T, "params": 3 * n, "steps": a.steps,
        "val_nmse": {k: hist[k] for k in arms},
        "max_grad_relerr": max(g["grad_relerr"] for g in gap),
        "final_param_maxdiff": gap[-1]["param_maxdiff"],
        "test_nmse": {"init": score(feats0, te),
                      **{k: score(F_fin[k], te) for k in arms}},
        "trace": str(trace), "wall_s": time.time() - t0, "gap": gap,
    }
    out = Path(f"{a.out}_{n}spin.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"max grad rel.err over run = {summary['max_grad_relerr']:.1e}; "
          f"final |p_psr - p_ag| = {summary['final_param_maxdiff']:.1e}")
    print(f"test NMSE init={summary['test_nmse']['init']:.4f} psr={summary['test_nmse']['psr']:.4f} "
          f"ag={summary['test_nmse']['autograd']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
