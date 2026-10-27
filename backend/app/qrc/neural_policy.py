"""Module B arm (ii): a model-free neural adaptive-estimation policy (Stage 3).

The policy never sees the likelihood or the nuisance. It is trained in simulation only:
episodes draw (theta, nu) from the prior, shots come from ``TableSim`` (noiseless engine
signals on a (theta, nu) grid + tensor Lagrange interpolation + Gaussian shot noise), and
the networks see only the measurement record, summarised by its sufficient statistics
for i.i.d. Gaussian shot noise: per-probe sample means and counts.

  * ``Estimator``  : record -> (theta_hat, nu_hat), trained by weighted squared error.
  * ``PolicyNet``  : record -> logits over the probe bank, trained by REINFORCE on the
                     reward -mean_t log err_t^2 (scale-free across theta), with a
                     privileged critic V(theta, nu) as baseline (simulation only).

Evaluation: ``rollout(..., greedy=True)`` takes the argmax probe.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from app.qrc.adaptive_batch import lagrange_weights


class TableSim:
    """Noiseless signals mu(theta, nu) (P, L) from a (n_th, n_nu, P, L) table."""

    def __init__(self, table, th_grid, nu_grid, sigma, A, device="cuda", k=6,
                 dtype=torch.complex64):
        self.dev = torch.device(device)
        self.T = torch.as_tensor(table, dtype=dtype, device=self.dev)
        self.th0, self.hth = float(th_grid[0]), float(th_grid[1] - th_grid[0])
        self.nu0, self.hnu = float(nu_grid[0]), float(nu_grid[1] - nu_grid[0])
        self.th_lo, self.th_hi = float(th_grid[0]), float(th_grid[-1])
        self.nu_lo, self.nu_hi = float(nu_grid[0]), float(nu_grid[-1])
        self.n_th, self.n_nu, self.P, self.L = self.T.shape
        self.sigma, self.A, self.k = float(sigma), float(A), k

    def mu(self, th, nu, chunk=1024):
        """th, nu (B,) float64 tensors -> (B, P, L)."""
        out = []
        for c in range(0, len(th), chunk):
            t, v = th[c:c + chunk], nu[c:c + chunk]
            it, wt, _ = lagrange_weights(t, self.th0, self.hth, self.n_th, self.k)
            iv, wv, _ = lagrange_weights(v, self.nu0, self.hnu, self.n_nu, self.k)
            acc = torch.zeros(len(t), self.P, self.L, dtype=self.T.dtype, device=self.dev)
            for a in range(self.k):
                for b in range(self.k):
                    w = (wt[:, a] * wv[:, b]).to(self.T.real.dtype)
                    acc += w[:, None, None] * self.T[it[:, a], iv[:, b]]
            out.append(acc)
        return torch.cat(out)

    def sample_prior(self, B, gen):
        th = self.th_lo + (self.th_hi - self.th_lo) * torch.rand(B, generator=gen, device=self.dev, dtype=torch.float64)
        nu = self.nu_lo + (self.nu_hi - self.nu_lo) * torch.rand(B, generator=gen, device=self.dev, dtype=torch.float64)
        return th, nu


def features(S, n, t, N, A):
    """Sufficient statistics -> network input (B, P*(2L+2)+1), float32."""
    nn_ = n.clamp(min=1.0)[..., None]
    ybar = (S / nn_) / A                                              # per-probe sample mean
    x = torch.cat([ybar.real, ybar.imag, (n / N)[..., None], (n > 0).float()[..., None]], -1)
    B = x.shape[0]
    return torch.cat([x.reshape(B, -1).float(),
                      torch.full((B, 1), t / N, device=x.device)], -1)


def mlp(d_in, d_hid, d_out, depth):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, d_hid), nn.LayerNorm(d_hid), nn.SiLU()]
        d = d_hid
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class Estimator(nn.Module):
    def __init__(self, d_in, th_range, nu_range, d_hid=512, depth=4):
        super().__init__()
        self.net = mlp(d_in, d_hid, 2, depth)
        self.register_buffer("c", torch.tensor([sum(th_range) / 2, sum(nu_range) / 2]))
        self.register_buffer("s", torch.tensor([(th_range[1] - th_range[0]) / 2,
                                                (nu_range[1] - nu_range[0]) / 2]))

    def forward(self, x):
        return self.c + self.s * self.net(x)                         # (B, 2): theta, nu


class PolicyNet(nn.Module):
    def __init__(self, d_in, P, d_hid=256, depth=3):
        super().__init__()
        self.net = mlp(d_in, d_hid, P, depth)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """Privileged baseline V(theta, nu) for the policy gradient (training only)."""

    def __init__(self, sim, d_hid=128):
        super().__init__()
        self.net = mlp(2, d_hid, 1, 2)
        self.lo = (sim.th_lo, sim.nu_lo)
        self.hi = (sim.th_hi, sim.nu_hi)

    def forward(self, th, nu):
        x = torch.stack([(th - self.lo[0]) / (self.hi[0] - self.lo[0]),
                         (nu - self.lo[1]) / (self.hi[1] - self.lo[1])], -1).float()
        return self.net(x)[:, 0]


def rollout(sim, est, pol, th, nu, N, gen, mode, mu=None, noise=None):
    """One batch of episodes. mode: "sample" (training), "greedy" (argmax) or
    "fixed-cycle". Returns dict with theta estimates (B, N), chosen (B, N), log-probs
    (B, N) and entropies (B, N). ``mu`` / ``noise`` can be given (evaluation)."""
    B = len(th)
    if mu is None:
        mu = sim.mu(th, nu)
    S = torch.zeros(B, sim.P, sim.L, dtype=mu.dtype, device=sim.dev)
    n = torch.zeros(B, sim.P, device=sim.dev)
    ests, chosen, logps, ents = [], [], [], []
    ar = torch.arange(B, device=sim.dev)
    for t in range(N):
        x = features(S, n, t, N, sim.A)
        if mode == "fixed-cycle":
            a = torch.full((B,), t % sim.P, dtype=torch.long, device=sim.dev)
        else:
            logits = pol(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = logits.argmax(1) if mode == "greedy" else dist.sample()
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
        if noise is None:
            z = torch.complex(torch.randn(B, sim.L, generator=gen, device=sim.dev),
                              torch.randn(B, sim.L, generator=gen, device=sim.dev)).to(mu.dtype)
        else:
            z = noise[:, t]
        S = S.clone()
        S[ar, a] += mu[ar, a] + sim.sigma * z
        n = n.clone()
        n[ar, a] += 1
        ests.append(est(features(S, n, t + 1, N, sim.A)))
        chosen.append(a)
    out = {"est": torch.stack(ests, 1), "chosen": torch.stack(chosen, 1)}
    if logps:
        out["logp"] = torch.stack(logps, 1)
        out["ent"] = torch.stack(ents, 1)
    return out


def est_loss(est_out, th, nu):
    """Per-step MSE, each step normalised by its own (detached) batch MSE, so a late step
    (MSE ~1e-6) weighs as much as an early one (~1e-2): sum_t MSE_t / stopgrad(MSE_t)."""
    e_th = ((est_out[..., 0] - th[:, None].float()) ** 2).mean(0)
    e_nu = ((est_out[..., 1] - nu[:, None].float()) ** 2).mean(0)
    return (e_th / e_th.detach().clamp(min=1e-12)).mean() + 0.1 * (e_nu / e_nu.detach().clamp(min=1e-12)).mean()


def reward(est_out, th, eval_steps, eps=1e-12):
    e = (est_out[:, eval_steps, 0].detach().double() - th[:, None]) ** 2
    return -torch.log(e + eps).mean(1)


def train(sim, N, steps, B, seed, mode="adaptive", lr=3e-4, ent_coef=1e-3, log_every=500,
          est=None, pol=None, log=print):
    """Train the estimator (and, if mode == "adaptive", the policy). Returns est, pol, history."""
    gen = torch.Generator(device=sim.dev).manual_seed(seed)
    torch.manual_seed(seed)
    d_in = sim.P * (2 * sim.L + 2) + 1
    est = est or Estimator(d_in, (sim.th_lo, sim.th_hi), (sim.nu_lo, sim.nu_hi)).to(sim.dev)
    pol = pol or PolicyNet(d_in, sim.P).to(sim.dev)
    crit = Critic(sim).to(sim.dev)
    params = list(est.parameters()) + (list(pol.parameters()) + list(crit.parameters())
                                       if mode == "adaptive" else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    eval_steps = sorted({N // 4 - 1, N // 2 - 1, N - 1})
    hist = []
    for step in range(steps):
        th, nu = sim.sample_prior(B, gen)
        out = rollout(sim, est, pol, th, nu, N, gen, "sample" if mode == "adaptive" else "fixed-cycle")
        loss = est_loss(out["est"], th, nu)
        if mode == "adaptive":
            R = reward(out["est"], th, eval_steps)
            V = crit(th, nu)
            adv = (R - V.detach().double()).float()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = loss + (-(adv[:, None] * out["logp"]).mean() - ent_coef * out["ent"].mean()
                           + 1e-2 * ((V - R.float()) ** 2).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                rmse = float(((out["est"][:, -1, 0] - th.float()) ** 2).mean().sqrt())
                h = {"step": step, "loss": float(loss), "rmse_final": rmse}
                if mode == "adaptive":
                    h["entropy"] = float(out["ent"].mean())
                    h["n_distinct"] = int(torch.unique(out["chosen"]).numel())
            hist.append(h)
            log(h)
    return est, pol, hist
