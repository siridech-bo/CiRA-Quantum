"""Module B, batched: many adaptive episodes at once in torch (GPU), for one known model on a
tabulated theta grid. Same algorithm as ``adaptive.GridPosterior`` + ``run_episode`` (v2,
zoom-on-underresolved), vectorised over episodes; the shot noise is passed in, so it can be
drawn with the per-episode numpy seeds of the CPU runner (paired / reproducible).

Model: ``table`` (n_tab, P, L) complex signals on the uniform grid ``tab_grid``; mu(theta)
and dmu/dtheta come from a k-point Lagrange interpolant of the table (as the CPU runner).
"""
from __future__ import annotations

import numpy as np
import torch


def lagrange_weights(theta, g0, h, n_tab, k=6):
    """theta (...,) -> idx (..., k) long, w (..., k), dw (..., k): k-point Lagrange
    interpolation weights and their theta-derivative on the uniform grid g0 + h*i."""
    c = torch.clamp(torch.floor((theta - g0) / h).long() - k // 2 + 1, 0, n_tab - k)
    idx = c[..., None] + torch.arange(k, device=theta.device)
    x = g0 + h * idx.to(theta.dtype)
    d = theta[..., None] - x
    w = torch.ones_like(d)
    dw = torch.zeros_like(d)
    for i in range(k):
        den = torch.ones_like(theta)
        for j in range(k):
            if j != i:
                den = den * (x[..., i] - x[..., j])
                w[..., i] = w[..., i] * d[..., j]
        for l in range(k):
            if l == i:
                continue
            t = torch.ones_like(theta)
            for j in range(k):
                if j not in (i, l):
                    t = t * d[..., j]
            dw[..., i] = dw[..., i] + t
        w[..., i] = w[..., i] / den
        dw[..., i] = dw[..., i] / den
    return idx, w, dw


class BatchedEpisodes:
    """B independent grid posteriors (uniform prior on [lo, hi]) sharing one model."""

    def __init__(self, table, tab_grid, sigma, lo, hi, n_grid=201, width=10.0, k=6,
                 device="cuda"):
        self.dev = torch.device(device)
        self.T = torch.as_tensor(table, dtype=torch.complex128, device=self.dev)
        self.g0, self.h = float(tab_grid[0]), float(tab_grid[1] - tab_grid[0])
        self.n_tab, self.P, self.L = self.T.shape
        self.sigma, self.lo, self.hi = float(sigma), float(lo), float(hi)
        self.n_grid, self.width, self.k = n_grid, width, k

    def _model(self, grid, need_d):
        idx, w, dw = lagrange_weights(grid, self.g0, self.h, self.n_tab, self.k)
        mu = torch.zeros(*grid.shape, self.P, self.L, dtype=torch.complex128, device=self.dev)
        dmu = torch.zeros_like(mu) if need_d else None
        for i in range(self.k):
            Ti = self.T[idx[..., i]]                                   # (B, G, P, L)
            mu += w[..., i, None, None] * Ti
            if need_d:
                dmu += dw[..., i, None, None] * Ti
        return mu, dmu

    def _eval(self, grid, S, n, need_d):
        mu, dmu = self._model(grid, need_d)
        cross = torch.einsum("bpl,bgpl->bg", S.conj(), mu).real
        nn = torch.einsum("bgp,bp->bg", (mu.abs() ** 2).sum(-1), n)
        ll = (2 * cross - nn) / (2 * self.sigma ** 2)
        w = torch.softmax(ll, dim=1)
        fi = ((dmu.abs() ** 2).sum(-1) / self.sigma ** 2) if need_d else None   # (B, G, P)
        return w, fi

    @staticmethod
    def _moments(w, grid):
        m = (w * grid).sum(1)
        s = torch.sqrt(torch.clamp((w * (grid - m[:, None]) ** 2).sum(1), min=0.0))
        return m, s

    def run(self, true_signal, noise, policy, static_j=None, max_zoom=40):
        """true_signal (P, L) complex; noise (B, n_shots, L) complex standard (unit per real
        component); policy in {"fixed-cycle", "greedy-fisher", "static-best"}.
        Returns numpy means, stds (B, n_shots) and chosen (B, n_shots)."""
        dev = self.dev
        Y0 = torch.as_tensor(true_signal, dtype=torch.complex128, device=dev)
        Z = torch.as_tensor(noise, dtype=torch.complex128, device=dev)
        B, n_shots, _ = Z.shape
        greedy = policy == "greedy-fisher"
        grid = torch.linspace(self.lo, self.hi, self.n_grid, dtype=torch.float64,
                              device=dev).expand(B, -1).contiguous()
        S = torch.zeros(B, self.P, self.L, dtype=torch.complex128, device=dev)
        n = torch.zeros(B, self.P, dtype=torch.float64, device=dev)
        w, fi = self._eval(grid, S, n, greedy)                         # prior: w uniform
        means = torch.empty(B, n_shots, dtype=torch.float64, device=dev)
        stds = torch.empty_like(means)
        chosen = torch.empty(B, n_shots, dtype=torch.long, device=dev)
        ar = torch.arange(B, device=dev)
        for t in range(n_shots):
            if policy == "fixed-cycle":
                j = torch.full((B,), t % self.P, dtype=torch.long, device=dev)
            elif greedy:
                j = torch.argmax(torch.einsum("bg,bgp->bp", w, fi), dim=1)
            else:
                j = torch.full((B,), int(static_j), dtype=torch.long, device=dev)
            S[ar, j] += Y0[j] + self.sigma * Z[:, t]
            n[ar, j] += 1
            # update: eval on current grid, then re-centre/zoom (>= 2 passes; see adaptive.py)
            w, fi = self._eval(grid, S, n, greedy)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            for it in range(max_zoom):
                m, s = self._moments(w, grid)
                half = self.width * torch.maximum(s, grid[:, 1] - grid[:, 0])
                a = torch.clamp(m - half, min=self.lo)
                b = torch.clamp(m + half, max=self.hi)
                act = ~done & (b - a >= 1e-14)
                done = done | ~act
                if not bool(act.any()):
                    break
                u = torch.linspace(0, 1, self.n_grid, dtype=torch.float64, device=dev)
                grid = torch.where(act[:, None], a[:, None] + (b - a)[:, None] * u, grid)
                w2, fi2 = self._eval(grid, S, n, greedy)
                w = torch.where(act[:, None], w2, w)
                if greedy:
                    fi = torch.where(act[:, None, None], fi2, fi)
                if it >= 1:
                    _, s2 = self._moments(w, grid)
                    done = done | (act & (s2 >= 3 * (grid[:, 1] - grid[:, 0])))
            m, s = self._moments(w, grid)
            means[:, t], stds[:, t], chosen[:, t] = m, s, j
        return means.cpu().numpy(), stds.cpu().numpy(), chosen.cpu().numpy()
