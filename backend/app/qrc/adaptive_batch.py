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


class BatchedEpisodes2D:
    """Known model *family* with a nuisance: joint grid posterior over (theta, nu), uniform
    prior on a box, B episodes at once. The grid is affine and follows the posterior:
    g = m + R u, u on an n x n square [-W, W]^2, R R^T = posterior cov + shrink^2 R_old R_old^T,
    i.e. the window spans ~W posterior stds (W / sqrt(1 + shrink^2) at the fixed point). The
    floor keeps the new half-width >= ``shrink`` x the old one, so a posterior narrower than a
    grid cell (all weight on one point, e.g. a one-shot ridge on the prior grid) is zoomed into
    gradually instead of truncated; if the boundary ring holds > ``edge_tol`` of the mass the
    window doubles instead. >= 2 passes, stop once the posterior spans >= ``resolve`` spacings in
    every direction and the ring mass is <= ``edge_tol``. (v2: v1 had R = chol(.)/W, a window of
    only ~1 std, so the ring test never passed and every shot ran ``max_zoom`` passes.)
    Greedy design for theta with nu as nuisance: the probe minimising the posterior-averaged
    [(Sigma_post^-1 + F_j)^-1]_theta,theta (Bayesian c-optimal, myopic)."""

    def __init__(self, table, th_grid, nu_grid, sigma, box, n=41, W=7.0, k=6, resolve=2.0,
                 shrink=0.5, edge_tol=1e-6, device="cuda"):
        self.dev = torch.device(device)
        self.T = torch.as_tensor(table, dtype=torch.complex128, device=self.dev)
        self.th0, self.hth = float(th_grid[0]), float(th_grid[1] - th_grid[0])
        self.nu0, self.hnu = float(nu_grid[0]), float(nu_grid[1] - nu_grid[0])
        self.n_th, self.n_nu, self.P, self.L = self.T.shape
        self.sigma, self.box, self.n, self.W, self.k = float(sigma), box, n, W, k
        self.resolve, self.shrink, self.edge_tol = resolve, shrink, edge_tol
        self.passes, self.unresolved = [], []    # per shot (last run): max zoom passes, fraction hitting max_zoom
        u = torch.linspace(-W, W, n, dtype=torch.float64, device=self.dev)
        U1, U2 = torch.meshgrid(u, u, indexing="ij")
        self.U = torch.stack([U1.ravel(), U2.ravel()], -1)              # (G, 2)
        self.hu = 2 * W / (n - 1)
        self.edge = (self.U.abs() >= W - 1e-9).any(-1)                 # boundary ring of the square

    def _model(self, pts, derivs=True):
        """pts (B, G, 2) -> mu, dmu_th, dmu_nu (B, G, P, L) (derivatives None if not derivs)."""
        it, wt, dwt = lagrange_weights(pts[..., 0], self.th0, self.hth, self.n_th, self.k)
        iv, wv, dwv = lagrange_weights(pts[..., 1], self.nu0, self.hnu, self.n_nu, self.k)
        shape = (*pts.shape[:2], self.P, self.L)
        mu = torch.zeros(shape, dtype=torch.complex128, device=self.dev)
        d1 = d2 = None
        if derivs:
            d1, d2 = torch.zeros_like(mu), torch.zeros_like(mu)
        for a in range(self.k):
            for b in range(self.k):
                Tab = self.T[it[..., a], iv[..., b]]
                mu += (wt[..., a] * wv[..., b])[..., None, None] * Tab
                if derivs:
                    d1 += (dwt[..., a] * wv[..., b])[..., None, None] * Tab
                    d2 += (wt[..., a] * dwv[..., b])[..., None, None] * Tab
        return mu, d1, d2

    def _eval(self, pts, S, n):
        """Posterior weights on the grid points (B, G)."""
        mu, _, _ = self._model(pts, derivs=False)
        cross = torch.einsum("bpl,bgpl->bg", S.conj(), mu).real
        nn = torch.einsum("bgp,bp->bg", (mu.abs() ** 2).sum(-1), n)
        ll = (2 * cross - nn) / (2 * self.sigma ** 2)
        (tl, th), (vl, vh) = self.box
        inside = (pts[..., 0] >= tl) & (pts[..., 0] <= th) & (pts[..., 1] >= vl) & (pts[..., 1] <= vh)
        ll = torch.where(inside, ll, torch.full_like(ll, -torch.inf))
        return torch.softmax(ll, 1)

    def _fbar(self, w, pts):
        """Posterior-averaged one-shot Fisher matrices (B, P, 2, 2)."""
        _, d1, d2 = self._model(pts)
        F = torch.stack([torch.stack([(d1.conj() * d1).real.sum(-1), (d1.conj() * d2).real.sum(-1)], -1),
                         torch.stack([(d2.conj() * d1).real.sum(-1), (d2.conj() * d2).real.sum(-1)], -1)],
                        -2) / self.sigma ** 2                            # (B, G, P, 2, 2)
        return torch.einsum("bg,bgpij->bpij", w, F)

    @staticmethod
    def _moments(w, pts):
        m = torch.einsum("bg,bgi->bi", w, pts)
        d = pts - m[:, None]
        C = torch.einsum("bg,bgi,bgj->bij", w, d, d)
        return m, C

    def run(self, true_signal, noise, policy, static_j=None, max_zoom=30):
        """true_signal (P, L) or (B, P, L); noise (B, n_shots, L) unit complex; policy in
        {"fixed-cycle", "greedy-c", "static-best"}. Returns theta means, theta stds, chosen."""
        dev = self.dev
        Z = torch.as_tensor(noise, dtype=torch.complex128, device=dev)
        B, n_shots, _ = Z.shape
        Y0 = torch.as_tensor(true_signal, dtype=torch.complex128, device=dev)
        if Y0.dim() == 2:
            Y0 = Y0.expand(B, -1, -1)
        (tl, th), (vl, vh) = self.box
        # initial grid: the prior box mapped onto the square
        m = torch.tensor([(tl + th) / 2, (vl + vh) / 2], dtype=torch.float64, device=dev).expand(B, 2).clone()
        R = torch.diag(torch.tensor([(th - tl) / 2, (vh - vl) / 2], dtype=torch.float64,
                                    device=dev) / self.W).expand(B, 2, 2).clone()
        pts = m[:, None] + torch.einsum("bij,gj->bgi", R, self.U)
        S = torch.zeros(B, self.P, self.L, dtype=torch.complex128, device=dev)
        n = torch.zeros(B, self.P, dtype=torch.float64, device=dev)
        w = self._eval(pts, S, n)
        greedy = policy == "greedy-c"
        Fbar = self._fbar(w, pts) if greedy else None
        means = torch.empty(B, n_shots, dtype=torch.float64, device=dev)
        stds = torch.empty_like(means)
        chosen = torch.empty(B, n_shots, dtype=torch.long, device=dev)
        ar = torch.arange(B, device=dev)
        eye = torch.eye(2, dtype=torch.float64, device=dev)
        self.passes, self.unresolved = [], []
        for t in range(n_shots):
            if policy == "fixed-cycle":
                j = torch.full((B,), t % self.P, dtype=torch.long, device=dev)
            elif policy == "greedy-c":
                _, C = self._moments(w, pts)
                Cinv = torch.linalg.inv(C + 1e-30 * eye + 1e-12 * C.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None] * eye)
                M = Cinv[:, None] + Fbar                                     # (B, P, 2, 2)
                det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
                j = torch.argmin(M[..., 1, 1] / det, dim=1)
            else:
                j = torch.full((B,), int(static_j), dtype=torch.long, device=dev)
            S[ar, j] += Y0[ar, j] + self.sigma * Z[:, t]
            n[ar, j] += 1
            w = self._eval(pts, S, n)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            for it in range(max_zoom):
                mm, C = self._moments(w, pts)
                floor = self.shrink ** 2 * R @ R.transpose(1, 2)              # half-width shrinks <= 1/shrink per pass
                Rn = torch.linalg.cholesky(C + floor)
                trunc = w[:, self.edge].sum(1) > self.edge_tol                 # window cuts the posterior
                Rn = torch.where(trunc[:, None, None], 2.0 * R, Rn)             # -> expand instead
                act = ~done
                m = torch.where(act[:, None], mm, m)
                R = torch.where(act[:, None, None], Rn, R)
                pts = m[:, None] + torch.einsum("bij,gj->bgi", R, self.U)
                w = torch.where(act[:, None], self._eval(pts, S, n), w)
                if it >= 1:
                    _, C2 = self._moments(w, pts)
                    Ri = torch.linalg.inv(R)
                    Cu = Ri @ C2 @ Ri.transpose(1, 2)                         # cov in u units
                    ok = torch.linalg.eigvalsh(Cu)[:, 0] >= (self.resolve * self.hu) ** 2
                    ok = ok & (w[:, self.edge].sum(1) <= self.edge_tol)
                    done = done | ok
                if bool(done.all()):
                    break
            self.passes.append(it + 1)
            self.unresolved.append(float((~done).double().mean()))
            if greedy:
                Fbar = self._fbar(w, pts)
            mm, C = self._moments(w, pts)
            means[:, t], stds[:, t], chosen[:, t] = mm[:, 0], C[:, 0, 0].clamp(min=0).sqrt(), j
        return means.cpu().numpy(), stds.cpu().numpy(), chosen.cpu().numpy()
