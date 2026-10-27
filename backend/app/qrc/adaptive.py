"""Adaptive controller for single-parameter quantum estimation (Module B).

State: posterior over theta + measurement history. Action: the next probe (index into a
discrete bank). Reward: information (Fisher) gain / posterior-variance reduction.

Measurement model per shot of probe j: y = mu_j(theta) + complex white noise, std
``sigma`` per real component. Because the likelihood is Gaussian, the per-probe sums
S_j = sum y and counts n_j are sufficient:

    log L(theta) = sum_j [ 2 Re<S_j, mu_j(theta)> - n_j ||mu_j(theta)||^2 ] / (2 sigma^2)

so the posterior can be re-gridded at any time without replaying the history. The grid
re-centres on the current posterior each update, so it resolves widths far below a
model-table spacing.

``model(thetas) -> (mu, dmu)`` with shapes (n, P, L) complex: model signals and their
theta-derivatives for every probe (e.g. a Lagrange interpolant of a simulated table).

Two timescales of the hardware loop: ``Controller.step`` is one between-shot decision
(posterior update, then choose); within-shot (single-coherence-window) choices are the
probe's own sequence, fixed by the bank here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Stats:
    S: np.ndarray                   # (P, L) complex, sum of shots per probe
    n: np.ndarray                   # (P,) shots per probe
    history: list = field(default_factory=list)

    @classmethod
    def empty(cls, P, L):
        return cls(np.zeros((P, L), complex), np.zeros(P, int))

    def add(self, j, y):
        self.S[j] += y
        self.n[j] += 1
        self.history.append(int(j))


class GridPosterior:
    """Posterior on an adaptive 1-D grid, uniform prior on [lo, hi]."""

    def __init__(self, model: Callable, sigma: float, lo: float, hi: float, n_grid: int = 401,
                 width: float = 10.0):
        self.model, self.sigma, self.lo, self.hi = model, sigma, lo, hi
        self.n_grid, self.width = n_grid, width
        self.grid = np.linspace(lo, hi, n_grid)
        self._eval(None)

    def loglik(self, thetas, stats):
        mu, dmu = self.model(thetas)
        if stats is None:
            return np.zeros(len(thetas)), mu, dmu
        cross = np.einsum("pl,gpl->g", stats.S.conj(), mu).real
        nn = (np.abs(mu) ** 2).sum(2) @ stats.n
        return (2 * cross - nn) / (2 * self.sigma ** 2), mu, dmu

    def _eval(self, stats):
        ll, self.mu, self.dmu = self.loglik(self.grid, stats)
        w = np.exp(ll - ll.max())
        self.w = w / w.sum()

    def update(self, stats):
        """Re-evaluate on the current grid, then re-centre the grid to mean +/- width*std
        (clipped to the prior) and re-evaluate there. Two passes keep it stable."""
        self._eval(stats)
        for _ in range(2):
            m, s = self.mean, self.std
            a = max(self.lo, m - self.width * s)
            b = min(self.hi, m + self.width * s)
            if b - a < 1e-14:
                break
            self.grid = np.linspace(a, b, self.n_grid)
            self._eval(stats)

    @property
    def mean(self):
        return float(self.w @ self.grid)

    @property
    def std(self):
        return float(np.sqrt(max(self.w @ (self.grid - self.mean) ** 2, 0.0)))

    def fisher_per_probe(self):
        """Posterior-averaged FI of one shot of each probe, (P,)."""
        fi = (np.abs(self.dmu) ** 2).sum(2) / self.sigma ** 2          # (G, P)
        return self.w @ fi


class Policy:
    name = "policy"

    def choose(self, post: GridPosterior, stats: Stats, t: int) -> int:
        raise NotImplementedError


class FixedCycle(Policy):
    """The fixed bank: cycle through every probe in order (the non-adaptive baseline)."""
    name = "fixed-cycle"

    def __init__(self, P):
        self.P = P

    def choose(self, post, stats, t):
        return t % self.P


class GreedyFisher(Policy):
    """Bayesian-adaptive greedy design: the probe with the largest posterior-expected
    one-shot Fisher information (Bayesian D-optimal, myopic)."""
    name = "greedy-fisher"

    def choose(self, post, stats, t):
        return int(np.argmax(post.fisher_per_probe()))


class GreedyEIG(Policy):
    """Greedy expected information gain, Monte Carlo over the posterior (n_mc draws).
    Exact-in-expectation reference for GreedyFisher at low information."""
    name = "greedy-eig"

    def __init__(self, n_mc=64, seed=0):
        self.n_mc = n_mc
        self.rng = np.random.default_rng(seed)

    def choose(self, post, stats, t):
        G = post.grid
        P = post.mu.shape[1]
        logw = np.log(post.w + 1e-300)
        H0 = -(post.w * logw).sum()
        idx = self.rng.choice(len(G), self.n_mc, p=post.w)
        gain = np.zeros(P)
        s = post.sigma
        for j in range(P):
            mu = post.mu[:, j]                                         # (G, L)
            for i in idx:
                y = mu[i] + s * (self.rng.standard_normal(mu.shape[1])
                                 + 1j * self.rng.standard_normal(mu.shape[1]))
                ll = -(np.abs(y[None, :] - mu) ** 2).sum(1) / (2 * s ** 2)
                lw = logw + ll
                lw -= lw.max()
                w = np.exp(lw)
                w /= w.sum()
                gain[j] += (H0 + (w * np.log(w + 1e-300)).sum()) / self.n_mc
        return int(np.argmax(gain))


def run_episode(policy: Policy, post: GridPosterior, true_signal: np.ndarray, n_shots: int,
                rng: np.random.Generator, record=None):
    """Simulate ``n_shots`` adaptive shots. ``true_signal`` (P, L) complex noiseless signals
    at the true theta. Returns arrays of posterior mean / std after each shot and the
    probe chosen."""
    P, L = true_signal.shape
    stats = Stats.empty(P, L)
    means, stds, chosen = np.empty(n_shots), np.empty(n_shots), np.empty(n_shots, int)
    s = post.sigma
    for t in range(n_shots):
        j = policy.choose(post, stats, t)
        y = true_signal[j] + s * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
        stats.add(j, y)
        post.update(stats)
        means[t], stds[t], chosen[t] = post.mean, post.std, j
    return means, stds, chosen
