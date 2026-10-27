"""Module B: adaptive controller. On a linear-Gaussian model mu_j(theta) = theta g_j the
posterior is Gaussian in closed form and the optimal probe is argmax ||g_j||."""
from __future__ import annotations

import numpy as np

from app.qrc import adaptive as ad

rng0 = np.random.default_rng(0)
P, L, SIG = 4, 30, 0.05
Gs = rng0.normal(size=(P, L)) + 1j * rng0.normal(size=(P, L))
Gs *= np.array([0.2, 1.0, 0.5, 0.1])[:, None]


def model(th):
    th = np.asarray(th, float)
    mu = th[:, None, None] * Gs[None]
    return mu, np.broadcast_to(Gs[None], mu.shape)


def test_posterior_matches_closed_form():
    rng = np.random.default_rng(1)
    post = ad.GridPosterior(model, SIG, -5.0, 5.0)
    stats = ad.Stats.empty(P, L)
    th0 = 0.37
    for t in range(12):
        j = t % P
        stats.add(j, th0 * Gs[j] + SIG * (rng.standard_normal(L) + 1j * rng.standard_normal(L)))
        post.update(stats)
    prec = (stats.n * (np.abs(Gs) ** 2).sum(1)).sum() / SIG ** 2
    mean = np.einsum("pl,pl->", stats.S.conj(), Gs).real / SIG ** 2 / prec
    assert abs(post.mean - mean) < 1e-2 / np.sqrt(prec)
    assert abs(post.std * np.sqrt(prec) - 1) < 1e-3


def test_policies():
    post = ad.GridPosterior(model, SIG, -5.0, 5.0)
    stats = ad.Stats.empty(P, L)
    assert ad.GreedyFisher().choose(post, stats, 0) == 1
    assert ad.GreedyEIG(n_mc=8).choose(post, stats, 0) == 1
    assert [ad.FixedCycle(P).choose(post, stats, t) for t in range(5)] == [0, 1, 2, 3, 0]


def test_adaptive_beats_fixed_on_linear_model():
    """Greedy picks the strongest probe every shot: posterior variance after n shots is
    1 / (n FI_max), vs 1 / (n mean FI) for the fixed cycle."""
    true = 0.8 * Gs
    fi = (np.abs(Gs) ** 2).sum(1) / SIG ** 2
    n = 40
    out = {}
    for pol in (ad.FixedCycle(P), ad.GreedyFisher()):
        post = ad.GridPosterior(model, SIG, -5.0, 5.0)
        _, stds, chosen = ad.run_episode(pol, post, true, n, np.random.default_rng(2))
        out[pol.name] = stds[-1] ** 2
        if pol.name == "greedy-fisher":
            assert set(chosen.tolist()) == {1}
    assert abs(out["greedy-fisher"] * n * fi.max() - 1) < 1e-3
    assert abs(out["fixed-cycle"] * n * fi.mean() - 1) < 1e-3
