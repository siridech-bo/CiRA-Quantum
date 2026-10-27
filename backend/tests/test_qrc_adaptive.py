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


def test_posterior_resolves_peak_narrower_than_prior_grid():
    """One shot far more informative than the prior grid spacing (std ~ 1e-5 vs spacing
    0.05): the grid must zoom in, not freeze on one grid point (Stage-2 v1 bug). The true
    value is placed off-grid and between two prior grid points."""
    sig, th0 = 1e-4, 0.3712345
    post = ad.GridPosterior(model, sig, -5.0, 5.0, n_grid=201)
    stats = ad.Stats.empty(P, L)
    rng = np.random.default_rng(3)
    for t in range(3):
        stats.add(1, th0 * Gs[1] + sig * (rng.standard_normal(L) + 1j * rng.standard_normal(L)))
        post.update(stats)
    prec = (stats.n * (np.abs(Gs) ** 2).sum(1)).sum() / sig ** 2
    mean = np.einsum("pl,pl->", stats.S.conj(), Gs).real / sig ** 2 / prec
    assert post.std > 0
    assert abs(post.std * np.sqrt(prec) - 1) < 1e-2
    assert abs(post.mean - mean) < 1e-2 / np.sqrt(prec)


def test_batched_torch_matches_numpy_episodes():
    """adaptive_batch (torch, run here on CPU) reproduces run_episode shot by shot, given the
    same noise, for all three policy kinds (tabulated model + Lagrange interpolation)."""
    from app.qrc import adaptive_batch as ab
    tab_grid = np.linspace(-1.0, 2.0, 301)
    table = np.sin(tab_grid[:, None, None] * (1 + np.arange(P))[None, :, None]
                   + 0.1 * np.arange(L)[None, None, :]) * Gs[None]

    def tab_model(th):
        import torch
        idx, w, dw = ab.lagrange_weights(torch.as_tensor(np.asarray(th, float)), tab_grid[0],
                                         tab_grid[1] - tab_grid[0], len(tab_grid))
        Ti = table[idx.numpy()]
        return (np.einsum("nk,nkpl->npl", w.numpy(), Ti), np.einsum("nk,nkpl->npl", dw.numpy(), Ti))

    sig, B, n_shots, th0 = 0.05, 3, 25, 0.613
    true = tab_model([th0])[0][0]
    noise = (np.random.default_rng(5).standard_normal((B, n_shots, L))
             + 1j * np.random.default_rng(6).standard_normal((B, n_shots, L)))

    class Replay:
        def __init__(self, z):
            self.z, self.k = z, 0

        def standard_normal(self, n):
            out = (self.z.real if self.k % 2 == 0 else self.z.imag)[self.k // 2]
            self.k += 1
            return out

    be = ab.BatchedEpisodes(table, tab_grid, sig, -1.0, 2.0, n_grid=101, device="cpu")
    for name, pol in (("fixed-cycle", ad.FixedCycle(P)), ("greedy-fisher", ad.GreedyFisher()),
                      ("static-best", None)):
        mb, sb, cb = be.run(true, noise, name, static_j=2)
        for b in range(B):
            post = ad.GridPosterior(tab_model, sig, -1.0, 2.0, n_grid=101)
            p = pol
            if p is None:
                p = ad.FixedCycle(P)
                p.choose = lambda post, stats, t: 2
            mn, sd, ch = ad.run_episode(p, post, true, n_shots, Replay(noise[b]))
            assert (ch == cb[b]).all()
            assert np.abs(mn - mb[b]).max() < 1e-9
            assert np.abs(sd / sb[b] - 1).max() < 1e-6


def test_batched_2d_posterior_matches_closed_form():
    """Joint (theta, nu) posterior on a linear-Gaussian model mu_j = theta g_j + nu h_j with
    strongly correlated directions: the affine grid must reproduce the closed-form Gaussian
    posterior mean and theta std (prior box far wider than the posterior)."""
    from app.qrc import adaptive_batch as ab
    r = np.random.default_rng(7)
    Pp, Ll, sig = 3, 4, 0.05
    g = r.normal(size=(Pp, Ll)) + 1j * r.normal(size=(Pp, Ll))
    h = 0.9 * g + 0.1 * (r.normal(size=(Pp, Ll)) + 1j * r.normal(size=(Pp, Ll)))   # nearly degenerate
    thg, nug = np.linspace(-2, 2, 81), np.linspace(-2, 2, 81)
    table = thg[:, None, None, None] * g + nug[None, :, None, None] * h
    th0, nu0 = 0.4321, -0.2345
    true = th0 * g + nu0 * h
    B, n_shots = 4, 12
    noise = r.standard_normal((B, n_shots, Ll)) + 1j * r.standard_normal((B, n_shots, Ll))
    be = ab.BatchedEpisodes2D(table, thg, nug, sig, ((-2, 2), (-2, 2)), device="cpu")
    mb, sb, cb = be.run(true, noise, "fixed-cycle")
    assert max(be.passes) < 20        # zoom converges (v1 window of ~1 std ran max_zoom every shot)
    for b in range(B):
        J = np.zeros((2, 2)); hvec = np.zeros(2)
        for t in range(n_shots):
            j = t % Pp
            y = true[j] + sig * noise[b, t]
            for a, ga in enumerate((g[j], h[j])):
                hvec[a] += np.real(np.vdot(ga, y)) / sig ** 2
                for c, gc in enumerate((g[j], h[j])):
                    J[a, c] += np.real(np.vdot(ga, gc)) / sig ** 2
        cov = np.linalg.inv(J)
        mean = cov @ hvec
        assert abs(mb[b, -1] - mean[0]) < 0.02 * np.sqrt(cov[0, 0])
        assert abs(sb[b, -1] / np.sqrt(cov[0, 0]) - 1) < 0.02
    mg, sg, cg = be.run(true, noise, "greedy-c")
    assert np.isfinite(mg).all() and (sg > 0).all()
