"""Module A gate: the Fisher / Cramer-Rao machinery must reproduce analytic Gaussian
bounds (<1%) and be attained by efficient estimators in Monte Carlo before any
multi-parameter or simulator output is trusted."""
from __future__ import annotations

import numpy as np
import pytest

from app.qrc import estimation as est

T = np.linspace(0, 0.05, 40)
A, F0, SIG = 1.3, 120.0, 0.02


def decay(theta, probe=None):
    """Complex decaying line, theta = [rate] or [rate, amplitude]."""
    th = theta
    amp = th[1] if len(th) > 1 else A
    if hasattr(th, "requires_grad"):
        import torch
        t = torch.tensor(T, dtype=torch.float64)
        return amp * torch.exp((-th[0] + 2j * np.pi * F0) * t)
    return amp * np.exp((-th[0] + 2j * np.pi * F0) * T)


def test_single_parameter_analytic_crb():
    g = 30.0
    fi_true = np.sum((A * T * np.exp(-g * T)) ** 2) / SIG ** 2
    for method in ("fd", "autograd"):
        if method == "autograd":
            pytest.importorskip("torch")
        FI = est.classical_fisher(decay, [g], [None], sigma=SIG, method=method)
        crb = est.cramer_rao(FI)
        assert abs(FI.sum() / fi_true - 1) < 1e-2
        assert abs(crb[0, 0] * fi_true - 1) < 1e-2


def test_two_parameter_analytic_fisher():
    g = 30.0
    e = np.exp(-g * T)
    dg, da = -A * T * e, e                       # |d mu|, phases identical -> real inner products
    FI_true = np.array([[dg @ dg, dg @ da], [da @ dg, da @ da]]) / SIG ** 2
    FI = est.classical_fisher(decay, [g, A], [None], sigma=SIG).sum(0)
    assert np.abs(FI / FI_true - 1).max() < 1e-2
    assert np.abs(est.cramer_rao(FI) @ FI_true - np.eye(2)).max() < 1e-2


def test_fisher_adds_over_probes():
    FI = est.classical_fisher(decay, [30.0], [None, None, None], sigma=SIG)
    assert FI.shape == (3, 1, 1)
    assert est.cramer_rao(FI)[0, 0] == pytest.approx(est.cramer_rao(FI[0])[0, 0] / 3)


def test_linear_model_mle_attains_crb_monte_carlo():
    """y = theta g + n: the MLE is efficient, Var = 1/FI exactly. M = 4e5 -> 0.22% s.e."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=20)
    theta, M = 0.7, 400_000
    y = theta * g[None, :] + SIG * rng.standard_normal((M, 20))
    th_hat = y @ g / (g @ g)
    crb = est.cramer_rao(est.classical_fisher(lambda t, p: t[0] * g, [theta], [None], sigma=SIG))
    r = est.report_vs_bound(th_hat[:, None], theta, crb)
    assert abs(r["efficiency_var"][0] - 1) < 0.01
    assert abs(r["bias"][0]) < 5 * np.sqrt(crb[0, 0] / M)


def test_two_parameter_linear_mle_covariance_matches_crb():
    rng = np.random.default_rng(1)
    G = rng.normal(size=(30, 2))
    th, M = np.array([0.4, -1.1]), 200_000
    y = (G @ th)[None, :] + SIG * rng.standard_normal((M, 30))
    th_hat = np.linalg.lstsq(G, y.T, rcond=None)[0].T
    crb = est.cramer_rao(est.classical_fisher(lambda t, p: G @ t, th, [None], sigma=SIG))
    cov = np.cov(th_hat, rowvar=False)
    assert np.abs(cov / crb - 1).max() < 0.02


def test_nonlinear_mle_attains_crb_high_snr():
    """Gauss-Newton MLE of the decay rate at high SNR reaches the CRB (asymptotic)."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(2)
    g, M, s = 30.0, 4000, 0.005
    mu = decay([g])
    crb = est.cramer_rao(est.classical_fisher(decay, [g], [None], sigma=s))
    hats = []
    for _ in range(M):
        y = mu + s * (rng.standard_normal(T.size) + 1j * rng.standard_normal(T.size))
        hats.append(least_squares(lambda x: est.realify(decay(x) - y), [g * 1.1]).x[0])
    r = est.report_vs_bound(np.array(hats)[:, None], g, crb)
    assert abs(r["efficiency_var"][0] - 1) < 0.07          # 2.2% s.e. at M=4000 -> ~3 s.e.


def test_linear_readout_fisher():
    rng = np.random.default_rng(3)
    J = rng.normal(size=(50, 2))
    FI_raw = est.fisher_from_jacobian(J, SIG)
    assert np.allclose(est.linear_readout_fisher(J, np.eye(50), SIG), FI_raw)
    Q = np.linalg.qr(rng.normal(size=(50, 50)))[0]
    assert np.allclose(est.linear_readout_fisher(J, Q, SIG), FI_raw)        # invertible: lossless
    Msum = np.ones((1, 50))
    FI_s = est.linear_readout_fisher(J, Msum, SIG)
    assert np.all(np.linalg.eigvalsh(FI_raw - FI_s) > -1e-9)                # compression loses
    assert np.allclose(est.fisher_from_jacobian(J, cov=SIG ** 2 * np.eye(50)), FI_raw)


def test_singular_fisher_raises():
    with pytest.raises(np.linalg.LinAlgError):
        est.cramer_rao(np.array([[1.0, 1.0], [1.0, 1.0]]))
