"""Estimation / metrology yardstick: classical Fisher information and the Cramer-Rao bound.

Measurement model: a probe returns a deterministic ensemble signal mu(theta, probe)
(complex or real), observed with additive Gaussian noise. Complex signals are split
into stacked (real, imag) components. With noise covariance C (default: iid, variance
sigma^2 per real component),

    FI_probe = J^T C^{-1} J,        J = d mu / d theta  (real-ified),

FI adds over independent probes, and CRB = FI^{-1} bounds the covariance of any
unbiased estimator. The derivative is taken by autograd (when ``signal_fn`` returns a
torch tensor that depends differentiably on a torch ``theta``) or by central finite
differences (default; 4th-order stencil).

Milestone-0 noise convention: complex white noise with std ``eta * A / sqrt(2)`` per
real component (``m0_sigma``).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def m0_sigma(eta: float, A: float) -> float:
    """Per-real-component noise std of the M0 convention (complex noise eta*A, |.|^2 split)."""
    return eta * A / np.sqrt(2.0)


def realify(x) -> np.ndarray:
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.concatenate([x.real.ravel(), x.imag.ravel()])
    return x.ravel().astype(float)


def jacobian(signal_fn: Callable, theta, probe=None, method: str = "fd", h: float = 1e-4):
    """Real-ified Jacobian (n_obs, k) of ``signal_fn(theta, probe)`` at ``theta``.

    method: "fd" (4th-order central differences, step ``h`` in theta units) or
    "autograd" (torch)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if method == "autograd":
        import torch
        t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)

        def f(tt):
            y = signal_fn(tt, probe)
            y = torch.view_as_real(y).movedim(-1, 0) if torch.is_complex(y) else y
            return y.reshape(-1)
        return torch.autograd.functional.jacobian(f, t).detach().cpu().numpy()
    cols = []
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h
        f = lambda d: realify(signal_fn(theta + d * e, probe))  # noqa: E731
        cols.append((-f(2) + 8 * f(1) - 8 * f(-1) + f(-2)) / (12 * h))
    return np.stack(cols, 1)


def fisher_from_jacobian(J: np.ndarray, sigma: float | None = None, cov: np.ndarray | None = None):
    """FI = J^T C^{-1} J for one probe. Give ``sigma`` (iid) or a full ``cov``."""
    if cov is None:
        if sigma is None:
            raise ValueError("need sigma or cov")
        return J.T @ J / sigma ** 2
    L = np.linalg.cholesky(cov)
    Jw = np.linalg.solve(L, J)
    return Jw.T @ Jw


def classical_fisher(signal_fn: Callable, theta, probes: Sequence, sigma: float | None = None,
                     cov=None, method: str = "fd", h: float = 1e-4) -> np.ndarray:
    """Per-probe Fisher information, shape (n_probes, k, k).

    ``signal_fn(theta, probe)`` -> measured observable model for one probe. ``cov`` may be
    one matrix for all probes or a list (one per probe)."""
    out = []
    for j, p in enumerate(probes):
        c = cov[j] if isinstance(cov, (list, tuple)) else cov
        out.append(fisher_from_jacobian(jacobian(signal_fn, theta, p, method, h), sigma, c))
    return np.array(out)


def linear_readout_fisher(J_raw: np.ndarray, M: np.ndarray, sigma: float) -> np.ndarray:
    """FI carried by a linear readout F = M y of the raw (real-ified) data y with iid
    noise sigma: J_F = M J_raw, C_F = sigma^2 M M^T (pseudo-inverse if rank-deficient).
    Bounds what any estimator built on F can achieve; FI_F <= FI_raw."""
    JF = M @ J_raw
    CF = sigma ** 2 * (M @ M.T)
    return JF.T @ np.linalg.pinv(CF, rcond=1e-12, hermitian=True) @ JF


def cramer_rao(FI: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """CRB = inv(sum over probes of FI). Accepts (k, k) or (n_probes, k, k). Raises if
    the total FI is singular to ``rcond`` (an unidentifiable parameter)."""
    F = np.asarray(FI, dtype=float)
    if F.ndim == 3:
        F = F.sum(0)
    F = 0.5 * (F + F.T)
    ev = np.linalg.eigvalsh(F)
    if ev.min() <= rcond * max(ev.max(), 1e-300):
        raise np.linalg.LinAlgError(f"singular Fisher information (eigenvalues {ev})")
    return np.linalg.inv(F)


def report_vs_bound(estimates: np.ndarray, theta_true, CRB: np.ndarray) -> dict:
    """Compare estimator samples (M, k) at a fixed true theta with the CRB (k, k).

    efficiency_var = CRB_ii / Var_ii (the unbiased-estimator comparison);
    efficiency_mse = CRB_ii / MSE_ii (includes bias). A value > 1 is possible only for a
    biased estimator (e.g. a prior-trained regressor) and is flagged."""
    E = np.atleast_2d(np.asarray(estimates, dtype=float))
    if E.shape[0] == 1 and E.shape[1] > 1 and np.ndim(theta_true) == 0:
        E = E.T
    th = np.atleast_1d(np.asarray(theta_true, dtype=float))
    crb = np.atleast_2d(CRB)
    bias = E.mean(0) - th
    cov = np.atleast_2d(np.cov(E, rowvar=False, ddof=1))
    var = np.diag(cov)
    mse = var + bias ** 2
    d = np.diag(crb)
    return {"n": int(E.shape[0]), "bias": bias.tolist(), "var": var.tolist(), "mse": mse.tolist(),
            "crb": d.tolist(), "efficiency_var": (d / var).tolist(),
            "efficiency_mse": (d / mse).tolist(),
            "below_bound": bool(np.any(mse < d)), "cov": cov.tolist()}
