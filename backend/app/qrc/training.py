"""QRC readout training — ridge regression on GPU (plan §7 step 6).

The reservoir does the nonlinear work; the readout is a *linear* map from
the feature matrix ``X`` to the target ``y``, fit by ridge regression:

    w* = (XᵀX + λI)⁻¹ Xᵀy

Torch handles this on the RTX 5070 (falling back to CPU if CUDA is
absent). λ is chosen by k-fold cross-validation over a log-spaced grid.
A bias column is appended to ``X`` so the readout has an intercept.

Torch is already a base dependency of the backend (cu128 wheel), so this
module — unlike the quantum core — has no optional-import guard beyond
CUDA availability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.qrc.config import TrainingConfig
from app.qrc.utils import nmse, r2_score, squared_correlation


def _torch():
    import torch
    return torch


@dataclass
class RidgeModel:
    """A fitted linear readout."""

    weights: np.ndarray          # (n_features + 1,) incl. bias
    lam: float
    device: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return Xb @ self.weights


def _fit_ridge(X, y, lam, device):
    """Closed-form ridge solve on ``device`` (torch)."""
    torch = _torch()
    Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.float64, device=device).reshape(-1, 1)
    ones = torch.ones((Xt.shape[0], 1), dtype=torch.float64, device=device)
    Xb = torch.cat([Xt, ones], dim=1)
    d = Xb.shape[1]
    ident = torch.eye(d, dtype=torch.float64, device=device)
    ident[-1, -1] = 0.0            # don't regularize the bias term
    A = Xb.T @ Xb + lam * ident
    b = Xb.T @ yt
    w = torch.linalg.solve(A, b)
    return w.reshape(-1).cpu().numpy()


def _resolve_device(requested: str) -> str:
    torch = _torch()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cross_val_lambda(X, y, cfg: TrainingConfig) -> tuple[float, dict]:
    """k-fold CV over the λ grid; returns ``(best_lambda, cv_scores)``."""
    device = _resolve_device(cfg.device)
    n = X.shape[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, cfg.cv_folds)

    scores: dict[float, float] = {}
    for lam in cfg.lambda_grid:
        errs = []
        for i in range(cfg.cv_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(cfg.cv_folds) if j != i])
            w = _fit_ridge(X[tr_idx], y[tr_idx], lam, device)
            model = RidgeModel(w, lam, device)
            errs.append(nmse(y[val_idx], model.predict(X[val_idx])))
        scores[lam] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    return best, scores


def train_readout(
    X_train, y_train, cfg: TrainingConfig, lam: float | None = None
) -> RidgeModel:
    """Fit the readout. If ``lam`` is None, choose it by CV."""
    device = _resolve_device(cfg.device)
    if lam is None:
        lam, _ = cross_val_lambda(X_train, y_train, cfg)
    w = _fit_ridge(X_train, y_train, lam, device)
    return RidgeModel(w, lam, device)


def evaluate(model: RidgeModel, X, y) -> dict:
    """Standard metric bundle (plan §6 validation)."""
    pred = model.predict(X)
    return {
        "r2": r2_score(y, pred),
        "corr2": squared_correlation(y, pred),
        "nmse": nmse(y, pred),
        "lambda": model.lam,
        "device": model.device,
    }
