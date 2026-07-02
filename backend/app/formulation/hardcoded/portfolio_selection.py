"""Portfolio Selection — Markowitz-shape mean/variance with a budget.

Given per-asset expected returns ``μ_i``, a covariance matrix ``Σ``, and
a maximum-assets budget ``K``, pick a subset of assets to maximize
mean minus (risk-scaled) variance:

    maximize   Σ μ_i · x_i  -  λ · Σ_{i,j} Σ_{i,j} · x_i · x_j
    subject to Σ x_i ≤ K                    (budget)
              x_i ∈ {0, 1}

For binary variables ``x_i² = x_i``, so the variance-diagonal folds
into the linear coefficient. Expanded:
    linear[x_i]  =  μ_i  -  λ · Σ_{i,i}
    quadratic[x_i·x_j] (i<j)  =  -2 · λ · Σ_{i,j}      (symmetric Σ)
    constraint  Σ x_i ≤ K  labeled ``budget``

The budget constraint is a real (non-natural-form) CQM constraint, so
this formulation is NOT pure-QUBO — the QAOA pipeline will lower it
into a penalty automatically via dimod's ``cqm_to_bqm``, incurring one
slack qubit per unit of budget slack. That's expected; portfolios with
K < N are inherently constrained and there's no simpler encoding that
avoids the slack qubits without changing the problem.
"""

from __future__ import annotations

from typing import Any


def formulate_portfolio_selection(
    returns: list[float],
    covariance: list[list[float]],
    max_assets: int,
    risk_aversion: float = 1.0,
) -> dict[str, Any]:
    """Emit a valid ``cqm_v1`` document for the budget-constrained
    Markowitz-shape portfolio problem. All coefficients are computed
    exactly.

    Parameters
    ----------
    returns
        Length-N vector of expected returns per asset.
    covariance
        N-by-N symmetric covariance matrix (list-of-lists). Off-diagonal
        asymmetries are averaged silently — some data pipelines emit
        upper-triangle-only matrices with zeros below the diagonal.
    max_assets
        Budget: at most this many assets may be selected. Must be in
        ``[1, N]``.
    risk_aversion
        Non-negative λ scaling the variance term. 0 recovers the pure
        return-maximization (linear) problem; larger values push toward
        low-variance portfolios.

    Returns
    -------
    dict
        A ``cqm_v1`` JSON dict ready to feed to ``compile_cqm_json``.
    """
    n = len(returns)
    if n < 2:
        raise ValueError(
            f"Portfolio selection requires at least 2 assets; got {n}",
        )
    if len(covariance) != n or any(len(row) != n for row in covariance):
        raise ValueError(
            f"Covariance must be a {n}x{n} matrix; got shape "
            f"{len(covariance)}x{len(covariance[0]) if covariance else 0}",
        )
    if not (1 <= max_assets <= n):
        raise ValueError(
            f"max_assets must be in [1, {n}]; got {max_assets}",
        )
    lam = float(risk_aversion)
    if lam < 0:
        raise ValueError(
            f"risk_aversion must be non-negative; got {lam}",
        )

    # Symmetrize covariance defensively — upper/lower-triangle-only
    # inputs shouldn't silently drop terms.
    sigma: list[list[float]] = [
        [0.5 * (float(covariance[i][j]) + float(covariance[j][i])) for j in range(n)]
        for i in range(n)
    ]

    variables = [
        {
            "name": f"x_{i}",
            "type": "binary",
            "description": (
                f"Asset {i} selected into the portfolio "
                f"(return μ={returns[i]:g})"
            ),
        }
        for i in range(n)
    ]

    # linear[x_i] = μ_i - λ · Σ_{i,i}
    linear = {
        f"x_{i}": float(returns[i]) - lam * sigma[i][i] for i in range(n)
    }

    # quadratic[x_i x_j] = -2 · λ · Σ_{i,j}   (i<j)
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            coef = -2.0 * lam * sigma[i][j]
            if coef != 0.0:
                quadratic[f"x_{i}*x_{j}"] = coef

    constraints = [
        {
            "label": "budget",
            "type": "inequality_le",
            "linear": {f"x_{i}": 1.0 for i in range(n)},
            "quadratic": {},
            "rhs": float(max_assets),
        }
    ]

    expected_optimum = _brute_force_portfolio(
        returns, sigma, max_assets, lam,
    )

    return {
        "version": "1",
        "description": (
            f"Portfolio selection over {n} assets, budget K = {max_assets}, "
            f"risk aversion λ = {lam:g}. Objective = expected return "
            "minus λ·variance. Binary selection with one budget "
            "constraint."
        ),
        "variables": variables,
        "objective": {
            "sense": "maximize",
            "linear": linear,
            "quadratic": quadratic,
        },
        "constraints": constraints,
        "test_instance": {
            "description": (
                f"Optimum objective (return - λ·variance) = "
                f"{expected_optimum:g}."
            ),
            "expected_optimum": float(expected_optimum),
        },
    }


def _brute_force_portfolio(
    returns: list[float],
    sigma: list[list[float]],
    max_assets: int,
    lam: float,
) -> float:
    """Return the maximum ``μ·x - λ·xᵀΣx`` over all 2^N binary
    selections respecting the budget. Formulator-time only. Exponential
    in ``len(returns)``; fine for qpu_ready-scale templates but not
    intended for the 20-asset knapsack-shape instances."""
    n = len(returns)
    best = float("-inf")
    for mask in range(1 << n):
        selected = [(mask >> i) & 1 for i in range(n)]
        k = sum(selected)
        if k > max_assets:
            continue
        ret = sum(returns[i] for i in range(n) if selected[i])
        var = 0.0
        for i in range(n):
            if not selected[i]:
                continue
            for j in range(n):
                if not selected[j]:
                    continue
                var += sigma[i][j]
        obj = ret - lam * var
        if obj > best:
            best = obj
    return best if best != float("-inf") else 0.0
