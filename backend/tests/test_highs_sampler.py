"""Phase 8 — HiGHS adapter tests.

Same shape as the CP-SAT tests, plus a quadratic-objective rejection check
(HiGHS is linear-objective only by design).
"""

from __future__ import annotations

import dimod
import pytest

from app.optimization.highs_sampler import HiGHSSampler


def _knapsack_cqm(weights, values, capacity, *, sense="maximize"):
    cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{i}") for i in range(len(weights))]
    if sense == "maximize":
        obj = sum(-v * xi for v, xi in zip(values, x, strict=True))
    else:
        obj = sum(v * xi for v, xi in zip(values, x, strict=True))
    cqm.set_objective(obj)
    cqm.add_constraint(
        sum(w * xi for w, xi in zip(weights, x, strict=True)) <= capacity,
        label="capacity",
    )
    return cqm


def test_solves_canonical_knapsack_to_optimum():
    cqm = _knapsack_cqm(
        weights=[2, 3, 4, 5, 9],
        values=[3, 4, 5, 8, 10],
        capacity=20,
    )
    s = HiGHSSampler()
    result = s.sample_cqm(cqm, time_limit=10.0, seed=42)
    assert len(result) == 1
    assert result.first.energy == pytest.approx(-26.0, abs=1e-6)


def test_integer_variable_domain():
    cqm = dimod.ConstrainedQuadraticModel()
    i = dimod.Integer("i", lower_bound=0, upper_bound=10)
    j = dimod.Integer("j", lower_bound=0, upper_bound=10)
    cqm.set_objective(-3 * i - 4 * j)
    cqm.add_constraint(2 * i + 3 * j <= 12, label="cap")

    s = HiGHSSampler()
    result = s.sample_cqm(cqm, time_limit=5.0)
    assert result.first.energy == pytest.approx(-18.0, abs=1e-6)


def test_agreement_with_exact_cqm():
    cqm = _knapsack_cqm(
        weights=[2, 3, 4],
        values=[3, 4, 5],
        capacity=5,
    )

    highs = HiGHSSampler().sample_cqm(cqm, time_limit=5.0).first.energy
    exact = dimod.ExactCQMSolver().sample_cqm(cqm).filter(lambda d: d.is_feasible).first.energy

    assert highs == pytest.approx(exact, abs=1e-6)


def test_quadratic_objective_rejected():
    """HiGHS is linear-objective only. The adapter must reject quadratic
    objectives loudly rather than silently dropping the quadratic terms."""
    cqm = dimod.ConstrainedQuadraticModel()
    x = dimod.Binary("x")
    y = dimod.Binary("y")
    cqm.set_objective(x * y - x - y)  # quadratic term x*y
    s = HiGHSSampler()
    with pytest.raises(ValueError, match="quadratic"):
        s.sample_cqm(cqm)


def test_declares_cqm_native():
    assert HiGHSSampler._CQM_NATIVE is True


def test_sample_bqm_raises():
    s = HiGHSSampler()
    bqm = dimod.BinaryQuadraticModel({"x": -1.0}, {}, 0.0, dimod.BINARY)
    with pytest.raises(NotImplementedError):
        s.sample(bqm)
