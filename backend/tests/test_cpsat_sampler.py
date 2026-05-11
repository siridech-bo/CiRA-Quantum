"""Phase 8 — CP-SAT adapter tests.

Mirrors the structure of test_gpu_sa.py: a tiny knapsack with a known
optimum, an integer-domain instance, and an agreement check against
``dimod.ExactCQMSolver`` for a problem small enough to enumerate.
"""

from __future__ import annotations

import dimod
import pytest

from app.optimization.cpsat_sampler import CPSATSampler


def _knapsack_cqm(weights, values, capacity, *, sense="maximize"):
    """Build a CQM for 0/1 knapsack. ``sense`` is encoded by sign-flipping
    the objective (CQM is always minimize in this codebase)."""
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
    """The canonical 5-item knapsack from tests/instances/knapsack_5item.json
    has expected_optimum=26. CP-SAT must find it exactly."""
    cqm = _knapsack_cqm(
        weights=[2, 3, 4, 5, 9],
        values=[3, 4, 5, 8, 10],
        capacity=20,
    )
    s = CPSATSampler()
    result = s.sample_cqm(cqm, time_limit=10.0, seed=42)
    assert len(result) == 1
    # Best energy in CQM minimize-space is -26 (because we sign-flipped).
    assert result.first.energy == pytest.approx(-26.0, abs=1e-9)


def test_integer_variable_domain():
    """An integer-domain bounded LP. Minimize -3i - 4j s.t. 2i + 3j <= 12,
    i, j integer in [0, 10]. Optimal: i=6, j=0 → -18 (the i-bound at 10
    is non-tight; the 2i ≤ 12 slack lets i=6 land on the constraint and
    beats the obvious-looking i=0, j=4 → -16)."""
    cqm = dimod.ConstrainedQuadraticModel()
    i = dimod.Integer("i", lower_bound=0, upper_bound=10)
    j = dimod.Integer("j", lower_bound=0, upper_bound=10)
    cqm.set_objective(-3 * i - 4 * j)
    cqm.add_constraint(2 * i + 3 * j <= 12, label="cap")

    s = CPSATSampler()
    result = s.sample_cqm(cqm, time_limit=5.0)
    assert result.first.energy == pytest.approx(-18.0, abs=1e-9)
    sample = result.first.sample
    # 2i + 3j <= 12, integer, minimize -3i - 4j → i=6, j=0.
    assert 2 * sample["i"] + 3 * sample["j"] <= 12
    assert sample["i"] == 6
    assert sample["j"] == 0


def test_agreement_with_exact_cqm_on_tiny_knapsack():
    """For a problem small enough for ExactCQMSolver to enumerate, both
    solvers must report the same optimal energy."""
    cqm = _knapsack_cqm(
        weights=[2, 3, 4],
        values=[3, 4, 5],
        capacity=5,
    )

    cpsat = CPSATSampler().sample_cqm(cqm, time_limit=5.0).first.energy
    exact = dimod.ExactCQMSolver().sample_cqm(cqm).filter(lambda d: d.is_feasible).first.energy

    assert cpsat == pytest.approx(exact, abs=1e-9)


def test_declares_cqm_native():
    """The dispatcher in records.record_run reads this attribute to pick
    the CQM-native path. Regression test in case someone accidentally
    removes it."""
    assert CPSATSampler._CQM_NATIVE is True


def test_sample_bqm_raises():
    """CP-SAT should not be called via sample(bqm); the dispatcher should
    pick the CQM path. If it ever isn't, sample(bqm) must fail loudly
    rather than silently produce nonsense."""
    s = CPSATSampler()
    bqm = dimod.BinaryQuadraticModel({"x": -1.0}, {}, 0.0, dimod.BINARY)
    with pytest.raises(NotImplementedError):
        s.sample(bqm)
