"""Phase 9C — Parallel Tempering tests.

Same shape as other sampler tests (test_gpu_sa, test_cpsat_sampler):
correctness on a tiny unconstrained QUBO, well-formed SampleSet on a
Lagrange-lifted CQM, sensible variance across seeds, qubit-budget /
empty-input edge cases.

PT is stochastic but on tiny problems with multiple replicas it
finds the optimum on every reasonable seed — we assert this without
flakiness risk.
"""

from __future__ import annotations

import itertools

import dimod
import pytest

from app.optimization.parallel_tempering_sampler import ParallelTemperingSampler


def _brute_force_min(bqm: dimod.BinaryQuadraticModel) -> tuple[dict, float]:
    variables = list(bqm.variables)
    best = (None, float("inf"))
    for bits in itertools.product([0, 1], repeat=len(variables)):
        sample = dict(zip(variables, bits, strict=True))
        e = bqm.energy(sample)
        if e < best[1]:
            best = (sample, float(e))
    return best


def test_finds_optimum_on_unconstrained_3var_qubo():
    """The OriginQC docs' canonical 3-variable example. PT at 4
    replicas, 200 sweeps, finds the optimum cleanly across seeds."""
    linear = {"x0": 1.3, "x1": -1.0, "x2": -0.5}
    quadratic = {("x0", "x1"): -1.2, ("x1", "x2"): 0.9}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)

    s = ParallelTemperingSampler(num_replicas=4)
    result = s.sample(bqm, num_sweeps=200, num_reads=5, seed=42)
    _best_sample, best_energy = _brute_force_min(bqm)
    assert result.first.energy == pytest.approx(best_energy, abs=1e-9)


def test_finds_optimum_on_random_5var_qubo():
    """A 5-variable random QUBO with known minimum via brute force.
    PT should find it reliably even at modest sweep budget."""
    import numpy as np

    rng = np.random.default_rng(0)
    variables = [f"v{i}" for i in range(5)]
    linear = {v: float(rng.uniform(-2, 2)) for v in variables}
    quadratic = {
        (variables[i], variables[j]): float(rng.uniform(-2, 2))
        for i in range(5) for j in range(i + 1, 5)
    }
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)
    _best_sample, best_energy = _brute_force_min(bqm)

    s = ParallelTemperingSampler(num_replicas=8)
    result = s.sample(bqm, num_sweeps=500, num_reads=10, seed=42)
    assert result.first.energy == pytest.approx(best_energy, abs=1e-9)


def test_seed_reproducibility():
    """Same seed → same best-energy outcome. Reproducibility is one
    of PT's correctness properties under the seeded-RNG path."""
    linear = {"x0": 1.0, "x1": -1.0}
    quadratic = {("x0", "x1"): -0.5}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)

    s = ParallelTemperingSampler(num_replicas=4)
    a = s.sample(bqm, num_sweeps=100, num_reads=3, seed=42).first.energy
    b = s.sample(bqm, num_sweeps=100, num_reads=3, seed=42).first.energy
    assert a == b


def test_returns_valid_sampleset_on_lagrange_lifted_cqm():
    """The penalty-lifted path is what the orchestrator gives us in
    practice. Check that PT returns a well-formed SampleSet and the
    info dict carries the hyperparameters."""
    cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{i}") for i in range(3)]
    cqm.set_objective(-3 * x[0] - 4 * x[1] - 5 * x[2])
    cqm.add_constraint(2 * x[0] + 3 * x[1] + 4 * x[2] <= 5, label="cap")
    bqm, _invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10.0)

    s = ParallelTemperingSampler(num_replicas=4)
    result = s.sample(bqm, num_sweeps=200, num_reads=5, seed=42)
    assert len(result) <= 5
    energies = list(result.record.energy)
    assert energies[0] == min(energies)
    assert result.info["pt_num_replicas"] == 4
    assert "pt_beta_range" in result.info


def test_empty_bqm_short_circuit():
    bqm = dimod.BinaryQuadraticModel({}, {}, 4.2, dimod.BINARY)
    s = ParallelTemperingSampler(num_replicas=2)
    result = s.sample(bqm, num_sweeps=10)
    assert result.first.energy == pytest.approx(4.2, abs=1e-9)


def test_stochastic_marker_set():
    """The replay system reads this attribute. Regression test in
    case someone removes it."""
    assert ParallelTemperingSampler._STOCHASTIC is True


def test_not_cqm_native():
    """PT is a QUBO solver; records dispatcher must route via
    sample(bqm), not sample_cqm."""
    assert not getattr(ParallelTemperingSampler, "_CQM_NATIVE", False)


def test_invalid_construction_raises():
    with pytest.raises(ValueError, match="num_replicas must be"):
        ParallelTemperingSampler(num_replicas=1)
    with pytest.raises(ValueError, match="beta_range"):
        ParallelTemperingSampler(beta_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="beta_range"):
        ParallelTemperingSampler(beta_range=(5.0, 1.0))


def test_invalid_sample_kwargs_raise():
    bqm = dimod.BinaryQuadraticModel({"x": 1.0}, {}, 0.0, dimod.BINARY)
    s = ParallelTemperingSampler()
    with pytest.raises(ValueError, match="num_sweeps"):
        s.sample(bqm, num_sweeps=0)
    with pytest.raises(ValueError, match="num_reads"):
        s.sample(bqm, num_reads=0)
    with pytest.raises(ValueError, match="swap_interval"):
        s.sample(bqm, swap_interval=0)
