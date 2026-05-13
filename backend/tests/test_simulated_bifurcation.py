"""Phase 9C — Simulated Bifurcation tests.

Same shape as the Parallel Tempering tests. SB has two modes (ballistic
and discrete); both should find the optimum on tiny problems.

The package emits a ``ConvergenceWarning`` on small problems where
none of the agents formally converge before max_steps — we ignore
this warning class.
"""

from __future__ import annotations

import itertools
import warnings

import dimod
import pytest

from app.optimization.simulated_bifurcation_sampler import SimulatedBifurcationSampler


def _brute_force_min(bqm: dimod.BinaryQuadraticModel) -> tuple[dict, float]:
    variables = list(bqm.variables)
    best = (None, float("inf"))
    for bits in itertools.product([0, 1], repeat=len(variables)):
        sample = dict(zip(variables, bits, strict=True))
        e = bqm.energy(sample)
        if e < best[1]:
            best = (sample, float(e))
    return best


@pytest.fixture(autouse=True)
def _silence_sb_warnings():
    """The package emits ConvergenceWarning on tiny problems where
    agents don't formally converge before max_steps. The result is
    still valid (sign of final position), so we hide the warning
    rather than dialing up max_steps unnecessarily."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("mode", ["ballistic", "discrete"])
def test_finds_optimum_on_unconstrained_3var_qubo(mode):
    linear = {"x0": 1.3, "x1": -1.0, "x2": -0.5}
    quadratic = {("x0", "x1"): -1.2, ("x1", "x2"): 0.9}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)

    s = SimulatedBifurcationSampler(mode=mode)
    result = s.sample(bqm, agents=16, max_steps=300, num_reads=3, seed=42)
    _best, best_energy = _brute_force_min(bqm)
    assert result.first.energy == pytest.approx(best_energy, abs=1e-9)
    assert result.info["sb_mode"] == mode


def test_finds_optimum_on_random_5var_qubo():
    import numpy as np

    rng = np.random.default_rng(0)
    variables = [f"v{i}" for i in range(5)]
    linear = {v: float(rng.uniform(-2, 2)) for v in variables}
    quadratic = {
        (variables[i], variables[j]): float(rng.uniform(-2, 2))
        for i in range(5) for j in range(i + 1, 5)
    }
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)
    _best, best_energy = _brute_force_min(bqm)

    s = SimulatedBifurcationSampler(mode="discrete")
    result = s.sample(bqm, agents=32, max_steps=400, num_reads=5, seed=42)
    assert result.first.energy == pytest.approx(best_energy, abs=1e-9)


def test_returns_valid_sampleset_on_lagrange_lifted_cqm():
    """Penalty-lifted CQM — orchestrator's standard path. SB needs to
    return a well-formed SampleSet with metadata."""
    cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{i}") for i in range(3)]
    cqm.set_objective(-3 * x[0] - 4 * x[1] - 5 * x[2])
    cqm.add_constraint(2 * x[0] + 3 * x[1] + 4 * x[2] <= 5, label="cap")
    bqm, _invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10.0)

    s = SimulatedBifurcationSampler()
    result = s.sample(bqm, agents=16, max_steps=200, num_reads=3, seed=42)
    assert len(result) <= 3
    assert "sb_mode" in result.info
    assert "sb_unique_solutions" in result.info


def test_empty_bqm_short_circuit():
    bqm = dimod.BinaryQuadraticModel({}, {}, 4.2, dimod.BINARY)
    s = SimulatedBifurcationSampler()
    result = s.sample(bqm, agents=2, max_steps=10)
    assert result.first.energy == pytest.approx(4.2, abs=1e-9)


def test_stochastic_marker_set():
    assert SimulatedBifurcationSampler._STOCHASTIC is True


def test_not_cqm_native():
    assert not getattr(SimulatedBifurcationSampler, "_CQM_NATIVE", False)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="ballistic"):
        SimulatedBifurcationSampler(mode="quantum")


def test_invalid_sample_kwargs_raise():
    bqm = dimod.BinaryQuadraticModel({"x": 1.0}, {}, 0.0, dimod.BINARY)
    s = SimulatedBifurcationSampler()
    with pytest.raises(ValueError, match="agents"):
        s.sample(bqm, agents=0)
    with pytest.raises(ValueError, match="max_steps"):
        s.sample(bqm, max_steps=0)
    with pytest.raises(ValueError, match="num_reads"):
        s.sample(bqm, num_reads=0)


def test_ballistic_and_discrete_both_run():
    """Sanity that the mode dispatch actually flips inside the package
    — not that the modes produce different *results* on tiny inputs."""
    bqm = dimod.BinaryQuadraticModel({"x": 1.0, "y": -1.0}, {("x", "y"): -0.5}, 0.0, dimod.BINARY)

    sb_b = SimulatedBifurcationSampler(mode="ballistic")
    sb_d = SimulatedBifurcationSampler(mode="discrete")

    rb = sb_b.sample(bqm, agents=8, max_steps=200, seed=42)
    rd = sb_d.sample(bqm, agents=8, max_steps=200, seed=42)

    assert rb.info["sb_mode"] == "ballistic"
    assert rd.info["sb_mode"] == "discrete"
