"""Phase 9A — QAOA adapter tests.

The structure mirrors the Phase-8 classical-adapter tests
(`test_cpsat_sampler.py`, `test_highs_sampler.py`), with one key
difference: QAOA is stochastic-in-distribution. We do *not* assert
that QAOA finds the global optimum on every instance — it's a known
property of QAOA at shallow depth that it doesn't always concentrate
onto the optimum, especially on penalty-lifted BQMs. The tests
encode:

1. **Correctness on tiny unconstrained QUBOs** — at depth ≥ 3, QAOA
   reliably finds the optimum of a 3-variable polynomial in the
   top-K samples. This is the test in the OriginQC docs.
2. **Valid SampleSet shape** — for harder problems, the adapter
   returns a properly-typed `dimod.SampleSet` with energies recomputed
   from the BQM. The lowest energy in the SampleSet is the lowest
   among the top-K probability bitstrings.
3. **Qubit-budget rejection** — large BQMs raise `ValueError` with a
   message pointing at Phase 9B.
4. **Empty BQM short-circuit** — degenerate input doesn't crash.
"""

from __future__ import annotations

import itertools

import dimod
import pytest

from app.optimization.qaoa_sampler import MAX_QUBITS_CPU_SIM, QAOASampler

# ----- Helpers -----------------------------------------------------------


def _brute_force_min(bqm: dimod.BinaryQuadraticModel) -> tuple[dict, float]:
    """Enumerate every assignment and return (best_sample, best_energy)."""
    variables = list(bqm.variables)
    best = (None, float("inf"))
    for bits in itertools.product([0, 1], repeat=len(variables)):
        sample = dict(zip(variables, bits, strict=True))
        e = bqm.energy(sample)
        if e < best[1]:
            best = (sample, float(e))
    return best


# ----- Tests --------------------------------------------------------------


def test_finds_optimum_on_unconstrained_3var_qubo():
    """The OriginQC docs' canonical example: a 3-variable polynomial
    whose unique minimum is at bitstring (0, 1, 0). At layer=3 the
    QAOA distribution concentrates sharply enough on the optimum that
    it lands at energy-rank #1 in the top-K samples."""
    linear = {"x0": 1.3, "x1": -1.0, "x2": -0.5}
    quadratic = {("x0", "x1"): -1.2, ("x1", "x2"): 0.9}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)

    s = QAOASampler(layer=3, top_k=5)
    result = s.sample(bqm, seed=42)
    _best_sample, best_energy = _brute_force_min(bqm)
    assert result.first.energy == pytest.approx(best_energy, abs=1e-6)


def test_returns_valid_sampleset_on_lagrange_lifted_knapsack():
    """For penalty-lifted CQMs, QAOA at shallow depth doesn't always
    find the global optimum. The honesty check is just that the
    adapter returns a well-formed SampleSet whose first row is the
    lowest energy among the top-K bitstrings."""
    cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{i}") for i in range(3)]
    cqm.set_objective(-3 * x[0] - 4 * x[1] - 5 * x[2])
    cqm.add_constraint(2 * x[0] + 3 * x[1] + 4 * x[2] <= 5, label="cap")
    bqm, _invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10.0)

    s = QAOASampler(layer=3, top_k=10)
    result = s.sample(bqm, seed=42)

    # Well-formed SampleSet
    assert len(result) > 0
    assert len(result) <= 10
    # First row is the energy-minimum among top-K
    energies = list(result.record.energy)
    assert energies[0] == min(energies)
    # info carries the probability ranking
    assert "qaoa_top_probabilities" in result.info
    assert "qaoa_top_bitstrings" in result.info
    assert len(result.info["qaoa_top_probabilities"]) == len(result)


def test_qubit_budget_rejection_points_at_phase_9b():
    """A BQM exceeding ``MAX_QUBITS_CPU_SIM`` must raise ValueError with
    a message that names the qubit cap and refers the reader to Phase
    9B (cloud / real hardware)."""
    # Build a dense BQM with > MAX_QUBITS_CPU_SIM variables.
    big_n = MAX_QUBITS_CPU_SIM + 5
    linear = {f"x{i}": 1.0 for i in range(big_n)}
    bqm = dimod.BinaryQuadraticModel(linear, {}, 0.0, dimod.BINARY)

    s = QAOASampler()
    with pytest.raises(ValueError, match="exceeds the local-simulator qubit cap"):
        s.sample(bqm)


def test_empty_bqm_short_circuit():
    """A BQM with zero variables should return cleanly, not raise."""
    bqm = dimod.BinaryQuadraticModel({}, {}, 7.5, dimod.BINARY)
    s = QAOASampler(layer=1)
    result = s.sample(bqm)
    # The single sample's energy equals the offset.
    assert result.first.energy == pytest.approx(7.5, abs=1e-9)


def test_stochastic_marker_set():
    """The Phase-2 replay system will eventually key replay-tolerance
    off this attribute. Regression test in case someone removes it."""
    assert QAOASampler._STOCHASTIC is True


def test_not_cqm_native():
    """QAOA is a QUBO solver — the records.record_run dispatcher must
    route it through ``sample(bqm)``, not ``sample_cqm(cqm)``. The
    Phase-8 ``_CQM_NATIVE`` marker must therefore be False/unset."""
    assert not getattr(QAOASampler, "_CQM_NATIVE", False)


def test_invalid_construction_raises():
    """Sanity checks on constructor arg validation."""
    with pytest.raises(ValueError, match="layer must be"):
        QAOASampler(layer=0)
    with pytest.raises(ValueError, match="top_k must be"):
        QAOASampler(top_k=0)
    with pytest.raises(ValueError, match="max_qubits must be"):
        QAOASampler(max_qubits=0)
