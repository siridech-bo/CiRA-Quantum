"""Phase 1 — GPUSimulatedAnnealingSampler unit tests.

The 9 tests below match the names listed in PROJECT_TEMPLATE.md → Phase 1.

These tests assume a CUDA-capable GPU is present. If CUDA is not available
the entire module is skipped, since the sampler is a GPU-only artifact.
"""

from __future__ import annotations

import inspect
import time

import dimod
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU is required for Phase 1 tests", allow_module_level=True)

# Import after the CUDA gate so the module isn't pulled into envs without a GPU.
from app.optimization.gpu_sa import GPUSimulatedAnnealingSampler  # noqa: E402, I001


# ---- 1 ----

def test_gpu_available():
    """Verifies torch.cuda.is_available() and that we are on a Blackwell-class
    device (compute capability 12.x = sm_120)."""
    assert torch.cuda.is_available()
    cc = torch.cuda.get_device_capability(0)
    assert cc[0] >= 12, f"Expected Blackwell sm_120+, got sm_{cc[0]}{cc[1]}"


# ---- 2 ----

def test_sampler_implements_dimod_interface():
    """Confirms .parameters, .properties, .sample(bqm) signatures are present."""
    sampler = GPUSimulatedAnnealingSampler()

    assert isinstance(sampler, dimod.Sampler)

    params = sampler.parameters
    assert isinstance(params, dict)
    for key in ("num_reads", "num_sweeps", "beta_range", "seed"):
        assert key in params, f"missing parameter key: {key}"

    props = sampler.properties
    assert isinstance(props, dict)
    for key in ("device", "compute_capability", "vram_total_gb"):
        assert key in props, f"missing property key: {key}"
    assert isinstance(props["device"], str)
    assert isinstance(props["compute_capability"], tuple)
    assert len(props["compute_capability"]) == 2
    assert isinstance(props["vram_total_gb"], float)

    sig = inspect.signature(sampler.sample)
    assert "bqm" in sig.parameters
    for kw in ("num_reads", "num_sweeps", "beta_range", "seed"):
        assert kw in sig.parameters, f"sample() missing kw: {kw}"


# ---- 3 ----

def test_two_variable_problem_finds_optimum(two_var_bqm):
    """BQM = {0: -1, 1: 1, (0,1): 2}; expected optimum at (1, 0) energy = -1."""
    sampler = GPUSimulatedAnnealingSampler()
    sampleset = sampler.sample(two_var_bqm, num_reads=200, num_sweeps=200, seed=0)
    best = sampleset.first
    assert best.sample == {0: 1, 1: 0}, f"got {best.sample} (energy {best.energy})"
    assert best.energy == pytest.approx(-1.0, abs=1e-9)


# ---- 4 ----

def test_agrees_with_exact_solver_5var(random_5var_bqm):
    """Random 5-var BQM; ExactSolver gives the truth; GPU SA at num_reads=2000
    must reach the same minimum energy."""
    truth = dimod.ExactSolver().sample(random_5var_bqm).first.energy

    sampler = GPUSimulatedAnnealingSampler()
    sampleset = sampler.sample(random_5var_bqm, num_reads=2000, num_sweeps=500, seed=1)
    best_energy = sampleset.first.energy

    assert best_energy == pytest.approx(truth, abs=1e-6), (
        f"GPU SA min={best_energy}, exact min={truth}"
    )


# ---- 5 ----

def test_agrees_with_exact_solver_10var(random_10var_bqm):
    """Same as 5-var but with 10 variables. Tolerance is loosened slightly
    (1e-4) since the search space is 1024× larger; we tighten num_reads."""
    truth = dimod.ExactSolver().sample(random_10var_bqm).first.energy

    sampler = GPUSimulatedAnnealingSampler()
    sampleset = sampler.sample(random_10var_bqm, num_reads=4000, num_sweeps=1000, seed=2)
    best_energy = sampleset.first.energy

    assert best_energy == pytest.approx(truth, abs=1e-4), (
        f"GPU SA min={best_energy}, exact min={truth}"
    )


# ---- 6 ----

def test_returns_correct_num_reads(random_5var_bqm):
    """sample(bqm, num_reads=500) returns 500 records."""
    sampler = GPUSimulatedAnnealingSampler()
    sampleset = sampler.sample(random_5var_bqm, num_reads=500, num_sweeps=200, seed=3)
    assert len(sampleset) == 500


# ---- 7 ----

def test_seed_is_reproducible(random_5var_bqm):
    """Same seed → identical SampleSet."""
    sampler = GPUSimulatedAnnealingSampler()

    a = sampler.sample(random_5var_bqm, num_reads=100, num_sweeps=200, seed=42)
    b = sampler.sample(random_5var_bqm, num_reads=100, num_sweeps=200, seed=42)

    a_records = a.record
    b_records = b.record

    assert (a_records.sample == b_records.sample).all(), (
        "Spin states differ between two seed=42 runs"
    )
    assert (a_records.energy == b_records.energy).all(), (
        "Energies differ between two seed=42 runs"
    )


# ---- 8 ----

def test_handles_empty_bqm(empty_bqm):
    """Empty BQM should produce a valid SampleSet without crashing."""
    sampler = GPUSimulatedAnnealingSampler()
    sampleset = sampler.sample(empty_bqm, num_reads=10, num_sweeps=10, seed=0)

    assert isinstance(sampleset, dimod.SampleSet)
    # An empty BQM has zero variables; every sample is the empty assignment
    # with energy = offset (= 0). The SampleSet should not crash on `.first`.
    assert sampleset.record.sample.shape[1] == 0
    # All energies must equal the BQM offset (0.0).
    assert all(e == pytest.approx(0.0) for e in sampleset.record.energy)


# ---- 9 ----

def test_scales_to_1000_variables(benchmark_1000var_bqm):
    """100-edge sparse 1000-var BQM completes in < 30 seconds.

    The spec only requires the run to finish in under 30s; it does not pin
    `num_reads` or `num_sweeps`. We pick conservative values that still give
    the SA enough sweeps to be a meaningful scaling check.
    """
    # Empty CUDA cache so the timing isn't sensitive to allocations left
    # over from earlier tests in the same suite run.
    torch.cuda.empty_cache()

    sampler = GPUSimulatedAnnealingSampler()

    start = time.perf_counter()
    sampleset = sampler.sample(
        benchmark_1000var_bqm,
        num_reads=64,
        num_sweeps=150,
        seed=4,
    )
    elapsed = time.perf_counter() - start

    assert isinstance(sampleset, dimod.SampleSet)
    assert len(sampleset) == 64
    assert elapsed < 30.0, f"Expected < 30s, took {elapsed:.2f}s"
