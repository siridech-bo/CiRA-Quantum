"""Phase 2 v2 — solver registry tests."""

from __future__ import annotations

import dimod
import pytest
import torch

from app.benchmarking import registry as reg
from app.benchmarking.registry import (
    SolverIdentity,
    bootstrap_default_solvers,
    get_solver,
    list_solvers,
    register_solver,
)

GPU_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts with the bootstrapped baseline registry."""
    snapshot = dict(reg._REGISTRY)  # noqa: SLF001 — testing the singleton
    yield
    reg._REGISTRY.clear()           # noqa: SLF001
    reg._REGISTRY.update(snapshot)  # noqa: SLF001


def test_register_solver_succeeds():
    identity = SolverIdentity(
        name="exact_cqm_test",
        version="1.0.0",
        source="dimod-test",
        hardware=None,
        parameter_schema={},
    )
    register_solver(identity, dimod.ExactCQMSolver)
    fetched_id, fetched_cls = get_solver("exact_cqm_test")
    assert fetched_id == identity
    assert fetched_cls is dimod.ExactCQMSolver


def test_register_duplicate_name_fails():
    identity = SolverIdentity(
        name="dup_test", version="1.0.0", source="x", hardware=None, parameter_schema={},
    )
    register_solver(identity, dimod.ExactCQMSolver)
    with pytest.raises(ValueError, match=r"already registered"):
        register_solver(identity, dimod.ExactCQMSolver)


def test_get_solver_returns_identity_and_class():
    identity, sampler_cls = get_solver("exact_cqm")
    assert identity.name == "exact_cqm"
    assert sampler_cls is dimod.ExactCQMSolver


def test_phase_1_gpu_sa_registers_correctly():
    if not GPU_AVAILABLE:
        # On CUDA-less hosts the GPU SA tier is skipped from the bootstrap.
        names = {ident.name for ident in list_solvers()}
        assert "gpu_sa" not in names
        return
    identity, sampler_cls = get_solver("gpu_sa")
    assert identity.name == "gpu_sa"
    assert identity.source == "cira-quantum"
    assert identity.hardware is not None
    from app.optimization.gpu_sa import GPUSimulatedAnnealingSampler
    assert sampler_cls is GPUSimulatedAnnealingSampler


def test_dwave_neal_registers_correctly():
    identity, sampler_cls = get_solver("cpu_sa_neal")
    assert identity.name == "cpu_sa_neal"
    assert identity.source == "dwave-samplers"
    from dwave.samplers import SimulatedAnnealingSampler
    assert sampler_cls is SimulatedAnnealingSampler


def test_exact_cqm_registers_correctly():
    identity, sampler_cls = get_solver("exact_cqm")
    assert identity.name == "exact_cqm"
    assert identity.source == "dimod"
    assert sampler_cls is dimod.ExactCQMSolver


def test_bootstrap_is_idempotent():
    before = {ident.name for ident in list_solvers()}
    bootstrap_default_solvers()
    after = {ident.name for ident in list_solvers()}
    assert before == after


def test_get_unknown_solver_raises():
    with pytest.raises(KeyError, match=r"unknown_solver"):
        get_solver("unknown_solver")
