"""Phase 2 v2 — RunRecord schema and serialization tests."""

from __future__ import annotations

from pathlib import Path

import dimod
import pytest

from app.benchmarking import (
    SolverIdentity,
    load_record,
    record_run,
)
from app.benchmarking.records import (
    compute_repro_hash,
    load_archived_sample_set,
)


@pytest.fixture
def tmp_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIRA_BENCH_ARCHIVE", str(tmp_path))
    return tmp_path


@pytest.fixture
def tiny_bqm() -> dimod.BinaryQuadraticModel:
    # Optimum at x_0=0, x_1=1, x_2=0, energy = -3.
    return dimod.BinaryQuadraticModel(
        {0: 1.0, 1: -3.0, 2: 1.0},
        {(0, 1): 1.0, (1, 2): 1.0},
        0.0,
        dimod.BINARY,
    )


def test_record_includes_all_required_fields(tmp_archive, tiny_bqm):
    record = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/tiny",
        bqm=tiny_bqm,
        parameters={"num_reads": 50, "num_sweeps": 50, "seed": 7},
    )
    for field in (
        "record_id", "code_version", "solver", "parameters", "instance_id",
        "hardware_id", "started_at", "completed_at", "repro_hash", "results",
    ):
        assert getattr(record, field) is not None, field
    assert isinstance(record.solver, SolverIdentity)
    assert record.results["num_samples"] == 50
    assert record.results["best_energy"] == pytest.approx(-3.0, abs=1e-9)


def test_record_id_is_sortable_and_unique(tmp_archive, tiny_bqm):
    ids = [
        record_run(
            solver_name="cpu_sa_neal",
            instance_id="test/tiny",
            bqm=tiny_bqm,
            parameters={"num_reads": 10, "num_sweeps": 10, "seed": i},
            archive_samples=False,
        ).record_id
        for i in range(5)
    ]
    assert len(set(ids)) == 5
    assert ids == sorted(ids), "record IDs must be naturally sortable"


def test_repro_hash_is_deterministic():
    identity = SolverIdentity(
        name="x", version="1", source="y", hardware=None, parameter_schema={}
    )
    h1 = compute_repro_hash("v1", "inst/a", identity, {"num_reads": 100, "seed": 0})
    h2 = compute_repro_hash("v1", "inst/a", identity, {"seed": 0, "num_reads": 100})
    assert h1 == h2, "parameter dict order must not affect the hash"


def test_repro_hash_differs_on_parameter_change():
    identity = SolverIdentity(
        name="x", version="1", source="y", hardware=None, parameter_schema={}
    )
    h1 = compute_repro_hash("v1", "inst/a", identity, {"num_reads": 100})
    h2 = compute_repro_hash("v1", "inst/a", identity, {"num_reads": 200})
    h3 = compute_repro_hash("v1", "inst/b", identity, {"num_reads": 100})
    assert len({h1, h2, h3}) == 3


def test_sample_set_round_trips_through_archive(tmp_archive, tiny_bqm):
    record = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/tiny",
        bqm=tiny_bqm,
        parameters={"num_reads": 10, "num_sweeps": 20, "seed": 1},
    )
    assert record.sample_set_path is not None
    archived = load_archived_sample_set(record.record_id)
    assert isinstance(archived, dimod.SampleSet)
    assert len(archived) == 10


def test_load_record_round_trip(tmp_archive, tiny_bqm):
    record = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/tiny",
        bqm=tiny_bqm,
        parameters={"num_reads": 10, "num_sweeps": 10, "seed": 1},
    )
    reloaded = load_record(record.record_id)
    assert reloaded.record_id == record.record_id
    assert reloaded.solver == record.solver
    assert reloaded.repro_hash == record.repro_hash
    assert reloaded.results["best_energy"] == record.results["best_energy"]


def test_record_rejects_non_json_parameters(tmp_archive, tiny_bqm):
    with pytest.raises(TypeError, match=r"JSON-serializable"):
        record_run(
            solver_name="cpu_sa_neal",
            instance_id="test/tiny",
            bqm=tiny_bqm,
            parameters={"num_reads": 10, "broken": object()},
            archive_samples=False,
        )


def test_record_marks_convergence_when_optimum_reached(tmp_archive, tiny_bqm):
    """When the run lands on the known optimum, the record must say so —
    that's the dashboard's primary 'is this an actual optimum or a
    heuristic estimate?' signal."""
    record = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/tiny",
        bqm=tiny_bqm,
        parameters={"num_reads": 200, "num_sweeps": 200, "seed": 0},
        archive_samples=False,
        expected_optimum=-3.0,        # the actual BQM minimum
    )
    assert record.results["converged_to_expected"] is True
    assert record.results["expected_optimum"] == -3.0
    assert record.results["gap_to_expected"] == pytest.approx(0.0, abs=1e-9)


def test_record_marks_under_converged_when_optimum_missed(tmp_archive, tiny_bqm):
    """If the recorded best is materially worse than expected, the record
    must say so honestly — no rounding the gap away."""
    record = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/tiny",
        bqm=tiny_bqm,
        parameters={"num_reads": 200, "num_sweeps": 200, "seed": 0},
        archive_samples=False,
        expected_optimum=-100.0,      # impossible to reach on this BQM
    )
    assert record.results["converged_to_expected"] is False
    assert record.results["expected_optimum"] == -100.0
    assert record.results["gap_to_expected"] == pytest.approx(97.0, abs=1e-9)


def test_record_convergence_is_none_without_expected_optimum(tmp_archive, tiny_bqm):
    """No ground truth → no convergence claim, ever."""
    record = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/tiny",
        bqm=tiny_bqm,
        parameters={"num_reads": 50, "num_sweeps": 50, "seed": 0},
        archive_samples=False,
        expected_optimum=None,
    )
    assert record.results["converged_to_expected"] is None
    assert record.results["expected_optimum"] is None
    assert record.results["gap_to_expected"] is None
