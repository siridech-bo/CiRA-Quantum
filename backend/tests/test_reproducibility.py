"""Phase 2 v2 — replay and three-solver agreement tests.

Replay verifies that re-executing a recorded run on the same code version
with the same seed lands on the same energy. Spec drift (a different
code version) is reported, not asserted away.
"""

from __future__ import annotations

from pathlib import Path

import dimod
import pytest

from app.benchmarking import record_run, replay_record


@pytest.fixture
def tmp_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIRA_BENCH_ARCHIVE", str(tmp_path))
    return tmp_path


@pytest.fixture
def tiny_bqm() -> dimod.BinaryQuadraticModel:
    return dimod.BinaryQuadraticModel(
        {0: 1.0, 1: -3.0, 2: 1.0},
        {(0, 1): 1.0, (1, 2): 1.0},
        0.0,
        dimod.BINARY,
    )


def test_replay_with_same_seed_produces_identical_record(tmp_archive, tiny_bqm):
    original = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/repro",
        bqm=tiny_bqm,
        parameters={"num_reads": 30, "num_sweeps": 200, "seed": 42},
        archive_samples=False,
    )
    result = replay_record(original.record_id, bqm=tiny_bqm)
    assert result.agree, result.notes
    assert result.replayed.repro_hash == original.repro_hash
    assert result.replayed.results["best_energy"] == pytest.approx(
        original.results["best_energy"], abs=1e-12
    )


def test_replay_with_different_code_version_warns(tmp_archive, tiny_bqm, monkeypatch):
    """Simulate spec drift by hand-mutating the archived record's
    code_version. ``replay_record`` is expected to log it in ``notes`` and
    set ``agree=False`` — drift is the signal, not the failure."""
    original = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/repro",
        bqm=tiny_bqm,
        parameters={"num_reads": 20, "num_sweeps": 100, "seed": 7},
        archive_samples=False,
    )
    # Rewrite the archived record with a stale code_version.
    import json

    from app.benchmarking import archive_path
    record_path = archive_path(original.record_id)
    payload = json.loads(record_path.read_text())
    payload["code_version"] = "ancient-sha-deadbeef"
    record_path.write_text(json.dumps(payload))

    result = replay_record(original.record_id, bqm=tiny_bqm)
    assert not result.agree
    assert any("code_version drift" in n for n in result.notes), result.notes


def test_three_solver_agreement_recorded_correctly(tmp_archive, tiny_bqm):
    """Run all three baseline solvers on the same tiny BQM and verify each
    record reaches the same minimum energy."""
    cqm = dimod.ConstrainedQuadraticModel.from_bqm(tiny_bqm)

    expected = -3.0  # the BQM's optimum: x_1 = 1, others 0
    energies: dict[str, float | None] = {}

    cpu_record = record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/three_solver",
        bqm=tiny_bqm,
        parameters={"num_reads": 200, "num_sweeps": 200, "seed": 0},
        archive_samples=False,
    )
    energies["cpu_sa_neal"] = cpu_record.results["best_energy"]

    exact_record = record_run(
        solver_name="exact_cqm",
        instance_id="test/three_solver",
        bqm=tiny_bqm,
        parameters={},
        archive_samples=False,
        cqm=cqm,
    )
    energies["exact_cqm"] = exact_record.results["best_energy"]

    try:
        import torch

        if torch.cuda.is_available():
            gpu_record = record_run(
                solver_name="gpu_sa",
                instance_id="test/three_solver",
                bqm=tiny_bqm,
                parameters={"num_reads": 200, "num_sweeps": 200, "seed": 0},
                archive_samples=False,
            )
            energies["gpu_sa"] = gpu_record.results["best_energy"]
    except ImportError:
        pass

    for solver, e in energies.items():
        assert e == pytest.approx(expected, abs=1e-6), f"{solver} reported {e}"
