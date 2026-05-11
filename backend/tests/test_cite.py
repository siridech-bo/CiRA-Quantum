"""Phase 2 v2 — citation generator tests."""

from __future__ import annotations

from pathlib import Path

import dimod
import pytest

from app.benchmarking import record_run
from app.benchmarking.cite import bibtex_entry, cite, short_citation


@pytest.fixture
def tmp_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIRA_BENCH_ARCHIVE", str(tmp_path))
    return tmp_path


@pytest.fixture
def example_record(tmp_archive):
    bqm = dimod.BinaryQuadraticModel({0: -1.0, 1: 1.0}, {(0, 1): 2.0}, 0.0, dimod.BINARY)
    return record_run(
        solver_name="cpu_sa_neal",
        instance_id="test/cite",
        bqm=bqm,
        parameters={"num_reads": 5, "num_sweeps": 10, "seed": 0},
        archive_samples=False,
    )


def test_bibtex_entry_well_formed(example_record):
    out = bibtex_entry(example_record)
    assert out.startswith("@misc{")
    assert out.rstrip().endswith("}")
    assert example_record.repro_hash in out
    assert example_record.solver.name in out
    assert example_record.instance_id in out
    # @misc + cite key + year + month + at least 2 more fields
    assert out.count("=") >= 5


def test_citation_string_includes_record_id_and_date(example_record):
    out = short_citation(example_record)
    assert example_record.record_id in out
    assert example_record.repro_hash in out
    assert example_record.started_at.strftime("%Y-%m-%d") in out


def test_cite_unknown_record_returns_clear_error(tmp_archive):
    with pytest.raises(FileNotFoundError, match=r"No archived record"):
        cite("bogus_id_does_not_exist")


def test_cli_emits_bibtex_to_stdout(tmp_archive, example_record, capsys):
    from app.benchmarking.cite import _main

    rc = _main([example_record.record_id])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("@misc{")
    assert example_record.repro_hash in captured.out
