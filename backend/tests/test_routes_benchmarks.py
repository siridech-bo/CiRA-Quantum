"""Phase 5C — benchmarks route tests.

These exercise the public read surface against the *live* archive
on disk (``benchmarks/archive/``). Phase 2's foundation generates real
records as a side effect of running the suite CLI; these tests use the
same data the dashboard renders, so a regression in archive shape or
route projection lights up here.

If the archive is empty (fresh checkout), the suite-listing endpoint
must still respond with a well-formed empty body — a brand-new
contributor cloning the repo shouldn't 500 on the dashboard.
"""

from __future__ import annotations

import os

import pytest

from app.benchmarking.records import archive_path


@pytest.fixture
def app():
    from app import create_app
    return create_app(test_config={"TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


def _archive_has_records() -> bool:
    """True when the archive on disk contains at least one record JSON.
    Used to skip detail-endpoint tests on a fresh checkout."""
    archive_dir = archive_path("__placeholder__").parent
    if not archive_dir.exists():
        return False
    return any(
        p.endswith(".json") and "samples" not in p
        for p in os.listdir(archive_dir)
    )


# ---- /suites ---------------------------------------------------------------


def test_suites_endpoint_lists_all_registered(client):
    """Even on an empty archive the suites endpoint must return all
    registered suites (with 0 records each), since the dashboard's
    suite picker is built from this response."""
    r = client.get("/api/benchmarks/suites")
    assert r.status_code == 200
    data = r.get_json()
    assert "suites" in data
    suite_ids = {s["suite_id"] for s in data["suites"]}
    # The Phase-2 manifest registers these five.
    assert {
        "knapsack/small",
        "setcover/small",
        "graph_coloring/small",
        "maxcut/gset_subset",
        "jss/small",
    } <= suite_ids
    for s in data["suites"]:
        assert "num_records" in s
        assert "solvers_seen" in s
        assert isinstance(s["solvers_seen"], list)


def test_suites_endpoint_is_public(client):
    """No auth header / cookie. Must still return 200."""
    r = client.get("/api/benchmarks/suites")
    assert r.status_code == 200


# ---- /suites/<suite_id> ----------------------------------------------------


def test_suite_detail_unknown_returns_404(client):
    r = client.get("/api/benchmarks/suites/no/such/suite")
    assert r.status_code == 404
    assert r.get_json()["code"] == "NOT_FOUND"


def test_suite_detail_known_returns_instances_and_records(client):
    r = client.get("/api/benchmarks/suites/knapsack/small")
    assert r.status_code == 200
    data = r.get_json()
    assert data["suite_id"] == "knapsack/small"
    # The Phase-2 manifest has 2 instances in this suite.
    instance_ids = {i["instance_id"] for i in data["instances"]}
    assert "knapsack/small/knapsack_5item" in instance_ids
    # Records list is always present (may be empty on fresh checkout).
    assert isinstance(data["records"], list)


# ---- /solvers/<name> -------------------------------------------------------


def test_solver_detail_known_solver(client):
    """``cpu_sa_neal`` is registered unconditionally; if no records are
    archived for it yet, ``records_by_suite`` is empty but the solver
    identity still renders."""
    r = client.get("/api/benchmarks/solvers/cpu_sa_neal")
    assert r.status_code == 200
    data = r.get_json()
    assert data["solver"]["name"] == "cpu_sa_neal"
    assert data["is_currently_registered"] is True
    assert "records_by_suite" in data


def test_solver_detail_unknown_solver_returns_404_when_no_records(client):
    r = client.get("/api/benchmarks/solvers/nonexistent_solver_xyz")
    assert r.status_code == 404


# ---- /instances/<instance_id> ---------------------------------------------


@pytest.mark.skipif(not _archive_has_records(), reason="archive is empty")
def test_instance_leaderboard_sorted(client):
    """When records exist, the leaderboard is sorted by best_user_energy
    ascending (lower-is-better in minimize encoding), tie-broken by
    elapsed_ms ascending."""
    r = client.get("/api/benchmarks/instances/knapsack/small/knapsack_5item")
    if r.status_code == 404:
        pytest.skip("no records for this instance yet")
    data = r.get_json()
    leaderboard = data["leaderboard"]
    energies = [row["best_user_energy"] for row in leaderboard if row["best_user_energy"] is not None]
    assert energies == sorted(energies)


def test_instance_unknown_returns_404(client):
    r = client.get("/api/benchmarks/instances/no/such/instance/_id")
    assert r.status_code == 404


# ---- /records/<id> + /cite -------------------------------------------------


@pytest.mark.skipif(not _archive_has_records(), reason="archive is empty")
def test_record_detail_and_cite_roundtrip(client):
    """Pick one archived record, GET its detail, then GET its citation.
    Both must succeed and the citation must reference the record_id."""
    r = client.get("/api/benchmarks/suites/knapsack/small")
    data = r.get_json()
    if not data["records"]:
        pytest.skip("no archived records for knapsack/small yet")
    record_id = data["records"][0]["record_id"]

    detail = client.get(f"/api/benchmarks/records/{record_id}")
    assert detail.status_code == 200
    record = detail.get_json()["record"]
    assert record["record_id"] == record_id
    assert "repro_hash" in record
    assert "parameters" in record
    assert "solver" in record

    cite_bibtex = client.get(f"/api/benchmarks/records/{record_id}/cite")
    assert cite_bibtex.status_code == 200
    bib = cite_bibtex.get_json()
    assert bib["kind"] == "bibtex"
    # Cite key in the BibTeX entry has the dots and colons stripped from
    # the record_id (per app.benchmarking.cite.bibtex_entry).
    assert "@misc{" in bib["citation"]

    cite_string = client.get(
        f"/api/benchmarks/records/{record_id}/cite?kind=string"
    )
    assert cite_string.status_code == 200
    assert cite_string.get_json()["kind"] == "string"
    assert record_id in cite_string.get_json()["citation"]


def test_record_unknown_returns_404(client):
    r = client.get("/api/benchmarks/records/no-such-record")
    assert r.status_code == 404
    assert r.get_json()["code"] == "NOT_FOUND"


# ---- /findings -------------------------------------------------------------


def test_findings_endpoint_is_public(client):
    """Findings aggregation is part of the public scoreboard surface."""
    r = client.get("/api/benchmarks/findings")
    assert r.status_code == 200


def test_findings_endpoint_shape(client):
    """The findings response must carry the keys the dashboard's
    BenchmarkFindingsPage reads."""
    r = client.get("/api/benchmarks/findings")
    assert r.status_code == 200
    data = r.get_json()
    assert "cells" in data
    assert "solver_summaries" in data
    assert isinstance(data["cells"], list)
    assert isinstance(data["solver_summaries"], list)


@pytest.mark.skipif(not _archive_has_records(), reason="archive is empty")
def test_findings_cells_contain_aggregations(client):
    """When records exist, each cell carries best/mean/worst energy plus
    n_runs and a latest_record_id pointer."""
    r = client.get("/api/benchmarks/findings")
    cells = r.get_json()["cells"]
    if not cells:
        pytest.skip("no aggregated cells yet")

    sample = cells[0]
    for key in (
        "solver_name",
        "instance_id",
        "n_runs",
        "best_user_energy",
        "mean_user_energy",
        "worst_user_energy",
        "best_elapsed_ms",
        "latest_record_id",
    ):
        assert key in sample, f"missing key {key!r}"

    # Best ≤ mean ≤ worst for any cell with energy data and n_runs >= 1.
    for c in cells:
        if c["best_user_energy"] is None:
            continue
        assert c["best_user_energy"] <= c["mean_user_energy"] <= c["worst_user_energy"]


@pytest.mark.skipif(not _archive_has_records(), reason="archive is empty")
def test_findings_solver_summaries_consistency(client):
    """instances_with_match must be <= instances_attempted; total_runs
    must equal the sum of n_runs across that solver's cells."""
    data = client.get("/api/benchmarks/findings").get_json()
    cells = data["cells"]
    summaries = {s["solver_name"]: s for s in data["solver_summaries"]}

    by_solver_cells: dict = {}
    by_solver_runs: dict = {}
    for c in cells:
        by_solver_cells.setdefault(c["solver_name"], 0)
        by_solver_cells[c["solver_name"]] += 1
        by_solver_runs.setdefault(c["solver_name"], 0)
        by_solver_runs[c["solver_name"]] += c["n_runs"]

    for solver, summary in summaries.items():
        assert summary["instances_with_match"] <= summary["instances_attempted"]
        assert summary["instances_attempted"] == by_solver_cells.get(solver, 0)
        assert summary["total_runs"] == by_solver_runs.get(solver, 0)


def test_cite_unknown_record_returns_404(client):
    r = client.get("/api/benchmarks/records/no-such-record/cite")
    assert r.status_code == 404


def test_cite_unknown_kind_returns_400(client):
    """A request with ``kind=unknown`` must 400, not silently fall back
    to the bibtex default."""
    if not _archive_has_records():
        pytest.skip("archive empty")
    suite = client.get("/api/benchmarks/suites/knapsack/small").get_json()
    if not suite["records"]:
        pytest.skip("no records in suite")
    record_id = suite["records"][0]["record_id"]
    r = client.get(f"/api/benchmarks/records/{record_id}/cite?kind=invalid")
    assert r.status_code == 400
    assert r.get_json()["code"] == "BAD_REQUEST"
