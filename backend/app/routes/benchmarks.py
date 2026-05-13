"""Phase 5C — Benchmark dashboard API.

Read-only public endpoints that surface the contents of
``benchmarks/archive/`` to the frontend dashboard. **All endpoints are
public — no ``@login_required``.** The Benchmark archive is the
platform's public scoreboard; reading it does not require
authentication. Writing to it (running a new benchmark) still requires
the Phase-2 CLI and a logged-in user with access to the host.

Endpoints
~~~~~~~~~

* ``GET /api/benchmarks/suites``
    List of registered instance suites with summary counts. The
    dashboard's landing page uses this to populate the suite picker.

* ``GET /api/benchmarks/suites/<suite_id>``
    For one suite, the full grid of ``(instance × solver)`` records.
    Path components in suite IDs use ``/`` so the URL takes them as
    nested path segments (``/api/benchmarks/suites/knapsack/small``).

* ``GET /api/benchmarks/solvers/<solver_name>``
    All records produced by one solver, grouped by suite. Powers the
    Solver profile page (version history, performance over time).

* ``GET /api/benchmarks/instances/<instance_id>``
    Leaderboard for one instance — every solver run on it, sorted by
    best energy.

* ``GET /api/benchmarks/records/<record_id>``
    Full RunRecord JSON, with all the reproducibility metadata.

* ``GET /api/benchmarks/records/<record_id>/cite[?kind=bibtex|string]``
    Generates a citation entry for the record.

Performance
~~~~~~~~~~~

The archive is read at request time on every call — there is no cache.
44 records loads in ~30 ms. When the archive crosses ~10k records the
hot path will need a per-suite cache; that's tracked as a Phase-5C
follow-up.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from app.benchmarking import bootstrap_default_solvers, list_solvers
from app.benchmarking.cite import bibtex_entry, short_citation
from app.benchmarking.instances import get_suite, list_suites
from app.benchmarking.records import (
    RunRecord,
    archive_path,
    load_record,
)

benchmarks_bp = Blueprint("benchmarks", __name__)


def _archive_dir() -> Path:
    """Return the path to ``benchmarks/archive/``. Kept as a function so
    test fixtures can monkey-patch without import-time binding."""
    return archive_path("__placeholder__").parent


def _list_records_iter():
    """Iterate every archived ``RunRecord`` once. The implementation
    reads the JSON file directly (skipping the gzipped sample sets,
    which are large and not needed for the dashboard's overview)."""
    pattern = str(_archive_dir() / "*.json")
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_samples.json") or path.endswith("_samples.json.gz"):
            continue
        try:
            with open(path) as f:
                yield RunRecord.from_dict(json.load(f))
        except Exception:  # pragma: no cover — corrupt file should not 500 the page
            continue


def _record_summary(r: RunRecord) -> dict[str, Any]:
    """Compact projection used by list endpoints. Excludes the
    parameters / sample_set_path / reproducibility fields — those land
    on the detail endpoint."""
    return {
        "record_id": r.record_id,
        "solver_name": r.solver.name,
        "solver_version": r.solver.version,
        "instance_id": r.instance_id,
        "hardware_id": r.hardware_id,
        "started_at": r.started_at.isoformat(),
        "best_user_energy": r.results.get("best_user_energy"),
        "best_energy": r.results.get("best_energy"),
        "num_feasible": r.results.get("num_feasible"),
        "num_samples": r.results.get("num_samples"),
        "elapsed_ms": r.results.get("elapsed_ms"),
        "converged_to_expected": r.results.get("converged_to_expected"),
        "gap_to_expected": r.results.get("gap_to_expected"),
        "expected_optimum": r.results.get("expected_optimum"),
    }


@benchmarks_bp.route("/suites", methods=["GET"])
def suites():
    """List every registered suite, with the count of records and the
    set of solvers seen in the archive for it."""
    bootstrap_default_solvers()

    counts: dict[str, int] = defaultdict(int)
    solvers_per_suite: dict[str, set[str]] = defaultdict(set)
    for r in _list_records_iter():
        suite_id = r.instance_id.rsplit("/", 1)[0]
        counts[suite_id] += 1
        solvers_per_suite[suite_id].add(r.solver.name)

    suites_list = []
    for suite_id in list_suites():
        suites_list.append({
            "suite_id": suite_id,
            "num_records": counts.get(suite_id, 0),
            "solvers_seen": sorted(solvers_per_suite.get(suite_id, set())),
        })
    return jsonify({"suites": suites_list})


@benchmarks_bp.route("/suites/<path:suite_id>", methods=["GET"])
def suite_detail(suite_id: str):
    """Return all RunRecord summaries belonging to one suite, plus the
    set of instances in the suite (so the frontend can render empty
    cells for instances that no solver has been run on yet)."""
    try:
        instances = get_suite(suite_id)
    except KeyError:
        return jsonify({
            "error": f"unknown suite {suite_id!r}",
            "code": "NOT_FOUND",
        }), 404

    records = [
        _record_summary(r)
        for r in _list_records_iter()
        if r.instance_id.startswith(suite_id + "/")
    ]
    # The instance manifest gives us the canonical instance list and any
    # expected_optimum / metadata the dashboard wants to surface.
    instance_summaries = [
        {
            "instance_id": inst.instance_id,
            "problem_class": inst.problem_class,
            "expected_optimum": inst.expected_optimum,
            "expected_optimum_kind": inst.expected_optimum_kind,
            "tags": list(inst.tags),
        }
        for inst in instances
    ]
    return jsonify({
        "suite_id": suite_id,
        "instances": instance_summaries,
        "records": records,
    })


@benchmarks_bp.route("/solvers/<solver_name>", methods=["GET"])
def solver_detail(solver_name: str):
    """Return the SolverIdentity and every record produced by it, grouped
    by suite. The frontend renders this as a per-solver profile."""
    bootstrap_default_solvers()
    identity = next(
        (s for s in list_solvers() if s.name == solver_name),
        None,
    )

    records_by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in _list_records_iter():
        if r.solver.name != solver_name:
            continue
        suite = r.instance_id.rsplit("/", 1)[0]
        records_by_suite[suite].append(_record_summary(r))

    # 404 only if the solver has no records AND isn't registered — a
    # currently-unregistered solver with archived records is still a
    # legitimate page to render (this is the "spec drift" signal from
    # Phase 2 — old runs from a since-removed solver still cite-able).
    if identity is None and not records_by_suite:
        return jsonify({
            "error": f"unknown solver {solver_name!r}",
            "code": "NOT_FOUND",
        }), 404

    return jsonify({
        "solver": identity.to_dict() if identity else {"name": solver_name},
        "is_currently_registered": identity is not None,
        "records_by_suite": dict(records_by_suite),
    })


@benchmarks_bp.route("/instances/<path:instance_id>", methods=["GET"])
def instance_detail(instance_id: str):
    """Return the leaderboard for one instance: every record sorted by
    best_user_energy (ascending — lower is better in minimize-encoded
    energy, regardless of the original problem's sense)."""
    leaderboard = [
        _record_summary(r)
        for r in _list_records_iter()
        if r.instance_id == instance_id
    ]
    if not leaderboard:
        return jsonify({
            "error": f"no records found for instance {instance_id!r}",
            "code": "NOT_FOUND",
        }), 404

    # Sort by best_user_energy; treat None as +inf so unsolved instances
    # land at the bottom rather than mid-list.
    leaderboard.sort(
        key=lambda r: (
            r["best_user_energy"] if r["best_user_energy"] is not None else float("inf"),
            r["elapsed_ms"] if r["elapsed_ms"] is not None else float("inf"),
        )
    )
    return jsonify({
        "instance_id": instance_id,
        "leaderboard": leaderboard,
    })


@benchmarks_bp.route("/records/<record_id>", methods=["GET"])
def record_detail(record_id: str):
    """Return the full ``RunRecord.to_dict()``, including parameters,
    repro_hash, and the path to the archived SampleSet."""
    try:
        record = load_record(record_id)
    except FileNotFoundError:
        return jsonify({
            "error": f"record {record_id!r} not found in archive",
            "code": "NOT_FOUND",
        }), 404
    return jsonify({"record": record.to_dict()})


@benchmarks_bp.route("/findings", methods=["GET"])
def findings():
    """Aggregated per-(solver, instance) stats across all archived
    records. Powers the Phase-5C Findings page — the dashboard's
    "what does the archive actually say" view.

    For each (solver, instance) pair with at least one record:
      * ``n_runs`` — total records
      * ``best_user_energy`` — minimum across runs (lower is better in
        the minimize-encoded space)
      * ``mean_user_energy`` — arithmetic mean across runs
      * ``worst_user_energy`` — maximum across runs
      * ``std_user_energy`` — sample standard deviation (None for n<2)
      * ``best_elapsed_ms`` / ``mean_elapsed_ms`` — same for wall time
      * ``convergence_rate`` — fraction of runs where
        ``converged_to_expected`` is True (None if the instance has
        no documented optimum)

    The response also surfaces per-instance metadata
    (``expected_optimum``, ``problem_class``) so the frontend can render
    "X out of N runs hit the documented optimum" labels.
    """
    from collections import defaultdict
    from statistics import StatisticsError, stdev

    bootstrap_default_solvers()

    # First pass: bucket records by (solver, instance).
    cells: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    instances_seen: dict[str, dict[str, Any]] = {}
    for r in _list_records_iter():
        cells[(r.solver.name, r.instance_id)].append(r)
        if r.instance_id not in instances_seen:
            # Source the canonical expected_optimum off the record (the
            # instance manifest is the ground truth, and the record was
            # tagged with it at run time).
            instances_seen[r.instance_id] = {
                "expected_optimum": r.results.get("expected_optimum"),
            }

    # Second pass: aggregate per cell.
    results: list[dict[str, Any]] = []
    for (solver_name, instance_id), recs in sorted(cells.items()):
        energies = [
            float(r.results["best_user_energy"])
            for r in recs
            if r.results.get("best_user_energy") is not None
        ]
        elapsed = [
            float(r.results["elapsed_ms"])
            for r in recs
            if r.results.get("elapsed_ms") is not None
        ]
        converged = [
            bool(r.results["converged_to_expected"])
            for r in recs
            if r.results.get("converged_to_expected") is not None
        ]

        try:
            energy_std = stdev(energies) if len(energies) >= 2 else None
        except StatisticsError:  # pragma: no cover
            energy_std = None

        results.append({
            "solver_name": solver_name,
            "instance_id": instance_id,
            "expected_optimum": instances_seen[instance_id]["expected_optimum"],
            "n_runs": len(recs),
            "best_user_energy": min(energies) if energies else None,
            "mean_user_energy": (sum(energies) / len(energies)) if energies else None,
            "worst_user_energy": max(energies) if energies else None,
            "std_user_energy": energy_std,
            "best_elapsed_ms": min(elapsed) if elapsed else None,
            "mean_elapsed_ms": (sum(elapsed) / len(elapsed)) if elapsed else None,
            "convergence_rate": (
                sum(converged) / len(converged) if converged else None
            ),
            # Pin the most-recent record id so the dashboard can cite
            # something specific when displaying a cell.
            "latest_record_id": max(recs, key=lambda r: r.started_at).record_id,
        })

    # Per-solver headline stats: how many instances has this solver run
    # on, and on how many of them did it hit the documented optimum
    # at least once.
    solver_summaries: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"instances_attempted": 0, "instances_with_match": 0, "total_runs": 0}
    )
    for cell in results:
        s = cell["solver_name"]
        solver_summaries[s]["instances_attempted"] += 1
        solver_summaries[s]["total_runs"] += cell["n_runs"]
        if cell["convergence_rate"] is not None and cell["convergence_rate"] > 0:
            solver_summaries[s]["instances_with_match"] += 1

    return jsonify({
        "cells": results,
        "solver_summaries": [
            {"solver_name": s, **stats}
            for s, stats in sorted(solver_summaries.items())
        ],
    })


@benchmarks_bp.route("/cloud-jobs/pending", methods=["GET"])
def cloud_jobs_pending():
    """List currently-pending cloud jobs with their live cloud status.

    For each pending entry, queries the cloud for the latest
    ``JobStatus``, error message (if any), and whether probability
    counts are available. The dashboard polls this endpoint every 30 s
    and auto-fires ``POST /materialize`` when a job becomes ready.

    No auth required (matches the rest of ``/api/benchmarks/*``).
    """
    from app.benchmarking import pending_jobs as pj

    api_key_path = os.environ.get("QPANDA_API_KEY_FILE")
    if not api_key_path or not Path(api_key_path).exists():
        return jsonify({
            "error": "QPANDA_API_KEY_FILE not configured; can't query cloud status",
            "code": "NO_CREDENTIAL",
            "pending": [],
        }), 503

    api_key = Path(api_key_path).read_text(encoding="utf-8").strip()
    try:
        import pyqpanda3.qcloud as qcloud_mod
        qcloud_mod.QCloudService(api_key=api_key)  # auth
    except Exception as e:  # noqa: BLE001
        return jsonify({
            "error": f"cloud auth failed: {type(e).__name__}: {e}",
            "code": "AUTH_FAILED",
            "pending": [],
        }), 502

    pendings = pj.list_pending()
    out = []
    for entry in pendings:
        cell = {
            "job_id": entry.job_id,
            "solver_name": entry.solver_name,
            "instance_id": entry.instance_id,
            "parameters": entry.parameters,
            "submitted_at": entry.submitted_at,
            "notes": entry.notes,
            "live_status": "UNKNOWN",
            "live_error": "",
            "has_probs": False,
        }
        try:
            import pyqpanda3.qcloud as qcloud_mod
            job = qcloud_mod.QCloudJob(entry.job_id)
            q = job.query()
            status = q.job_status()
            cell["live_status"] = (
                str(status).rsplit(".", 1)[-1] if hasattr(status, "name") else str(status)
            )
            err = q.error_message() or ""
            cell["live_error"] = err
            cell["has_probs"] = bool(q.get_probs())
        except Exception as e:  # noqa: BLE001
            cell["live_error"] = f"{type(e).__name__}: {e}"
        out.append(cell)

    return jsonify({"pending": out})


@benchmarks_bp.route("/cloud-jobs/<job_id>/materialize", methods=["POST"])
def cloud_jobs_materialize(job_id: str):
    """Convert a completed pending job into a ``RunRecord`` in the
    archive. Removes the entry from the pending list on success.

    Idempotent: a second call after the job is already archived
    returns ``404 NOT_FOUND`` because the pending entry is gone.
    """
    from app.benchmarking import pending_jobs as pj
    from app.benchmarking.cloud_materialize import (
        JobErroredError,
        JobNotReadyError,
        materialize,
    )

    api_key_path = os.environ.get("QPANDA_API_KEY_FILE")
    if not api_key_path or not Path(api_key_path).exists():
        return jsonify({
            "error": "QPANDA_API_KEY_FILE not configured",
            "code": "NO_CREDENTIAL",
        }), 503

    pending = pj.get(job_id)
    if pending is None:
        return jsonify({
            "error": f"no pending job with id {job_id!r}",
            "code": "NOT_FOUND",
        }), 404

    api_key = Path(api_key_path).read_text(encoding="utf-8").strip()
    try:
        record_id = materialize(pending, api_key)
    except JobNotReadyError as e:
        return jsonify({
            "error": str(e),
            "code": "NOT_READY",
        }), 409
    except JobErroredError as e:
        # Don't auto-remove from pending — let the operator decide
        # whether to drop it or wait.
        return jsonify({
            "error": str(e),
            "code": "CLOUD_ERROR",
        }), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({
            "error": f"materialize failed: {type(e).__name__}: {e}",
            "code": "INTERNAL",
        }), 500

    return jsonify({
        "success": True,
        "record_id": record_id,
        "job_id": job_id,
    })


@benchmarks_bp.route("/cloud-jobs/<job_id>", methods=["DELETE"])
def cloud_jobs_remove(job_id: str):
    """Drop a pending entry without materializing it. Used to clear
    entries for jobs that errored out on the cloud side and won't
    ever produce a usable result."""
    from app.benchmarking import pending_jobs as pj

    if pj.remove(job_id):
        return jsonify({"success": True, "removed": job_id})
    return jsonify({
        "error": f"no pending job with id {job_id!r}",
        "code": "NOT_FOUND",
    }), 404


@benchmarks_bp.route("/records/<record_id>/cite", methods=["GET"])
def record_cite(record_id: str):
    """Generate a citation for one record. ``kind=bibtex`` (default) or
    ``kind=string`` (short inline citation)."""
    try:
        record = load_record(record_id)
    except FileNotFoundError:
        return jsonify({
            "error": f"record {record_id!r} not found in archive",
            "code": "NOT_FOUND",
        }), 404

    kind = (request.args.get("kind") or "bibtex").lower()
    if kind == "string":
        return jsonify({"kind": "string", "citation": short_citation(record)})
    if kind == "bibtex":
        return jsonify({"kind": "bibtex", "citation": bibtex_entry(record)})
    return jsonify({
        "error": f"unknown citation kind {kind!r}; expected 'bibtex' or 'string'",
        "code": "BAD_REQUEST",
    }), 400
