"""Phase 9B+ — pending cloud-job manager.

The synchronous ``record_run`` path works for solvers that return
quickly (CP-SAT in milliseconds, GPU SA in seconds). It does not
work for the real-QPU path: Origin Quantum's Wukong queue can hold
a job for tens of minutes, and a Flask request handler can't block
that long.

The pending-jobs module is the workaround. When a cloud sampler
submits a job that won't return immediately, the caller saves a
``PendingJob`` record describing what the job *would* eventually
become — the solver, the instance, the parameters used, plus enough
context to rebuild the lowered BQM and recompute per-sample energies
when the result arrives.

A polling endpoint (``GET /api/benchmarks/cloud-jobs/pending``) reads
this list, queries each job's current cloud status, and reports back.
When a job transitions to ``COMPLETED``, the materialization endpoint
(``POST /api/benchmarks/cloud-jobs/<id>/materialize``) converts the
cloud probability distribution into a proper ``RunRecord`` in
``benchmarks/archive/`` and removes the entry from the pending list.

Storage is a single JSON file at ``backend/data/pending_cloud_jobs.json``
so a fresh checkout doesn't carry someone else's pending jobs. The
schema is forward-compatible: unknown fields are preserved on
round-trip.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# State file lives next to the SQLite DB. Same directory the schema
# expects to exist (kept alive via .gitkeep).
_STATE_FILE = Path(__file__).parent.parent.parent / "data" / "pending_cloud_jobs.json"
_LOCK = threading.Lock()


@dataclass
class PendingJob:
    """A cloud-submitted job we're tracking until it completes."""

    job_id: str                       # cloud-side identifier
    solver_name: str                  # e.g. "qaoa_originqc"
    instance_id: str                  # e.g. "setcover/small/setcover_4item"
    parameters: dict[str, Any]        # the params dict passed to record_run
    lagrange_multiplier: float        # used to re-lower the CQM at materialize time
    submitted_at: str                 # ISO-8601 UTC
    notes: str = ""                   # optional free-form annotation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PendingJob:
        # Forward-compat: allow extra keys we don't know about
        known = {f for f in cls.__dataclass_fields__}
        kept = {k: v for k, v in d.items() if k in known}
        return cls(**kept)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _state_file() -> Path:
    """Allow tests to monkey-patch the location via this hook."""
    return _STATE_FILE


def _load_all() -> list[PendingJob]:
    path = _state_file()
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [PendingJob.from_dict(d) for d in data.get("jobs", [])]


def _save_all(jobs: list[PendingJob]) -> None:
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"jobs": [j.to_dict() for j in jobs]}, f, indent=2)


def list_pending() -> list[PendingJob]:
    """Snapshot of currently-pending jobs. Order is the order they
    were added (oldest first)."""
    with _LOCK:
        return list(_load_all())


def add(job: PendingJob) -> None:
    """Append a new pending job. Idempotent on ``job_id`` — a
    re-add with the same job_id replaces the existing entry."""
    with _LOCK:
        jobs = _load_all()
        jobs = [j for j in jobs if j.job_id != job.job_id]
        jobs.append(job)
        _save_all(jobs)


def remove(job_id: str) -> bool:
    """Drop the given pending entry. Returns True if removed."""
    with _LOCK:
        jobs = _load_all()
        before = len(jobs)
        jobs = [j for j in jobs if j.job_id != job_id]
        if len(jobs) == before:
            return False
        _save_all(jobs)
        return True


def get(job_id: str) -> PendingJob | None:
    """Lookup a single pending entry by job_id."""
    with _LOCK:
        for j in _load_all():
            if j.job_id == job_id:
                return j
    return None
