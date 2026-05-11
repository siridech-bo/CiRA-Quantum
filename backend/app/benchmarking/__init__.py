"""CiRA Quantum benchmarking foundation (Phase 2 v2).

Public surface:
    SolverIdentity, register_solver, get_solver, list_solvers
    RunRecord, record_run, replay_record, archive_path, load_record

The submodules (`registry`, `records`) are kept lightweight on import so
the heavy adapters (GPU SA, dwave-samplers) are only loaded when their
solver is actually requested.
"""

from __future__ import annotations

from app.benchmarking.records import (
    RunRecord,
    archive_path,
    load_record,
    record_run,
    replay_record,
)
from app.benchmarking.registry import (
    SolverIdentity,
    bootstrap_default_solvers,
    get_solver,
    list_solvers,
    register_solver,
)

__all__ = [
    "RunRecord",
    "SolverIdentity",
    "archive_path",
    "bootstrap_default_solvers",
    "get_solver",
    "list_solvers",
    "load_record",
    "record_run",
    "register_solver",
    "replay_record",
]

# Auto-register the three baseline solver tiers on first import. This keeps
# `app.benchmarking` self-bootstrapping for tests and CLI alike, but the
# registration is guarded so a second import (or an explicit re-bootstrap)
# is a no-op.
bootstrap_default_solvers()
