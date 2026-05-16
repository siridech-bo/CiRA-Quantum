"""Phase 11 M3 — materialize a terminal pending cloud job into the
``solver_results`` column of its parent live-solve job.

When the orchestrator submits a real-QPU solver asynchronously, it
parks a ``PendingJob`` with ``target="solve_job:<parent_id>:<solver>"``
and immediately marks the row as ``status='queued'`` so the user's
solve completes with classical results visible. The pending-jobs
poller picks the entry up minutes-to-hours later when Origin or IBM
finishes, and calls into here to fill in the row.

This is intentionally separate from ``cloud_materialize.materialize``
(which writes to the benchmark archive). Different destination, same
upstream cloud result.
"""

from __future__ import annotations

import json
from typing import Any

import dimod
import numpy as np


def materialize_into_solve_job(entry, poll_cell: dict[str, Any]) -> None:
    """Convert a terminal cloud job into a completed solver_results row.

    ``entry`` is the ``PendingJob`` whose ``target`` starts with
    ``solve_job:``; ``poll_cell`` is the latest poll snapshot (carries
    the live cloud status + any error). Reads the BYOK key fresh, calls
    the right ``try_materialize()`` on the sampler, and patches the
    parent solve job's ``solver_results.solvers[<name>]`` row.
    """
    target = getattr(entry, "target", "")
    if not target.startswith("solve_job:"):
        raise RuntimeError(f"not a solve-job target: {target!r}")
    try:
        _, parent_job_id, solver_name = target.split(":", 2)
    except ValueError:
        raise RuntimeError(f"malformed target string: {target!r}")

    # Re-fetch the API key from the parent job's owning user; the key
    # might have been rotated between submit and materialize.
    api_key = _resolve_key_for_parent(parent_job_id, solver_name)
    if api_key is None:
        # Mark the row as error so the UI stops showing 'queued'.
        _patch_parent_job_row(
            parent_job_id, solver_name,
            row={
                "status": "error",
                "error": (
                    "Lost access to the BYOK key after submit. "
                    "Re-save your key and re-run the solve."
                ),
                "elapsed_ms": 0,
            },
        )
        return

    # If the cloud-side ended in error, surface that on the row.
    if poll_cell.get("live_error") and not poll_cell.get("has_probs"):
        _patch_parent_job_row(
            parent_job_id, solver_name,
            row={
                "status": "error",
                "error": f"Cloud-side error: {poll_cell['live_error']}",
                "elapsed_ms": 0,
                "cloud_job_id": entry.job_id,
                "backend_name": (entry.materialize_context or {}).get("backend_name"),
            },
        )
        return

    # Build the SampleSet from the cloud result using the same sampler
    # class that submitted (qaoa_originqc or qaoa_ibmq).
    sampleset_info = _try_materialize_via_sampler(
        solver_name=solver_name,
        cloud_job_id=entry.job_id,
        api_key=api_key,
        materialize_context=entry.materialize_context or {},
    )

    if sampleset_info is None:
        # The sampler said "not terminal yet" — leave the entry alone;
        # next poll cycle will retry.
        return

    if sampleset_info.get("status") == "error":
        _patch_parent_job_row(
            parent_job_id, solver_name,
            row={
                "status": "error",
                "error": sampleset_info.get("error", "materialization failed"),
                "elapsed_ms": 0,
                "cloud_job_id": entry.job_id,
            },
        )
        return

    # Success — build the public solver_results row.
    _patch_parent_job_row(
        parent_job_id, solver_name,
        row=sampleset_info["row"],
    )


def _resolve_key_for_parent(parent_job_id: str, solver_name: str) -> str | None:
    from app.config import KEY_ENCRYPTION_SECRET
    from app.crypto import decrypt_api_key
    from app.models import get_api_key_ciphertext, get_job
    provider_by_solver = {
        "qaoa_originqc": "originqc",
        "qaoa_ibmq": "ibm_quantum",
    }
    provider = provider_by_solver.get(solver_name)
    if provider is None:
        return None
    job = get_job(parent_job_id, is_admin=True)
    if job is None:
        return None
    cipher = get_api_key_ciphertext(int(job["user_id"]), provider)
    if cipher is None:
        return None
    try:
        return decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET)
    except ValueError:
        return None


def _try_materialize_via_sampler(
    *,
    solver_name: str,
    cloud_job_id: str,
    api_key: str,
    materialize_context: dict[str, Any],
) -> dict[str, Any] | None:
    """Call the sampler's try_materialize, then turn the result into
    a public solver_results row dict."""
    if solver_name == "qaoa_ibmq":
        from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
        sampler = QAOAIBMQSampler(
            api_key=api_key,
            backend_name=materialize_context.get("backend_name"),
            layer=int(materialize_context.get("layer") or 1),
            shots=int(materialize_context.get("shots") or 200),
        )
        try_result = sampler.try_materialize(cloud_job_id)
        if try_result is None or not try_result.get("terminal"):
            return None
        if try_result.get("status") == "error":
            return {"status": "error", "error": try_result.get("error", "")}
        return _row_from_ibmq_result(
            try_result, materialize_context, sampler=sampler,
        )

    if solver_name == "qaoa_originqc":
        from app.optimization.qaoa_cloud_sampler import QAOACloudSampler  # noqa: F401
        # Origin async parity ships in a later milestone; for now this
        # path isn't exercised because the orchestrator's submit_async
        # check fails for QAOACloudSampler.
        return {
            "status": "error",
            "error": "qaoa_originqc async materialization not yet implemented",
        }

    return {"status": "error", "error": f"unknown solver {solver_name!r}"}


def _row_from_ibmq_result(
    try_result: dict[str, Any],
    ctx: dict[str, Any],
    *,
    sampler,
) -> dict[str, Any]:
    """Convert a SamplerV2 PrimitiveResult into a public solver_results
    row, including the qaoa_extras payload Phase 10A's explainer panel
    consumes."""
    from app.pipeline.orchestrator import (
        _build_qaoa_extras,
        _finite_or_none,
        deserialize_bqm,
    )

    # Rebuild the BQM the submit step serialized.
    bqm_blob = ctx.get("bqm") or {}
    bqm = deserialize_bqm(bqm_blob)
    sense = ctx.get("sense", "minimize")
    variables = list(bqm.variables)
    n = len(variables)

    # Bridge: rebuild a "prepared"-like dict the sampler's
    # _sampleset_from_result expects.
    prepared = {
        "trivial": False,
        "n": n,
        "variables": variables,
        "linear_terms": [
            (variables.index(v), float(c)) for v, c in bqm.linear.items() if c != 0.0
        ],
        "quadratic_terms": [
            (
                sorted([variables.index(u), variables.index(v)])[0],
                sorted([variables.index(u), variables.index(v)])[1],
                float(c),
            )
            for (u, v), c in bqm.quadratic.items() if c != 0.0
        ],
        "gammas": ctx.get("trained_gammas", []),
        "betas": ctx.get("trained_betas", []),
        "train_loss": 0.0,
        "bqm_bin": bqm.change_vartype(dimod.BINARY, inplace=False),
    }
    submitted = {
        "trivial": False,
        "job_id": try_result.get("cloud_job_id", ""),
        "backend_name": ctx.get("backend_name", ""),
    }

    sampleset = sampler._sampleset_from_result(
        prepared, submitted, try_result["primitive_result"],
    )

    raw_energy = _finite_or_none(sampleset.first.energy)
    best_sample = dict(sampleset.first.sample)
    # CQM feasibility check isn't available here (we only have the BQM),
    # so we report feasibility=True when the BQM energy is finite. The
    # orchestrator's normal path uses cqm.check_feasible; for async
    # materialization that's not in scope — the cloud result is the
    # cloud result.
    feasible = raw_energy is not None
    obj_energy = raw_energy
    user_energy = (
        None if obj_energy is None
        else (obj_energy if sense == "minimize" else -obj_energy)
    )

    qaoa_extras = _build_qaoa_extras(
        sampleset, prepared["bqm_bin"],
        variables_in_bqm=list(prepared["bqm_bin"].variables),
        sense=sense,
        num_logical_vars=None,  # poller doesn't have access to the CQM
    )

    row: dict[str, Any] = {
        "status": "complete",
        "energy": user_energy,
        "raw_energy": raw_energy,
        "feasible": feasible,
        "elapsed_ms": 0,  # cloud wait time is open-ended; we don't capture it
        "cloud_job_id": try_result.get("cloud_job_id"),
        "backend_name": ctx.get("backend_name"),
        "tier_source": "qiskit-ibm-runtime",
        "version": ctx.get("version") or "",
        "hardware": "ibm-quantum-cloud",
    }
    if qaoa_extras is not None:
        row["qaoa_extras"] = qaoa_extras
    return {"status": "complete", "row": row}


def _patch_parent_job_row(
    parent_job_id: str, solver_name: str, *, row: dict[str, Any],
) -> None:
    """Atomically merge ``row`` into the parent job's
    ``solver_results.solvers[<solver_name>]`` field in SQLite, and
    emit a solver_progress SSE so any live viewers update."""
    from app.models import get_job, update_job
    from app.pipeline import get_event_bus

    job = get_job(parent_job_id, is_admin=True)
    if job is None:
        return  # parent was deleted — drop the result silently

    sr_raw = job.get("solver_results")
    if isinstance(sr_raw, str):
        try:
            sr = json.loads(sr_raw)
        except json.JSONDecodeError:
            sr = {"solvers": {}, "primary": None}
    elif isinstance(sr_raw, dict):
        sr = sr_raw
    else:
        sr = {"solvers": {}, "primary": None}

    solvers = sr.get("solvers") or {}
    existing = solvers.get(solver_name) or {}
    # Merge — keep any pre-existing fields (e.g. tier_source) that the
    # orchestrator stamped at submit time.
    merged = {**existing, **row}
    solvers[solver_name] = merged
    sr["solvers"] = solvers

    update_job(parent_job_id, solver_results=json.dumps(sr))
    try:
        get_event_bus().emit(parent_job_id, "solver_progress")
    except Exception:  # pragma: no cover — event bus is best-effort
        pass
