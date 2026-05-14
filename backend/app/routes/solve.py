"""Solve & jobs endpoints.

    POST   /api/solve                    submit a problem, returns job_id
    GET    /api/jobs                     paginated list (current user)
    GET    /api/jobs/<id>                full detail
    GET    /api/jobs/<id>/stream         SSE stream of status updates
    DELETE /api/jobs/<id>                delete one job

The POST returns immediately with ``{"job": {"id", "status": "queued"}}``;
the pipeline runs on a background daemon thread (Phase 6 swaps this for
an RQ worker). The SSE route is the live-progress channel; the GET-by-id
route is the resting state for clients that don't keep an EventSource
open.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from flask import Blueprint, Response, jsonify, request

from app.auth import get_current_user, login_required
from app.config import KEY_ENCRYPTION_SECRET, MAX_PROMPT_LENGTH
from app.crypto import decrypt_api_key
from app.formulation import list_providers
from app.models import (
    create_job,
    delete_job,
    get_api_key_ciphertext,
    get_job,
    list_jobs,
)
from app.pipeline import Orchestrator, get_event_bus

# Phase 5D — multi-solver pickers. Tier metadata isn't carried on
# ``SolverIdentity`` itself (the registry is intentionally minimal), so
# we keep the name→tier map here. Adding a new solver to the registry
# without a tier entry falls through to "other" — the UI handles that.
_SOLVER_TIERS: dict[str, dict[str, Any]] = {
    "exact_cqm":             {"tier": "classical_exact",   "tier_label": "Classical SOTA",         "tier_color": "success", "recommended_default": False, "warning": "Exponential time on large CQMs — keep instances small.", "requires_key": None},
    "cpsat":                 {"tier": "classical_exact",   "tier_label": "Classical SOTA",         "tier_color": "success", "recommended_default": True,  "warning": None, "requires_key": None},
    "highs":                 {"tier": "classical_exact",   "tier_label": "Classical SOTA",         "tier_color": "success", "recommended_default": True,  "warning": None, "requires_key": None},
    "gpu_sa":                {"tier": "qubo_heuristic",    "tier_label": "QUBO heuristic (GPU)",   "tier_color": "primary", "recommended_default": True,  "warning": None, "requires_key": None},
    "cpu_sa_neal":           {"tier": "qubo_heuristic",    "tier_label": "QUBO heuristic (CPU)",   "tier_color": "primary", "recommended_default": True,  "warning": None, "requires_key": None},
    "parallel_tempering":    {"tier": "quantum_inspired",  "tier_label": "Quantum-inspired",       "tier_color": "info",    "recommended_default": True,  "warning": None, "requires_key": None},
    "simulated_bifurcation": {"tier": "quantum_inspired",  "tier_label": "Quantum-inspired",       "tier_color": "info",    "recommended_default": True,  "warning": None, "requires_key": None},
    "qaoa_sim":              {"tier": "quantum_simulator", "tier_label": "Quantum (simulator)",    "tier_color": "warning", "recommended_default": True,  "warning": "Slow above ~12 qubits.", "requires_key": None},
    "qaoa_originqc":         {"tier": "quantum_qpu",       "tier_label": "Quantum (real QPU)",     "tier_color": "error",   "recommended_default": False, "warning": "Real superconducting QPU — minutes per shot and uses Origin cloud quota. Keep problems ≤7 qubits.", "requires_key": "originqc"},
}

logger = logging.getLogger(__name__)

solve_bp = Blueprint("solve", __name__)


def _known_provider(name: str) -> bool:
    try:
        return name in list_providers()
    except Exception:
        return name in {"claude", "openai", "local"}


def _resolve_api_key(user_id: int, provider: str, payload: dict) -> tuple[str | None, tuple]:
    """Pick the right key for the request. Returns ``(key, ())`` on success
    or ``(None, (error_dict, status))`` on failure."""
    inline = payload.get("api_key")
    use_stored = bool(payload.get("use_stored_key", False))

    if inline and not use_stored:
        return inline, ()

    if provider == "local":
        # Ollama needs no key — accept "" to satisfy the provider interface.
        return "", ()

    if use_stored or not inline:
        cipher = get_api_key_ciphertext(user_id, provider)
        if cipher is None:
            return None, (
                {"error": f"no stored key for {provider!r}", "code": "NO_API_KEY"},
                402,
            )
        try:
            return decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET), ()
        except ValueError as e:
            return None, (
                {"error": f"stored key for {provider!r} is unreadable: {e}",
                 "code": "STORED_KEY_BROKEN"},
                402,
            )

    return inline, ()


def _launch_pipeline_in_background(
    *,
    job_id: str,
    user_id: int,
    problem_statement: str,
    provider_name: str,
    api_key: str,
    solvers: list[str],
) -> None:
    """Spawn a daemon thread that runs the async pipeline. Tests
    monkeypatch this with a synchronous stub."""
    bus = get_event_bus()
    orchestrator = Orchestrator()

    def target() -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                orchestrator.run(
                    job_id=job_id, user_id=user_id,
                    problem_statement=problem_statement,
                    provider_name=provider_name, api_key=api_key,
                    event_bus=bus,
                    solvers=solvers,
                )
            )
        except Exception:
            # The orchestrator already catches everything internally; if
            # we're here it's an event-loop or logging-level failure.
            # Don't crash the thread silently — log it so an operator can
            # find it in the journal.
            logger.exception("pipeline thread for job %s crashed", job_id)
        finally:
            loop.close()

    threading.Thread(target=target, daemon=True).start()


# ---- GET /api/solvers --------------------------------------------------------


@solve_bp.route("/solvers", methods=["GET"])
@login_required
def solvers_index():
    """List registered solvers + their tier metadata so the frontend's
    SolverPicker can render dynamically. Bootstraps the registry on
    first call (safe to call repeatedly — bootstrap is idempotent)."""
    from app.benchmarking.registry import bootstrap_default_solvers, list_solvers

    bootstrap_default_solvers()
    out = []
    for ident in list_solvers():
        meta = _SOLVER_TIERS.get(ident.name, {})
        out.append({
            "name": ident.name,
            "version": ident.version,
            "source": ident.source,
            "hardware": ident.hardware,
            "tier": meta.get("tier", "other"),
            "tier_label": meta.get("tier_label", "Other"),
            "tier_color": meta.get("tier_color", "default"),
            "recommended_default": meta.get("recommended_default", False),
            "warning": meta.get("warning"),
            "requires_key": meta.get("requires_key"),
        })
    return jsonify({"solvers": out})


def _validate_solvers(payload: dict) -> tuple[list[str], tuple]:
    """Normalize the ``solvers`` field. Returns ``(solvers, ())`` on success
    or ``(_, (error_dict, status))`` on failure.

    Defaults to the legacy single-solver behavior (``["gpu_sa"]``) when
    the field is absent. Empty list is treated the same as absent. A cap
    of 9 keeps a misbehaving client from queuing the whole registry on
    a single problem instance.
    """
    from app.benchmarking.registry import bootstrap_default_solvers, list_solvers

    raw = payload.get("solvers")
    if raw in (None, []):
        return ["gpu_sa"], ()
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        return [], ({"error": "solvers must be a list of strings",
                     "code": "BAD_SOLVERS"}, 400)
    # Dedup but preserve order — the first occurrence wins so the UI's
    # ordering choice is what the user sees in the result panel.
    seen: set[str] = set()
    requested = []
    for s in raw:
        if s not in seen:
            seen.add(s)
            requested.append(s)
    if len(requested) > 9:
        return [], ({"error": "at most 9 solvers per request",
                     "code": "TOO_MANY_SOLVERS"}, 400)

    bootstrap_default_solvers()
    registered = {ident.name for ident in list_solvers()}
    unknown = [s for s in requested if s not in registered]
    if unknown:
        return [], ({"error": f"unknown solver(s): {', '.join(unknown)}",
                     "code": "UNKNOWN_SOLVER"}, 400)
    return requested, ()


# ---- POST /api/solve ---------------------------------------------------------


@solve_bp.route("/solve", methods=["POST"])
@login_required
def solve():
    user = get_current_user()
    payload = request.get_json(silent=True) or {}

    statement = (payload.get("problem_statement") or "").strip()
    provider = (payload.get("provider") or "").strip()

    if not statement:
        return jsonify({"error": "problem_statement is required",
                        "code": "MISSING_FIELDS"}), 400
    if len(statement) > MAX_PROMPT_LENGTH:
        return jsonify({
            "error": f"problem_statement exceeds {MAX_PROMPT_LENGTH} characters",
            "code": "PROBLEM_TOO_LONG",
        }), 400
    if not _known_provider(provider):
        return jsonify({"error": f"unknown provider {provider!r}",
                        "code": "UNKNOWN_PROVIDER"}), 400

    api_key, err = _resolve_api_key(user["id"], provider, payload)
    if err:
        body, status = err
        return jsonify(body), status

    solvers, err = _validate_solvers(payload)
    if err:
        body, status = err
        return jsonify(body), status

    job_id = create_job(
        user["id"], statement, provider,
        solvers_requested=json.dumps(solvers),
    )
    _launch_pipeline_in_background(
        job_id=job_id, user_id=user["id"],
        problem_statement=statement,
        provider_name=provider, api_key=api_key or "",
        solvers=solvers,
    )

    # The Phase-5 frontend's solve store reads .job.id then opens SSE.
    job = get_job(job_id, user_id=user["id"])
    return jsonify({"success": True, "job": _public_job(job)})


# ---- GET /api/jobs[/...] -----------------------------------------------------


def _public_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    """Hand a job row to a client; decode JSON-string columns inline so
    the frontend doesn't have to parse strings-of-strings."""
    if job is None:
        return None
    out = dict(job)
    for column in (
        "cqm_json", "variable_registry", "validation_report", "solution",
        "solver_results", "solvers_requested",
    ):
        if out.get(column):
            try:
                out[column] = json.loads(out[column])
            except (TypeError, json.JSONDecodeError):
                pass
    return out


@solve_bp.route("/jobs", methods=["GET"])
@login_required
def jobs_index():
    user = get_current_user()
    page = int(request.args.get("page", "1") or "1")
    page_size = int(request.args.get("page_size", "20") or "20")
    is_admin = user["role"] == "admin" and request.args.get("scope") == "all"
    payload = list_jobs(
        user["id"], is_admin=is_admin, page=page, page_size=page_size
    )
    payload["jobs"] = [_public_job(j) for j in payload["jobs"]]
    return jsonify(payload)


@solve_bp.route("/jobs/<job_id>", methods=["GET"])
@login_required
def job_detail(job_id: str):
    user = get_current_user()
    is_admin = user["role"] == "admin"
    job = get_job(job_id, user_id=user["id"], is_admin=is_admin)
    if job is None:
        return jsonify({"error": "job not found", "code": "NOT_FOUND"}), 404
    return jsonify({"job": _public_job(job)})


@solve_bp.route("/jobs/<job_id>", methods=["DELETE"])
@login_required
def job_delete(job_id: str):
    user = get_current_user()
    is_admin = user["role"] == "admin"
    removed = delete_job(job_id, user_id=user["id"], is_admin=is_admin)
    if not removed:
        return jsonify({"error": "job not found", "code": "NOT_FOUND"}), 404
    return jsonify({"success": True})


# ---- GET /api/jobs/<id>/stream  (SSE) ---------------------------------------


@solve_bp.route("/jobs/<job_id>/stream", methods=["GET"])
@login_required
def job_stream(job_id: str):
    user = get_current_user()
    is_admin = user["role"] == "admin"
    job = get_job(job_id, user_id=user["id"], is_admin=is_admin)
    if job is None:
        return jsonify({"error": "job not found", "code": "NOT_FOUND"}), 404

    bus = get_event_bus()

    def generate():
        # SSE wire format: ``event: <type>\ndata: <json>\n\n``. The type
        # is always ``status``; the data dict carries the status string +
        # any extra fields (num_variables, solve_time_ms, error, …).
        for event in bus.subscribe(job_id):
            yield "event: status\n"
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx proxy buffering
        },
    )
