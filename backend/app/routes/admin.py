"""Phase 7 — Admin read-only endpoints.

Surfaces operational visibility for the ``admin`` role: who's using the
platform, what's been submitted recently, what's queued, what credentials
are on file (without leaking any plaintext).

All endpoints are GET-only and require the ``admin_required`` gate. No
mutation endpoints in this first cut — admins can already manage their
own user via the existing keys/jobs routes; deletion / suspension of
other users would warrant a separate review and live in a future Phase 7B.

Endpoints
~~~~~~~~~

* ``GET /api/admin/users``
    Every registered user, with their role, last-login, created-at, and
    the set of BYOK providers they have on file (provider names only —
    ciphertext + plaintext never echoed).

* ``GET /api/admin/jobs?page=N&page_size=M&status=...``
    Cross-user paginated job list, newest first, with optional status
    filter (``solving``, ``complete``, ``error``, ``queued``). Includes
    user_id + username for triage.

* ``GET /api/admin/overview``
    Headline counters: user count, total jobs, jobs-by-status, jobs in
    the last 24 h, currently-pending cloud jobs. Powers the admin
    landing page.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request

from app.auth import admin_required
from app.models import get_db_connection, list_jobs

admin_bp = Blueprint("admin", __name__)


def _row_or_none(row):
    return dict(row) if row else None


# ---- GET /users -------------------------------------------------------------


@admin_bp.route("/users", methods=["GET"])
@admin_required
def admin_users():
    conn = get_db_connection()
    try:
        users = conn.execute(
            "SELECT id, username, email, display_name, role, is_active, "
            "created_at, last_login FROM users ORDER BY id ASC"
        ).fetchall()
        # Pull the BYOK providers each user has on file. Aggregated per
        # user_id in one query so we don't N+1 against the api_keys table.
        keys_rows = conn.execute(
            "SELECT user_id, provider FROM api_keys ORDER BY user_id, provider"
        ).fetchall()
        keys_by_user: dict[int, list[str]] = {}
        for r in keys_rows:
            keys_by_user.setdefault(int(r["user_id"]), []).append(r["provider"])

        # Job counts per user — same idea: one query, group in Python.
        job_count_rows = conn.execute(
            "SELECT user_id, COUNT(*) AS n FROM jobs GROUP BY user_id"
        ).fetchall()
        jobs_by_user = {int(r["user_id"]): int(r["n"]) for r in job_count_rows}

        out = []
        for u in users:
            d = dict(u)
            out.append({
                "id": d["id"],
                "username": d["username"],
                "email": d.get("email"),
                "display_name": d.get("display_name"),
                "role": d["role"],
                "is_active": bool(d["is_active"]),
                "created_at": d["created_at"],
                "last_login": d.get("last_login"),
                "providers_on_file": keys_by_user.get(int(d["id"]), []),
                "total_jobs": jobs_by_user.get(int(d["id"]), 0),
            })
        return jsonify({"users": out, "total": len(out)})
    finally:
        conn.close()


# ---- GET /jobs --------------------------------------------------------------


@admin_bp.route("/jobs", methods=["GET"])
@admin_required
def admin_jobs():
    page = int(request.args.get("page", "1") or "1")
    page_size = int(request.args.get("page_size", "30") or "30")
    status_filter = (request.args.get("status") or "").strip().lower() or None

    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    offset = (page - 1) * page_size

    conn = get_db_connection()
    try:
        if status_filter:
            total = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?",
                (status_filter,),
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT j.id, j.user_id, j.problem_statement, j.provider, "
                "j.status, j.created_at, j.completed_at, j.solve_time_ms, "
                "j.template_id, u.username "
                "FROM jobs j LEFT JOIN users u ON u.id = j.user_id "
                "WHERE j.status = ? "
                "ORDER BY j.created_at DESC LIMIT ? OFFSET ?",
                (status_filter, page_size, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = conn.execute(
                "SELECT j.id, j.user_id, j.problem_statement, j.provider, "
                "j.status, j.created_at, j.completed_at, j.solve_time_ms, "
                "j.template_id, u.username "
                "FROM jobs j LEFT JOIN users u ON u.id = j.user_id "
                "ORDER BY j.created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()

        jobs = []
        for r in rows:
            d = dict(r)
            jobs.append({
                "id": d["id"],
                "user_id": d["user_id"],
                "username": d.get("username"),
                "problem_statement": (d.get("problem_statement") or "")[:200],
                "provider": d["provider"],
                "status": d["status"],
                "created_at": d["created_at"],
                "completed_at": d.get("completed_at"),
                "solve_time_ms": d.get("solve_time_ms"),
                "template_id": d.get("template_id"),
            })
        return jsonify({
            "jobs": jobs,
            "total": int(total),
            "page": page,
            "page_size": page_size,
            "status_filter": status_filter,
        })
    finally:
        conn.close()


# ---- GET /overview ----------------------------------------------------------


@admin_bp.route("/overview", methods=["GET"])
@admin_required
def admin_overview():
    conn = get_db_connection()
    try:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active_user_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE is_active = 1"
        ).fetchone()[0]
        admin_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin'"
        ).fetchone()[0]

        job_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        # Group jobs by status in one pass.
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        by_status = {r["status"]: int(r["n"]) for r in status_rows}

        # Jobs in last 24 h. SQLite ISO-8601 timestamps sort lexically.
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        recent_count = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()[0]

        # Top-3 providers by job count (LLM provider).
        provider_rows = conn.execute(
            "SELECT provider, COUNT(*) AS n FROM jobs GROUP BY provider "
            "ORDER BY n DESC LIMIT 5"
        ).fetchall()
        top_providers = [
            {"provider": r["provider"], "count": int(r["n"])}
            for r in provider_rows
        ]

        # Pending cloud jobs from the JSON file.
        pending_cloud_count = 0
        try:
            from app.benchmarking import pending_jobs as pj
            pending_cloud_count = len(pj.list_pending())
        except Exception:
            pass

        return jsonify({
            "users": {
                "total": int(user_count),
                "active": int(active_user_count),
                "admins": int(admin_count),
            },
            "jobs": {
                "total": int(job_count),
                "by_status": by_status,
                "last_24h": int(recent_count),
                "top_providers": top_providers,
            },
            "pending_cloud_jobs": int(pending_cloud_count),
        })
    finally:
        conn.close()
