"""SQLite schema + thin model helpers.

Three tables ship in Phase 0:

* ``users``    — authentication subjects.
* ``api_keys`` — BYOK ciphertext per (user, provider). Encrypted via
                 ``app.crypto`` under ``KEY_ENCRYPTION_SECRET``.
* ``jobs``     — the optimization-pipeline run history. Phase 0 creates
                 the table; Phase 4 wires the orchestrator that writes
                 into it.

The helpers below are deliberately function-shaped (no ORM, no class
hierarchy) — the schema is small, the query surface is narrow, and a
plain ``sqlite3`` cursor is the simplest thing that works. When Phase 7
hardening or Phase 11 multi-tenancy demands more (migrations, indexes,
analytics), this is the file to grow.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from werkzeug.security import check_password_hash, generate_password_hash

from app.config import (
    DATABASE_PATH,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
    ROLE_ADMIN,
)

# ---- Connection ----


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Enforce foreign-key constraints on every connection (sqlite default is OFF).
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---- Schema bootstrap ----


def init_db() -> None:
    """Create tables and seed the default admin if no users exist yet."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                email         TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                display_name  TEXT,
                role          TEXT NOT NULL DEFAULT 'user',
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT NOT NULL,
                last_login    TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                provider      TEXT NOT NULL,
                encrypted_key BLOB NOT NULL,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE (user_id, provider)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id                  TEXT PRIMARY KEY,
                user_id             INTEGER NOT NULL,
                problem_statement   TEXT NOT NULL,
                provider            TEXT NOT NULL,
                status              TEXT NOT NULL,
                cqm_json            TEXT,
                variable_registry   TEXT,
                validation_report   TEXT,
                solution            TEXT,
                interpreted_solution TEXT,
                error               TEXT,
                num_variables       INTEGER,
                num_constraints     INTEGER,
                solve_time_ms       INTEGER,
                created_at          TEXT NOT NULL,
                completed_at        TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # Idempotent ALTERs for upgrades from earlier schemas.
        # ``ALTER TABLE ... ADD COLUMN`` is a no-op against a fresh
        # CREATE, so we only run each when the column is missing.
        existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(jobs)")}
        if "validation_report" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN validation_report TEXT")
        if "template_id" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN template_id TEXT")
        if "expected_optimum" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN expected_optimum REAL")
        # Phase 5D multi-solver: per-solver result map + the requested set.
        if "solver_results" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN solver_results TEXT")
        if "solvers_requested" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN solvers_requested TEXT")

        # 2026-06-30 approval gate — preflight summary the orchestrator
        # writes after compile + validate so the approval UI can show
        # lowered qubit count + per-QAOA-tier verdict before the user
        # commits to a solve.
        if "preflight" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN preflight TEXT")

        # 2026-07-01 per-solve overrides — currently used for the Origin
        # backend selector (Simulator / Wukong / Hanyuan). Persisted so
        # the approval-gate resume path picks up the same choice the
        # user made pre-pause. JSON-encoded dict shaped like
        # ``{solver_name: {param: value}}``.
        if "solver_params_overrides" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN solver_params_overrides TEXT")

        # 2026-07-01 plain-English summary — LLM rewrite of the
        # deterministic interpreter output, so the user gets an answer
        # phrased in the same vocabulary as their original question.
        # Generated best-effort during stage 5 by calling the
        # formulation provider's ``summarize_solution`` method; NULL
        # when the call failed or the provider doesn't support it (the
        # frontend falls back to the technical view in that case).
        if "plain_english_solution" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN plain_english_solution TEXT")

        # QML-6: per-provider state on QPU runs (e.g. Origin's
        # sample_index — which test point we evaluated). IBM ignores it
        # because it batches the whole test set. Idempotent ALTER so
        # databases created before QML-6 don't fail to upgrade.
        try:
            qpu_columns = {row[1] for row in cursor.execute("PRAGMA table_info(qml_qpu_runs)")}
            if qpu_columns and "submission_context" not in qpu_columns:
                cursor.execute("ALTER TABLE qml_qpu_runs ADD COLUMN submission_context TEXT")
        except Exception:
            # Table doesn't exist yet on a fresh DB — the CREATE below
            # will include the column.
            pass

        # QML-1: Quantum Machine Learning sister app. Distinct table so
        # the optimization "jobs" table stays focused. Schema mirrors
        # the optimization side where it overlaps (id, user_id, status,
        # timestamps, error) and adds QML-specific columns (dataset,
        # model, hyperparameters, training history, final metrics).
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS qml_jobs (
                id              TEXT PRIMARY KEY,
                user_id         INTEGER NOT NULL,
                dataset_id      TEXT NOT NULL,
                model           TEXT NOT NULL,
                status          TEXT NOT NULL,
                hyperparameters TEXT,
                training_history TEXT,
                metrics         TEXT,
                error           TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT,
                train_time_ms   INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # QML-5: real-QPU inference runs. Decoupled from qml_jobs so a
        # single trained VQC can be evaluated on several backends (IBM
        # ibmq_qasm_simulator + ibm_brisbane, say) for comparison. The
        # ``provider`` column will accept "ibmq" today and "originqc"
        # once QML-6 ships.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS qml_qpu_runs (
                id              TEXT PRIMARY KEY,
                qml_job_id      TEXT NOT NULL,
                user_id         INTEGER NOT NULL,
                provider        TEXT NOT NULL,
                backend_name    TEXT,
                shots           INTEGER NOT NULL,
                status          TEXT NOT NULL,
                cloud_job_id    TEXT,
                queue_position  INTEGER,
                live_status     TEXT,
                metrics         TEXT,
                error           TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT,
                wall_time_ms    INTEGER,
                submission_context TEXT,
                FOREIGN KEY (qml_job_id) REFERENCES qml_jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            _seed_default_admin(conn)
    finally:
        conn.close()


def _seed_default_admin(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO users (username, email, password_hash, display_name, role, created_at)
        VALUES (?, NULL, ?, ?, ?, ?)
        """,
        (
            DEFAULT_ADMIN_USERNAME,
            generate_password_hash(DEFAULT_ADMIN_PASSWORD),
            "Administrator",
            ROLE_ADMIN,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


# ---- User helpers ----


def create_user(
    username: str,
    password: str,
    *,
    email: str | None = None,
    display_name: str | None = None,
    role: str = "user",
) -> dict[str, Any]:
    """Insert a new user. Raises ``ValueError`` if username or email already
    exists (caller maps that to HTTP 409)."""
    conn = get_db_connection()
    try:
        try:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, display_name, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    generate_password_hash(password),
                    display_name or username,
                    role,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"username or email already taken: {e}") from e
        # Re-query so we return whatever AUTOINCREMENT picked.
        row = conn.execute(
            "SELECT id, username, email, display_name, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def verify_user(username: str, password: str) -> dict[str, Any] | None:
    """Return the user row (as a dict) if credentials check out and the
    account is active; ``None`` otherwise."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None or not row["is_active"]:
            return None
        if not check_password_hash(row["password_hash"], password):
            return None
        # Touch last_login on success.
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), row["id"]),
        )
        conn.commit()
        return dict(row)
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id, username, email, display_name, role, is_active "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    """Verify the current password, then store a fresh hash for the new one.
    Returns True on success, False if the current password was wrong."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None or not check_password_hash(row["password_hash"], current_password):
            return False
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(new_password), user_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


# ---- Job helpers (used by Phase 4; declared here so all schema-aware code
# lives in one place). ----


def create_job(
    user_id: int,
    problem_statement: str,
    provider: str,
    *,
    template_id: str | None = None,
    expected_optimum: float | None = None,
    solvers_requested: str | None = None,
) -> str:
    """Insert a fresh job row in ``queued`` state. Returns the new job ID.

    ``template_id`` and ``expected_optimum`` are populated only when the
    job was launched from a template (Phase 5B route
    ``POST /api/solve/from-template/<id>``). The detail view's match
    badge consumes them to compare the solver's actual answer to the
    template's documented one.

    ``solvers_requested`` is a JSON-encoded list of solver names from
    Phase 5D's multi-solver flow. ``NULL`` means the legacy path (the
    orchestrator picks GPU SA on its own).
    """
    job_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO jobs (
                id, user_id, problem_statement, provider, status,
                template_id, expected_optimum, solvers_requested, created_at
            )
            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)
            """,
            (
                job_id, user_id, problem_statement, provider,
                template_id, expected_optimum, solvers_requested,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    conn = get_db_connection()
    try:
        keys = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE jobs SET {keys} WHERE id = ?",
            (*fields.values(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_job(
    job_id: str,
    *,
    user_id: int | None = None,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    """Fetch a job. Non-admins can only see their own; admins see any."""
    conn = get_db_connection()
    try:
        if is_admin:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_jobs(
    user_id: int | None,
    *,
    is_admin: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Paginated job list. Non-admins are filtered by ``user_id``.
    Returns ``{"jobs": [...], "page": <n>, "page_size": <n>, "total": <n>}``.
    """
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 100))
    offset = (page - 1) * page_size

    conn = get_db_connection()
    try:
        if is_admin:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM jobs WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, page_size, offset),
            ).fetchall()
        return {
            "jobs": [dict(r) for r in rows],
            "page": page,
            "page_size": page_size,
            "total": int(total),
        }
    finally:
        conn.close()


def delete_job(
    job_id: str, *, user_id: int | None = None, is_admin: bool = False
) -> bool:
    """Delete a job. Returns ``True`` if a row was removed."""
    conn = get_db_connection()
    try:
        if is_admin:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        else:
            cur = conn.execute(
                "DELETE FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---- BYOK API-key helpers ----


def list_api_keys(user_id: int) -> list[dict[str, Any]]:
    """List a user's stored BYOK providers. Never returns ciphertext —
    the route layer can't accidentally leak keys."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT provider, created_at FROM api_keys WHERE user_id = ? "
            "ORDER BY provider",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def put_api_key(user_id: int, provider: str, encrypted_key: bytes) -> None:
    """Upsert the ciphertext for ``(user_id, provider)``."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO api_keys (user_id, provider, encrypted_key, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (user_id, provider)
            DO UPDATE SET encrypted_key = excluded.encrypted_key,
                          created_at = excluded.created_at
            """,
            (user_id, provider, encrypted_key, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_api_key_ciphertext(user_id: int, provider: str) -> bytes | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT encrypted_key FROM api_keys WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        ).fetchone()
        return bytes(row["encrypted_key"]) if row else None
    finally:
        conn.close()


def delete_api_key(user_id: int, provider: str) -> bool:
    """Remove a stored key. Returns ``True`` if a row was deleted."""
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "DELETE FROM api_keys WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---- QML job helpers (sister-app counterpart to the optimization jobs API) --


def create_qml_job(
    user_id: int,
    dataset_id: str,
    model: str,
    *,
    hyperparameters: str | None = None,
) -> str:
    """Insert a QML training job in ``queued`` state. Returns the new id.

    ``hyperparameters`` is a JSON-encoded blob so the schema doesn't have
    to grow a column every time a model adds a knob.
    """
    job_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO qml_jobs (
                id, user_id, dataset_id, model, status,
                hyperparameters, created_at
            )
            VALUES (?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                job_id, user_id, dataset_id, model,
                hyperparameters, datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return job_id


def update_qml_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    conn = get_db_connection()
    try:
        keys = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE qml_jobs SET {keys} WHERE id = ?",
            (*fields.values(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_qml_job(
    job_id: str,
    *,
    user_id: int | None = None,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        if is_admin:
            row = conn.execute(
                "SELECT * FROM qml_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM qml_jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_qml_jobs(
    user_id: int | None,
    *,
    is_admin: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 100))
    offset = (page - 1) * page_size

    conn = get_db_connection()
    try:
        if is_admin:
            total = conn.execute("SELECT COUNT(*) FROM qml_jobs").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM qml_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) FROM qml_jobs WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM qml_jobs WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, page_size, offset),
            ).fetchall()
        return {
            "jobs": [dict(r) for r in rows],
            "page": page,
            "page_size": page_size,
            "total": int(total),
        }
    finally:
        conn.close()


def delete_qml_job(
    job_id: str, *, user_id: int | None = None, is_admin: bool = False
) -> bool:
    conn = get_db_connection()
    try:
        if is_admin:
            cur = conn.execute("DELETE FROM qml_jobs WHERE id = ?", (job_id,))
        else:
            cur = conn.execute(
                "DELETE FROM qml_jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---- QML real-QPU run helpers (QML-5+) -------------------------------------


def create_qml_qpu_run(
    qml_job_id: str,
    user_id: int,
    provider: str,
    shots: int,
    *,
    backend_name: str | None = None,
    submission_context: str | None = None,
) -> str:
    """Create a queued QPU-inference run linked to ``qml_job_id``.

    Status starts as ``queued`` and transitions through ``submitted`` →
    ``running`` → ``complete`` / ``error`` as the cloud job progresses.
    The submitter populates ``cloud_job_id`` once the cloud accepts
    the submission. ``submission_context`` is a JSON blob for any
    per-provider state — e.g. Origin stores the test-point index it
    evaluated, so the materializer can reconstruct the predicted label.
    """
    run_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO qml_qpu_runs (
                id, qml_job_id, user_id, provider, backend_name,
                shots, status, created_at, submission_context
            )
            VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                run_id, qml_job_id, user_id, provider, backend_name,
                int(shots), datetime.utcnow().isoformat(),
                submission_context,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def update_qml_qpu_run(run_id: str, **fields: Any) -> None:
    if not fields:
        return
    conn = get_db_connection()
    try:
        keys = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE qml_qpu_runs SET {keys} WHERE id = ?",
            (*fields.values(), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_qml_qpu_run(
    run_id: str,
    *,
    user_id: int | None = None,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        if is_admin:
            row = conn.execute(
                "SELECT * FROM qml_qpu_runs WHERE id = ?", (run_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM qml_qpu_runs WHERE id = ? AND user_id = ?",
                (run_id, user_id),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_qml_qpu_runs_for_job(qml_job_id: str) -> list[dict[str, Any]]:
    """All QPU runs against a single parent VQC training job, newest first."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM qml_qpu_runs WHERE qml_job_id = ? "
            "ORDER BY created_at DESC",
            (qml_job_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_qml_qpu_run_by_cloud_job_id(cloud_job_id: str) -> dict[str, Any] | None:
    """Lookup a run by its cloud-side job id (used by the poller)."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM qml_qpu_runs WHERE cloud_job_id = ?",
            (cloud_job_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_unsettled_qml_qpu_runs() -> list[dict[str, Any]]:
    """Runs the poller still needs to check on. Used by the periodic
    materializer to drive non-terminal runs to completion."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM qml_qpu_runs "
            "WHERE status IN ('queued', 'submitted', 'running') "
            "ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
